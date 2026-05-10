"""CBIC v2 re-ingest dashboard.
Live-updating single-page app. Polls /state.json every 2s; only changed
sections re-render (no full page refresh, no scroll jump).
Usage: python progress_server.py  ->  http://127.0.0.1:8765/
"""
import json, time, subprocess, os, threading, re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

STATE = Path(__file__).parent / "progress_state.json"
SSH_HOST = os.environ.get("RIG_SSH", "root@192.168.1.107")
SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519_rig")
LIVE_CACHE = {"text": "(probe pending)", "ts": 0}
LIVE_TTL = 3  # seconds — slightly longer than the 1.6s GPU sample window so we get fresh samples each cycle
STATE_LOCK = threading.Lock()

def ssh(cmd, timeout=15):
    try:
        r = subprocess.run(
            ["ssh","-o","ControlPath=none","-o","ConnectTimeout=5","-i",SSH_KEY,SSH_HOST,cmd],
            capture_output=True, text=True, timeout=timeout)
        return r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return "", f"timeout({timeout}s)"
    except Exception as e:
        return "", str(e)

def writer_loop():
    """Autonomous observer: polls rig every 30s, writes state.json."""
    while True:
        try:
            update_state_from_rig()
        except Exception as e:
            print(f"[writer] error: {e}")
        time.sleep(30)

def update_state_from_rig():
    """Probe rig for: cbic_v2 pts, ingest_v2 PID + elapsed, recent log lines, gate result files.
    Translate into state.json fields. Idempotent — writes only if changed."""
    out, _ = ssh(
        "echo ===PTS===; curl -s http://127.0.0.1:6343/collections/cbic_v2 2>/dev/null | "
        "python3 -c 'import sys,json;d=json.load(sys.stdin);print(d[\"result\"][\"points_count\"])' 2>/dev/null;"
        "echo ===INGEST===; ps -ef | grep -E 'ingest_v2.py|run_batch_loop' | grep -v grep | "
        "awk '{print $2,$NF,substr($0,index($0,$5))}';"
        "echo ===LOG===; tail -3 /tmp/ingest_loop.log 2>/dev/null;"
        "echo ===GATES===; for f in /opt/indian-legal-ai/data/eval/cp*_g*.json /opt/indian-legal-ai/data/eval/cp1_smokes.json; do "
        "  [ -f \"$f\" ] && python3 -c \"import json,os;d=json.load(open('$f'));n=os.path.basename('$f');print(n, json.dumps({k:d.get(k) for k in ['pass_gate','combined_recall','refusal_rate','errors','n']} if isinstance(d,dict) else {'n':len(d)}))\";"
        "done"
    )
    sections = {}
    cur = None
    for line in out.splitlines():
        m = re.match(r"^===(\w+)===$", line.strip())
        if m: cur = m.group(1); sections[cur] = []; continue
        if cur: sections[cur].append(line)

    pts = None
    if sections.get("PTS"):
        for ln in sections["PTS"]:
            if ln.strip().isdigit(): pts = int(ln.strip()); break

    ingest_proc = None
    for ln in sections.get("INGEST", []):
        ln = ln.strip()
        if "ingest_v2.py" in ln:
            parts = ln.split()
            if parts and parts[0].isdigit(): ingest_proc = {"pid": int(parts[0]), "cmd": ln}
            break

    log_tail = "\n".join(sections.get("LOG", [])[-3:])

    # Detect current batch by looking at FULL ingest_loop.log for last "BATCH N START" line
    # (separately from log_tail which is just last 3 lines)
    out2, _ = ssh("grep -E '=== BATCH [0-9]+ (START|DONE) ===|CP-[0-9] GATES (START|DONE)' /tmp/ingest_loop.log 2>/dev/null | tail -8", timeout=8)
    cur_batch_id = None
    cur_step = None
    batches_done = 0
    for ln in out2.splitlines():
        m = re.search(r"BATCH (\d+) START", ln)
        if m: cur_batch_id = int(m.group(1)); cur_step = f"E (Phase 2 chunk) — batch {cur_batch_id} ingesting"
        m = re.search(r"BATCH (\d+) DONE", ln)
        if m:
            batches_done = max(batches_done, int(m.group(1)))
            cur_step = f"O — batch {m.group(1)} done"
        m = re.search(r"CP-(\d+) GATES START", ln)
        if m: cur_step = f"L — CP-{m.group(1)} gates running"
        m = re.search(r"CP-(\d+) GATES DONE", ln)
        if m: cur_step = f"L done — CP-{m.group(1)} complete"

    with STATE_LOCK:
        try:
            st = json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            return

        changed = False
        # Update pts + batches_done (from log, more reliable than pts heuristic)
        if pts and st["global"].get("points") != pts:
            st["global"]["points"] = pts
            changed = True
        if batches_done > st["global"].get("batches_done", 0):
            st["global"]["batches_done"] = batches_done
            changed = True
        # Update current_batch.id + current_step from log
        if cur_batch_id and cur_batch_id != st.get("current_batch", {}).get("id"):
            st["current_batch"]["id"] = cur_batch_id
            changed = True
        if cur_step and cur_step != st.get("current_batch", {}).get("current_step"):
            st["current_batch"]["current_step"] = cur_step
            changed = True
        # Sync batches[] status with rig truth
        for b in st.get("batches", []):
            bid = b.get("id")
            if not bid: continue
            new_status = "pending"
            if bid <= batches_done:
                new_status = "done"
            elif bid == cur_batch_id:
                new_status = "running"
            if b.get("status") != new_status:
                b["status"] = new_status
                changed = True
            if new_status == "pending" and b.get("steps"):
                b["steps"] = {}
                changed = True

        # Per-step (A-O) timing for current batch — simple presence-based detection
        if cur_batch_id:
            log_out, _ = ssh(
                "grep -E '\\[phase|RESOURCE_OK|PREFLIGHT_OK|built [0-9]+ doc_ids|ingest_v2 PID|reconcile|post-ingest pts|=== BATCH|=== CP-|\\[lint\\]|snapshot|warmup|HALT' /tmp/ingest_loop.log 2>/dev/null | tail -150",
                timeout=8)
            # Use simple substring/regex flags
            now_ts = int(time.time())
            steps = {}
            text = log_out
            if "RESOURCE_OK" in text or "PREFLIGHT_OK" in text:
                steps["A"] = {"status":"done","duration_s":1,"gpus":"all (probe)"}
            if "built " in text and "doc_ids" in text:
                steps["B"] = {"status":"done","duration_s":2}
                steps["C"] = {"status":"done","duration_s":1}
            if "ingest_v2 PID=" in text:
                steps["D"] = {"status":"done","duration_s":1}
                steps["E"] = {"status":"running","gpus":"2 (qwen3 classify)"}
            if "[phase2 DONE]" in text or re.search(r"phase 2.*done", text, re.I):
                steps["E"] = {"status":"done","gpus":"2 (qwen3) + 4-6 (chunker)"}
            elif "[phase2]" in text:
                # phase2 still running — find latest progress
                pm = re.findall(r"\[phase2\] (\d+)/(\d+) docs.*?rate=([0-9.]+)", text)
                if pm:
                    cur, total, rate = pm[-1]
                    steps["E"] = {"status":"running","gpus":"2 (qwen3) + 4-6","detail":f"{cur}/{total} docs @ {rate} doc/s"}
            if "[phase3-5]" in text or "[phase3_4_5]" in text or "phase3" in text.lower():
                if "E" in steps and steps["E"].get("status")!="done":
                    steps["E"]["status"]="done"
                steps["F"] = {"status":"running","gpus":"0,1,3,4,5,6 (BGE-M3 pool)"}
                pm = re.findall(r"\[phase3[-_]?[0-9_]*\].*?rate=([0-9.]+)", text)
                if pm: steps["F"]["detail"] = f"{pm[-1]} ch/s"
            if "reconcile" in text.lower() and ("PASS" in text or "ok" in text.lower()):
                if "F" in steps: steps["F"]["status"]="done"
                steps["G"] = {"status":"done","duration_s":2}
            if "post-ingest pts=" in text:
                m_pts = re.findall(r"post-ingest pts=(\d+).*?\(\+(\d+)\)", text)
                if m_pts:
                    pts_val, delta = m_pts[-1]
                    steps["G"] = {"status":"done","detail":f"pts={pts_val} (+{delta})"}
                    steps["H"] = {"status":"done","duration_s":2}
            if "[lint]" in text:
                steps["H"] = {"status":"done"}
                # I=spot check, J=manifest, K=qdrant — hard to detect individually; mark all done if lint ran
                steps["I"] = {"status":"done"}
            if "snapshot" in text.lower() or "snapshots done" in text:
                steps["J"] = {"status":"done"}
                steps["K"] = {"status":"done"}
            if f"=== CP-" in text and "START" in text:
                steps["L"] = {"status":"running","gpus":"all (gates)"}
            if "=== CP-" in text and "DONE" in text:
                steps["L"] = {"status":"done"}
                steps["M"] = {"status":"done"}
                steps["N"] = {"status":"done"}
            if f"=== BATCH {cur_batch_id} DONE ===" in text:
                steps["O"] = {"status":"done"}
                # Mark all preceding as done
                for k in "ABCDEFGHIJKLMN":
                    steps.setdefault(k, {"status":"done"})
            now = int(time.time())
            for ln in log_out.splitlines():
                m = re.match(r"^\[(\d{2}):(\d{2}):(\d{2})\] (.*)$", ln)
                if not m: continue
                hh, mm, ss, body = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
                # crude: today's epoch
                tm = time.localtime()
                ts = int(time.mktime(time.struct_time((tm.tm_year, tm.tm_mon, tm.tm_mday, hh, mm, ss, 0, 0, -1))))
                if "RESOURCE_OK" in body or "preflight" in body.lower(): steps.setdefault("A", {})["status"]="done"; steps["A"].setdefault("ts_start",ts); steps["A"]["ts_end"]=ts
                if "built" in body and "doc_ids" in body: steps["B"]={"status":"done","ts_start":ts,"ts_end":ts}; steps["C"]={"status":"done","ts_start":ts,"ts_end":ts}
                if "pre-ingest" in body or "ingest_v2 PID" in body: steps["D"]={"status":"done","ts_start":ts,"ts_end":ts}; steps["E"]={"status":"running","ts_start":ts,"gpus":"4,5,6 (chunker)"}
                if "[phase2]" in body or "phase2" in body.lower(): steps.setdefault("E",{})["status"]="running"; steps["E"]["gpus"]="2 (qwen3 classify) + 4,5,6"
                if "[phase3-5]" in body or "[phase3_4_5]" in body:
                    if "E" in steps: steps["E"]["status"]="done"; steps["E"]["ts_end"]=ts
                    steps["F"]={"status":"running","ts_start":ts,"gpus":"0,1,3,4,5,6 (BGE-M3 pool)"}
                if "reconcile" in body.lower():
                    if "F" in steps: steps["F"]["status"]="done"; steps["F"]["ts_end"]=ts
                    steps["G"]={"status":"running","ts_start":ts}
                if "post-ingest pts" in body:
                    if "G" in steps: steps["G"]["status"]="done"; steps["G"]["ts_end"]=ts
                    steps["H"]={"status":"running","ts_start":ts}
                if "[lint]" in body and "wrote" in body: steps["H"]={"status":"done","ts_start":ts,"ts_end":ts}; steps.setdefault("I",{"status":"done","ts_start":ts,"ts_end":ts})
                if "snapshot" in body.lower(): steps.setdefault("J",{"status":"done","ts_start":ts,"ts_end":ts}); steps.setdefault("K",{"status":"done","ts_start":ts,"ts_end":ts})
                if "CP-" in body and "START" in body: steps["L"]={"status":"running","ts_start":ts,"gpus":"all (gates)"}
                if "CP-" in body and "DONE" in body and "L" in steps: steps["L"]["status"]="done"; steps["L"]["ts_end"]=ts
                if f"BATCH {cur_batch_id} DONE" in body: steps["O"]={"status":"done","ts_start":ts,"ts_end":ts}
            # Compute durations
            tgt = next((b for b in st["batches"] if b["id"]==cur_batch_id), None)
            if tgt and tgt.get("steps") != steps:
                tgt["steps"] = steps
                changed = True

        # ---- Pull latest gate result files from rig and populate trust block ----
        gate_out, _ = ssh(
            "python3 -c '"
            "import json, os, glob;"
            "out = {};"
            "for f in glob.glob(\"/opt/indian-legal-ai/data/eval/cp*_g*.json\") + glob.glob(\"/opt/indian-legal-ai/data/eval/cp*smokes*.json\"):"
            "    try:"
            "        d = json.load(open(f));"
            "        n = os.path.basename(f);"
            "        out[n] = ({k: d.get(k) for k in [\"n\",\"recall_at_k\",\"combined_recall\",\"refusal_rate\",\"hits\",\"errors\",\"pass_gate\"]} if isinstance(d, dict) else {\"n\": len(d)});"
            "    except Exception:"
            "        pass;"
            "print(json.dumps(out))"
            "'", timeout=8)
        latest = {}
        try:
            # Output may have other text; find first { and parse
            idx = gate_out.find("{")
            if idx >= 0:
                latest = json.loads(gate_out[idx:].strip().split("\n")[0])
        except Exception:
            latest = {}

        def pick(prefix):
            # Prefer cp1v2 (this restart) over cp2/cp1
            for k in latest:
                if k.startswith("cp1v2_") and prefix in k: return latest[k]
            for k in latest:
                if prefix in k: return latest[k]
            return None

        trust = st.setdefault("trust", {})
        for gate, prefix in [("G1","g1"),("G3","g3"),("G4","g4")]:
            d = pick(prefix)
            if not d: continue
            n = d.get("n")
            score = d.get("recall_at_k") or d.get("combined_recall") or d.get("refusal_rate")
            passing = d.get("pass_gate")
            if score is None: continue
            new_latest = f"{score:.4f} ({d.get('hits','?')}/{n}, errors={d.get('errors',0)})"
            cur = trust.get(gate, {})
            if cur.get("latest") != new_latest or cur.get("passing") != passing:
                trust[gate] = {**cur, "latest": new_latest, "passing": bool(passing) if passing is not None else None,
                               "threshold": cur.get("threshold", ">=0.95")}
                changed = True
        # Smokes
        sm = latest.get("cp1_smokes.json")
        if sm and isinstance(sm, dict):
            n = sm.get("n", 0)
            if n:
                # not a "trust gate" but still show a count somewhere — append to agent_log
                pass

        # Update active jobs
        new_active = []
        if ingest_proc:
            # Try to find existing entry by pid
            found = next((a for a in st.get("active",[]) if a.get("pid")==ingest_proc["pid"]), None)
            started_ts = found.get("started_ts") if found else int(time.time())
            new_active.append({
                "name": f"ingest_v2 batch ~{cur_batch_id or '?'}",
                "status": "running",
                "started_ts": started_ts,
                "pid": ingest_proc["pid"],
                "eta": "~6-7 min",
                "detail": f"PID {ingest_proc['pid']} · {pts or '?'} pts in cbic_v2",
            })
        if st.get("active") != new_active:
            st["active"] = new_active
            changed = True

        # Append heartbeat to agent_log if we haven't logged in a while
        log = st.get("agent_log", [])
        last_hb = next((e for e in reversed(log) if e.get("source")=="dashboard_writer"), None)
        if not last_hb or (time.time() - last_hb.get("epoch", 0)) > 180:
            entry = {
                "ts": time.strftime("%H:%M:%S"),
                "epoch": int(time.time()),
                "source": "dashboard_writer",
                "severity": "P2",
                "what": f"Heartbeat: cbic_v2={pts} pts · ingest={'PID '+str(ingest_proc['pid']) if ingest_proc else 'idle'} · batch~{cur_batch_id or '?'}",
                "rec": None,
                "evidence": log_tail.strip()[:300] or "(no log lines)",
                "guardrail_ok": True,
            }
            log.append(entry)
            st["agent_log"] = log[-100:]  # cap at 100 entries
            changed = True

        if changed:
            STATE.write_text(json.dumps(st, indent=2), encoding="utf-8")

def fetch_live():
    """Single SSH probe: GPU busy%, process-per-GPU, current phase, recent events. Timeout 12s for parallel GPU sampling."""
    try:
        r = subprocess.run(
            ["ssh","-o","ControlPath=none","-o","ConnectTimeout=5","-i",SSH_KEY,SSH_HOST,
             # GPU busy% — sample each GPU 6× over 1.2s in PARALLEL (captures embed bursts).
             "GPU_TMP=$(mktemp); "
             "for i in 0 1 2 3 4 5 6; do "
             "  ( f=/sys/class/drm/card$i/device/gpu_busy_percent; "
             "    if [ -r $f ]; then "
             "      mx=0; for n in 1 2 3 4 5 6; do v=$(cat $f); [ $v -gt $mx ] && mx=$v; sleep 0.2; done; "
             "      echo \"$i,$mx\" >> $GPU_TMP; "
             "    else echo \"$i,-1\" >> $GPU_TMP; fi ) & "
             "done; "
             # GPU procs — read GGML_VK_VISIBLE_DEVICES env per process. That's the
             # GPU(s) actually being used (Vulkan opens all renderDs for enumeration,
             # so FD scanning is misleading).
             "echo ---GPU_PROCS---; for pid in $(pgrep -f 'ingest_v2.py|llama-server'); do "
             "  cmd=$(tr '\\0' ' ' < /proc/$pid/cmdline 2>/dev/null | head -c 60); "
             "  vis=$(tr '\\0' '\\n' < /proc/$pid/environ 2>/dev/null | grep -E '^GGML_VK_VISIBLE_DEVICES=|^EMBED_GPUS=' | head -1 | cut -d= -f2); "
             "  for i in $(echo $vis | tr ',' ' '); do "
             "    printf '%s,%s,%s\\n' \"$i\" \"$pid\" \"$(echo $cmd | head -c 30)\"; "
             "  done; "
             "done | sort -u; "
             # Current phase: parse last meaningful ingest line (skip noise)
             "echo ---PHASE---; "
             "tail -200 /tmp/ingest_loop.log 2>/dev/null | "
             "grep -E '\\[phase[0-9]\\]|\\[phase[0-9]_[0-9]_[0-9]\\]|\\[G1\\]|\\[G3\\]|\\[G4\\]|=== BATCH|=== CP-|post-ingest pts|reconcile|HALT|RESOURCE_OK|warmup' | "
             "tail -8; "
             # Qdrant pts
             "echo ---PTS---; "
             "curl -s -m 3 http://127.0.0.1:6343/collections/cbic_v2 2>/dev/null | "
             "python3 -c 'import sys,json;d=json.load(sys.stdin);print(d[\"result\"][\"points_count\"])' 2>/dev/null || echo 0; "
             # Reranker traffic
             "echo ---RERANK_RECENT---; "
             "journalctl -u cbic-rag-api --since '60 sec ago' --no-pager 2>/dev/null | "
             "grep -cE '(\\[rerank\\]|/rerank|/retrieve.*200|POST /retrieve)';"
             "wait; echo ---GPU---; sort -n $GPU_TMP; rm -f $GPU_TMP"],
            capture_output=True, text=True, timeout=12)
        LIVE_CACHE["text"] = (r.stdout or "") + ("\n[stderr]\n"+r.stderr if r.stderr else "")
    except Exception as e:
        LIVE_CACHE["text"] = f"(SSH probe failed: {e})"
    LIVE_CACHE["ts"] = time.time()

def maybe_refresh_live():
    if time.time() - LIVE_CACHE["ts"] > LIVE_TTL:
        threading.Thread(target=fetch_live, daemon=True).start()

INDEX = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<title>CBIC v2 re-ingest</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--border:#30363d;--fg:#c9d1d9;--mute:#8b949e;
 --blue:#58a6ff;--green:#3fb950;--red:#f85149;--yellow:#d29922;--purple:#a371f7}
*{box-sizing:border-box}
body{font:14px/1.45 -apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:18px;max-width:1280px;margin-inline:auto}
h1{font-size:18px;margin:0 0 4px;color:var(--blue)}
h2{font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute);margin:20px 0 8px;font-weight:600}
.muted{color:var(--mute);font-size:12px}
.row{display:flex;gap:14px;flex-wrap:wrap}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px 14px}
.now{background:linear-gradient(120deg,#161b22,#1c2129);border-left:3px solid var(--yellow)}
.now b{color:#fff;font-size:15px}
.now .timer{font-variant-numeric:tabular-nums;color:var(--yellow);font-weight:600}

/* progress bar 1..10 */
.bar{display:flex;gap:4px;margin:8px 0 6px}
.bar > div{flex:1;height:24px;border-radius:4px;background:#21262d;display:flex;align-items:center;justify-content:center;font-size:11px;color:var(--mute);position:relative}
.bar > div.done{background:#0a3d22;color:#7ee2a8}
.bar > div.run{background:#3d2f0a;color:#f0c674;animation:pulse 1.6s infinite}
.bar > div.fail{background:#3d0a0a;color:#ff8b80}
.bar > div.cp{outline:2px solid var(--purple);outline-offset:-2px}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.55}}

/* trust gates */
.gates{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}
.gate{padding:12px;border-radius:8px;background:var(--card);border:1px solid var(--border);position:relative}
.gate .name{font-size:11px;color:var(--mute);letter-spacing:.08em}
.gate .val{font-size:22px;font-weight:700;margin-top:2px}
.gate .thr{font-size:10px;color:var(--mute);margin-top:2px}
.gate.pass{border-color:var(--green)}
.gate.pass .val{color:var(--green)}
.gate.fail{border-color:var(--red)}
.gate.fail .val{color:var(--red)}
.gate.unk .val{color:var(--mute)}
.gate.skip{border-color:#484f58;border-style:dashed}
.gate.skip .val{color:#8b949e;font-size:13px}
.gate .latest{font-size:11px;color:var(--mute);margin-top:6px;line-height:1.3}

/* tables */
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:6px 10px;text-align:left;border-bottom:1px solid var(--border);vertical-align:top}
th{color:var(--blue);font-weight:600;font-size:11px;letter-spacing:.06em;text-transform:uppercase}
tr:last-child td{border-bottom:none}
code{color:var(--blue);font-size:12px}
.tag{display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600}
.tag.run{background:#3d2f0a;color:#f0c674}
.tag.done{background:#0a3d22;color:#7ee2a8}
.tag.fail{background:#3d0a0a;color:#ff8b80}
.tag.pending{background:#21262d;color:var(--mute)}

/* current batch step strip */
.steps{display:flex;gap:3px;margin-top:8px;flex-wrap:wrap}
.steps .s{flex:0 0 52px;padding:6px 4px;text-align:center;border-radius:4px;background:#21262d;font-size:10px;color:var(--mute);cursor:help}
.steps .s b{display:block;font-size:13px;color:var(--fg)}
.steps .s.done{background:#0a3d22;color:#7ee2a8}
.steps .s.done b{color:#7ee2a8}
.steps .s.run{background:#3d2f0a;color:#f0c674;animation:pulse 1.6s infinite}
.steps .s.run b{color:#f0c674}
.steps .s.fail{background:#3d0a0a;color:#ff8b80}

/* event log */
.log{max-height:260px;overflow:auto;font-size:12px}
.log .e{padding:4px 0;border-bottom:1px dotted #21262d}
.log .e .ts{color:var(--mute);margin-right:8px;font-variant-numeric:tabular-nums}
.log .e.p0{color:#ff8b80}
.log .e.p1{color:#f0c674}
.log .e.p2{color:var(--fg)}

.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 16px;font-size:13px}
.kv dt{color:var(--mute)}
.kv dd{margin:0;font-variant-numeric:tabular-nums}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--green);margin-right:6px;animation:pulse 1.4s infinite}
.dead{background:var(--red)!important;animation:none!important}
.head{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:14px}
</style></head><body>

<div class=head>
  <div>
    <h1>CBIC v2 re-ingest</h1>
    <div class=muted><span id=conn class=dot></span><span id=clock>--:--:--</span> · auto-poll 2s · <span id=lag>—</span></div>
  </div>
  <div class=muted style="text-align:right" id=eta>ETA —</div>
</div>

<div class="card now" id=now>
  <div class=muted>NOW</div>
  <div id=now_what><b>loading…</b></div>
  <div style="margin-top:6px"><span class=timer id=now_timer>—</span> elapsed · <span id=now_eta class=muted>ETA —</span></div>
  <div class=muted style="margin-top:4px;font-size:11px" id=now_detail></div>
</div>

<h2>Batches (1 → 10)</h2>
<div class=bar id=bar></div>
<div class=muted><span id=batch_summary>—</span></div>

<h2>Trust gates <span class=muted style="font-size:11px;font-weight:400;text-transform:none;letter-spacing:0">— concurrency: G1+G3 may pair · G4/G2/G5 must run solo</span></h2>
<div class=gates id=gates></div>
<div class=muted style="font-size:11px;margin-top:6px">
  <b style="color:#3fb950">━━━</b> can run in parallel
  <b style="color:#f85149;margin-left:14px">━SOLO━</b> must run alone (codified hard rule, gate_preflight.sh enforces)
</div>

<h2>GPU activity <span class=muted style="font-size:11px;font-weight:400;text-transform:none;letter-spacing:0">— pulse = active in last sample · click for role</span></h2>
<div class=card><div id=gpus style="display:flex;gap:14px;flex-wrap:wrap"></div></div>

<h2>Recent events <span class=muted style="font-size:11px;font-weight:400;text-transform:none;letter-spacing:0">— last 8 meaningful lines from /tmp/ingest_loop.log</span></h2>
<div class=card><pre id=phase_log style="font:12px Consolas,monospace;color:#c9d1d9;margin:0;white-space:pre-wrap;max-height:200px;overflow:auto">…</pre></div>

<h2>Current batch · steps A–O</h2>
<div class=card>
  <div class=muted id=cur_batch_hdr>—</div>
  <div class=steps id=steps></div>
</div>

<h2>Active jobs</h2>
<div class=card><table id=active><thead><tr><th>Job</th><th>Status</th><th>Elapsed</th><th>ETA</th><th>Detail</th></tr></thead><tbody></tbody></table></div>

<h2>Performance &amp; optimization agent</h2>
<div class=card>
<div class=muted style="margin-bottom:8px;font-size:11px"><b>Mandates:</b> (1) Detect stalls — alert P0 if any step duration &gt; expected × 2.0 with no log advance. (2) Flag rule violations (gate concurrency, cold-load &gt;2 cards, proxy use). (3) Surface idle GPUs while jobs run. (4) Watch reranker/embed pool fan-out skew. (5) Recommend action with evidence — never just observe.</div>
<div id=agent_findings></div>
</div>

<h2>Other events</h2>
<div class="card log" id=log></div>

<h2>Halts &amp; fixes</h2>
<div class=card><table id=halts><thead><tr><th>Time</th><th>What</th><th>Fix</th></tr></thead><tbody></tbody></table></div>

<script>
const STEP_NAMES={A:"pre-flight",B:"build",C:"sanity",D:"launch",E:"phase 2 chunk",F:"phase 3-5 embed+upsert",G:"verify",H:"record",I:"spot-check",J:"manifest snap",K:"qdrant snap",L:"CP gate",M:"journal",N:"opt log",O:"green-light"};
const fmtDur=s=>{if(s==null||isNaN(s))return"—";s=Math.max(0,s|0);const h=s/3600|0,m=(s%3600/60)|0,r=s%60;return h?`${h}h${String(m).padStart(2,"0")}`:m?`${m}m${String(r).padStart(2,"0")}`:`${r}s`};
const $ = id => document.getElementById(id);
let last_started_ts=null, last_active_starts={}, server_now=0, local_anchor=0, last_state=null;

function tickClocks(){
  const now=Math.floor(Date.now()/1000) + (server_now-local_anchor);
  $("clock").textContent=new Date(now*1000).toLocaleTimeString("en-GB",{hour12:false});
  if(last_started_ts) $("now_timer").textContent=fmtDur(now-last_started_ts);
  // update each active row's elapsed
  document.querySelectorAll("tr[data-startts]").forEach(tr=>{
    const ts=+tr.dataset.startts; if(ts) tr.querySelector(".elapsed").textContent=fmtDur(now-ts);
  });
}
setInterval(tickClocks,1000);

function renderBar(batches,curId){
  const bar=$("bar"); bar.innerHTML="";
  const CP_AT={1:"CP1",5:"CP2",10:"CP3"};
  for(let i=1;i<=10;i++){
    const b=batches.find(x=>x.id===i)||{};
    const cpTag=CP_AT[i];
    const cls=b.status==="done"?"done":b.status==="running"?"run":b.status==="failed"?"fail":"";
    const div=document.createElement("div");
    div.className=cls+(cpTag?" cp":"");
    div.title=`Batch ${i} · ${b.status||"pending"}${cpTag?` · checkpoint ${cpTag}`:""}`;
    div.innerHTML=`${i}${cpTag?`<span style="position:absolute;bottom:1px;right:3px;font-size:9px;color:#a371f7;font-weight:600">${cpTag}</span>`:""}`;
    bar.appendChild(div);
  }
}

function renderGates(trust){
  const GATE_INFO={
    G1:{label:"recall",concurrency:"PAIR",pair_with:"G3",hits:"/retrieve",runs_at:"CP-1, CP-2, CP-3"},
    G3:{label:"levenshtein",concurrency:"PAIR",pair_with:"G1",hits:"/retrieve",runs_at:"CP-1, CP-2, CP-3"},
    G4:{label:"groundedness",concurrency:"SOLO",pair_with:null,hits:"/retrieve+qwen3",runs_at:"CP-1, CP-2, CP-3"},
    G2:{label:"dual-judge",concurrency:"SOLO",pair_with:null,hits:"Gemini+Claude+/retrieve",runs_at:"CP-3 only ($4/run)"},
    G5:{label:"latency",concurrency:"SOLO",pair_with:null,hits:"/query (full)",runs_at:"CP-3 only (full-corpus only)"},
  };
  const order=["G1","G3","G4","G2","G5"], el=$("gates"); el.innerHTML="";
  order.forEach(g=>{
    const t=trust[g]||{}, p=t.passing, info=GATE_INFO[g];
    const runsAtCP3Only = info.runs_at && info.runs_at.startsWith("CP-3 only");
    const cls = p===true?"pass":p===false?"fail":(runsAtCP3Only && p===null?"skip":"unk");
    const val = p===true?"PASS":p===false?"FAIL":(runsAtCP3Only?"DEFERRED":"—");
    const concBadge=info.concurrency==="PAIR"?
      `<span style="background:#0a3d22;color:#7ee2a8;padding:1px 5px;border-radius:3px;font-size:9px;font-weight:600">↔ ${info.pair_with}</span>`:
      `<span style="background:#3d0a0a;color:#ff8b80;padding:1px 5px;border-radius:3px;font-size:9px;font-weight:600">SOLO</span>`;
    const d=document.createElement("div");
    d.className="gate "+cls;
    const latestText = t.latest || (runsAtCP3Only ? `runs at: ${info.runs_at}` : "not measured");
    d.innerHTML=`<div style="display:flex;justify-content:space-between;align-items:flex-start"><div class=name>${g} · ${info.label}</div>${concBadge}</div><div class=val>${val}</div><div class=thr>thr ${t.threshold||"—"} · runs at: ${info.runs_at||"every CP"}</div><div class=latest>${latestText}</div>`;
    el.appendChild(d);
  });
}

const STEP_DESCRIPTIONS={
  A:"Pre-flight: run status.sh, verify API active, qdrant reachable, no leftover ingest/gate processes, embed pool GPUs (4,5,6) responsive, reranker (:9085) and qwen3 (:9082) listening.",
  B:"Build batch: generate doc-id list for batch N (1500 docs, seed=43, 7 categories represented).",
  C:"Sanity: verify category distribution within ±15% of expected mix.",
  D:"Launch: nohup ingest_v2.py --doclist /tmp/batch_N.txt --collection cbic_v2 ; record PID.",
  E:"Phase 2 chunk: chunker_v2 splits documents → ~4200 canonical chunks (~39% dedupe). qwen3 classifier on GPU 2. Watch for 'phase2 done'. ~2.5 min.",
  F:"Phase 3-5 embed+upsert: BGE-M3 pool {4,5,6} embeds chunks → upsert to Qdrant. Watch for 'reconcile PASS'. 0 NaN, 0 Qdrant 400s, ≥21 ch/s. ~3.5 min.",
  G:"Verify: curl Qdrant — points_count must equal previous + ~7800.",
  H:"Record row: append batch row to INGEST_TRACKER.md with timing, ch/s, pts_after.",
  I:"Spot-check: pick 3 random batch-N doc_ids from manifest, hit /retrieve, verify all 3 retrieve correctly.",
  J:"Manifest snap: cp _manifest.sqlite to /opt/snapshots/manifest_after_batchN.sqlite.",
  K:"Qdrant snap: POST :6343/collections/cbic_v2/snapshots → snapshot id.",
  L:"CP gate (only batches 1, 5, 10): run G1+G3 parallel pair (preflight each), then G4 solo (preflight), CP-3 also G2 + G5. Acceptance per checkpoint thresholds.",
  M:"Journal: append entry to JOURNAL.md (1-2 lines: batch N pts time gate panel result).",
  N:"Optimization log: append P2 finding noting any drift (timing variance, fan-out skew, GPU idle while jobs run).",
  O:"Green-light: flip batch N status=done, increment global.batches_done, proceed to N+1."
};
function showStepDetail(letter, stepData){
  const desc = STEP_DESCRIPTIONS[letter] || "(no description)";
  const status = stepData.status || "pending";
  const dur = stepData.duration_s ? fmtDur(stepData.duration_s) : "—";
  const html = `<div style="position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.7);z-index:1000;display:flex;align-items:center;justify-content:center" id=stepmodal onclick="this.remove()">
    <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:18px 22px;max-width:560px" onclick="event.stopPropagation()">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px"><h3 style="color:#58a6ff;margin:0">Step ${letter} · ${STEP_NAMES[letter]}</h3><button onclick="document.getElementById('stepmodal').remove()" style="background:#21262d;border:1px solid #30363d;color:#c9d1d9;cursor:pointer;padding:2px 8px;border-radius:4px">close</button></div>
      <div style="font-size:13px;color:#c9d1d9;line-height:1.5;margin-bottom:10px">${esc(desc)}</div>
      <div style="display:flex;gap:14px;font-size:12px;color:#8b949e"><div>status: <b style="color:${status==='done'?'#3fb950':status==='running'?'#d29922':status==='failed'?'#f85149':'#8b949e'}">${esc(status)}</b></div><div>duration: ${esc(dur)}</div></div>
    </div>
  </div>`;
  document.body.insertAdjacentHTML("beforeend", html);
}
function renderSteps(batch){
  const el=$("steps"); el.innerHTML="";
  "ABCDEFGHIJKLMNO".split("").forEach(k=>{
    const s=(batch.steps||{})[k]||{};
    const cls=s.status==="done"?"done":s.status==="running"?"run":s.status==="failed"?"fail":"";
    const d=document.createElement("div");
    d.className="s "+cls;
    d.style.cursor="pointer";
    d.title=`${STEP_NAMES[k]}${s.gpus?` · GPUs ${s.gpus}`:""}${s.duration_s?` · ${fmtDur(s.duration_s)}`:""} · click for full detail`;
    const sub = s.duration_s ? fmtDur(s.duration_s) : (s.status==="running" ? "…" : "·");
    const gpu = s.gpus ? `<div style="font-size:8px;color:#79c0ff;line-height:1.1">${esc(s.gpus.replace(/\s*\([^)]+\)/g,"").slice(0,12))}</div>` : "";
    d.innerHTML=`<b>${k}</b>${gpu}<small>${sub}</small>`;
    d.addEventListener("click", ()=>showStepDetail(k, s));
    el.appendChild(d);
  });
}

function renderActive(active){
  const tb=$("active").querySelector("tbody"); tb.innerHTML="";
  if(!active.length){ tb.innerHTML="<tr><td colspan=5 class=muted>nothing active</td></tr>"; return; }
  active.forEach(a=>{
    const tr=document.createElement("tr");
    tr.dataset.startts=a.started_ts||"";
    tr.innerHTML=`<td><b>${a.name||"—"}</b></td><td><span class="tag run">${a.status||""}</span></td><td class=elapsed>—</td><td>${a.eta||"—"}</td><td class=muted>${a.detail||""}</td>`;
    tb.appendChild(tr);
  });
}

function esc(s){return String(s==null?"":s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}

function renderAgent(log){
  // Agent-mandated findings: entries with severity/what/rec/evidence (structured agent output)
  const findings=(log||[]).filter(e=>e.severity&&(e.what||e.rec));
  const el=$("agent_findings"); el.innerHTML="";
  if(!findings.length){ el.innerHTML='<div class=muted>no findings yet — agent will append as it runs</div>'; return; }
  const order={P0:0,P1:1,P2:2,P3:3};
  const sorted=[...findings].sort((a,b)=>(order[(a.severity||"P2").toUpperCase()]??9)-(order[(b.severity||"P2").toUpperCase()]??9)||(b.ts||"").localeCompare(a.ts||""));
  sorted.forEach(e=>{
    const sev=(e.severity||"P2").toString().toUpperCase();
    const sevColor={P0:"#f85149",P1:"#d29922",P2:"#58a6ff",P3:"#8b949e"}[sev]||"#8b949e";
    const guard=e.guardrail_ok===true?"✓ within guardrails":(e.guardrail_ok===false?"⚠ guardrail concern":"");
    const d=document.createElement("div");
    d.style.cssText="border-left:3px solid "+sevColor+";padding:8px 10px;margin-bottom:8px;background:#0d1117;border-radius:0 4px 4px 0";
    d.innerHTML=`
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:3px">
        <div><span style="background:${sevColor};color:#0d1117;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:700;letter-spacing:.05em">${sev}</span>
        <span class=muted style="margin-left:8px;font-size:11px">${esc(e.ts||"")}</span></div>
        <span class=muted style="font-size:10px">${esc(guard)}</span>
      </div>
      <div style="font-weight:600;color:#fff">${esc(e.what||"")}</div>
      ${e.rec?`<div style="margin-top:4px;font-size:12px;color:#79c0ff"><b>→ Recommend:</b> ${esc(e.rec)}</div>`:""}
      ${e.evidence?`<div class=muted style="margin-top:4px;font-size:10px;font-style:italic;line-height:1.4">${esc(e.evidence).slice(0,400)}${e.evidence.length>400?"…":""}</div>`:""}
    `;
    el.appendChild(d);
  });
}

function renderLog(log){
  // Non-agent events (free-form msg entries without severity)
  const other=(log||[]).filter(e=>!e.severity||(!e.what&&!e.rec));
  const el=$("log"); el.innerHTML="";
  other.slice(-15).reverse().forEach(e=>{
    const d=document.createElement("div");
    d.className="e p2";
    d.innerHTML=`<span class=ts>${esc(e.ts||"")}</span>${esc(e.msg||e.headline||"")}`;
    el.appendChild(d);
  });
  if(!other.length) el.innerHTML='<div class="muted">none</div>';
}

const GPU_ROLES={
  0:{role:"BGE-M3 (direct)",port:"+ bge-reranker",color:"#3fb950"},
  1:{role:"BGE-M3 (direct)",port:"embed pool",color:"#3fb950"},
  2:{role:"qwen3-14b",port:"classifier+grounded",color:"#d29922"},
  3:{role:"BGE-M3 (direct)",port:"embed pool",color:"#3fb950"},
  4:{role:"BGE-M3 (direct)",port:"embed pool",color:"#3fb950"},
  5:{role:"BGE-M3 (direct)",port:"embed pool",color:"#3fb950"},
  6:{role:"BGE-M3 (direct)",port:"embed pool",color:"#3fb950"},
};
function renderGPUs(gpuText, procsText){
  const el=$("gpus"); el.innerHTML="";
  const busy={};
  (gpuText||"").trim().split("\n").forEach(line=>{
    const m=line.trim().match(/^(\d+),(-?\d+)/);
    if(m) busy[+m[1]]=+m[2];
  });
  // Parse procs: "i,pid,cmd"
  const procs={};
  (procsText||"").trim().split("\n").forEach(line=>{
    const m=line.trim().match(/^(\d+),(\d+),(.+)$/);
    if(m){ if(!procs[+m[1]]) procs[+m[1]]=[]; procs[+m[1]].push({pid:+m[2],cmd:m[3].trim()}); }
  });
  for(let i=0;i<7;i++){
    const info=GPU_ROLES[i];
    const b=busy[i]||0;
    const myProcs=procs[i]||[];
    // Process-presence is the reliable activity signal on this AMDGPU stack
    // (sysfs gpu_busy_percent reads 0 even during active embed/inference)
    const isActive = myProcs.length>0 || b>5;
    const isPython = myProcs.some(p => /python|llama|server/i.test(p.cmd));
    const pulse=isActive?"animation:pulse 1s infinite":"";
    const bgColor = isActive ? info.color : "#21262d";
    const shadow = isActive?`box-shadow:0 0 14px ${info.color}`:"";
    const dot=`<div style="width:14px;height:14px;border-radius:50%;background:${bgColor};${shadow};${pulse};margin:0 auto 4px"></div>`;
    const card=document.createElement("div");
    card.style.cssText="text-align:center;min-width:108px;padding:8px;border-radius:6px;background:#0d1117;border:1px solid #30363d";
    const stateText = isActive ? `<span style="color:#3fb950;font-weight:700">ACTIVE</span>${b>5?` <span class=muted>${b}%</span>`:""}` :
                      `<span style="color:#484f58">idle</span>`;
    // Map process cmdline to clean service names (no binary paths exposed)
    const procText = myProcs.length ? myProcs.slice(0,2).map(p=>{
      let svc = "?";
      if (/ingest_v2/i.test(p.cmd)) svc = "ingest_v2 (BGE-M3 direct)";
      else if (/9082|qwen3/i.test(p.cmd)) svc = "qwen3-14b";
      else if (/9085|reranker/i.test(p.cmd)) svc = "bge-reranker";
      else if (/llama-server/i.test(p.cmd) && i===0) svc = "bge-reranker";
      else if (/llama-server/i.test(p.cmd) && i===2) svc = "qwen3-14b";
      else svc = p.cmd.slice(0,16);
      return `<div style="font-size:10px;color:#79c0ff" title="PID ${p.pid}">${esc(svc)}</div>`;
    }).join("") : "";
    card.innerHTML=`${dot}<div style="font-size:11px;color:${info.color};font-weight:600">GPU ${i}</div><div style="font-size:10px;color:#8b949e;margin-top:1px">${info.role}</div><div style="font-size:10px;margin-top:3px">${stateText}</div>${procText}`;
    el.appendChild(card);
  }
}
function renderHalts(halts){
  const tb=$("halts").querySelector("tbody"); tb.innerHTML="";
  if(!halts||!halts.length){ tb.innerHTML="<tr><td colspan=3 class=muted>none</td></tr>"; return; }
  halts.slice(-10).reverse().forEach(h=>{
    const tr=document.createElement("tr");
    tr.innerHTML=`<td><code>${h.ts||""}</code></td><td>${h.what||""}</td><td class=muted>${h.fix||""}</td>`;
    tb.appendChild(tr);
  });
}

async function poll(){
  const t0=performance.now();
  try{
    const r=await fetch("/state.json",{cache:"no-store"});
    if(!r.ok) throw new Error(r.status);
    const st=await r.json();
    $("conn").classList.remove("dead");
    $("lag").textContent=`${(performance.now()-t0)|0}ms`;
    server_now=Math.floor(Date.now()/1000); local_anchor=server_now;

    const g=st.global||{}, cur=st.current_batch||{}, trust=st.trust||{};
    last_started_ts=cur.started_ts||null;

    $("now_what").innerHTML=`<b>Batch ${cur.id||"—"}</b> · step ${cur.current_step||"—"}`;
    $("now_eta").textContent=`ETA ${fmtDur((cur.expected_s||0) - (server_now-(cur.started_ts||server_now)))}`;
    $("now_detail").textContent=`${g.batches_done||0}/10 batches · ${(g.points||0).toLocaleString()} pts in cbic_v2`;
    $("eta").textContent=`ETA done · ${g.eta_done||"—"}`;
    $("batch_summary").textContent=`${g.batches_done||0} done · halts ${g.halts_count||0} · projected total ${fmtDur(g.time_total_expected_s)}`;
    $("cur_batch_hdr").textContent=`Batch #${cur.id||"—"} · started ${cur.started||"—"} · expected ${fmtDur(cur.expected_s)}`;

    renderBar(st.batches||[], cur.id);
    renderGates(trust);
    const curB=(st.batches||[]).find(b=>b.id===cur.id)||{};
    renderSteps(curB);
    renderActive(st.active||[]);
    renderAgent(st.agent_log||[]);
    renderLog(st.agent_log||[]);
    if(st.live){
      // Parse new sections: ---GPU---, ---GPU_PROCS---, ---PHASE---, ---PTS---, ---RERANK_RECENT---
      const sections = {};
      let cur = "head";
      sections[cur] = [];
      st.live.split("\n").forEach(line=>{
        const m = line.match(/^---(\w+)---$/);
        if (m) { cur = m[1]; sections[cur] = []; }
        else { (sections[cur] = sections[cur] || []).push(line); }
      });
      renderGPUs((sections.GPU||[]).join("\n"), (sections.GPU_PROCS||[]).join("\n"));
      $("phase_log").textContent = (sections.PHASE||[]).join("\n").trim() || "(no recent meaningful events)";
    }
    renderHalts(st.halts||[]);
    last_state=st;
  }catch(e){
    $("conn").classList.add("dead");
    $("lag").textContent="offline: "+e.message;
  }
}
poll(); setInterval(poll,2000);
</script>
</body></html>"""

class H(BaseHTTPRequestHandler):
    def log_message(self, *a, **k): pass
    def do_GET(self):
        if self.path.startswith("/state.json"):
            maybe_refresh_live()
            try:
                st = json.loads(STATE.read_text(encoding="utf-8"))
                st["live"] = LIVE_CACHE["text"]
                st["live_age_s"] = int(time.time() - LIVE_CACHE["ts"]) if LIVE_CACHE["ts"] else None
                body = json.dumps(st).encode()
            except Exception as e: body = json.dumps({"error": str(e)}).encode()
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Cache-Control","no-store")
            self.end_headers(); self.wfile.write(body); return
        body = INDEX.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type","text/html; charset=utf-8")
        self.send_header("Cache-Control","no-store")
        self.end_headers(); self.wfile.write(body)

if __name__ == "__main__":
    print("CBIC v2 dashboard at http://127.0.0.1:8765/")
    threading.Thread(target=writer_loop, daemon=True).start()
    print("[writer] autonomous rig observer started, polls every 30s")
    HTTPServer(("127.0.0.1", 8765), H).serve_forever()
