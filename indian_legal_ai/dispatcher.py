#!/usr/bin/env python3
"""
Overnight RAG index dispatcher.

Maintains a pool of 6 worker slots (one per BGE-serving GPU) and pulls
files from a priority queue. Survives worker crashes via retry limit.

Run as: nohup sudo -u user python3 /opt/indian-legal-ai/scripts/dispatcher.py \
          > /opt/indian-legal-ai/logs/dispatcher.log 2>&1 &
"""

import os, sys, json, time, subprocess, datetime, signal

WORK_DIR   = "/opt/indian-legal-ai"
LOG_DIR    = f"{WORK_DIR}/logs"
SCRIPT     = f"{WORK_DIR}/scripts/build_rag_worker_tiered.py"
STATE_FILE = f"{LOG_DIR}/dispatcher_state.json"
CURATED    = f"{WORK_DIR}/datasets/_curated"

# slot -> (port, gpu, worker_id_base). worker_id = base + iteration.
SLOTS = [
    ("9090", 1, 1000),
    ("9093", 2, 2000),
    ("9091", 3, 3000),
    ("9092", 4, 4000),
    ("9094", 5, 5000),
    ("9095", 6, 6000),
]

# Queue: heaviest / most important first.
# Already-running files are skipped at startup via skip_files check.
WORK_QUEUE = [
    "indian_laws",              # Tier 1 - was failing
    "courts_cases_02",          # Tier 2 split
    "courts_cases_03",          # Tier 2 split
    "prarabdha_sft",            # Tier 3 200k
    "kshitij_law",              # Tier 3 25k
    "indian_law",               # Tier 3 24k
    "indian_law_9b",            # Tier 3 24k
    "deshmukh_law",             # Tier 3 24k
    "tech_legal",               # Tier 3 14k
    "jizzu_law_v4",             # Tier 3 13k
    "ipc_insights",             # Tier 3 5k
    "varma_law",                # Tier 3 3k
    "shreyas_legal",
    "indian_lawyer",
    "traffic_law",
    "lawyer_gpt",
    "legal_asst",
    "karthi_law",
    "gst_faqs",
]

NEXT_ID_BASE = 200_000_000   # start above all hand-assigned bases
ID_STEP      = 10_000_000
MAX_RETRIES  = 2
POLL_SEC     = 20
IDLE_EXIT    = 5    # consecutive idle loops with empty queue -> exit


def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}][DISP] {msg}", flush=True)


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "queue": list(WORK_QUEUE),
        "done": [],
        "failed": [],
        "attempts": {},
        "next_id_base": NEXT_ID_BASE,
        "slot_iter": {str(i): 0 for i in range(len(SLOTS))},
    }


def save_state(state):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def _pgrep_workers():
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", "build_rag_worker"], text=True
        )
    except subprocess.CalledProcessError:
        return []
    # Match both build_rag_worker.py and build_rag_worker_tiered.py
    return [l for l in out.splitlines() if "build_rag_worker" in l]


def running_files():
    """Return set of dataset names currently being processed by build_rag_worker."""
    files = set()
    for line in _pgrep_workers():
        if "--files" in line:
            parts = line.split()
            try:
                i = parts.index("--files")
                for name in parts[i + 1].split(","):
                    files.add(name.strip())
            except ValueError:
                pass
    return files


def busy_ports(my_pids):
    """Return set of ports currently hit by build_rag_worker procs we didn't spawn."""
    busy = set()
    for line in _pgrep_workers():
        parts = line.split()
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid in my_pids:
            continue
        if "--endpoint" in parts:
            i = parts.index("--endpoint")
            url = parts[i + 1]
            # http://localhost:9090/v1/embeddings
            try:
                port = url.split(":")[2].split("/")[0]
                busy.add(port)
            except Exception:
                pass
    return busy


def worker_log_status(worker_id):
    """Inspect worker log; return 'finished', 'failed', or 'unknown'."""
    path = f"{LOG_DIR}/worker{worker_id}.log"
    if not os.path.exists(path):
        return "unknown"
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4000))
            tail = f.read().decode("utf-8", "ignore")
    except Exception:
        return "unknown"
    if "WORKER FINISHED" in tail:
        return "finished"
    if "RuntimeError: embed failed" in tail or "Traceback" in tail:
        return "failed"
    return "unknown"


def spawn_worker(slot_idx, filename, id_base, worker_id):
    port, gpu, _ = SLOTS[slot_idx]
    endpoint = f"http://localhost:{port}/v1/embeddings"
    logpath  = f"{LOG_DIR}/worker{worker_id}.log"
    cmd = [
        "python3", SCRIPT,
        "--endpoint", endpoint,
        "--id-base", str(id_base),
        "--worker-id", str(worker_id),
        "--files", filename,
        "--tiers", "1,2,3",
    ]
    with open(logpath, "ab") as lf:
        lf.write(f"\n===== dispatcher launch {datetime.datetime.now()} =====\n".encode())
        lf.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=lf, stderr=lf,
            cwd=WORK_DIR,
            preexec_fn=os.setsid,
        )
    log(f"spawned W{worker_id} slot={slot_idx} gpu={gpu} port={port} file={filename} id_base={id_base:,} pid={proc.pid}")
    return proc


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    state = load_state()

    # Skip anything currently already in-flight.
    in_flight = running_files()
    if in_flight:
        log(f"already running at start: {sorted(in_flight)}")
    state["queue"] = [f for f in state["queue"]
                      if f not in in_flight and f not in state["done"]]
    save_state(state)

    slot_procs = {i: None for i in range(len(SLOTS))}
    slot_files = {i: None for i in range(len(SLOTS))}
    slot_wids  = {i: None for i in range(len(SLOTS))}

    idle_loops = 0

    while True:
        # Reap finished slots.
        for i, proc in list(slot_procs.items()):
            if proc is None:
                continue
            ret = proc.poll()
            if ret is None:
                continue
            filename = slot_files[i]
            wid      = slot_wids[i]
            status   = worker_log_status(wid)
            if status == "finished":
                log(f"W{wid} DONE file={filename} slot={i} ret={ret}")
                state["done"].append(filename)
            else:
                attempts = state["attempts"].get(filename, 0) + 1
                state["attempts"][filename] = attempts
                if attempts >= MAX_RETRIES:
                    log(f"W{wid} FAILED file={filename} slot={i} ret={ret} status={status} attempts={attempts} -> giving up")
                    state["failed"].append(filename)
                else:
                    log(f"W{wid} FAILED file={filename} slot={i} ret={ret} status={status} attempts={attempts} -> requeuing")
                    state["queue"].insert(0, filename)
            slot_procs[i] = None
            slot_files[i] = None
            slot_wids[i]  = None
            save_state(state)

        # Fill idle slots.
        in_flight = running_files()
        my_pids = {p.pid for p in slot_procs.values() if p is not None}
        blocked = busy_ports(my_pids)
        for i in range(len(SLOTS)):
            if slot_procs[i] is not None:
                continue
            port = SLOTS[i][0]
            if port in blocked:
                # External worker still using this port; skip until it finishes.
                continue
            if not state["queue"]:
                break
            # pull next file, skip if somehow still in flight.
            while state["queue"] and state["queue"][0] in in_flight:
                skip = state["queue"].pop(0)
                log(f"skip {skip} (already in flight)")
            if not state["queue"]:
                break
            # verify file exists.
            next_file = state["queue"][0]
            path = f"{CURATED}/{next_file}.jsonl"
            if not os.path.exists(path):
                log(f"skip missing file {next_file}")
                state["queue"].pop(0)
                state["failed"].append(next_file + " (missing)")
                save_state(state)
                continue
            state["queue"].pop(0)

            id_base = state["next_id_base"]
            state["next_id_base"] = id_base + ID_STEP
            state["slot_iter"][str(i)] = state["slot_iter"].get(str(i), 0) + 1
            wid = SLOTS[i][2] + state["slot_iter"][str(i)]

            proc = spawn_worker(i, next_file, id_base, wid)
            slot_procs[i] = proc
            slot_files[i] = next_file
            slot_wids[i]  = wid
            in_flight.add(next_file)
            save_state(state)

        # Status heartbeat.
        active = [(i, slot_files[i], slot_wids[i])
                  for i in range(len(SLOTS)) if slot_procs[i] is not None]
        if active:
            idle_loops = 0
            summary = "  ".join(f"slot{i}:W{w}:{f}" for i, f, w in active)
            log(f"active={len(active)} queue_left={len(state['queue'])} done={len(state['done'])} failed={len(state['failed'])} | {summary}")
        else:
            if not state["queue"]:
                idle_loops += 1
                log(f"idle (queue empty) idle_loops={idle_loops}/{IDLE_EXIT}")
                if idle_loops >= IDLE_EXIT:
                    log("ALL WORK DONE; dispatcher exiting.")
                    log(f"done ({len(state['done'])}): {state['done']}")
                    log(f"failed ({len(state['failed'])}): {state['failed']}")
                    break
            else:
                log(f"no active workers but queue has {len(state['queue'])}; will refill next loop")

        time.sleep(POLL_SEC)

    save_state(state)


if __name__ == "__main__":
    main()
