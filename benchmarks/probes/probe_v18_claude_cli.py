#!/usr/bin/env python3
"""V18: Claude CLI for extraction (Max plan subscription; no API cost).
Pipes 50 prompts through `claude -p` (non-interactive), measures parse-rate + latency.
Pass: parse-rate >= qwen3-14b's (V1 result); latency <=5s/call.
Run on rig where claude CLI is authed under user home.
"""
import json, subprocess, time, urllib.request
from pathlib import Path

QDRANT = "http://127.0.0.1:6343"
OUT = Path("/opt/indian-legal-ai/data/probes/v18_result.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

SYS = ("Return STRICT JSON: {\"section_ref\": \"Section N(x)(y)\" or null, "
       "\"confidence\": 0.0-1.0}. No prose. No markdown.")

def get_samples(n=50):
    body = {"limit": n, "with_payload": ["text"], "with_vector": False}
    req = urllib.request.Request(f"{QDRANT}/collections/cbic_v1/points/scroll",
        method="POST", data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=30).read())
    return [p["payload"]["text"][:2000] for p in r["result"]["points"] if p.get("payload",{}).get("text")]

def ask_claude(text):
    prompt = f"{SYS}\n\nTEXT:\n{text}\n\nExtract section_ref as JSON."
    t0 = time.time()
    try:
        r = subprocess.run(["claude", "-p", prompt], capture_output=True, timeout=60, text=True)
        dt = time.time() - t0
        return r.stdout.strip(), dt, (r.stderr if r.returncode!=0 else None)
    except Exception as e:
        return None, time.time()-t0, str(e)

def main():
    samples = get_samples(50)
    print(f"[V18] {len(samples)} samples")
    results = []; parsed = 0; lats = []
    for i, t in enumerate(samples):
        raw, dt, err = ask_claude(t); lats.append(dt)
        if err:
            results.append({"i": i, "err": err, "dt": dt}); continue
        r = raw.strip()
        if r.startswith("```"): r = r.split("\n",1)[1].rsplit("```",1)[0]
        try:
            obj = json.loads(r)
            ok = "section_ref" in obj and "confidence" in obj
            if ok: parsed += 1
            results.append({"i": i, "parsed": ok, "ref": obj.get("section_ref"), "dt": dt})
        except Exception as e:
            results.append({"i": i, "parsed": False, "raw": raw[:200], "dt": dt})
        if i % 10 == 0: print(f"  [{i}] ok={parsed}")

    lats.sort()
    summary = {
        "probe": "V18",
        "n": len(samples), "parsed_ok": parsed,
        "parse_rate": round(parsed/len(samples), 3),
        "latency_p50": round(lats[len(lats)//2], 2) if lats else 0,
        "latency_p95": round(lats[int(len(lats)*0.95)], 2) if lats else 0,
        "pass_gate_latency": (lats[len(lats)//2] if lats else 99) <= 5.0,
        "pass_gate_parse": parsed >= 45,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
