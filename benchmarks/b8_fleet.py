#!/usr/bin/env python3
"""B8: 5-GPU qwen3-8B fleet (GPU 0,1,3,4,6 via ports 8768,8769,8779,8766,8767).
-np 4 per server, 4 parallel requests each = 20 concurrent total."""
import json, time, threading, urllib.request
from pathlib import Path

OUT = Path("/mnt/d/_gpu_rig_ai/benchmarks/b8_results.json")
PORTS = [8766, 8767, 8768, 8769, 8779]
PER_PORT = 4
PROMPT = "Generate one Q: A: pair about GST input tax credit section 16."
MAX_TOK = 200

results = []
lock = threading.Lock()

def req(port, i):
    body = {"prompt": PROMPT, "n_predict": MAX_TOK, "temperature": 0.2, "cache_prompt": False}
    t = time.perf_counter()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/completion", method="POST",
        headers={"Content-Type": "application/json"}, data=json.dumps(body).encode())
    resp = json.loads(urllib.request.urlopen(r, timeout=600).read())
    dt = time.perf_counter() - t
    toks = resp.get("tokens_predicted", resp.get("timings", {}).get("predicted_n", 0))
    with lock:
        results.append((port, i, dt, toks))

t0 = time.perf_counter()
ths = [threading.Thread(target=req, args=(p, i)) for p in PORTS for i in range(PER_PORT)]
for t in ths: t.start()
for t in ths: t.join()
wall = time.perf_counter() - t0
total_tok = sum(r[3] for r in results)
agg = total_tok / wall
per_port = {}
for p, i, dt, toks in results:
    per_port.setdefault(str(p), {"toks": 0, "max_dt": 0})
    per_port[str(p)]["toks"] += toks
    per_port[str(p)]["max_dt"] = max(per_port[str(p)]["max_dt"], dt)
for p, d in per_port.items():
    d["port_tok_per_s"] = d["toks"] / d["max_dt"]
result = {
    "wall_s": wall, "total_tokens": total_tok, "aggregate_tok_per_s": agg,
    "per_port": per_port,
    "projected_20k_QA_hours": (20000 * 300) / agg / 3600,
    "passed": agg >= 200,  # 2.4x single-GPU 83 tok/s
}
OUT.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
print("VERDICT:", "PASS" if result["passed"] else "FAIL")
