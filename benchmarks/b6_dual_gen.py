#!/usr/bin/env python3
"""B6: 2× llama-server (GPU 4 port 8766, GPU 6 port 8767) concurrent.
Each gets 8 parallel requests — total 16 in-flight."""
import json, time, threading, urllib.request
from pathlib import Path

OUT = Path("/mnt/d/_gpu_rig_ai/benchmarks/b6_results.json")
PORTS = [8766, 8767]
PROMPT = "Generate one Q: A: pair about GST input tax credit under section 16 of CGST Act. Be concise."
MAX_TOK = 200
PER_PORT = 8

lock = threading.Lock()
results = []

def req(port, idx):
    body = {"prompt": PROMPT, "n_predict": MAX_TOK, "temperature": 0.2, "cache_prompt": False}
    t = time.perf_counter()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/completion", method="POST",
        headers={"Content-Type": "application/json"}, data=json.dumps(body).encode())
    resp = json.loads(urllib.request.urlopen(r, timeout=300).read())
    dt = time.perf_counter() - t
    toks = resp.get("tokens_predicted", resp.get("timings", {}).get("predicted_n", 0))
    with lock:
        results.append((port, idx, dt, toks))

t0 = time.perf_counter()
threads = []
for p in PORTS:
    for i in range(PER_PORT):
        threads.append(threading.Thread(target=req, args=(p, i)))
for t in threads: t.start()
for t in threads: t.join()
wall = time.perf_counter() - t0

total_tok = sum(r[3] for r in results)
agg = total_tok / wall
per_port = {}
for p, i, dt, toks in results:
    per_port.setdefault(p, {"toks": 0, "dt_max": 0})
    per_port[p]["toks"] += toks
    per_port[p]["dt_max"] = max(per_port[p]["dt_max"], dt)
for p, d in per_port.items():
    d["port_tok_per_s"] = d["toks"] / d["dt_max"]

result = {
    "wall_s": wall,
    "total_tokens": total_tok,
    "aggregate_tok_per_s": agg,
    "per_port": per_port,
    "passed": agg >= 120,  # ≥1.5x single-GPU 83 tok/s
}
OUT.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
print("VERDICT:", "PASS" if result["passed"] else "FAIL")
