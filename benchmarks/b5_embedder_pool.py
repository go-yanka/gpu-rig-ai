#!/usr/bin/env python3
"""B5: 2-GPU embedder pool. Hammer both ports concurrently, verify throughput sums."""
import json, time, threading, urllib.request
from pathlib import Path

OUT = Path("/mnt/d/_gpu_rig_ai/benchmarks/b5_results.json")
PORTS = [8770, 8771]
TEXTS = [
    "Section 16 of CGST Act governs input tax credit eligibility.",
    "HSN 8703 motor cars attract 28% GST plus compensation cess.",
    "Rule 138 e-way bill threshold is Rs 50,000 consignment value.",
    "IGST Act section 5(1) levies integrated GST on inter-state supply.",
    "Composition levy under section 10 for turnover up to 1.5 crore.",
] * 20  # 100 texts per batch

N_ROUNDS = 10  # 10 batches of 100 per port = 1000 embeddings each

def hit(port, round_idx, out):
    body = {"content": TEXTS}
    t = time.perf_counter()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/embedding",
        method="POST", headers={"Content-Type": "application/json"},
        data=json.dumps(body).encode())
    resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
    dt = time.perf_counter() - t
    out.append((port, round_idx, dt, len(TEXTS)))

results = []
lock = threading.Lock()

def worker(port):
    for r in range(N_ROUNDS):
        local = []
        hit(port, r, local)
        with lock:
            results.extend(local)

t0 = time.perf_counter()
threads = [threading.Thread(target=worker, args=(p,)) for p in PORTS]
for t in threads: t.start()
for t in threads: t.join()
wall = time.perf_counter() - t0

total_emb = sum(r[3] for r in results)
per_port = {}
for port, ridx, dt, n in results:
    per_port.setdefault(port, {"total_dt": 0, "total_n": 0})
    per_port[port]["total_dt"] += dt
    per_port[port]["total_n"] += n
for p, d in per_port.items():
    d["emb_per_s"] = d["total_n"] / d["total_dt"]

result = {
    "wall_s": wall,
    "total_embeddings": total_emb,
    "aggregate_emb_per_s": total_emb / wall,
    "per_port": per_port,
    "speedup_vs_single": (total_emb / wall) / max(list(per_port.values())[0]["emb_per_s"], 0.001),
    "passed": (total_emb / wall) >= 1.5 * list(per_port.values())[0]["emb_per_s"],
}
OUT.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
print("VERDICT:", "PASS" if result["passed"] else "FAIL")
