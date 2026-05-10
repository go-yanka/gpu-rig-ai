#!/usr/bin/env python3
"""B-Gen: qwen3-8B Vulkan throughput with -np 8 -cb continuous batching.
Send 16 concurrent Q&A-style requests, measure aggregate tok/s."""
import json
import time
import threading
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8766/completion"
OUT = Path("/mnt/d/_gpu_rig_ai/benchmarks/b_gen_results.json")

PROMPT = """Given the following CBIC notification extract, generate one question and one answer.
Notification: Section 16 of CGST Act provides conditions for claiming input tax credit including possession of tax invoice, receipt of goods/services, payment to supplier within 180 days, and filing of GSTR-3B.
Output format: Q: ... A: ...
"""

N_CONCURRENT = 16
MAX_TOKENS = 200

results = []
lock = threading.Lock()

def one_request(idx):
    body = {"prompt": PROMPT, "n_predict": MAX_TOKENS, "temperature": 0.2, "cache_prompt": False}
    t = time.perf_counter()
    req = urllib.request.Request(URL, method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(body).encode())
    resp = json.loads(urllib.request.urlopen(req, timeout=300).read())
    dt = time.perf_counter() - t
    toks = resp.get("tokens_predicted", resp.get("timings", {}).get("predicted_n", 0))
    with lock:
        results.append((idx, dt, toks))
    print(f"  req {idx}: {dt:.2f}s, {toks} tok, {toks/dt:.1f} tok/s single", flush=True)

t0 = time.perf_counter()
threads = [threading.Thread(target=one_request, args=(i,)) for i in range(N_CONCURRENT)]
for t in threads: t.start()
for t in threads: t.join()
wall = time.perf_counter() - t0

total_tok = sum(r[2] for r in results)
aggregate = total_tok / wall
single_avg = sum(r[2]/r[1] for r in results) / len(results)

result = {
    "n_concurrent": N_CONCURRENT,
    "max_tokens": MAX_TOKENS,
    "wall_time_s": wall,
    "total_tokens": total_tok,
    "aggregate_tok_per_s": aggregate,
    "single_req_avg_tok_per_s": single_avg,
    "passed": aggregate >= 150,
    "target": "≥150 tok/s aggregate per GPU (consult prediction 150-200)",
}
OUT.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
print("VERDICT:", "PASS" if result["passed"] else "FAIL")
