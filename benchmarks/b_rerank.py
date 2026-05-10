#!/usr/bin/env python3
"""B-Rerank: bge-reranker-v2-m3 GGUF via llama-server --reranking on GPU 3 Vulkan.
Measure latency for top-50 rerank (simulating live query load)."""
import json
import time
import statistics
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8765/reranking"
OUT = Path("/mnt/d/_gpu_rig_ai/benchmarks/b_rerank_results.json")

# Representative CBIC chunks (512 chars each ~ 128 tokens)
CHUNKS = [
    "Section 16 of CGST Act provides that every registered person shall be entitled to take credit of input tax charged on any supply of goods or services subject to conditions and restrictions specified.",
    "Notification No. 11/2017-Central Tax (Rate) prescribes GST rates on services. SAC 9954 construction services attract 18% with ITC.",
    "HSN code 8703 covers motor cars and other motor vehicles principally designed for the transport of persons. GST rate is 28% plus compensation cess.",
    "Rule 36 of CGST Rules governs documentary requirements and conditions for claiming input tax credit including tax invoice, debit note, bill of entry.",
    "Section 9(3) of CGST Act provides for reverse charge mechanism on notified categories of supply of goods or services by unregistered supplier to registered recipient.",
    "Customs Tariff First Schedule lists basic customs duty rates. Chapter 85 covers electrical machinery and equipment.",
    "E-way bill is required under rule 138 for inter-state movement of goods of consignment value exceeding Rs 50,000.",
    "Duty drawback rates are notified under section 75 of Customs Act 1962 read with rule 3 of Customs and Central Excise Duties Drawback Rules.",
    "IGST Act Section 5(1) levies integrated GST on all inter-state supplies of goods or services or both except on supply of alcoholic liquor for human consumption.",
    "Composition levy under section 10 of CGST Act is available to registered persons whose aggregate turnover in preceding financial year did not exceed Rs 1.5 crore.",
] * 5  # 50 chunks total

QUERY = "What are the conditions for claiming input tax credit under GST?"

def bench(n_queries=20):
    # warmup
    req = urllib.request.Request(URL, method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"query": QUERY, "documents": CHUNKS[:10]}).encode())
    urllib.request.urlopen(req, timeout=60).read()

    latencies = []
    for i in range(n_queries):
        t = time.perf_counter()
        req = urllib.request.Request(URL, method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"query": QUERY, "documents": CHUNKS}).encode())
        resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
        dt = (time.perf_counter() - t) * 1000
        latencies.append(dt)
        print(f"query {i}: {dt:.1f} ms, {len(resp.get('results', []))} results")

    result = {
        "n_queries": n_queries,
        "n_chunks_per_query": len(CHUNKS),
        "latency_ms": {
            "p50": statistics.median(latencies),
            "p95": sorted(latencies)[int(0.95 * n_queries)],
            "mean": statistics.mean(latencies),
            "min": min(latencies),
            "max": max(latencies),
        },
        "passed": statistics.median(latencies) < 500,  # target <500 ms per consult
        "target_ms": 500,
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("VERDICT:", "PASS" if result["passed"] else "FAIL")

if __name__ == "__main__":
    bench()
