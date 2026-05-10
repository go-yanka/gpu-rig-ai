#!/usr/bin/env python3
"""G5 — Latency / Cost gate.

Times /query end-to-end across N gold queries. Reports p50/p95/p99 wall-clock
and per-query cost (local qwen3-14b ≈ $0; LiteLLM proxies removed per no-proxy
rule). p95 must be ≤ --p95-threshold (default 8s per SPEC).

Cost model: assumes local qwen3 = $0 (electricity off-budget). If the operator
sets a per-query cost ceiling, supply --cost-per-query; default $0.0 (free).

Pass: p95 ≤ p95-threshold AND avg_cost ≤ cost-threshold AND errors ≤ allow.
"""
from __future__ import annotations
import argparse, json, sys, time, urllib.request, statistics, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API = "http://127.0.0.1:9500/query"


def _query(q, collection, k=10, grounded=False):
    body = {"question": q, "k": k, "collection": collection}
    if grounded: body["grounded"] = True
    req = urllib.request.Request(API, method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read())
    return time.time() - t0, resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="cbic_v2_set5_dfix")
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--p95-threshold", type=float, default=8.0)
    ap.add_argument("--cost-per-query-usd", type=float, default=0.0,
                    help="Local qwen3 = $0; this is a placeholder unless paid backends used.")
    ap.add_argument("--cost-threshold", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=1,
                    help="qwen3 --parallel 1 — keep 1 unless 2nd LLM stood up.")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--grounded", action="store_true")
    ap.add_argument("--allow-errors", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "gate_g5_result.json")
    args = ap.parse_args()

    queries = json.loads(args.gold.read_text()).get("queries", [])
    if args.limit > 0: queries = queries[:args.limit]

    latencies = []; errors = []; per = []
    lock = threading.Lock(); done = 0

    def _run(i, q):
        try:
            lat, resp = _query(q["query"], args.collection, grounded=args.grounded)
            return i, lat, resp, None
        except Exception as e:
            return i, None, None, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(_run, i, q) for i, q in enumerate(queries)]
        for fut in as_completed(futs):
            i, lat, resp, err = fut.result()
            with lock:
                done += 1
                if err:
                    errors.append({"i": i, "error": err})
                    per.append({"i": i, "error": err})
                else:
                    latencies.append(lat)
                    per.append({"i": i, "latency_s": round(lat, 3)})
                if done % 5 == 0:
                    p95 = (statistics.quantiles(latencies, n=20)[18]
                           if len(latencies) >= 5 else (max(latencies) if latencies else 0))
                    print(f"[G5] {done}/{len(queries)} p95~{p95:.2f}s errors={len(errors)}", flush=True)

    n = len(queries)
    if not latencies:
        out = {"gate":"G5", "n":n, "errors":len(errors), "pass_gate":False}
        args.out.write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2)); sys.exit(3)

    latencies.sort()
    p50 = latencies[len(latencies)//2]
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else latencies[-1]
    p99 = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else latencies[-1]
    avg_cost = args.cost_per_query_usd

    passed = (p95 <= args.p95_threshold
              and avg_cost <= args.cost_threshold
              and len(errors) <= args.allow_errors)
    out = {"gate":"G5", "collection":args.collection, "n":n, "ok":len(latencies),
           "errors":len(errors),
           "p50_s":round(p50,3), "p95_s":round(p95,3), "p99_s":round(p99,3),
           "max_s":round(max(latencies),3), "min_s":round(min(latencies),3),
           "p95_threshold":args.p95_threshold,
           "avg_cost_usd":avg_cost, "cost_threshold":args.cost_threshold,
           "grounded":args.grounded, "workers":args.workers,
           "pass_gate":passed, "ts":time.time(),
           "per_item":per}
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k:v for k,v in out.items() if k!="per_item"}, indent=2))
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
