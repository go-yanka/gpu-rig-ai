"""Standalone 6-GPU embed pool bench.

Solo per-GPU 50-call timing → pair {fast, slow} → quad → six → 600-call burst.
Writes results to /tmp/embed_pool_bench_results.json + prints summary.

Run as:
  PYTHONPATH=/opt/indian-legal-ai/rag/cbic_rag:/home/user/.local/lib/python3.10/site-packages \
  EMBED_GPUS=0,1,3,4,5,6 \
  EMBED_PROFILES=/opt/indian-legal-ai/embed_pool_profiles.json \
  python3 /tmp/embed_pool_bench.py
"""
import os, sys, time, json
sys.path.insert(0, "/opt/indian-legal-ai/rag/cbic_rag")
from embedder_direct import get_pool

SAMPLE = "Section 17(5) of the CGST Act 2017 specifies items on which input tax credit is blocked, including motor vehicles for personal use."

def bench_solo(p, gpu_id, n=50):
    t0 = time.time()
    errs = 0
    for _ in range(n):
        try:
            v = p.embed_on(gpu_id, [SAMPLE])
            if not v or len(v[0]) != 1024:
                errs += 1
        except Exception as e:
            errs += 1
            print(f"  gpu{gpu_id} err: {type(e).__name__}: {e}", flush=True)
    dt = time.time() - t0
    s = p.workers[gpu_id].stats()
    return {"gpu": gpu_id, "calls": n, "elapsed_s": round(dt, 2),
            "qps": round(n / dt, 2), "errs": errs,
            "p50_ms": s["p50_ms"], "p95_ms": s["p95_ms"]}

def bench_burst(p, n=600):
    t0 = time.time()
    errs = 0
    for _ in range(n):
        try:
            v = p.embed([SAMPLE])
            if not v or len(v[0]) != 1024:
                errs += 1
        except Exception as e:
            errs += 1
    dt = time.time() - t0
    return {"calls": n, "elapsed_s": round(dt, 2),
            "qps": round(n / dt, 2), "errs": errs}

def main():
    print("[bench] initializing pool...", flush=True)
    p = get_pool()
    h0 = p.health()
    print(f"[bench] pool live: {h0['ready']}", flush=True)

    results = {"ts": time.time(), "initial_health": h0, "solo": {}, "burst": {}}

    print("\n[bench] === SOLO per-GPU 50 calls ===", flush=True)
    for gid in sorted(p.workers.keys()):
        if p.workers[gid].state != "ready":
            print(f"  gpu{gid}: state={p.workers[gid].state}, skipping", flush=True)
            continue
        r = bench_solo(p, gid, n=50)
        print(f"  gpu{gid}: {r['elapsed_s']}s for {r['calls']} = {r['qps']} q/s, p50={r['p50_ms']}ms p95={r['p95_ms']}ms errs={r['errs']}", flush=True)
        results["solo"][str(gid)] = r

    print("\n[bench] === BURST 600 round-robin retrieves (n=1) ===", flush=True)
    # Reset call counters so fairness check is clean
    for h in p.workers.values():
        h.calls = 0
        h.errors = 0
    r = bench_burst(p, n=600)
    print(f"  total: {r['elapsed_s']}s for {r['calls']} = {r['qps']} q/s errs={r['errs']}", flush=True)
    results["burst"] = r

    print("\n[bench] === FAIRNESS (per-GPU calls during burst) ===", flush=True)
    for gid in sorted(p.workers.keys()):
        s = p.workers[gid].stats()
        print(f"  gpu{gid}: state={s['state']} weight={s['weight']} calls={s['calls']} errs={s['errors']} p50={s['p50_ms']}ms", flush=True)
    results["final_health"] = p.health()

    out = "/tmp/embed_pool_bench_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[bench] wrote {out}", flush=True)

if __name__ == "__main__":
    main()
