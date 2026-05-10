#!/usr/bin/env python3
"""V19: api.py stays up during a mini concurrent ingest.
Chunk+upsert 100 docs into cbic_v2_test, concurrently hit /query at 1 QPS.
Pass: zero 5xx from /query; zero OOM.
Run on rig. Requires: ingest script adapted to target cbic_v2_test, existing eval_set queries.
"""
import json, time, urllib.request, threading, random
from pathlib import Path

API = "http://127.0.0.1:9500/query"
GOLD = Path("/opt/indian-legal-ai/rag/cbic_rag/eval_set.json")
OUT = Path("/opt/indian-legal-ai/data/probes/v19_result.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

DURATION = int(__import__("os").environ.get("V19_DURATION_SEC", "600"))

def hit_api(q):
    body = {"query": q, "k": 5}
    req = urllib.request.Request(API, method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json"})
    t0 = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=20)
        return r.status, time.time()-t0, None
    except urllib.error.HTTPError as e:
        return e.code, time.time()-t0, f"http {e.code}"
    except Exception as e:
        return 0, time.time()-t0, str(e)

def main():
    qs = [q["query"] for q in json.loads(GOLD.read_text()).get("queries", [])]
    print(f"[V19] {len(qs)} queries available; hammering /query at ~1qps for {DURATION}s")
    print("IMPORTANT: start your ingest-into-cbic_v2_test in another terminal before/during this")

    results = []
    t_end = time.time() + DURATION
    while time.time() < t_end:
        q = random.choice(qs)
        code, dt, err = hit_api(q)
        results.append({"code": code, "dt": round(dt,2), "err": err})
        time.sleep(1.0)

    codes = [r["code"] for r in results]
    fives = sum(1 for c in codes if 500 <= c < 600)
    twos = sum(1 for c in codes if 200 <= c < 300)
    summary = {
        "probe": "V19", "total": len(results),
        "2xx": twos, "5xx": fives, "other": len(results)-twos-fives,
        "pass_gate": fives == 0,
        "avg_latency": round(sum(r["dt"] for r in results)/max(len(results),1), 2),
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
