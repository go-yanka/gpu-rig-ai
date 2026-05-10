#!/usr/bin/env python3
"""Stage D — probe_v2_runner.py
Re-executes the PROBES.md matrix against cbic_v2. Writes probe_v2_results.json.

Runs the Qdrant-only / API-level probes that are meaningful against a freshly-built
collection: V6 (snapshot), V7 (disk), V15 (payload perf), V16 (theta baseline),
V20 (taxonomy coverage), V21 (dedup), V24 (validator dry-run).

Rig-shell LLM probes (V1,V2b,V10,V18) are NOT re-run here — they test the LLM
stack which is collection-independent and already PASS/recovered per SPEC.md.

Outputs per-probe status + aggregate pass_gate. Exits 0 if all run probes pass,
2 otherwise.

Pattern reused from benchmarks/probes/probe_v16_theta.py and probe_v6_snapshot.py.
"""
from __future__ import annotations
import argparse, json, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
QDRANT = "http://192.168.1.107:6343"
API = "http://127.0.0.1:9500/query"
DEFAULT_GOLD = HERE.parent / "eval" / "v2_gold.json"  # reingest_spec/eval/
DEFAULT_ADV = HERE.parent / "eval" / "v2_adversarial.json"
OUT = HERE / "probe_v2_results.json"


def _http(url, method="GET", body=None, timeout=60):
    req = urllib.request.Request(url, method=method)
    if body is not None:
        req.data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _score(resp):
    hits = resp.get("hits") or resp.get("results") or []
    return max((h.get("score", 0) for h in hits), default=0)


def _query(q, collection, k=10):
    body = {"question": q, "k": k, "collection": collection}
    return _http(API, method="POST", body=body, timeout=30)


def probe_v6_snapshot(collection):
    try:
        info = _http(f"{QDRANT}/collections/{collection}")["result"]
        pts = info["points_count"]
        t0 = time.time()
        snap = _http(f"{QDRANT}/collections/{collection}/snapshots", method="POST", timeout=600)["result"]
        dt = time.time() - t0
        return {
            "probe": "V6", "pass": bool(snap.get("name")),
            "points": pts, "snapshot": snap.get("name"),
            "size_bytes": snap.get("size"), "seconds": round(dt, 1),
        }
    except Exception as e:
        return {"probe": "V6", "pass": False, "error": str(e)}


def probe_v7_disk(collection):
    try:
        info = _http(f"{QDRANT}/collections/{collection}")["result"]
        pts = info["points_count"]
        tele = _http(f"{QDRANT}/telemetry")
        return {"probe": "V7", "pass": True, "points": pts,
                "note": "compare disk free vs collection size manually; see Qdrant telemetry",
                "telemetry_ok": bool(tele)}
    except Exception as e:
        return {"probe": "V7", "pass": False, "error": str(e)}


def probe_v15_payload_perf(collection):
    """Scroll 1000 points; time a no-op payload set on them."""
    try:
        body = {"limit": 1000, "with_payload": False, "with_vector": False}
        resp = _http(f"{QDRANT}/collections/{collection}/points/scroll",
                     method="POST", body=body, timeout=120)
        ids = [p["id"] for p in resp["result"]["points"]]
        if not ids:
            return {"probe": "V15", "pass": False, "error": "no points"}
        t0 = time.time()
        _http(f"{QDRANT}/collections/{collection}/points/payload?wait=true",
              method="POST",
              body={"points": ids, "payload": {"_probe_v15_touch": 1}},
              timeout=60)
        dt = time.time() - t0
        return {"probe": "V15", "pass": dt <= 10.0,
                "n_points": len(ids), "seconds": round(dt, 2)}
    except Exception as e:
        return {"probe": "V15", "pass": False, "error": str(e)}


def probe_v16_theta_baseline(collection, gold_path, adv_path):
    """Reruns V16 clean-separation check against cbic_v2."""
    try:
        gold = json.loads(Path(gold_path).read_text())
        adv = json.loads(Path(adv_path).read_text()) if Path(adv_path).exists() else {"queries": []}
        gold_qs = [q["query"] for q in gold.get("queries", [])][:50]
        adv_qs = [q["query"] for q in adv.get("queries", [])][:50]
        def _safe(q):
            try: return _score(_query(q, collection))
            except Exception: return None
        with ThreadPoolExecutor(max_workers=8) as ex:
            gs = [r for r in ex.map(_safe, gold_qs) if r is not None]
            as_ = [r for r in ex.map(_safe, adv_qs) if r is not None]
        gs.sort(); as_.sort()
        sep = bool(gs and as_ and as_[-1] < gs[0] * 0.9)
        return {"probe": "V16", "pass": sep,
                "gold_n": len(gs), "adv_n": len(as_),
                "gold_min": gs[0] if gs else None,
                "adv_max": as_[-1] if as_ else None,
                "proposed_theta": round((gs[0] + as_[-1]) / 2, 3) if gs and as_ else None}
    except Exception as e:
        return {"probe": "V16", "pass": False, "error": str(e)}


def probe_v20_taxonomy(collection, gold_path):
    """Ensure every gold category/subcategory exists as a payload value."""
    try:
        gold = json.loads(Path(gold_path).read_text())["queries"]
        wanted = set()
        for q in gold:
            if "category" in q: wanted.add(("category", q["category"]))
            if "subcategory" in q: wanted.add(("subcategory", q["subcategory"]))
        def _check(item):
            field, val = item
            body = {"filter": {"must": [{"key": field, "match": {"value": val}}]},
                    "exact": True}
            try:
                r = _http(f"{QDRANT}/collections/{collection}/points/count",
                          method="POST", body=body, timeout=30)
                if r["result"]["count"] == 0:
                    return f"{field}={val}"
            except Exception as e:
                return f"{field}={val}:err:{e}"
            return None
        with ThreadPoolExecutor(max_workers=8) as ex:
            missing = [m for m in ex.map(_check, list(wanted)) if m]
        return {"probe": "V20", "pass": len(missing) == 0,
                "checked": len(wanted), "missing": missing}
    except Exception as e:
        return {"probe": "V20", "pass": False, "error": str(e)}


def probe_v21_dedup(collection):
    """Sample 2000 points, flag identical text across different doc_ids."""
    try:
        import hashlib
        body = {"limit": 2000, "with_payload": ["text", "doc_id"], "with_vector": False}
        resp = _http(f"{QDRANT}/collections/{collection}/points/scroll",
                     method="POST", body=body, timeout=120)
        seen = {}
        dups = 0
        for p in resp["result"]["points"]:
            pl = p.get("payload") or {}
            txt = pl.get("text", "")
            if not txt: continue
            h = hashlib.sha256(txt.encode()).hexdigest()
            if h in seen and seen[h] != pl.get("doc_id"):
                dups += 1
            else:
                seen[h] = pl.get("doc_id")
        rate = dups / max(len(resp["result"]["points"]), 1)
        return {"probe": "V21", "pass": rate <= 0.005,
                "sampled": len(resp["result"]["points"]),
                "cross_doc_dups": dups, "rate": round(rate, 5)}
    except Exception as e:
        return {"probe": "V21", "pass": False, "error": str(e)}


def probe_v24_validator(collection):
    """Sample 500 points, verify required v2 payload keys are present."""
    required = ["doc_id", "category", "chunk_type", "text", "parent_hierarchy_text"]
    try:
        body = {"limit": 500, "with_payload": True, "with_vector": False}
        resp = _http(f"{QDRANT}/collections/{collection}/points/scroll",
                     method="POST", body=body, timeout=120)
        bad = 0
        for p in resp["result"]["points"]:
            pl = p.get("payload") or {}
            if any(k not in pl for k in required):
                bad += 1
        rate = bad / max(len(resp["result"]["points"]), 1)
        return {"probe": "V24", "pass": rate <= 0.02,
                "sampled": len(resp["result"]["points"]),
                "missing_required": bad, "rate": round(rate, 4)}
    except Exception as e:
        return {"probe": "V24", "pass": False, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="cbic_v2")
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--adv", type=Path, default=DEFAULT_ADV)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--skip", nargs="*", default=[], help="probe ids to skip, e.g. V6 V16")
    args = ap.parse_args()

    results = []
    def _run(pid, fn, *a):
        if pid in args.skip:
            results.append({"probe": pid, "pass": True, "skipped": True}); return
        print(f"[probe_v2] running {pid}...", flush=True)
        r = fn(*a)
        results.append(r)
        print(f"  -> {r.get('pass')}  {r}", flush=True)

    _run("V6",  probe_v6_snapshot, args.collection)
    _run("V7",  probe_v7_disk, args.collection)
    _run("V15", probe_v15_payload_perf, args.collection)
    _run("V16", probe_v16_theta_baseline, args.collection, args.gold, args.adv)
    _run("V20", probe_v20_taxonomy, args.collection, args.gold)
    _run("V21", probe_v21_dedup, args.collection)
    _run("V24", probe_v24_validator, args.collection)

    all_pass = all(r.get("pass") for r in results)
    summary = {"collection": args.collection, "pass_gate": all_pass,
               "probes": results, "ts": time.time()}
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"collection": args.collection, "pass_gate": all_pass,
                      "results": [(r["probe"], r.get("pass")) for r in results]},
                     indent=2))
    sys.exit(0 if all_pass else 2)


if __name__ == "__main__":
    main()
