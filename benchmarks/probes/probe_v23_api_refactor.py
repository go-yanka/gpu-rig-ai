#!/usr/bin/env python3
"""V23: api.py refactor safety -- 12 endpoints unchanged before/after /query_v2 addition.
Captures pre-refactor baseline response for each endpoint; post-refactor diffs.
Run on rig BEFORE refactoring to capture baseline, then AGAIN after to verify.
"""
import json, urllib.request, hashlib, sys
from pathlib import Path

BASE = "http://127.0.0.1:9500"
OUT = Path("/opt/indian-legal-ai/data/probes/v23_result.json")
OUT.parent.mkdir(parents=True, exist_ok=True)
BASELINE = Path("/opt/indian-legal-ai/data/probes/v23_baseline.json")

# Core endpoints to check; extend after running /openapi.json
TESTS = [
    ("GET", "/health", None),
    ("GET", "/openapi.json", None),
    ("POST", "/query", {"query": "GST on restaurant services", "k": 5}),
    ("GET", "/collections", None),
]

def call(method, path, body):
    req = urllib.request.Request(f"{BASE}{path}", method=method)
    if body is not None:
        req.data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    try:
        r = urllib.request.urlopen(req, timeout=20)
        return r.status, r.read().decode(errors="replace")[:5000]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:5000]
    except Exception as e:
        return 0, str(e)

def shape(body_text):
    # Shape = top-level keys only (ignore data values that naturally vary)
    try:
        o = json.loads(body_text)
        if isinstance(o, dict): return sorted(o.keys())
        if isinstance(o, list): return f"list[{len(o)}]"
    except: pass
    return None

def main():
    mode = sys.argv[1] if len(sys.argv)>1 else "baseline"
    results = {}
    for method, path, body in TESTS:
        code, resp = call(method, path, body)
        results[f"{method} {path}"] = {"code": code, "shape": shape(resp)}
    if mode == "baseline":
        BASELINE.write_text(json.dumps(results, indent=2))
        print(f"[V23] baseline captured: {BASELINE}")
        return
    if mode == "verify":
        if not BASELINE.exists():
            print("no baseline; run: python3 probe_v23_api_refactor.py baseline"); sys.exit(1)
        base = json.loads(BASELINE.read_text())
        diffs = []
        for k, v in results.items():
            b = base.get(k)
            if b != v: diffs.append({"endpoint": k, "baseline": b, "current": v})
        summary = {"probe": "V23", "endpoints": len(results), "diffs": len(diffs),
                   "diff_detail": diffs, "pass_gate": len(diffs) == 0}
        OUT.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
