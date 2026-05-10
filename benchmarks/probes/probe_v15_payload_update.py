#!/usr/bin/env python3
"""V15: Qdrant payload update-in-place perf (1000 points).
Pass: <=10s for 1000-point batch update.
Runs from laptop.
"""
import json, urllib.request, time
from pathlib import Path

QDRANT = "http://192.168.1.107:6343"
COLL = "cbic_v1"
OUT = Path("D:/_gpu_rig_ai/reingest_spec/v15_result.json")

def q(path, method="GET", body=None, timeout=60):
    req = urllib.request.Request(f"{QDRANT}{path}", method=method)
    if body is not None:
        req.data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

def main():
    # Get 1000 point IDs
    body = {"limit": 1000, "with_payload": False, "with_vector": False}
    r = q(f"/collections/{COLL}/points/scroll", "POST", body)["result"]
    ids = [p["id"] for p in r["points"]]
    print(f"[V15] got {len(ids)} point IDs")

    # Set a temporary payload field on all
    update = {
        "payload": {"_v15_probe": int(time.time())},
        "points": ids,
    }
    t0 = time.time()
    res = q(f"/collections/{COLL}/points/payload?wait=true", "POST", update)
    dt = time.time() - t0
    print(f"[V15] updated {len(ids)} points in {dt:.2f}s")

    # Cleanup: delete the probe field
    clean = {"keys": ["_v15_probe"], "points": ids}
    try:
        q(f"/collections/{COLL}/points/payload/delete?wait=true", "POST", clean)
    except Exception as e:
        print(f"[V15] cleanup warn: {e}")

    summary = {
        "probe": "V15",
        "n_points": len(ids),
        "update_seconds": round(dt, 2),
        "rate_per_sec": round(len(ids)/max(dt, 0.01), 1),
        "pass_gate": dt <= 10.0,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
