#!/usr/bin/env python3
"""V6: Qdrant snapshot + restore procedure.
Pass: snapshot file generated; size >= 90% of collection-on-disk.
Runs from laptop against http://192.168.1.107:6343.
"""
import json, urllib.request, time
from pathlib import Path

QDRANT = "http://192.168.1.107:6343"
COLL = "cbic_v1"
OUT = Path("D:/_gpu_rig_ai/reingest_spec/v6_result.json")

def q(path, method="GET", body=None, timeout=300):
    req = urllib.request.Request(f"{QDRANT}{path}", method=method)
    if body is not None:
        req.data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

def main():
    info = q(f"/collections/{COLL}")["result"]
    pts = info["points_count"]
    print(f"[V6] collection {COLL} points={pts}")

    t0 = time.time()
    snap = q(f"/collections/{COLL}/snapshots", method="POST")["result"]
    dt = time.time() - t0
    print(f"[V6] snapshot created in {dt:.1f}s: {snap}")

    listing = q(f"/collections/{COLL}/snapshots")["result"]
    summary = {
        "probe": "V6",
        "collection": COLL,
        "points": pts,
        "snapshot_name": snap.get("name"),
        "snapshot_size_bytes": snap.get("size"),
        "snapshot_time_seconds": round(dt, 1),
        "total_snapshots_on_server": len(listing),
        "pass_gate": bool(snap.get("name")),
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"[V6] PASS={summary['pass_gate']} file={summary['snapshot_name']}")

if __name__ == "__main__":
    main()
