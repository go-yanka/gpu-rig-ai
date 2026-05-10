#!/usr/bin/env python3
"""V7: Disk space for dual-collection operation (v1 keep + v2 build).
Pass: free disk >= 1.5x current collection storage.
Needs rig shell for `du`. Laptop variant uses Qdrant telemetry only (coarse).
"""
import json, urllib.request, shutil, sys, os
from pathlib import Path

QDRANT = "http://192.168.1.107:6343"
COLL = "cbic_v1"
OUT_LAPTOP = Path("D:/_gpu_rig_ai/reingest_spec/v7_result.json")
OUT_RIG = Path("/opt/indian-legal-ai/data/probes/v7_result.json")
STORAGE_DIR = "/opt/indian-legal-ai/data/qdrant_storage"

def q(path):
    return json.loads(urllib.request.urlopen(f"{QDRANT}{path}", timeout=30).read())

def main():
    info = q(f"/collections/{COLL}")["result"]
    pts = info["points_count"]
    vec_size = info["config"]["params"]["vectors"]
    summary = {"probe": "V7", "points": pts, "vector_config": vec_size}

    if os.path.exists(STORAGE_DIR):
        # Rig-side run
        import subprocess
        du = subprocess.check_output(["du", "-sb", STORAGE_DIR]).split()[0]
        storage_bytes = int(du)
        total, used, free = shutil.disk_usage(STORAGE_DIR)
        summary.update({
            "storage_dir": STORAGE_DIR,
            "storage_bytes": storage_bytes,
            "disk_free_bytes": free,
            "disk_total_bytes": total,
            "headroom_ratio": round(free / max(storage_bytes, 1), 2),
            "pass_gate": free >= storage_bytes * 1.5,
        })
        OUT_RIG.parent.mkdir(parents=True, exist_ok=True)
        OUT_RIG.write_text(json.dumps(summary, indent=2))
    else:
        # Laptop coarse estimate: 115k points * (1024 floats * 4 bytes + payload ~2KB)
        est_bytes = pts * (1024 * 4 + 2048)
        summary.update({
            "estimated_storage_bytes_coarse": est_bytes,
            "note": "run on rig for real du measurement",
        })
        OUT_LAPTOP.parent.mkdir(parents=True, exist_ok=True)
        OUT_LAPTOP.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
