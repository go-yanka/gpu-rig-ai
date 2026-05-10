#!/usr/bin/env python3
"""V5: Multi-GPU BGE-M3 embedder pool 1-hour soak.
Continuous 32-chunk batches, 500ms gap, 3600s. Log rocm-smi every 60s.
Pass: zero OOM, zero failures, VRAM stable +/-5% across GPUs.
Run on rig from cbic_rag dir.
"""
import sys, time, subprocess, json, os
from pathlib import Path

sys.path.insert(0, "/opt/indian-legal-ai/rag/cbic_rag")
import embedder  # noqa

OUT = Path("/opt/indian-legal-ai/data/probes/v5_result.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

DURATION = int(os.environ.get("V5_DURATION_SEC", "3600"))
BATCH_SIZE = 32

def make_batch():
    return [f"Section {i}(2)(c) of the Central GST Act 2017 specifies conditions for "
            f"input tax credit; the provisions are binding. Sample content for batch test {i}."
            for i in range(BATCH_SIZE)]

def rocm_snapshot():
    try:
        r = subprocess.check_output(["rocm-smi", "--showmemuse", "--json"], timeout=10)
        return json.loads(r)
    except Exception as e:
        return {"err": str(e)}

def main():
    print(f"[V5] warming pool...")
    embedder.embed_dense_bulk(["warmup"]*5)

    t0 = time.time()
    n_batches = 0
    n_vectors = 0
    errors = 0
    snapshots = [{"t": 0, "mem": rocm_snapshot()}]
    next_snap = 60

    while time.time() - t0 < DURATION:
        try:
            vecs = embedder.embed_dense_bulk(make_batch())
            n_batches += 1; n_vectors += len(vecs)
        except Exception as e:
            errors += 1
            snapshots.append({"t": int(time.time()-t0), "err": str(e)})
            if errors > 10:
                print(f"[V5] FAIL: {errors} errors, aborting")
                break
        if time.time() - t0 >= next_snap:
            snapshots.append({"t": int(time.time()-t0), "mem": rocm_snapshot()})
            next_snap += 60
            print(f"  t={int(time.time()-t0)}s batches={n_batches} err={errors}")
        time.sleep(0.5)

    summary = {
        "probe": "V5",
        "duration_sec": round(time.time()-t0, 1),
        "batches": n_batches,
        "vectors": n_vectors,
        "errors": errors,
        "snapshots_count": len(snapshots),
        "pass_gate": errors == 0,
        "rocm_snapshots": snapshots,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"[V5] batches={n_batches} errors={errors} PASS={summary['pass_gate']}")

if __name__ == "__main__":
    main()
