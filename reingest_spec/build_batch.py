"""Batch builder for full re-ingest of cbic_v2 corpus (14,925 docs).

Codified 2026-04-26 — single shared `cbic_v2` Qdrant collection, batched
ingest of ~1500 docs/batch (~10 batches). Manifest sqlite tracks state.
NO collection merge needed — all batches upsert into one collection.

Usage:
  python3 reingest_spec/build_batch.py --batch 1 [--ingest]
    - default: writes /tmp/batch{N}_doc_ids.csv + report
    - --ingest: also kicks off phase-all into cbic_v2

Determinism:
  - SEED=42 + sorted prefix order = same docs picked across reruns
  - Already-ingested docs (from prior batches) skipped via manifest sqlite
  - Set 6 docs (cbic_v2_set6 collection) are NOT excluded — they get
    re-ingested into cbic_v2 cleanly
"""
import json, sqlite3, random, os, sys, time, subprocess, argparse
from collections import Counter, defaultdict

MANIFEST_SCRAPER = "/opt/indian-legal-ai/data/scraped/cbic/_manifest.sqlite"
# 2026-05-07: realigned with ingest_v2.py — both now point at the same manifest.
# Previously this was `_full.sqlite` (which never existed at runtime) while
# ingest_v2.py wrote `_v2.sqlite`. Result: get_already_ingested() returned an
# empty set every call, batches overlapped wildly, batch 8 produced 1500 docs
# all already-done → +0 pts → loop HALT (correct safety check, exposed the bug).
MANIFEST_V2 = "/opt/indian-legal-ai/data/ingest_manifest_v2.sqlite"
QDRANT_COLL_FULL = "cbic_v2"
SEED = 42
BATCH_SIZE = 1500


def get_already_ingested():
    """Read v2 manifest for docs that already have canonical chunks upserted."""
    if not os.path.exists(MANIFEST_V2):
        return set()
    c = sqlite3.connect(MANIFEST_V2)
    try:
        # Check if `chunks` table has `upserted` and `is_canonical` columns
        rows = c.execute(
            "SELECT DISTINCT doc_id FROM chunks WHERE is_canonical=1 AND upserted=1"
        ).fetchall()
        return {r[0] for r in rows if r[0]}
    except Exception as e:
        print(f"[warn] manifest read failed: {e}")
        return set()
    finally:
        c.close()


def stratify_pick(by_cat, n_target, seed):
    """Pick n_target docs across categories proportional to category size."""
    rnd = random.Random(seed)
    total = sum(len(v) for v in by_cat.values())
    picks_by_cat = {}
    remaining = n_target
    cats_sorted = sorted(by_cat.items(), key=lambda kv: -len(kv[1]))
    for i, (cat, pool) in enumerate(cats_sorted):
        if i == len(cats_sorted) - 1:
            n_pick = remaining
        else:
            n_pick = round(n_target * len(pool) / total)
            n_pick = min(n_pick, len(pool), remaining)
        rnd.shuffle(pool)
        picks_by_cat[cat] = pool[:n_pick]
        remaining -= n_pick
    return picks_by_cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, required=True, help="Batch number (1, 2, 3, ...)")
    ap.add_argument("--size", type=int, default=BATCH_SIZE)
    ap.add_argument("--ingest", action="store_true")
    args = ap.parse_args()

    # Read full corpus inventory
    c = sqlite3.connect(MANIFEST_SCRAPER)
    rows = c.execute("SELECT doc_id, category FROM docs WHERE doc_id IS NOT NULL").fetchall()
    c.close()
    print(f"[batch{args.batch}] full corpus: {len(rows)} docs")

    already = get_already_ingested()
    print(f"[batch{args.batch}] already ingested into cbic_v2: {len(already)} docs")

    by_cat = defaultdict(list)
    for did, cat in rows:
        if did not in already:
            by_cat[cat or "unk"].append(did)
    avail = sum(len(v) for v in by_cat.values())
    print(f"[batch{args.batch}] available for batching: {avail} docs across {len(by_cat)} categories")

    # Deterministic per-batch seed: SEED + batch_num so each batch picks
    # disjoint docs in a fixed order across reruns
    picks_by_cat = stratify_pick(by_cat, args.size, SEED + args.batch)
    final = sorted(d for picks in picks_by_cat.values() for d in picks)
    print(f"[batch{args.batch}] FINAL: {len(final)} docs picked")
    for cat, picks in sorted(picks_by_cat.items(), key=lambda kv: -len(kv[1])):
        print(f"  {cat}: {len(picks)}")

    # Persist
    os.makedirs("/opt/indian-legal-ai/data/batches", exist_ok=True)
    csv_path = f"/opt/indian-legal-ai/data/batches/batch{args.batch}_doc_ids.csv"
    open(csv_path, "w").write(",".join(final))
    print(f"[batch{args.batch}] wrote {csv_path}")
    # Per-doc audit log
    log_path = f"/opt/indian-legal-ai/data/batches/batch{args.batch}_audit.json"
    audit = {
        "batch": args.batch,
        "size_target": args.size,
        "size_actual": len(final),
        "seed": SEED + args.batch,
        "ts": int(time.time()),
        "by_category": {k: len(v) for k, v in picks_by_cat.items()},
        "already_ingested_excluded": len(already),
        "doc_ids": final,
    }
    json.dump(audit, open(log_path, "w"), indent=1)
    print(f"[batch{args.batch}] wrote audit {log_path}")

    if args.ingest:
        ts = int(time.time())
        log = f"/opt/indian-legal-ai/logs/reingest_batch{args.batch}_{ts}.log"
        env = os.environ.copy()
        env.update({
            "QDRANT_COLL_V2": QDRANT_COLL_FULL,
            "MANIFEST_V2": MANIFEST_V2,
            "DENSE_ONLY": "1",
            "EMBED_GPUS": "4,5,6",
            "RADV_DEBUG": "nodcc",
            "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT": "1",
            "PYTHONPATH": "/home/user/.local/lib/python3.10/site-packages:/opt/indian-legal-ai/rag/cbic_rag",
            "PYTHONUNBUFFERED": "1",
        })
        cmd = [
            "/usr/bin/python3", "/opt/indian-legal-ai/reingest_spec/ingest_v2.py",
            "--phase", "all",
            "--doc-ids", ",".join(final),
            # NOTE: NO --no-resume — we want resume across batches in same manifest
        ]
        with open(log, "w") as logf:
            p = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                 cwd="/opt/indian-legal-ai")
        print(f"[batch{args.batch}] ingest launched PID={p.pid} log={log}")
        print(f"[batch{args.batch}] expected ~22 min wall-clock at observed rates")


if __name__ == "__main__":
    main()
