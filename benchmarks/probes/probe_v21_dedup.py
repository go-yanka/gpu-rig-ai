#!/usr/bin/env python3
"""V21 (rewritten): Dedup audit + simulated-post-dedup verification.
Phase 1: scan cbic_v1, compute hash stats (pre-dedup measurement).
Phase 2: run ChunkDeduper on all chunks; verify post-dedup has 0 cross-doc duplicates.
Pass: pre-dedup dup_rate known; post-dedup dup_rate == 0.
Run on rig.
"""
import sys, json, hashlib, time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/opt/indian-legal-ai/reingest_spec")
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "reingest_spec"))
from dedupe_chunks import ChunkDeduper, text_hash

try:
    from qdrant_client import QdrantClient
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "qdrant-client"])
    from qdrant_client import QdrantClient

QDRANT_URL = "http://127.0.0.1:6343"
COLLECTION = "cbic_v1"
OUT = Path("/opt/indian-legal-ai/data/probes/v21_result.json")
OUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    client = QdrantClient(url=QDRANT_URL)
    t0 = time.time()
    dedup = ChunkDeduper()
    hash_to_docs = defaultdict(set)   # hash -> set of doc_ids (for pre-dedup cross-doc count)
    total = 0
    empty = 0
    offset = None
    sample_dups = {}

    while True:
        pts, offset = client.scroll(
            collection_name=COLLECTION,
            limit=2000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not pts: break
        for p in pts:
            total += 1
            pl = p.payload or {}
            text = pl.get("text") or ""
            if not text.strip():
                empty += 1; continue
            doc_id = f"{pl.get('source','?')}:{pl.get('doc_id','?')}"
            chunk_id = pl.get("chunk_id") or str(p.id)
            chunk_obj = {"chunk_id": chunk_id, "doc_id": doc_id, "text": text}
            dedup.add(chunk_obj)
            h = text_hash(text)
            hash_to_docs[h].add(doc_id)
        if offset is None: break

    # Pre-dedup cross-doc stats
    cross_doc_bodies = sum(1 for docs in hash_to_docs.values() if len(docs) > 1)
    unique_bodies = len(hash_to_docs)
    pre_dup_rate = 1 - (unique_bodies / max(total - empty, 1))
    # Sample 3 duplicates
    for h, docs in hash_to_docs.items():
        if len(docs) > 1:
            sample_dups[h[:12]] = sorted(docs)[:10]
            if len(sample_dups) >= 3: break

    # Phase 2: post-dedup verification
    canon = dedup.canonical_chunks()
    seen_hashes = set()
    post_cross_doc = 0
    for c in canon:
        h = c["_text_hash"]
        if h in seen_hashes:
            post_cross_doc += 1
        seen_hashes.add(h)
    post_dup_rate = post_cross_doc / max(len(canon), 1)

    elapsed = round(time.time() - t0, 1)
    summary = {
        "probe": "V21",
        "elapsed_s": elapsed,
        "total_points": total,
        "empty_text_points": empty,
        "pre_dedup": {
            "unique_text_bodies": unique_bodies,
            "cross_doc_dup_bodies": cross_doc_bodies,
            "dup_rate": round(pre_dup_rate, 4),
            "sample_dups_first3": sample_dups,
        },
        "post_dedup": {
            "canonical_chunks": len(canon),
            "duplicates_remaining": post_cross_doc,
            "dup_rate": round(post_dup_rate, 6),
            "saved_chunks": total - empty - len(canon),
            "saved_pct": round(1 - len(canon) / max(total - empty, 1), 4),
        },
        "dedup_stats": dedup.stats,
        "pass_gate": post_dup_rate == 0.0,
    }
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    # Short print
    short = {k: summary[k] for k in ("probe","total_points","pre_dedup","post_dedup","pass_gate")}
    print(json.dumps(short, indent=2))

if __name__ == "__main__":
    main()
