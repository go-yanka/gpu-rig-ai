#!/usr/bin/env python3
"""Sparse-backfill that bypasses Qdrant scroll (which panics on corrupt segments
in `red` collections — observed 2026-05-08 cbic_v2).

Strategy: read all chunk identity tuples (doc_id, page, char_start) from the
manifest, derive the SHA256 point_id deterministically, retrieve text + missing
points in chunks via /points (not scroll), embed sparse, update_vectors.

Idempotent: a point that already has sparse will just get rewritten.
"""
import os, sys, json, time, hashlib, sqlite3
sys.path.insert(0, '/opt/indian-legal-ai/rag/cbic_rag')
from embedder import embed_sparse_batch
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

QDRANT_URL = 'http://127.0.0.1:6343'
COLLECTION = 'cbic_v2'
MANIFEST = '/opt/indian-legal-ai/data/ingest_manifest_v2.sqlite'
BATCH = 64


def derive_pid(doc_id, page, char_start):
    """Match ingest.py's deterministic point_id formula."""
    key = f"{doc_id}|{page or 0}|{char_start or 0}".encode()
    return int(hashlib.sha256(key).hexdigest()[:15], 16) % (10**15)


def main():
    qc = QdrantClient(url=QDRANT_URL, timeout=120)
    n_total = qc.get_collection(COLLECTION).points_count
    print(f'cbic_v2: {n_total} points; backfilling sparse via manifest IDs')

    # Build full list of point_ids from manifest's canonical chunks
    c = sqlite3.connect(MANIFEST)
    rows = list(c.execute("""
        SELECT chunk_id, payload_json FROM chunks
        WHERE is_canonical=1 AND upserted=1
    """))
    print(f'manifest: {len(rows)} canonical+upserted chunks')

    pids = []
    pid_to_text = {}
    for chunk_id, pj in rows:
        try:
            p = json.loads(pj)
        except Exception:
            continue
        doc_id = p.get('doc_id')
        page = p.get('page', 0)
        char_start = p.get('char_start', 0)
        if not doc_id:
            continue
        pid = derive_pid(doc_id, page, char_start)
        text = p.get('text') or ''
        if not text:
            continue
        pids.append(pid)
        pid_to_text[pid] = text

    print(f'derived {len(pids)} point_ids with text; starting backfill')

    t0 = time.time()
    n_done = 0; n_err = 0
    for i in range(0, len(pids), BATCH):
        sub_pids = pids[i:i+BATCH]
        sub_texts = [pid_to_text[p] for p in sub_pids]
        try:
            sparse_vecs = embed_sparse_batch(sub_texts)
        except Exception as e:
            print(f'  embed fail at i={i}: {type(e).__name__}: {e}', flush=True)
            n_err += len(sub_pids); continue
        try:
            qc.update_vectors(
                collection_name=COLLECTION,
                points=[
                    qm.PointVectors(
                        id=pid,
                        vector={'sparse': qm.SparseVector(
                            indices=[int(k) for k in sv.keys()],
                            values=[float(v) for v in sv.values()],
                        )}
                    )
                    for pid, sv in zip(sub_pids, sparse_vecs)
                ],
            )
            n_done += len(sub_pids)
        except Exception as e:
            print(f'  update fail at i={i}: {type(e).__name__}: {str(e)[:200]}', flush=True)
            n_err += len(sub_pids)

        if (i // BATCH) % 20 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (len(pids) - n_done - n_err) / rate if rate > 0 else 0
            print(f'  i={i} done={n_done} err={n_err} rate={rate:.1f}/s '
                  f'elapsed={elapsed:.0f}s eta={eta:.0f}s', flush=True)

    elapsed = time.time() - t0
    print(f'\nfinished: {n_done} backfilled, {n_err} errors, {elapsed:.0f}s')


if __name__ == '__main__':
    main()
