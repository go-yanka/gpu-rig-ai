"""Backfill sparse vectors by iterating Qdrant directly (uses real point IDs).
Codified 2026-05-08.
"""
import os, sys, json, time
sys.path.insert(0, '/opt/indian-legal-ai/rag/cbic_rag')
from embedder import embed_sparse_batch
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

QDRANT_URL = 'http://127.0.0.1:6343'
COLLECTION = 'cbic_v2'
BATCH = 64

def main():
    qc = QdrantClient(url=QDRANT_URL, timeout=120)
    cinfo = qc.get_collection(COLLECTION)
    n_total = cinfo.points_count
    print(f'cbic_v2: {n_total} points to backfill sparse for')

    t0 = time.time()
    n_done = 0; n_err = 0; n_skip = 0
    next_offset = None
    batch_count = 0

    while True:
        scroll = qc.scroll(collection_name=COLLECTION, limit=BATCH,
                           offset=next_offset,
                           with_payload=['text','chunk_id'], with_vectors=False)
        points, next_offset = scroll
        if not points: break
        batch_count += 1

        ids = []; texts = []
        for p in points:
            txt = (p.payload or {}).get('text') or ''
            if not txt:
                n_skip += 1; continue
            ids.append(p.id)
            texts.append(txt)

        if not ids: continue

        try:
            sparse_vecs = embed_sparse_batch(texts)
        except Exception as e:
            print(f'sparse batch err at batch {batch_count}: {type(e).__name__}: {e}')
            n_err += len(ids); 
            if next_offset is None: break
            continue

        # Build update points
        upd = []
        for pid, sv in zip(ids, sparse_vecs):
            if isinstance(sv, dict):
                indices = list(sv.keys()); values = list(sv.values())
            else:
                indices = list(sv.indices); values = list(sv.values)
            if not indices:
                n_skip += 1; continue
            upd.append(qm.PointVectors(
                id=pid,
                vector={'sparse': qm.SparseVector(indices=[int(i) for i in indices],
                                                    values=[float(v) for v in values])},
            ))
        if upd:
            try:
                qc.update_vectors(collection_name=COLLECTION, points=upd)
                n_done += len(upd)
            except Exception as e:
                print(f'qdrant update err at batch {batch_count}: {type(e).__name__}: {str(e)[:200]}')
                n_err += len(upd)

        if batch_count % 20 == 0:
            elapsed = time.time() - t0
            rate = (n_done+n_err)/elapsed if elapsed > 0 else 0
            print(f'  batch {batch_count} done={n_done} err={n_err} skip={n_skip} '
                  f'rate={rate:.1f}/s elapsed={elapsed:.0f}s', flush=True)

        if next_offset is None: break

    elapsed = time.time() - t0
    print(f'\nfinished: {n_done} sparse-updated, {n_err} errors, {n_skip} skipped in {elapsed:.0f}s')

if __name__ == '__main__':
    main()
