#!/usr/bin/env python3
"""Ingest all CBIC PDFs into the cbic_v1 Qdrant collection.

Hybrid embedding:
- Dense (1024-d BGE-M3) via Ollama on GPU (port 11434)
- Sparse (BGE-M3 lexical weights) via FlagEmbedding on CPU

- Reads manifest at /opt/indian-legal-ai/data/scraped/cbic/_manifest.sqlite
- Picks every doc with a PDF path (path_en preferred; falls back to path_hi)
- Parallel extraction via ProcessPoolExecutor (CPU-bound pdftotext)
- Dense+sparse vectors written to Qdrant in batches

Run:
  python3 ingest.py --resume           # skip doc_ids already in collection
  python3 ingest.py --reset            # drop and recreate collection
  python3 ingest.py --limit 500        # debug: only do N docs

Environment:
  CBIC_MANIFEST=/opt/indian-legal-ai/data/scraped/cbic/_manifest.sqlite
  QDRANT_URL=http://127.0.0.1:6343
  QDRANT_COLL=cbic_v1
  OLLAMA_URL=http://127.0.0.1:11434
  OLLAMA_EMBED_MODEL=bge-m3
"""
from __future__ import annotations
import os, sys, argparse, sqlite3, time, threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from queue import Queue

sys.path.insert(0, str(Path(__file__).parent))
from chunker import chunk_doc
import embedder

MANIFEST = os.environ.get('CBIC_MANIFEST', '/opt/indian-legal-ai/data/scraped/cbic/_manifest.sqlite')
QURL     = os.environ.get('QDRANT_URL', 'http://127.0.0.1:6343')
QCOLL    = os.environ.get('QDRANT_COLL', 'cbic_v1')
BATCH    = int(os.environ.get('EMBED_BATCH', '48'))
# Ollama batch size for dense. Shards across N hosts, so larger batch exploits parallelism.
DENSE_BATCH = int(os.environ.get('DENSE_BATCH', '48'))


def load_docs(limit=None):
    c = sqlite3.connect(MANIFEST)
    c.row_factory = sqlite3.Row
    q = """SELECT doc_id, category, subcategory, title,
                  path_en, path_hi, download_source, source_url, url_en, url_hi
           FROM docs
           WHERE path_en IS NOT NULL OR path_hi IS NOT NULL"""
    if limit:
        q += f' LIMIT {int(limit)}'
    out = []
    for r in c.execute(q):
        path = r['path_en'] or r['path_hi']
        if not path or not os.path.exists(path):
            continue
        out.append({
            'doc_id': r['doc_id'],
            'category': r['category'],
            'subcategory': r['subcategory'],
            'title': r['title'],
            'file_path': path,
            'download_source': r['download_source'] or 'cbic_primary',
            'source_url': r['source_url'] or r['url_en'] or r['url_hi'] or '',
        })
    return out


def chunk_worker(doc):
    try:
        chunks = list(chunk_doc(doc, doc['file_path']))
        return doc['doc_id'], [asdict(c) for c in chunks]
    except Exception:
        return doc['doc_id'], []


def ensure_collection(qc, dim):
    from qdrant_client.http import models as qm
    colls = {c.name for c in qc.get_collections().collections}
    if QCOLL in colls:
        return
    qc.create_collection(
        collection_name=QCOLL,
        vectors_config={'dense': qm.VectorParams(size=dim, distance=qm.Distance.COSINE)},
        sparse_vectors_config={'sparse': qm.SparseVectorParams(index=qm.SparseIndexParams(on_disk=False))},
    )
    for field in ('doc_id', 'category', 'subcategory'):
        try:
            qc.create_payload_index(QCOLL, field_name=field, field_schema='keyword')
        except Exception:
            pass


def existing_doc_ids(qc):
    seen = set()
    off = None
    try:
        while True:
            pts, off = qc.scroll(collection_name=QCOLL, limit=2000,
                                 with_payload=['doc_id'], with_vectors=False, offset=off)
            for p in pts:
                seen.add(p.payload.get('doc_id'))
            if off is None:
                break
    except Exception:
        pass
    return seen


def embed_batch(texts):
    """Parallel: dense via Ollama, sparse via CPU FlagEmbedding, together."""
    dense_holder = {}
    sparse_holder = {}
    err = []
    def d_job():
        try:
            dense_holder['v'] = embedder.embed_dense_bulk(texts)
        except Exception as e:
            err.append(('dense', e))
    def s_job():
        try:
            sparse_holder['v'] = embedder.embed_sparse_batch(texts)
        except Exception as e:
            err.append(('sparse', e))
    t1 = threading.Thread(target=d_job); t2 = threading.Thread(target=s_job)
    t1.start(); t2.start(); t1.join(); t2.join()
    if err:
        raise RuntimeError(f'embed failed: {err}')
    return dense_holder['v'], sparse_holder['v']


def upsert_chunks(qc, chunks, dense, sparse):
    from qdrant_client.http import models as qm
    points = []
    for c, dv, sv in zip(chunks, dense, sparse):
        idx = list(sv.keys()); vals = [float(v) for v in sv.values()]
        pid = abs(hash((c['doc_id'], c['page'], c['char_start']))) % (10**15)
        points.append(qm.PointStruct(
            id=pid,
            vector={'dense': list(dv),
                    'sparse': qm.SparseVector(indices=[int(k) for k in idx], values=vals)},
            payload=c,
        ))
    qc.upsert(QCOLL, points=points, wait=False)
    return len(points)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reset', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()

    from qdrant_client import QdrantClient
    qc = QdrantClient(url=QURL, timeout=120)

    if args.reset:
        try: qc.delete_collection(QCOLL)
        except Exception: pass

    ensure_collection(qc, dim=embedder.DENSE_DIM)

    skip = set()
    if args.resume:
        skip = existing_doc_ids(qc)
        print(f'[resume] {len(skip)} docs already in collection', flush=True)

    docs = [d for d in load_docs(limit=args.limit) if d['doc_id'] not in skip]
    print(f'[ingest] {len(docs)} docs to process', flush=True)
    if not docs:
        print('nothing to do'); return

    # Warm up sparse (BM25 via fastembed).
    print('[warmup] loading BM25...', flush=True)
    embedder.get_bm25()
    print('[warmup] BM25 ready. hitting Ollama for dense warmup...', flush=True)
    _ = embedder.embed_dense_bulk(['warmup'])
    print('[warmup] done. starting ingestion.', flush=True)

    total_chunks = 0
    done_docs = 0
    t0 = time.time()
    buf = []  # chunk dicts awaiting embed

    def flush(final=False):
        nonlocal buf, total_chunks
        while len(buf) >= BATCH or (final and buf):
            batch = buf[:BATCH]; buf = buf[BATCH:]
            texts = [b['text'] for b in batch]
            try:
                dense, sparse = embed_batch(texts)
                n = upsert_chunks(qc, batch, dense, sparse)
                total_chunks += n
            except Exception as e:
                print(f'[embed/upsert err] {e}', flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(chunk_worker, d): d for d in docs}
        for fu in as_completed(futs):
            done_docs += 1
            doc_id, chunks = fu.result()
            buf.extend(chunks)
            flush()
            if done_docs % 50 == 0:
                elapsed = time.time() - t0
                rate = done_docs / max(1, elapsed)
                print(f'[{done_docs}/{len(docs)}] chunks={total_chunks} '
                      f'rate={rate:.2f} docs/s  buffered={len(buf)}',
                      flush=True)
        flush(final=True)

    elapsed = time.time() - t0
    print(f'[DONE] docs={len(docs)} chunks={total_chunks} elapsed={elapsed:.1f}s', flush=True)


if __name__ == '__main__':
    main()
