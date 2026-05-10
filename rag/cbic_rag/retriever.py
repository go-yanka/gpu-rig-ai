"""Hybrid retrieval (dense + sparse BM25) with optional cross-encoder rerank.

Returns chunks with full provenance so the answer formatter can quote + cite.
Uses the same embedder module as ingest so sparse token-IDs align.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))
import embedder

QURL  = os.environ.get('QDRANT_URL', 'http://127.0.0.1:6343')
QCOLL = os.environ.get('QDRANT_COLL', 'cbic_v1')

_client = None
def get_client():
    global _client
    if _client is None:
        from qdrant_client import QdrantClient
        _client = QdrantClient(url=QURL, timeout=30)
    return _client


def retrieve(query: str, k: int = 20, filters: Dict[str, Any] | None = None,
             prefetch_mult: int = 4, collection: str | None = None) -> List[dict]:
    """Return k best chunks via RRF fusion of dense + sparse prefetches.
    Each chunk is a dict with payload fields + score.

    H1/H2 fix: accept explicit `collection` arg so callers (shadow dual-writer,
    v2 evaluators) can target a specific collection without mutating the
    process-wide QDRANT_COLL env var — which is a race under concurrency.
    """
    coll = collection or QCOLL
    import os as _os
    _dense_only = _os.environ.get('DENSE_ONLY','0') in ('1','true','True')
    if _dense_only:
        dense = embedder.embed_dense_batch([query])[0]
        sp_idx=[]; sp_val=[]
    else:
        q = embedder.embed_query(query)
        dense = q['dense']
        sp = q['sparse']
        sp_idx = list(sp.keys())
        sp_val = [float(v) for v in sp.values()]

    from qdrant_client.http import models as qm
    qfilter = None
    if filters:
        conds = []
        for field, val in filters.items():
            if isinstance(val, list):
                conds.append(qm.FieldCondition(key=field, match=qm.MatchAny(any=val)))
            else:
                conds.append(qm.FieldCondition(key=field, match=qm.MatchValue(value=val)))
        qfilter = qm.Filter(must=conds)

    qc = get_client()
    try:
        if _dense_only:
            res = qc.query_points(
                collection_name=coll,
                query=dense, using='dense',
                limit=k, query_filter=qfilter,
                with_payload=True,
            ).points
        else:
            res = qc.query_points(
                collection_name=coll,
                prefetch=[
                    qm.Prefetch(query=dense, using='dense',
                                limit=k * prefetch_mult, filter=qfilter),
                    qm.Prefetch(query=qm.SparseVector(indices=sp_idx, values=sp_val),
                                using='sparse', limit=k * prefetch_mult, filter=qfilter),
                ],
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                limit=k,
                with_payload=True,
            ).points
    except Exception:
        # fallback: dense-only
        res = qc.search(collection_name=coll,
                        query_vector=('dense', dense),
                        limit=k, query_filter=qfilter, with_payload=True)

    out = []
    for p in res:
        d = dict(p.payload)
        d['score'] = float(p.score)
        out.append(d)
    return out


_RERANK_URL = __import__('os').environ.get('RERANK_URL', 'http://127.0.0.1:9085/v1/rerank')
_RERANK_TIMEOUT = float(__import__('os').environ.get('RERANK_TIMEOUT_S', '15'))

def get_reranker():
    return True  # HTTP service health implied; errors handled in rerank()


# 2026-04-25 O4: per-doc truncation. bge-reranker uses c=8192 context.
# Query is small (<200 chars typical); reserve 7000 chars per doc to keep
# total tokens safely under context budget. Long chunks were silently
# producing degraded scores when total tokens > 8192.
_RERANK_DOC_MAX_CHARS = int(__import__('os').environ.get('RERANK_DOC_MAX_CHARS', '6000'))

def rerank(query: str, chunks: List[dict], top_n: int = 8) -> List[dict]:
    """CE rerank via bge-reranker-v2-m3 GGUF on Vulkan llama-server (port 9085). Falls back to dense score on error.

    2026-04-25 O1+O4 hardening:
      - Already batched: one HTTP call with all docs in `documents:[...]` (verified).
      - Per-doc truncation to RERANK_DOC_MAX_CHARS (default 6000) so we never blow
        the 8192-token reranker context. Truncation is for scoring only; original
        chunk text remains in the returned dict.
    """
    if not chunks:
        return []
    try:
        import requests
        docs = [(c.get('text') or '')[:_RERANK_DOC_MAX_CHARS] for c in chunks]
        r = requests.post(_RERANK_URL,
            json={'model': 'bge-reranker-v2-m3', 'query': query, 'documents': docs},
            timeout=_RERANK_TIMEOUT)
        r.raise_for_status()
        results = r.json().get('results', [])
        for item in results:
            idx = item['index']
            if 0 <= idx < len(chunks):
                chunks[idx]['rerank_score'] = float(item['relevance_score'])
        chunks.sort(key=lambda c: c.get('rerank_score', -1e9), reverse=True)
        return chunks[:top_n]
    except Exception as e:
        print(f'[rerank err] {e}')
        chunks.sort(key=lambda c: c['score'], reverse=True)
        return chunks[:top_n]
