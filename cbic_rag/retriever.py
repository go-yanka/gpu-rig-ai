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
    q = embedder.embed_query(query)
    dense = q['dense']
    sp = q['sparse']  # {tok_id: weight}
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


_reranker = None
def get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from FlagEmbedding import FlagReranker
            # CPU-only; the rerank cost on ~20 chunks is manageable
            _reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=False,
                                     device='cpu')
        except Exception as e:
            print(f'[rerank] unavailable: {e}')
            _reranker = False
    return _reranker


def rerank(query: str, chunks: List[dict], top_n: int = 8) -> List[dict]:
    """Cross-encoder rerank via bge-reranker-v2-m3 (CPU). Falls back to score order."""
    if not chunks:
        return []
    rr = get_reranker()
    if rr:
        try:
            pairs = [[query, c['text']] for c in chunks]
            scores = rr.compute_score(pairs, normalize=True)
            if not isinstance(scores, list):
                scores = [float(scores)]
            for c, s in zip(chunks, scores):
                c['rerank_score'] = float(s)
            chunks.sort(key=lambda c: c.get('rerank_score', c['score']), reverse=True)
            return chunks[:top_n]
        except Exception as e:
            print(f'[rerank err] {e}')
    chunks.sort(key=lambda c: c['score'], reverse=True)
    return chunks[:top_n]
