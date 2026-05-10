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

# b8b9p5_v1: LRU cache for query embedding (P5)
import functools as _ft_b8b9p5
@_ft_b8b9p5.lru_cache(maxsize=512)
def _cached_embed_query(q_norm: str):
    return embedder.embed_query(q_norm)
def _cache_info_b8b9p5():
    return _cached_embed_query.cache_info()

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
             prefetch_mult: int = 4, timings: Dict[str, float] | None = None) -> List[dict]:  # b2_subtimings
    """Return k best chunks via RRF fusion of dense + sparse prefetches.
    Each chunk is a dict with payload fields + score.
    """
    import time as _t  # b2_subtimings
    _t0 = _t.perf_counter()
    # b8b9p5_v1: normalize + LRU-cache query embedding
    _q_norm_b8 = (query or '').strip().lower()
    _ci_before = _cached_embed_query.cache_info()
    q = _cached_embed_query(_q_norm_b8)
    _ci_after = _cached_embed_query.cache_info()
    if timings is not None:
        timings['embed_query_ms'] = round((_t.perf_counter()-_t0)*1000, 2)
        timings['embed_cache_hit'] = bool(_ci_after.hits > _ci_before.hits)
        # b11b12a8_v1: expose running cache stats
        timings['embed_cache_stats'] = {'hits': _ci_after.hits, 'misses': _ci_after.misses, 'currsize': _ci_after.currsize}
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
    _t1 = _t.perf_counter()  # b2_subtimings
    try:
        res = qc.query_points(
            collection_name=QCOLL,
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
        res = qc.search(collection_name=QCOLL,
                        query_vector=('dense', dense),
                        limit=k, query_filter=qfilter, with_payload=True)

    if timings is not None: timings['qdrant_ms'] = round((_t.perf_counter()-_t1)*1000, 2)  # b2_subtimings
    out = []
    for p in res:
        d = dict(p.payload)
        d['score'] = float(p.score)
        out.append(d)
    return out


# b22_v1 B19: section-aware post-retrieval augmentation.
# Runs targeted queries for Section/Rule references and common GST legal
# phrases, unions into the hit set. Dedupes on doc_id+chunk_index.
import re as _b19_re
_B19_SEC_RE = _b19_re.compile(r'\b(?:[Ss]ection|[Rr]ule)\s+(\d+[A-Z]?(?:\(\d+\))?(?:\([a-z]\))?)\b')
_B19_PHRASES = [
    'bill-to-ship-to', 'bill to ship to', 'composite supply', 'mixed supply',
    'place of supply', 'time of supply', 'input tax credit',
    'reverse charge',
]

def _b19_chunk_key(c):
    return (c.get('doc_id') or '', c.get('chunk_index', c.get('char_start', '')))

def augment_section_aware(question: str, hits: list, filters=None,
                          k_per_ref: int = 3, max_total: int = 40,
                          timings=None) -> list:
    """Additive retrieval boost. Runs targeted queries for each Section/Rule
    reference and common GST phrases found in the question, unions them into
    `hits` deduped by (doc_id, chunk_index). Returns augmented list capped at
    `max_total`. Never breaks existing retrieval."""
    if not question:
        if timings is not None:
            timings['section_aware_added'] = 0
        return hits
    added = 0
    existing_keys = {_b19_chunk_key(c) for c in hits}
    out = list(hits)
    targets = []
    for m in _B19_SEC_RE.finditer(question):
        targets.append(question[m.start():m.end()])
    ql = question.lower()
    for ph in _B19_PHRASES:
        if ph in ql:
            targets.append(ph)
    seen = set(); uniq_targets = []
    for t in targets:
        tl = t.lower().strip()
        if tl in seen: continue
        seen.add(tl); uniq_targets.append(t)
    if not uniq_targets:
        if timings is not None:
            timings['section_aware_added'] = 0
        return out
    for tgt in uniq_targets:
        if len(out) >= max_total:
            break
        try:
            extra = retrieve(tgt, k=k_per_ref, filters=filters, timings=None)
        except Exception:
            continue
        for c in extra:
            key = _b19_chunk_key(c)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            c['_b19_added_via'] = tgt
            out.append(c)
            added += 1
            if len(out) >= max_total:
                break
    if timings is not None:
        timings['section_aware_added'] = added
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
