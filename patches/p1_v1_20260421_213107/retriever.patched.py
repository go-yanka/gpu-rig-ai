# p1_v1 — BM25 boost on statute queries, min-2-non-Act backfill.
"""Hybrid retrieval (dense + sparse BM25) with optional cross-encoder rerank.

Returns chunks with full provenance so the answer formatter can quote + cite.
Uses the same embedder module as ingest so sparse token-IDs align.

p1_v1 changes (A1 — retrieval-side fix):
  * classify_query(q) -> {'tier': 'hard'|'soft'|'none', 'refs': [...]}
  * retrieve() now accepts a qclass kwarg. When tier != 'none' we switch
    from server-side (Qdrant) RRF to client-side RRF so we can apply a
    BM25-side multiplier BEFORE fusion for chunks whose payload.doc_type
    is in ('act','rules'). Multiplier:  hard=3.5, soft=1.8, none=1.0 (unchanged).
  * Because Qdrant's FusionQuery is a black box, we implement the multiplier
    as an effective-rank shift in the BM25 prefetch list
    (new_rank = floor(rank / mult)) so RRF sees boosted chunks earlier.
  * Post-rerank backfill ensures >=2 non-(act|rules) chunks in the final
    top-12 — preserving visibility of clarifying circulars. Exposed as
    helper `enforce_min_non_act(top, pool, min_non_act=2)` and invoked by
    api.py after MMR.  No Qdrant hard payload filter is applied.
"""
from __future__ import annotations
import os, sys, re
from pathlib import Path
from typing import List, Dict, Any, Optional

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


# --------------------------------------------------------------------------
# p1_v1: query classifier (zero-LLM)
# --------------------------------------------------------------------------
STATUTE_REGEX = re.compile(
    r'\b(?:Section|Sec\.?|s\.|Rule|Article)\s+(\d+[A-Z]?)'
    r'(?:\s*\(\s*(\d+[A-Z]?)\s*\))?'
    r'(?:\s*\(\s*([a-z])\s*\))?',
    re.IGNORECASE,
)

LEGAL_PHRASE_SET = {
    'composite supply', 'mixed supply', 'place of supply', 'time of supply',
    'bill-to-ship-to', 'bill to ship to', 'input tax credit', 'reverse charge',
    'zero-rated', 'export of services', 'drc-01b', 'gstr-1', 'gstr-3b', 'gstr-2b',
    'section 75(12)', 'rule 86b', 'rule 88c', 'rule 88d', 'rule 36(4)',
}

_ACT_LIKE = ('act', 'rules')

def classify_query(q: str) -> dict:
    """Classify a query into retrieval tier.

    hard -> explicit Section/Rule/Article reference (e.g. 'Section 16(2)(b)')
    soft -> no explicit reference but a known legal phrase
    none -> ordinary natural-language query
    """
    if not q:
        return {'tier': 'none', 'refs': []}
    hard = STATUTE_REGEX.search(q)
    lq = q.lower()
    soft = any(p in lq for p in LEGAL_PHRASE_SET)
    if hard:
        refs = [g for g in hard.groups() if g]
        return {'tier': 'hard', 'refs': refs}
    if soft:
        return {'tier': 'soft', 'refs': []}
    return {'tier': 'none', 'refs': []}


# Multipliers per tier (spec v3).
_MULT_BY_TIER = {'hard': 3.5, 'soft': 1.8, 'none': 1.0}


def _rrf_fuse(dense_ranked: List[tuple], sparse_ranked: List[tuple],
              k: int, rrf_k: int = 60) -> List[tuple]:
    """Client-side RRF fusion.

    Each input is a list of (point_id, payload, rank) sorted ascending by rank.
    Returns top-k list of (point_id, payload, fused_score) sorted descending.
    """
    scores: Dict[Any, float] = {}
    payloads: Dict[Any, dict] = {}
    for pid, pl, rank in dense_ranked:
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (rrf_k + rank)
        payloads[pid] = pl
    for pid, pl, rank in sparse_ranked:
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (rrf_k + rank)
        payloads.setdefault(pid, pl)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(pid, payloads[pid], s) for pid, s in fused]


def retrieve(query: str, k: int = 20, filters: Dict[str, Any] | None = None,
             prefetch_mult: int = 4, timings: Dict[str, float] | None = None,
             qclass: Optional[Dict[str, Any]] = None) -> List[dict]:  # p1_v1
    """Return k best chunks via RRF fusion of dense + sparse prefetches.

    When qclass['tier'] in ('hard','soft'): fetch dense + sparse separately,
    apply a rank-shift boost to act/rules chunks on the BM25 side, then do
    client-side RRF. When tier == 'none' (default): keep the original fast
    server-side Qdrant FusionQuery path — zero behaviour/perf regression.
    """
    import time as _t  # b2_subtimings
    _t0 = _t.perf_counter()

    if qclass is None:
        qclass = {'tier': 'none', 'refs': []}
    tier = qclass.get('tier', 'none')
    mult = _MULT_BY_TIER.get(tier, 1.0)

    # b8b9p5_v1: normalize + LRU-cache query embedding
    _q_norm_b8 = (query or '').strip().lower()
    _ci_before = _cached_embed_query.cache_info()
    q = _cached_embed_query(_q_norm_b8)
    _ci_after = _cached_embed_query.cache_info()
    if timings is not None:
        timings['embed_query_ms'] = round((_t.perf_counter()-_t0)*1000, 2)
        timings['embed_cache_hit'] = bool(_ci_after.hits > _ci_before.hits)
        timings['embed_cache_stats'] = {
            'hits': _ci_after.hits, 'misses': _ci_after.misses,
            'currsize': _ci_after.currsize,
        }
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

    if tier == 'none':
        # ----- fast path: original server-side RRF (unchanged) -----
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
            res = qc.search(collection_name=QCOLL,
                            query_vector=('dense', dense),
                            limit=k, query_filter=qfilter, with_payload=True)
        if timings is not None:
            timings['qdrant_ms'] = round((_t.perf_counter()-_t1)*1000, 2)
            timings['fusion_mode'] = 'qdrant_rrf'
        out = []
        for p in res:
            d = dict(p.payload)
            d['score'] = float(p.score)
            out.append(d)
        return out

    # ----- boosted path: two separate prefetches + client-side RRF -----
    prefetch_limit = k * prefetch_mult
    try:
        dense_res = qc.query_points(
            collection_name=QCOLL,
            query=dense, using='dense',
            limit=prefetch_limit, query_filter=qfilter,
            with_payload=True,
        ).points
    except Exception:
        dense_res = qc.search(collection_name=QCOLL,
                              query_vector=('dense', dense),
                              limit=prefetch_limit, query_filter=qfilter,
                              with_payload=True)
    try:
        sparse_res = qc.query_points(
            collection_name=QCOLL,
            query=qm.SparseVector(indices=sp_idx, values=sp_val),
            using='sparse',
            limit=prefetch_limit, query_filter=qfilter,
            with_payload=True,
        ).points
    except Exception:
        sparse_res = []

    # Build ranked tuples (1-based ranks).
    dense_ranked = [(p.id, dict(p.payload or {}), i + 1)
                    for i, p in enumerate(dense_res)]

    # Apply BM25-side rank shift for act/rules chunks.
    raw_sparse = [(p.id, dict(p.payload or {}), i + 1)
                  for i, p in enumerate(sparse_res)]
    boosted: List[tuple] = []
    if mult > 1.0:
        for pid, pl, rank in raw_sparse:
            if str(pl.get('doc_type', '')).lower() in _ACT_LIKE:
                new_rank = max(1, int(rank // mult))
                boosted.append((pid, pl, new_rank))
            else:
                boosted.append((pid, pl, rank))
        # Re-sort so RRF sees monotonic ranks; ties broken by original rank.
        boosted.sort(key=lambda t: t[2])
        # Re-number to dense 1..N so RRF math is consistent.
        sparse_ranked = [(pid, pl, i + 1) for i, (pid, pl, _r) in enumerate(boosted)]
    else:
        sparse_ranked = raw_sparse

    fused = _rrf_fuse(dense_ranked, sparse_ranked, k=k)

    if timings is not None:
        timings['qdrant_ms'] = round((_t.perf_counter()-_t1)*1000, 2)
        timings['fusion_mode'] = f'client_rrf_boost_{tier}'
        timings['fusion_mult'] = mult

    out = []
    for pid, pl, score in fused:
        d = dict(pl)
        d['score'] = float(score)
        out.append(d)
    return out


# --------------------------------------------------------------------------
# p1_v1: post-rerank backfill rule
# --------------------------------------------------------------------------
def enforce_min_non_act(top: List[dict], pool: List[dict],
                        min_non_act: int = 2) -> List[dict]:
    """Ensure `top` contains at least `min_non_act` chunks whose doc_type
    is NOT in ('act','rules'). Backfill from `pool` (ordered best->worst)
    by swapping out the lowest-ranked act/rules chunks.

    Non-destructive: returns a new list. Preserves order of kept items.
    """
    if not top:
        return top
    def _is_act(c: dict) -> bool:
        return str(c.get('doc_type', '')).lower() in _ACT_LIKE

    have_non_act = sum(1 for c in top if not _is_act(c))
    if have_non_act >= min_non_act:
        return top

    need = min_non_act - have_non_act
    top_ids = {c.get('chunk_id') or c.get('id') or id(c) for c in top}
    # Candidates from pool: non-act and not already in top.
    candidates = [c for c in pool
                  if not _is_act(c)
                  and (c.get('chunk_id') or c.get('id') or id(c)) not in top_ids]

    if not candidates:
        return top

    # Indices of act/rules chunks in top, lowest-scoring first (end of list).
    act_positions = [i for i, c in enumerate(top) if _is_act(c)]
    act_positions.sort(key=lambda i: top[i].get('rerank_score',
                                                 top[i].get('score', 0.0)))
    # But we want to drop the WORST act chunks, so process lowest score first.
    # (already ascending — pop from the front: the worst.)

    new_top = list(top)
    swaps = min(need, len(act_positions), len(candidates))
    for k_i in range(swaps):
        drop_idx = act_positions[k_i]
        new_top[drop_idx] = candidates[k_i]
    return new_top


_reranker = None
def get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from FlagEmbedding import FlagReranker
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
