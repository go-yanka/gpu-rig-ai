"""ColBERT late-interaction rerank.

We DON'T index ColBERT vectors (3x storage overhead, slow indexing). Instead,
we use ColBERT at rerank time: after dense+sparse RRF gives us 20-30 candidates,
we score each (query, chunk) pair with ColBERT's MaxSim between per-token vecs.

fastembed exposes `colbert-ir/colbertv2.0` as an ONNX model — runs on CPU in
~50ms per pair, so reranking 24 candidates is ~1.2s. Acceptable.

Contract:
  rerank_colbert(query, chunks, top_n=8) -> list of chunks (sorted, top_n)
"""
from __future__ import annotations
import os
from typing import List
import numpy as np

COLBERT_MODEL = os.environ.get('COLBERT_MODEL', 'colbert-ir/colbertv2.0')

_model = None
def get_model():
    global _model
    if _model is None:
        try:
            from fastembed import LateInteractionTextEmbedding
            _model = LateInteractionTextEmbedding(COLBERT_MODEL)
        except Exception as e:
            print(f'[colbert] unavailable: {e}')
            _model = False
    return _model


def _maxsim(q_vec: np.ndarray, d_vec: np.ndarray) -> float:
    """ColBERT MaxSim: sum over query tokens of max similarity to any doc token."""
    # q_vec: (Lq, H), d_vec: (Ld, H), both L2-normalized by model
    sims = q_vec @ d_vec.T           # (Lq, Ld)
    return float(sims.max(axis=1).sum())


def rerank_colbert(query: str, chunks: List[dict], top_n: int = 8) -> List[dict]:
    if not chunks:
        return []
    m = get_model()
    if not m:
        # fallback: keep existing score order
        chunks.sort(key=lambda c: c.get('rerank_score', c.get('score', 0)), reverse=True)
        return chunks[:top_n]
    try:
        q_vecs = list(m.query_embed([query]))
        d_vecs = list(m.embed([c['text'] for c in chunks]))
        q_arr = np.asarray(q_vecs[0])
        for c, dv in zip(chunks, d_vecs):
            d_arr = np.asarray(dv)
            c['colbert_score'] = _maxsim(q_arr, d_arr)
        chunks.sort(key=lambda c: c['colbert_score'], reverse=True)
    except Exception as e:
        print(f'[colbert err] {e}')
        chunks.sort(key=lambda c: c.get('rerank_score', c.get('score', 0)), reverse=True)
    return chunks[:top_n]
