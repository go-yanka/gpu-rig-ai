"""Hybrid embedder for CBIC RAG.

Dense: BGE-M3 (1024-d) via Ollama on GPU. Multilingual, handles EN/HI well.
Sparse: BM25 via fastembed (ONNX, CPU). Fast, lexical, ideal for tax-specific
        terminology like section refs, notification numbers, statute names.

Design choice: we use BM25 rather than BGE-M3 sparse because
 1. CPU BGE-M3 sparse is ~1 doc/sec (PyTorch overhead) — too slow for 500K chunks.
 2. BM25 is purely tokenization+IDF — effectively free on CPU.
 3. For domain-specific tax/legal queries, exact-token BM25 outperforms learned
    sparse on precision-critical lookups (section numbers, rule citations).
 4. Combined with dense BGE-M3 (semantic), RRF fusion gives a strong hybrid.

ColBERT multi-vector is added later as a rerank step, not at index time.
"""
from __future__ import annotations
import os, time, threading, itertools
import concurrent.futures as _cf
import requests
from typing import List, Dict

# Comma-separated list of ollama hosts; we round-robin dense embedding requests across them.
# Default spreads across 6 GPUs (one idle GPU reserved for Hermes/LiteLLM).
OLLAMA_URLS = [u.strip() for u in os.environ.get(
    'OLLAMA_URLS',
    'http://127.0.0.1:11434,http://127.0.0.1:11435,http://127.0.0.1:11436,'
    'http://127.0.0.1:11437,http://127.0.0.1:11438,http://127.0.0.1:11439'
).split(',') if u.strip()]
OLLAMA_EMBED_MODEL = os.environ.get('OLLAMA_EMBED_MODEL', 'bge-m3')
DENSE_DIM = 1024

_rr = itertools.cycle(OLLAMA_URLS)
_rr_lock = threading.Lock()
def _next_url():
    with _rr_lock:
        return next(_rr)

_bm25 = None
_bm25_lock = threading.Lock()


def get_bm25():
    global _bm25
    if _bm25 is None:
        with _bm25_lock:
            if _bm25 is None:
                from fastembed import SparseTextEmbedding
                _bm25 = SparseTextEmbedding('Qdrant/bm25')
    return _bm25


def _post_embed(url: str, batch: List[str], retries: int = 2, timeout: int = 300) -> List[List[float]]:
    last = None
    for _ in range(retries + 1):
        try:
            r = requests.post(f'{url}/api/embed',
                              json={'model': OLLAMA_EMBED_MODEL, 'input': batch},
                              timeout=timeout)
            r.raise_for_status()
            embs = r.json().get('embeddings')
            if isinstance(embs, list) and len(embs) == len(batch):
                return embs
            raise RuntimeError(f'bad embed response length {len(embs) if embs else 0} != {len(batch)}')
        except Exception as e:
            last = e
            time.sleep(0.3)
    raise last


def embed_dense_batch(texts: List[str], retries: int = 3) -> List[List[float]]:
    """Simple per-text fallback; used rarely (query path)."""
    return _post_embed(_next_url(), texts, retries=retries, timeout=60)


def embed_dense_bulk(texts: List[str]) -> List[List[float]]:
    """Bulk dense via N parallel Ollama instances. Shards the batch across hosts.

    With N hosts and batch of B, each host gets ~B/N items. Since per-GPU
    throughput is roughly linear in small batches, this gives ~N× speedup.
    """
    n = len(texts)
    if n == 0:
        return []
    hosts = list(OLLAMA_URLS)
    shards = min(len(hosts), n)
    # split texts into `shards` contiguous groups
    step = (n + shards - 1) // shards
    groups = [texts[i:i + step] for i in range(0, n, step)]
    assigned = list(zip(hosts[:len(groups)], groups))

    out = [None] * shards
    def work(i, url, batch):
        out[i] = _post_embed(url, batch)

    with _cf.ThreadPoolExecutor(max_workers=len(assigned)) as ex:
        futs = [ex.submit(work, i, url, g) for i, (url, g) in enumerate(assigned)]
        for f in _cf.as_completed(futs):
            f.result()
    # flatten
    flat: List[List[float]] = []
    for g in out:
        if g is None:
            raise RuntimeError('embed shard missing')
        flat.extend(g)
    return flat


def embed_sparse_batch(texts: List[str]) -> List[Dict[int, float]]:
    """BM25 sparse weights via fastembed. Returns list of {token_hash: weight}.
    Hashes are u64 from mmh3; we keep them as-is (Qdrant sparse supports large ints)."""
    m = get_bm25()
    out = []
    for emb in m.embed(texts):
        d = {}
        # emb.indices and emb.values are numpy arrays
        for idx, val in zip(emb.indices.tolist(), emb.values.tolist()):
            d[int(idx)] = float(val)
        out.append(d)
    return out


def embed_query(text: str) -> dict:
    """Query-side: return both dense + sparse for hybrid search.
    Uses fastembed's BM25 `query_embed` for proper BM25 query scoring (no IDF reuse)."""
    dense = embed_dense_batch([text])[0]
    m = get_bm25()
    # BM25 has separate query_embed that doesn't apply tf-weighting
    q_embs = list(m.query_embed([text]))
    qe = q_embs[0]
    sparse = {int(i): float(v) for i, v in zip(qe.indices.tolist(), qe.values.tolist())}
    return {'dense': dense, 'sparse': sparse}
