#!/usr/bin/env python3
"""Hybrid retrieval over indian_legal_t1_v2:
  1. Dense search   (BGE-small 384d)
  2. BM25 sparse    (Qdrant native sparse)
  3. RRF fusion     (Reciprocal Rank Fusion)
  4. Rerank         (bge-reranker-v2-m3 on :9096) with legal-context prefix

Usage:
  python3 rag_query_v2.py --q "punishment for theft under new criminal code"
  python3 rag_query_v2.py --q "..."  --status current
  python3 rag_query_v2.py --compare "..."  # show v1 vs v2 side-by-side
"""
import os, sys, json, argparse, math, hashlib, urllib.request, re
from collections import Counter

QDRANT = "http://localhost:6333"
COLL_V2 = "indian_legal_t1_v2"
COLL_V1 = "indian_legal_full"
EMBED_URL = "http://localhost:9092/v1/embeddings"
RERANK_URL = "http://localhost:9096/v1/rerank"
IDF_PATH = "/opt/indian-legal-ai/rag/bm25_idf_v2.json"

LEGAL_PREFIX = "Indian statutory law query: "

STOPWORDS = set("""a an the of in on at by for to from with and or but not is are was were be been
being have has had do does did can could would should shall will may might must this that these
those it its as if then than there here which who whom whose what when where why how""".split())
TOK_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{1,}")


def tokenize(text):
    return [t for t in TOK_RE.findall(text.lower())
            if t not in STOPWORDS and len(t) > 1]


def token_to_idx(tok):
    h = hashlib.md5(tok.encode()).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF


def http_json(url, body=None, method="POST", timeout=60):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 method=method)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def embed(text):
    return http_json(EMBED_URL, {"input": [text], "model": "bge"})["data"][0]["embedding"]


def bm25_query_sparse(text, idf):
    toks = tokenize(text)
    tf = Counter(toks)
    idx, val = [], []
    for t, f in tf.items():
        if t not in idf: continue
        w = idf[t] * f  # simple tf*idf at query time (good default for sparse query)
        idx.append(token_to_idx(t))
        val.append(float(w))
    return {"indices": idx, "values": val}


def rrf_fuse(runs, k=60):
    """Reciprocal Rank Fusion. runs = list of lists of point dicts with 'id'."""
    scores = {}
    meta = {}
    for run in runs:
        for rank, p in enumerate(run):
            pid = p["id"]
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
            meta[pid] = p
    ordered = sorted(scores.items(), key=lambda x: -x[1])
    out = []
    for pid, s in ordered:
        m = dict(meta[pid])
        m["rrf"] = s
        out.append(m)
    return out


def search_v2(query, limit=20, status=None, act_year_min=None, dataset=None):
    idf_blob = json.load(open(IDF_PATH))
    idf = idf_blob["idf"]

    must = []
    if status: must.append({"key": "status", "match": {"value": status}})
    if act_year_min: must.append({"key": "act_year", "range": {"gte": int(act_year_min)}})
    if dataset: must.append({"key": "dataset", "match": {"value": dataset}})
    flt = {"must": must} if must else None

    # dense
    dense_vec = embed(query)
    dense_body = {"query": dense_vec, "using": "dense", "limit": limit,
                  "with_payload": True}
    if flt: dense_body["filter"] = flt
    dense_res = http_json(f"{QDRANT}/collections/{COLL_V2}/points/query",
                          dense_body)["result"]["points"]

    # sparse BM25
    sparse = bm25_query_sparse(query, idf)
    if sparse["indices"]:
        sparse_body = {"query": sparse, "using": "bm25", "limit": limit,
                       "with_payload": True}
        if flt: sparse_body["filter"] = flt
        sparse_res = http_json(f"{QDRANT}/collections/{COLL_V2}/points/query",
                               sparse_body)["result"]["points"]
    else:
        sparse_res = []

    # RRF fuse
    fused = rrf_fuse([dense_res, sparse_res])[:limit]
    return fused, dense_res, sparse_res


def rerank(query, candidates, top_k=5):
    if not candidates: return []
    docs = [c["payload"]["text"][:1500] for c in candidates]
    q = LEGAL_PREFIX + query
    r = http_json(RERANK_URL, {"model": "bge-reranker", "query": q, "documents": docs})
    scored = []
    for item in r["results"]:
        idx = item["index"]
        c = dict(candidates[idx])
        c["rerank_score"] = item["relevance_score"]
        scored.append(c)
    scored.sort(key=lambda x: -x["rerank_score"])
    return scored[:top_k]


def fmt_hit(h, idx):
    p = h["payload"]
    act = p.get("act_name", "?")
    sec = p.get("section_no", "") or p.get("anchor", "")
    chap = p.get("chapter_no", "")
    status = p.get("status", "?")
    ds = p.get("dataset", "")
    rr = h.get("rerank_score")
    rrf = h.get("rrf", h.get("score"))
    head = f"[{idx+1}] {act}"
    if chap: head += f" Ch.{chap}"
    if sec: head += f" Sec.{sec}"
    head += f" [{status.upper()}] (ds={ds})"
    if rr is not None: head += f"  rerank={rr:.3f}"
    if rrf is not None: head += f"  rrf/score={rrf:.3f}"
    txt = p.get("text", "")[:300].replace("\n", " ")
    return head + "\n    " + txt


def cmd_query(q, top_k, status=None):
    print(f"\n=== query: {q!r} ===")
    print(f"=== filter: status={status} ===\n")
    fused, dense_res, sparse_res = search_v2(q, limit=25, status=status)
    print(f"--- DENSE top 5 ---")
    for i, h in enumerate(dense_res[:5]): print(fmt_hit(h, i))
    print(f"\n--- SPARSE BM25 top 5 ---")
    for i, h in enumerate(sparse_res[:5]): print(fmt_hit(h, i))
    print(f"\n--- RRF FUSED top 5 ---")
    for i, h in enumerate(fused[:5]): print(fmt_hit(h, i))
    print(f"\n--- RERANKED top {top_k} ---")
    reranked = rerank(q, fused[:20], top_k=top_k)
    for i, h in enumerate(reranked): print(fmt_hit(h, i))


def search_v1(q, limit=5):
    vec = embed(q)
    body = {"vector": vec, "limit": limit, "with_payload": True,
            "filter": {"must": [{"key": "tier", "match": {"value": 1}}]}}
    r = http_json(f"{QDRANT}/collections/{COLL_V1}/points/search", body)
    return r["result"]


def fmt_v1_hit(h, i):
    p = h["payload"]
    s = p.get("source", "?")
    t = p.get("text", "")[:250].replace("\n", " ")
    return f"[{i+1}] {s[:120]}  score={h['score']:.3f}\n    {t}"


def cmd_compare(q, top_k):
    print(f"\n=== COMPARE v1 (dense-only 384d) vs v2 (hybrid+rerank) ===")
    print(f"=== query: {q!r} ===\n")
    print("--- V1 (dense BGE-small only, tier=1 filter) ---")
    v1 = search_v1(q, limit=top_k)
    for i, h in enumerate(v1): print(fmt_v1_hit(h, i))
    print(f"\n--- V2 (dense + BM25 + rerank, legal prefix) ---")
    fused, *_ = search_v2(q, limit=25)
    reranked = rerank(q, fused[:20], top_k=top_k)
    for i, h in enumerate(reranked): print(fmt_hit(h, i))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", help="single query")
    ap.add_argument("--compare", help="v1 vs v2 side-by-side")
    ap.add_argument("--status", choices=["current", "legacy"], default=None)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    if args.compare:
        cmd_compare(args.compare, args.top)
    elif args.q:
        cmd_query(args.q, args.top, status=args.status)
    else:
        ap.error("need --q or --compare")


if __name__ == "__main__":
    main()
