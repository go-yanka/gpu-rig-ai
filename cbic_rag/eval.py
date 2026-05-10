"""Eval harness for CBIC RAG.

For each query in eval_set.json, run the full retrieval pipeline and compute:
  - recall@k  (does any retrieved chunk contain any expected_term?)
  - MRR       (rank of first hit)
  - category_route_accuracy (did the router pick the right category?)

We ship knobs: --use-hyde, --use-router, --rerank {colbert,ce,none}, --k.
Running with different settings shows marginal lift of each component.

Usage:
  python3 eval.py --k 10
  python3 eval.py --k 10 --no-hyde
  python3 eval.py --k 10 --rerank none   # raw dense+sparse RRF
  python3 eval.py --k 10 --rerank colbert
"""
from __future__ import annotations
import os, sys, json, argparse, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from retriever import retrieve
from hyde import hyde as hyde_rewrite
from router import route, filters_for


def hit_score(chunks, expected_terms):
    """Return (rank_of_first_hit or None, recall_flag)."""
    for i, c in enumerate(chunks, 1):
        t = (c.get('text') or '').lower()
        for term in expected_terms:
            if term.lower() in t:
                return i, 1
    return None, 0


def run_one(q, k, use_hyde, use_router, reranker):
    filters = None
    route_ok = None
    if use_router:
        cat = route(q['query'])
        if cat:
            route_ok = 1 if cat == q.get('category') else 0
            filters = filters_for(cat)
    search_text = hyde_rewrite(q['query']) if use_hyde else q['query']
    hits = retrieve(search_text, k=max(k * 3, 24), filters=filters)
    if not hits and filters:
        hits = retrieve(search_text, k=max(k * 3, 24), filters=None)
    if reranker == 'colbert':
        from colbert_rerank import rerank_colbert
        top = rerank_colbert(q['query'], hits, top_n=k)
    elif reranker == 'ce':
        from retriever import rerank
        top = rerank(q['query'], hits, top_n=k)
    else:
        hits.sort(key=lambda c: c.get('score', 0), reverse=True)
        top = hits[:k]
    rank, rec = hit_score(top, q['expected_terms'])
    return {'id': q['id'], 'rank': rank, 'recall': rec, 'route_ok': route_ok,
            'n_hits': len(top)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--no-hyde', action='store_true')
    ap.add_argument('--no-router', action='store_true')
    ap.add_argument('--rerank', choices=['colbert', 'ce', 'none'], default='colbert')
    ap.add_argument('--set', default=str(Path(__file__).parent / 'eval_set.json'))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    data = json.loads(Path(args.set).read_text(encoding='utf-8'))
    qs = data['queries']
    print(f'[eval] n={len(qs)}  k={args.k}  hyde={not args.no_hyde}  router={not args.no_router}  rerank={args.rerank}')

    results = []
    t0 = time.time()
    recs, mrrs, routes = [], [], []
    for q in qs:
        r = run_one(q, args.k, not args.no_hyde, not args.no_router, args.rerank)
        results.append(r)
        recs.append(r['recall'])
        mrrs.append(1.0 / r['rank'] if r['rank'] else 0.0)
        if r['route_ok'] is not None:
            routes.append(r['route_ok'])
        flag = '✓' if r['recall'] else '✗'
        rk = r['rank'] if r['rank'] else '-'
        print(f"  {flag} {q['id']:5s}  rank={rk}  | {q['query'][:70]}")

    elapsed = time.time() - t0
    recall = sum(recs) / len(recs)
    mrr = sum(mrrs) / len(mrrs)
    route_acc = (sum(routes) / len(routes)) if routes else None
    print(f'\n[SUMMARY] recall@{args.k}={recall:.3f}  MRR={mrr:.3f}  '
          f'route_acc={route_acc:.3f if route_acc is not None else "n/a"}  '
          f'elapsed={elapsed:.1f}s')

    if args.out:
        Path(args.out).write_text(json.dumps({
            'config': vars(args),
            'recall': recall, 'mrr': mrr, 'route_acc': route_acc,
            'results': results,
        }, indent=2))


if __name__ == '__main__':
    main()
