#!/usr/bin/env python3
"""Diagnose the 60 G1 misses on cbic_v2 (codified 2026-05-08).

Reads /tmp/g1_v3_result.json (from gate_g1_recall.py) plus v2_gold.json,
identifies which gold queries had a miss, then for each miss reports:
  - gold doc_id + category + subcategory
  - query text + length
  - what was retrieved (top-3 doc_ids) so we can see WHERE the model went wrong
  - whether the gold doc_id is in linked clusters
  - chunk type (act / circular / notification / form / ...)

This tells us if misses are:
  (a) concentrated in a specific category (gst vs customs vs central_excise)
  (b) all long-narrative scenario queries
  (c) all in the cluster-shared D-1 docs
  (d) random / no pattern → fine-tune is the right lever

Output: /tmp/g1_miss_analysis.json + console summary table.
"""
import json, sys, urllib.request
from collections import Counter, defaultdict

GOLD = '/opt/indian-legal-ai/reingest_spec/eval/v2_gold.json'
RESULT = '/tmp/g1_v3_result.json'
RETRIEVE_API = 'http://127.0.0.1:9500/retrieve'


def fetch_top3(question, k=3):
    body = json.dumps({'question': question, 'k': k}).encode()
    req = urllib.request.Request(RETRIEVE_API, method='POST', data=body,
                                  headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.load(r)
        hits = d.get('hits', [])
        return [h.get('doc_id', '?') for h in hits[:k]]
    except Exception as e:
        return [f'ERR:{type(e).__name__}']


def main():
    gold = json.load(open(GOLD))
    res = json.load(open(RESULT))

    # gate_g1_recall.py writes per_item but it's None in our run — we need to
    # recompute misses by re-querying. The gold list and the per_item list are
    # parallel by index, so we can map the boolean miss to the gold entry.
    # But per_item came back as nulls. We have to re-query.
    print(f'Re-running diagnostic against {len(gold)} gold queries...', flush=True)

    misses = []
    by_cat = Counter()
    by_subcat = Counter()
    by_len_band = Counter()
    miss_details = []

    n_done = 0
    for i, g in enumerate(gold):
        q = g.get('question') or g.get('query') or ''
        gold_doc = g.get('doc_id') or g.get('expected_doc_id') or ''
        cat = g.get('category', '?')
        subcat = g.get('subcategory', '?')
        # length band
        L = len(q)
        if L < 100: lb = 'short<100'
        elif L < 300: lb = 'mid100-300'
        elif L < 600: lb = 'long300-600'
        else: lb = 'xlong>=600'

        retrieved = fetch_top3(q, k=10)
        n_done += 1
        if gold_doc not in retrieved:
            # Miss — record details
            by_cat[cat] += 1
            by_subcat[f'{cat}/{subcat}'] += 1
            by_len_band[lb] += 1
            miss_details.append({
                'i': i,
                'question': q[:200],
                'q_len': L,
                'gold_doc_id': gold_doc,
                'category': cat,
                'subcategory': subcat,
                'retrieved_top3': retrieved[:3],
                'len_band': lb,
            })
        if n_done % 50 == 0:
            print(f'  {n_done}/{len(gold)} (misses so far: {len(miss_details)})', flush=True)

    summary = {
        'n_total': len(gold),
        'n_misses': len(miss_details),
        'recall_at_10': 1.0 - len(miss_details)/len(gold),
        'misses_by_category': dict(by_cat),
        'misses_by_subcategory': dict(by_subcat),
        'misses_by_length_band': dict(by_len_band),
        'misses': miss_details,
    }
    json.dump(summary, open('/tmp/g1_miss_analysis.json', 'w'), indent=2)

    print('\n=== SUMMARY ===')
    print(f'misses: {len(miss_details)}/{len(gold)}  recall: {summary["recall_at_10"]:.4f}')
    print('\nby category:')
    for cat, n in by_cat.most_common():
        print(f'  {cat}: {n}')
    print('\nby length band:')
    for lb, n in by_len_band.most_common():
        print(f'  {lb}: {n}')
    print('\nfull report at /tmp/g1_miss_analysis.json')


if __name__ == '__main__':
    main()
