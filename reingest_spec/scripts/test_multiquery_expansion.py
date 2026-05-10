#!/usr/bin/env python3
"""Multi-query expansion test on a 50-query subset of v2_gold.

Per consultant brief 2026-05-09: high-ROI lever before fine-tune.
For each gold query, use qwen3 to generate 3 distinct rewrites:
  1. Essential legal ask (strip narrative, keep statute/section/notification)
  2. Formal statutory rewrite (long → short legal terms)
  3. Step-back / general legal concept

Retrieve top-K for each variant, RRF-fuse, measure recall@10 vs gold.
Compare to baseline (single-query retrieval).

Output: /tmp/multiquery_test.json with per-query results and summary.
"""
import json, sys, urllib.request, urllib.error, time
from collections import defaultdict

GOLD = '/opt/indian-legal-ai/reingest_spec/eval/v2_gold.json'
RETRIEVE_API = 'http://127.0.0.1:9500/retrieve'
LLM_URL = 'http://127.0.0.1:9082/v1/chat/completions'
LLM_MODEL = 'qwen3-14b-q4_k_m.gguf'
N_SUBSET = 50
K_RETR = 10  # for recall@10
K_PREFETCH = 30  # per variant before RRF


REWRITE_SYS = (
    '/no_think\n'
    'You are an expert Indian indirect-tax consultant. Given a long client scenario / '
    'business query, output 3 retrieval-optimized rewrites for searching CBIC notifications, '
    'circulars, GST/CGST/IGST/Customs/Central Excise Acts. Output ONLY a JSON array of 3 strings.\n'
    'Variant 1: Essential legal ask — strip narrative (company names, fictional cities, amounts), '
    'keep section numbers, rule numbers, notification numbers, tax types, procedures.\n'
    'Variant 2: Formal statutory rewrite — single short sentence in formal legal language, '
    'as if searching the statute itself.\n'
    'Variant 3: Step-back / general legal concept — what general legal principle or category '
    'does this query touch (e.g., "anti-dumping duty calculation", "input tax credit refund").\n'
    'Output exactly 3 distinct strings as a JSON array. No preamble.'
)


def llm_rewrite(query, retries=2):
    """Return [v1, v2, v3] or [] on failure."""
    body = {
        'model': LLM_MODEL,
        'messages': [
            {'role': 'system', 'content': REWRITE_SYS},
            {'role': 'user', 'content': query},
        ],
        'max_tokens': 400,
        'temperature': 0.0,
    }
    for attempt in range(retries):
        try:
            req = urllib.request.Request(LLM_URL, method='POST',
                data=json.dumps(body).encode(),
                headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=30) as r:
                resp = json.load(r)
            txt = resp['choices'][0]['message'].get('content', '').strip()
            if not txt:
                continue
            arr = json.loads(txt)
            if isinstance(arr, list) and len(arr) >= 1:
                return [str(x).strip() for x in arr if str(x).strip()][:3]
        except Exception:
            if attempt < retries - 1:
                time.sleep(0.5)
                continue
    return []


def retrieve(question, k=10):
    body = json.dumps({'question': question, 'k': k}).encode()
    req = urllib.request.Request(RETRIEVE_API, method='POST', data=body,
                                  headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.load(r)
        return d.get('hits', [])
    except Exception as e:
        return [{'__err__': str(e)[:120]}]


def rrf_fuse(rankings, k=60):
    """Reciprocal Rank Fusion across multiple ranked lists.
    `rankings`: list of lists, each is [doc_id_in_rank_order]
    Returns [doc_id] sorted by fused score.
    """
    score = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            score[doc_id] += 1.0 / (k + rank + 1)
    return sorted(score.keys(), key=lambda d: -score[d])


def main():
    gold = json.load(open(GOLD))[:N_SUBSET]
    print(f'[multiquery] running {len(gold)} queries (subset)', flush=True)

    n_baseline_hit = 0
    n_mq_hit = 0
    n_only_baseline = 0   # baseline hit, mq missed (regression)
    n_only_mq = 0          # mq hit, baseline missed (lift)
    per_query = []

    t0 = time.time()
    for i, g in enumerate(gold):
        q = g.get('question') or g.get('query') or ''
        gold_doc = g.get('doc_id') or g.get('expected_doc_id') or ''
        if not q or not gold_doc:
            continue

        # Baseline retrieve
        baseline_hits = retrieve(q, k=K_RETR)
        baseline_docs = [h.get('doc_id', '?') for h in baseline_hits if '__err__' not in h][:K_RETR]
        baseline_hit = gold_doc in baseline_docs

        # Multi-query rewrites
        variants = llm_rewrite(q)
        if variants:
            # Retrieve K_PREFETCH per variant + original
            all_rankings = [
                [h.get('doc_id', '?') for h in retrieve(q, k=K_PREFETCH) if '__err__' not in h]
            ]
            for v in variants:
                all_rankings.append(
                    [h.get('doc_id', '?') for h in retrieve(v, k=K_PREFETCH) if '__err__' not in h]
                )
            mq_docs = rrf_fuse(all_rankings)[:K_RETR]
        else:
            mq_docs = baseline_docs  # rewrite failed → fall back to baseline
        mq_hit = gold_doc in mq_docs

        n_baseline_hit += int(baseline_hit)
        n_mq_hit += int(mq_hit)
        if baseline_hit and not mq_hit: n_only_baseline += 1
        if mq_hit and not baseline_hit: n_only_mq += 1

        per_query.append({
            'q': q[:200],
            'gold_doc': gold_doc,
            'baseline_hit': baseline_hit,
            'mq_hit': mq_hit,
            'variants': variants,
            'baseline_top3': baseline_docs[:3],
            'mq_top3': mq_docs[:3],
        })

        if (i+1) % 10 == 0:
            elapsed = time.time() - t0
            print(f'  [{i+1}/{len(gold)}] baseline={n_baseline_hit} mq={n_mq_hit} '
                  f'lift={n_only_mq-n_only_baseline:+d} elapsed={elapsed:.0f}s', flush=True)

    summary = {
        'n': len(gold),
        'baseline_recall_at_10': n_baseline_hit / len(gold),
        'mq_recall_at_10': n_mq_hit / len(gold),
        'lift': (n_mq_hit - n_baseline_hit) / len(gold),
        'mq_only_hits': n_only_mq,
        'baseline_only_hits': n_only_baseline,
        'net_lift_count': n_only_mq - n_only_baseline,
        'per_query': per_query,
    }
    json.dump(summary, open('/tmp/multiquery_test.json', 'w'), indent=2)

    print(f'\n=== SUMMARY ===')
    print(f'baseline recall@10: {summary["baseline_recall_at_10"]:.4f}')
    print(f'multi-query recall@10: {summary["mq_recall_at_10"]:.4f}')
    print(f'absolute lift: {summary["lift"]:+.4f}')
    print(f'mq saved {n_only_mq} that baseline missed')
    print(f'mq lost {n_only_baseline} that baseline got')
    print(f'net: {n_only_mq - n_only_baseline:+d}')
    print(f'\nfull report: /tmp/multiquery_test.json')


if __name__ == '__main__':
    main()
