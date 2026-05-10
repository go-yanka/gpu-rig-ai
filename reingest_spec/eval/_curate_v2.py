"""Curate v2 gold + adversarial eval sets from existing training_pairs.
No new queries are drafted; this file only samples and reshapes existing data.
Run once: python _curate_v2.py
"""
import json, os, random
from collections import Counter, defaultdict

random.seed(42)

TP = 'D:/_gpu_rig_ai/eval/training_pairs'
OUT_DIR = 'D:/_gpu_rig_ai/reingest_spec/eval'
os.makedirs(OUT_DIR, exist_ok=True)


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def norm_category(c):
    if not c:
        return 'unknown'
    c = c.lower().strip()
    # Keep native categories; map 'others' -> 'others'
    return c


# ------------------------------------------------------------------
# Build a pool of (category, query, expected_doc_id, expected_section, source)
# ------------------------------------------------------------------
gold_pool = []

# qa_gemini — use only answerable=1 AND specific=1
for d in load_jsonl(f'{TP}/qa_gemini.jsonl'):
    g = d.get('grading') or {}
    if g.get('answerable') != 1 or g.get('specific') != 1:
        continue
    q = d.get('question')
    if not q:
        continue
    gold_pool.append({
        'query': q,
        'expected_doc_id': None,  # qa_gemini has chunk_id only, not doc_id
        'expected_chunk_id': d.get('chunk_id'),
        'expected_section': None,
        'category': norm_category(d.get('category')),
        '_source': 'qa_gemini.jsonl',
    })

# qa_sonnet_high — answerable=1 and specific=1
for d in load_jsonl(f'{TP}/qa_sonnet_high.jsonl'):
    g = d.get('grading') or {}
    if g.get('answerable') != 1 or g.get('specific') != 1:
        continue
    q = d.get('question')
    if not q:
        continue
    gold_pool.append({
        'query': q,
        'expected_doc_id': None,
        'expected_chunk_id': d.get('chunk_id'),
        'expected_section': None,
        'category': norm_category(d.get('category')),
        '_source': 'qa_sonnet_high.jsonl',
    })

# pairs_2000_20260422 — Gemini-generated, has doc_id + title + section_ref
for d in load_jsonl(f'{TP}/pairs_2000_20260422.jsonl'):
    cat = norm_category(d.get('category'))
    doc_id = d.get('doc_id')
    section = d.get('section_ref') or d.get('title') or ''
    for qobj in (d.get('questions') or []):
        q = qobj.get('q') or qobj.get('question')
        if not q:
            continue
        gold_pool.append({
            'query': q,
            'expected_doc_id': doc_id,
            'expected_chunk_id': d.get('chunk_id'),
            'expected_section': section,
            'category': cat,
            '_source': 'pairs_2000_20260422.jsonl',
        })

# pairs_opus_highcomplex — high-complexity scenario queries
for d in load_jsonl(f'{TP}/pairs_opus_highcomplex.jsonl'):
    cat = norm_category(d.get('category'))
    doc_id = d.get('doc_id')
    section = d.get('section_ref') or d.get('title') or ''
    for qobj in (d.get('questions') or []):
        q = qobj.get('q') or qobj.get('question')
        if not q:
            continue
        gold_pool.append({
            'query': q,
            'expected_doc_id': doc_id,
            'expected_chunk_id': d.get('chunk_id'),
            'expected_section': section,
            'category': cat,
            '_source': 'pairs_opus_highcomplex.jsonl',
        })

# pairs_sonnet_lowcomplex — simple factoid queries (definitions, dates, rates)
for d in load_jsonl(f'{TP}/pairs_sonnet_lowcomplex.jsonl'):
    cat = norm_category(d.get('category'))
    doc_id = d.get('doc_id')
    section = d.get('section_ref') or d.get('title') or ''
    for qobj in (d.get('questions') or []):
        q = qobj.get('q') or qobj.get('question')
        if not q:
            continue
        gold_pool.append({
            'query': q,
            'expected_doc_id': doc_id,
            'expected_chunk_id': d.get('chunk_id'),
            'expected_section': section,
            'category': cat,
            '_source': 'pairs_sonnet_lowcomplex.jsonl',
        })

print(f'[gold_pool] size={len(gold_pool)}')
src_counts = Counter(p['_source'] for p in gold_pool)
cat_counts = Counter(p['category'] for p in gold_pool)
print(f'[gold_pool] sources: {dict(src_counts)}')
print(f'[gold_pool] categories: {dict(cat_counts)}')

# ------------------------------------------------------------------
# Stratified sample ~380 gold queries across categories.
# Target proportional coverage but floor each category at 20 (if available).
# Prefer entries with doc_id populated (richer provenance), but don't exclude
# qa_gemini which lacks doc_id.
# ------------------------------------------------------------------
TARGET_TOTAL = 380
by_cat = defaultdict(list)
for p in gold_pool:
    by_cat[p['category']].append(p)

# Rank within category: entries with doc_id first, then others; shuffle each bucket.
for c in by_cat:
    with_doc = [p for p in by_cat[c] if p.get('expected_doc_id')]
    without_doc = [p for p in by_cat[c] if not p.get('expected_doc_id')]
    random.shuffle(with_doc)
    random.shuffle(without_doc)
    by_cat[c] = with_doc + without_doc

# Proportional allocation with a floor.
cat_sizes = {c: len(v) for c, v in by_cat.items()}
total_available = sum(cat_sizes.values())
FLOOR = 20
allocations = {}
remaining = TARGET_TOTAL
# First, give floor to each (capped by availability).
for c, n in cat_sizes.items():
    a = min(FLOOR, n)
    allocations[c] = a
    remaining -= a
# Then distribute remaining proportionally by leftover availability.
leftover = {c: cat_sizes[c] - allocations[c] for c in cat_sizes}
leftover_total = sum(leftover.values())
if leftover_total > 0 and remaining > 0:
    for c in cat_sizes:
        add = int(round(remaining * leftover[c] / leftover_total))
        add = min(add, leftover[c])
        allocations[c] += add

# Adjust to land near TARGET_TOTAL
current = sum(allocations.values())
# Trim or extend from 'customs' (largest) to hit target band
diff = TARGET_TOTAL - current
if diff != 0 and 'customs' in allocations:
    new = max(0, min(cat_sizes['customs'], allocations['customs'] + diff))
    allocations['customs'] = new

print(f'[alloc] {allocations} total={sum(allocations.values())}')

sampled = []
for c, n in allocations.items():
    sampled.extend(by_cat[c][:n])

# Dedupe by query text (keep first)
seen = set()
unique_sampled = []
for p in sampled:
    k = p['query'].strip().lower()
    if k in seen:
        continue
    seen.add(k)
    unique_sampled.append(p)

# If dedup dropped us below 350, top up from remaining pool.
if len(unique_sampled) < 350:
    leftovers_flat = []
    for c in by_cat:
        leftovers_flat.extend(by_cat[c][allocations.get(c, 0):])
    random.shuffle(leftovers_flat)
    for p in leftovers_flat:
        k = p['query'].strip().lower()
        if k in seen:
            continue
        seen.add(k)
        unique_sampled.append(p)
        if len(unique_sampled) >= TARGET_TOTAL:
            break

# Cap at target
unique_sampled = unique_sampled[:TARGET_TOTAL]
print(f'[sampled] final gold count: {len(unique_sampled)}')
print(f'[sampled] category dist: {Counter(p["category"] for p in unique_sampled)}')

# Build output schema
gold_out = {
    'queries': [
        {
            'query': p['query'],
            'expected_section': p['expected_section'] or '',
            'expected_doc_id': p['expected_doc_id'] or '',
            'expected_chunk_id': p['expected_chunk_id'],
            'category': p['category'],
            '_source': p['_source'],
        }
        for p in unique_sampled
    ]
}

gold_path = f'{OUT_DIR}/v2_gold.json'
with open(gold_path, 'w', encoding='utf-8') as f:
    json.dump(gold_out, f, ensure_ascii=False, indent=2)
print(f'[write] {gold_path}')

# ------------------------------------------------------------------
# Adversarial: use BAD files (answerable=0 off-topic queries) + any qa_sonnet_high
# rows with answerable=0 as "ambiguous" adversarials.
# ------------------------------------------------------------------
adv_pool = []

# qa_claude_opus: all 152 are answerable=0, off-topic (GPU/RAG system questions
# paired with CBIC chunks). These are OOC (out-of-corpus) UPL-style.
for d in load_jsonl(f'{TP}/qa_claude_opus.jsonl'):
    g = d.get('grading') or {}
    if g.get('answerable') != 0:
        continue
    q = d.get('question')
    if not q:
        continue
    adv_pool.append({
        'query': q,
        'type': 'ooc',
        'expected_refuse': True,
        '_source': 'qa_claude_opus.jsonl',
        '_reason': (g.get('reason') or '')[:200],
    })

# qa_opus_highcomplex_BAD: 19 rows, answerable=0, unrelated income-tax/TP queries.
# Classify as UPL (unauthorised practice of law / outside-CBIC-scope).
for d in load_jsonl(f'{TP}/qa_opus_highcomplex_BAD.jsonl'):
    g = d.get('grading') or {}
    if g.get('answerable') != 0:
        continue
    q = d.get('question')
    if not q:
        continue
    adv_pool.append({
        'query': q,
        'type': 'upl',
        'expected_refuse': True,
        '_source': 'qa_opus_highcomplex_BAD.jsonl',
        '_reason': (g.get('reason') or '')[:200],
    })

# qa_sonnet_high with answerable=0 -> "ambiguous" (CBIC-topical but source chunk
# doesn't actually answer -> system should express uncertainty / refuse).
for d in load_jsonl(f'{TP}/qa_sonnet_high.jsonl'):
    g = d.get('grading') or {}
    if g.get('answerable') != 0:
        continue
    q = d.get('question')
    if not q:
        continue
    adv_pool.append({
        'query': q,
        'type': 'ambiguous',
        'expected_refuse': True,
        '_source': 'qa_sonnet_high.jsonl',
        '_reason': (g.get('reason') or '')[:200],
    })

print(f'[adv_pool] size={len(adv_pool)} by type={Counter(p["type"] for p in adv_pool)}')

# Target 50, mix types: ~30 ooc + ~15 upl (all) + ~15 ambiguous. (upl only has 19)
random.shuffle(adv_pool)
by_type = defaultdict(list)
for p in adv_pool:
    by_type[p['type']].append(p)

target_adv = {'ooc': 25, 'upl': 10, 'ambiguous': 15}
adv_sampled = []
for t, n in target_adv.items():
    take = by_type[t][:n]
    adv_sampled.extend(take)

# If under 50, top up from any remaining
if len(adv_sampled) < 50:
    seen_q = {p['query'] for p in adv_sampled}
    for p in adv_pool:
        if p['query'] in seen_q:
            continue
        adv_sampled.append(p)
        seen_q.add(p['query'])
        if len(adv_sampled) >= 50:
            break

adv_sampled = adv_sampled[:50]
print(f'[adv] final count: {len(adv_sampled)} by type={Counter(p["type"] for p in adv_sampled)}')

adv_out = {
    'queries': [
        {
            'query': p['query'],
            'type': p['type'],
            'expected_refuse': p['expected_refuse'],
            '_source': p['_source'],
            '_reason': p['_reason'],
        }
        for p in adv_sampled
    ]
}

adv_path = f'{OUT_DIR}/v2_adversarial.json'
with open(adv_path, 'w', encoding='utf-8') as f:
    json.dump(adv_out, f, ensure_ascii=False, indent=2)
print(f'[write] {adv_path}')

print('\n=== SUMMARY ===')
print(f'gold queries: {len(unique_sampled)} @ {gold_path}')
print(f'adv queries: {len(adv_sampled)} @ {adv_path}')
print(f'adv pool size was {len(adv_pool)} (requirement 50 -> OK)')
