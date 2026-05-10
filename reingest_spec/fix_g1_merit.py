"""Compute G1 retrieval-merit recall and build a proper CP-2 gold subset.
Filters gold queries to those whose expected_doc IS in current cbic_v2."""
import json
from urllib.request import Request, urlopen

GOLD = '/opt/indian-legal-ai/reingest_spec/eval/v2_gold_cp2.json'
RESULT = '/opt/indian-legal-ai/data/eval/cp2_g1.json'
SUBSET_OUT = '/opt/indian-legal-ai/reingest_spec/eval/v2_gold_cp2_subset.json'

# Get all doc_ids in cbic_v2 via paginated scroll
got_doc_ids = set()
offset = None
while True:
    body = {'limit': 256, 'with_payload': True}
    if offset is not None:
        body['offset'] = offset
    req = Request(
        'http://127.0.0.1:6343/collections/cbic_v2/points/scroll',
        data=json.dumps(body).encode(),
        headers={'content-type': 'application/json'},
    )
    raw = urlopen(req, timeout=30).read()
    d = json.loads(raw)
    for p in d['result']['points']:
        did = p.get('payload', {}).get('doc_id')
        if did:
            got_doc_ids.add(did)
    offset = d['result'].get('next_page_offset')
    if not offset:
        break

print(f'cbic_v2 distinct doc_ids: {len(got_doc_ids)}')

# Load gold and filter
gold = json.load(open(GOLD))
queries = gold['queries'] if isinstance(gold, dict) and 'queries' in gold else gold

def expected_doc(q):
    for k in ('expected_doc_id', 'doc_id', 'expected_doc', 'gold_doc_id'):
        v = q.get(k)
        if isinstance(v, list) and v:
            return v[0]
        if isinstance(v, str):
            return v
    ld = q.get('linked_doc_ids')
    if isinstance(ld, list) and ld:
        return ld[0]
    return None

in_corpus = []
missing = []
for q in queries:
    ed = expected_doc(q)
    if ed and ed in got_doc_ids:
        in_corpus.append(q)
    else:
        missing.append((q.get('query', '')[:60], ed))

print(f'gold in cbic_v2: {len(in_corpus)}')
print(f'gold NOT in cbic_v2 (forced misses): {len(missing)}')

# Write the subset gold
json.dump({'queries': in_corpus}, open(SUBSET_OUT, 'w'), indent=2)
print(f'wrote {SUBSET_OUT} with {len(in_corpus)} queries')

# Compute retrieval-merit recall against original G1 result
res = json.load(open(RESULT))
hits = res.get('hits', 0)
total = res.get('n', len(queries))
print(f'G1 raw: {hits}/{total} = {hits/total:.4f}')
if len(in_corpus):
    print(f'G1 retrieval-merit estimate: {hits}/{len(in_corpus)} = {hits/len(in_corpus):.4f} (assumes all hits are on in-corpus queries)')
