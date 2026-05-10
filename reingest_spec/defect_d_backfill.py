'''Backfill linked_doc_ids on existing v2 SQLite manifest WITHOUT re-ingest.

Logic: chunks share content iff their source docs share path_en. For each
canonical chunk, find all docs whose path_en matches the canonical's
doc_id path_en, and attach the OTHER doc_ids as linked_doc_ids.

This recovers the linkage that INSERT OR REPLACE destroyed during phase2.
'''
import json, sqlite3, sys, time, os
DB = sys.argv[1] if len(sys.argv) > 1 else '/opt/indian-legal-ai/data/ingest_manifest_v2.sqlite'

c = sqlite3.connect(DB)
c.row_factory = sqlite3.Row

# 1. Build path_en -> [doc_ids that have phase2_done=1]
print('[backfill] indexing docs by path...', flush=True)
docs_by_path = {}
for r in c.execute("SELECT doc_id, path_en FROM docs WHERE phase2_done=1 AND phase2_status='ok'"):
    p = r['path_en']
    if not p: continue
    docs_by_path.setdefault(p, []).append(r['doc_id'])
shared_paths = {p: ids for p, ids in docs_by_path.items() if len(ids) > 1}
print(f'[backfill] {len(shared_paths)} shared PDFs across {sum(len(v) for v in shared_paths.values())} docs')

# 2. Build doc_id -> path
doc_path = {d: p for p, ids in docs_by_path.items() for d in ids}

# 3. For each canonical chunk whose doc_id is in a shared-path group, attach OTHER doc_ids
t0 = time.time()
n_updated = 0
n_canon = 0
for r in c.execute('SELECT chunk_id, doc_id, payload_json FROM chunks WHERE is_canonical=1'):
    n_canon += 1
    p = doc_path.get(r['doc_id'])
    if not p or p not in shared_paths:
        continue
    others = [d for d in shared_paths[p] if d != r['doc_id']]
    if not others:
        continue
    payload = json.loads(r['payload_json'])
    payload['linked_doc_ids'] = sorted(others)
    c.execute('UPDATE chunks SET payload_json=? WHERE chunk_id=?', (json.dumps(payload), r['chunk_id']))
    n_updated += 1
c.commit()
print(f'[backfill] scanned {n_canon} canonicals, updated {n_updated} in {time.time()-t0:.1f}s')
