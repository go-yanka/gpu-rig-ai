#!/usr/bin/env python3
"""One-shot patch for /opt/indian-legal-ai/rag/cbic_rag/ingest.py.

Four fixes:
1. Add hashlib import.
2. Move FILES_LOG from /tmp to /opt/indian-legal-ai/state/.
3. Replace random-seeded hash() with stable blake2b for point_id
   so re-ingests overwrite instead of stacking.
4. Make existing_doc_ids() raise on failure (no silent empty set).
"""
import re
import sys

P = '/opt/indian-legal-ai/rag/cbic_rag/ingest.py'
with open(P) as f:
    src = f.read()

# 1. hashlib import
if 'hashlib' not in src:
    old_imp = 'import os, sys, argparse, sqlite3, time, threading, queue, math'
    new_imp = 'import os, sys, argparse, sqlite3, time, threading, queue, math, hashlib'
    assert old_imp in src, 'import line not found'
    src = src.replace(old_imp, new_imp)
    assert 'hashlib' in src, 'could not add hashlib'

# 2. Persistent FILES_LOG
old_log = 'FILES_LOG = os.environ.get("CBIC_FILES_LOG", "/tmp/cbic-files.log")'
new_log = 'FILES_LOG = os.environ.get("CBIC_FILES_LOG", "/opt/indian-legal-ai/state/cbic-files.log")'
assert old_log in src, 'FILES_LOG line not found'
src = src.replace(old_log, new_log)

# 3. Stable point id
old_pid = "pid = abs(hash((c['doc_id'], c['page'], c['char_start']))) % (10**15)"
new_pid = (
    "_pk = f\"{c['doc_id']}|{c['page']}|{c['char_start']}\".encode()\n"
    "                pid = int.from_bytes(hashlib.blake2b(_pk, digest_size=8).digest(), 'big') % (10**15)"
)
assert old_pid in src, 'pid line not found'
src = src.replace(old_pid, new_pid)

# 4. existing_doc_ids: no silent failure
new_eid = '''def existing_doc_ids(qc):
    seen = set()
    off = None
    pages = 0
    while True:
        pts, off = qc.scroll(collection_name=QCOLL, limit=2000,
                             with_payload=['doc_id'], with_vectors=False, offset=off)
        for p in pts:
            seen.add(p.payload.get('doc_id'))
        pages += 1
        if pages % 50 == 0:
            print(f'[resume] scanned {pages*2000} points, {len(seen)} unique doc_ids so far', flush=True)
        if off is None:
            break
    return seen'''
eid_re = re.compile(r'def existing_doc_ids\(qc\):.*?return seen', re.DOTALL)
m = eid_re.search(src)
assert m, 'existing_doc_ids block not found'
src = src[: m.start()] + new_eid + src[m.end():]

with open(P, 'w') as f:
    f.write(src)

# Verify
import py_compile
py_compile.compile(P, doraise=True)
print('patched OK, compiles OK')
