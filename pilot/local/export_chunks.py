#!/usr/bin/env python3
"""Export the CBIC chunk corpus once, so the pilot can run on a desktop without the rig.

Run ON THE RIG (reads the v2 ingest manifest read-only; no Qdrant needed):

    python3 export_chunks.py --manifest /opt/indian-legal-ai/data/ingest_manifest_v2.sqlite \
        --out /mnt/d/_gpu_rig_ai/pilot_data

Writes corpus-000.jsonl.gz, corpus-001.jsonl.gz, ... (each well under 100 MB) plus
SHA256SUMS. The rig corrupts large local-disk writes (INCIDENTS_ARCHIVE 2026-05-09),
so parts are small and the desktop verifies every checksum before indexing; a part
that fails is simply re-exported with --only <n>.
"""
import argparse
import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

FIELDS = ('chunk_id', 'doc_id', 'title', 'section_ref', 'category', 'subcategory', 'doc_number',
          'doc_type', 'lang', 'page', 'source_url', 'linked_doc_ids', 'embed_text', 'text')
ROWS_PER_PART = 8000


def rows(manifest: str):
    con = sqlite3.connect(f'file:{manifest}?mode=ro', uri=True)
    q = ('SELECT c.chunk_id, c.doc_id, d.title, d.category, d.subcategory, c.payload_json '
         'FROM chunks c LEFT JOIN docs d ON d.doc_id = c.doc_id '
         'WHERE c.is_canonical = 1 ORDER BY c.doc_id, c.chunk_id')
    for chunk_id, doc_id, title, cat, sub, pj in con.execute(q):
        p = json.loads(pj or '{}')
        p.setdefault('chunk_id', chunk_id)
        p.setdefault('doc_id', doc_id)
        p['title'] = p.get('title') or title
        p['category'] = p.get('category') or cat
        p['subcategory'] = p.get('subcategory') or sub
        if p.get('text'):
            yield {k: p.get(k) for k in FIELDS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--only', type=int, help='re-export just this part number')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    part, buf, sums, total = 0, [], {}, 0

    def flush():
        nonlocal part, buf
        if buf and (a.only is None or a.only == part):
            path = out / f'corpus-{part:03d}.jsonl.gz'
            data = gzip.compress(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in buf).encode())
            path.write_bytes(data)
            sums[path.name] = hashlib.sha256(data).hexdigest()
        part, buf = part + 1, []

    for r in rows(a.manifest):
        buf.append(r)
        total += 1
        if len(buf) == ROWS_PER_PART:
            flush()
    flush()
    sums_path = out / 'SHA256SUMS'
    old = {}
    if a.only is not None and sums_path.exists():
        old = dict(line.split()[::-1] for line in sums_path.read_text().splitlines() if line.strip())
    old.update(sums)
    sums_path.write_text(''.join(f'{h}  {n}\n' for n, h in sorted(old.items())))
    print(f'exported {total} chunks into {part} parts -> {out}')


if __name__ == '__main__':
    main()
