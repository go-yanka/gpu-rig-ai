#!/usr/bin/env python3
"""Local CBIC search for the desktop pilot — same POST /retrieve contract as cbic-rag-api.

Keyword search: SQLite FTS5 BM25 (built into Python, no server, low memory).
Meaning search (optional): BGE-M3 embeddings from a local Ollama (`ollama pull bge-m3`),
cached in a float16 .npy file. Results of both are merged by reciprocal-rank fusion —
hybrid was the single biggest recall gain on the rig (+0.30 G1, L8).
There is no cross-encoder reranker here, so measure before trusting it:

    set G1_RETRIEVE_API=http://127.0.0.1:9601/retrieve
    python reingest_spec/evaluators/gate_g1_recall.py --retrieve-only

Build once:  python local_retrieve.py index --data D:/_gpu_rig_ai/pilot_data [--dense]
Serve:       python local_retrieve.py serve --data D:/_gpu_rig_ai/pilot_data --port 9601
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List

import httpx

OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://127.0.0.1:11434')
EMBED_MODEL = os.environ.get('EMBED_MODEL', 'bge-m3')
STOP = set('the of and to a in for is on by with or as be at an this that from are under any which it its '
           'shall such said may has have was were not no if into our we my can what how does do i he she '
           'they their there than then also all other been will would should could whether about'.split())


def verify(data: Path) -> List[Path]:
    sums = data / 'SHA256SUMS'
    if not sums.exists():
        raise SystemExit(f'{sums} missing — copy the whole export folder.')
    parts, bad = [], []
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        h, name = line.split()
        p = data / name
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
            bad.append(name)
        parts.append(p)
    if bad:
        raise SystemExit('Checksum mismatch (re-export these on the rig with --only N): ' + ', '.join(bad))
    return parts


def read_parts(parts: List[Path]):
    for p in parts:
        with gzip.open(p, 'rt', encoding='utf-8') as f:
            for line in f:
                yield json.loads(line)


def embed(texts: List[str]) -> List[List[float]]:
    r = httpx.post(f'{OLLAMA_URL}/api/embed', json={'model': EMBED_MODEL, 'input': texts}, timeout=600)
    r.raise_for_status()
    return r.json()['embeddings']


def index(data: Path, dense: bool, batch: int) -> None:
    parts = verify(data)
    db = data / 'search.sqlite'
    if db.exists():
        db.unlink()
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE chunk (rowid INTEGER PRIMARY KEY, doc TEXT)')
    con.execute("CREATE VIRTUAL TABLE fts USING fts5(title, body, tokenize='porter unicode61')")
    n = 0
    for r in read_parts(parts):
        n += 1
        body = r.get('embed_text') or r['text']
        con.execute('INSERT INTO chunk VALUES (?,?)', (n, json.dumps(r, ensure_ascii=False)))
        con.execute('INSERT INTO fts (rowid, title, body) VALUES (?,?,?)',
                    (n, f"{r.get('title') or ''} {r.get('doc_number') or ''} {r.get('section_ref') or ''}", body))
    con.commit()
    print(f'keyword index: {n} chunks')
    if dense:
        import numpy as np
        vecs = None
        t0, done = time.time(), 0
        for start in range(1, n + 1, batch):
            ids = list(range(start, min(n, start + batch - 1) + 1))
            docs = [json.loads(con.execute('SELECT doc FROM chunk WHERE rowid=?', (i,)).fetchone()[0]) for i in ids]
            out = np.asarray(embed([(d.get('embed_text') or d['text'])[:6000] for d in docs]), dtype=np.float32)
            out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
            if vecs is None:
                vecs = np.zeros((n, out.shape[1]), dtype=np.float16)
            vecs[ids[0] - 1:ids[-1]] = out.astype(np.float16)
            done += len(ids)
            if done % (batch * 20) == 0 or done == n:
                rate = done / (time.time() - t0)
                print(f'  embedded {done}/{n}  {rate:.1f}/s  eta {(n - done) / max(rate, 1e-9) / 60:.0f} min', flush=True)
        np.save(data / 'dense.npy', vecs)
        print('meaning index saved')
    con.close()


class Searcher:
    def __init__(self, data: Path):
        self.con = sqlite3.connect(f'file:{data / "search.sqlite"}?mode=ro', uri=True, check_same_thread=False)
        self.vecs = None
        if (data / 'dense.npy').exists():
            import numpy as np
            self.np = np
            self.vecs = np.load(data / 'dense.npy').astype(np.float32)

    @staticmethod
    def _fts_query(q: str) -> str:
        words = [w for w in re.findall(r'\w+', q.lower()) if w not in STOP and len(w) > 1]
        return ' OR '.join(f'"{w}"' for w in dict.fromkeys(words))[:2000]

    def keyword(self, q: str, n: int) -> List[int]:
        fq = self._fts_query(q)
        if not fq:
            return []
        return [r[0] for r in self.con.execute(
            'SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY bm25(fts, 3.0, 1.0) LIMIT ?', (fq, n))]

    def meaning(self, q: str, n: int) -> List[int]:
        if self.vecs is None:
            return []
        v = self.np.asarray(embed([q])[0], dtype=self.np.float32)
        v /= self.np.linalg.norm(v) + 1e-9
        sims = self.vecs @ v
        top = self.np.argpartition(-sims, min(n, len(sims) - 1))[:n]
        return [int(i) + 1 for i in top[self.np.argsort(-sims[top])]]

    def search(self, q: str, k: int) -> List[Dict]:
        lists = [self.keyword(q, 100), self.meaning(q, 100)]
        score: Dict[int, float] = {}
        for lst in lists:
            for rank, rid in enumerate(lst):
                score[rid] = score.get(rid, 0.0) + 1.0 / (60 + rank + 1)
        hits = []
        for rid in sorted(score, key=score.get, reverse=True)[:k]:
            d = json.loads(self.con.execute('SELECT doc FROM chunk WHERE rowid=?', (rid,)).fetchone()[0])
            hits.append({'chunk_id': d.get('chunk_id'), 'doc_id': d.get('doc_id'),
                         'linked_doc_ids': d.get('linked_doc_ids') or [], 'section_ref': d.get('section_ref'),
                         'title': d.get('title'), 'text': d.get('text'), 'page': d.get('page'),
                         'source_url': d.get('source_url'), 'score': round(score[rid], 5),
                         'rerank_score': round(score[rid], 5)})
        return hits


def serve(data: Path, port: int) -> None:
    import uvicorn
    from fastapi import Body, FastAPI

    s = Searcher(data)
    app = FastAPI(title='cbic-local-retrieve')

    @app.get('/health')
    def health():
        return {'status': 'ok', 'dense': s.vecs is not None}

    @app.post('/retrieve')
    def retrieve(body: dict = Body(...)):
        t0 = time.perf_counter()
        hits = s.search(body['question'], int(body.get('k') or 10))
        return {'hits': hits, 'router_category': None,
                'timings': {'total_ms': round((time.perf_counter() - t0) * 1000, 2)}}

    print(f'local search on http://127.0.0.1:{port}  (meaning search: {"on" if s.vecs is not None else "off"})')
    uvicorn.run(app, host='127.0.0.1', port=port, log_level='warning')


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    i = sub.add_parser('index')
    i.add_argument('--data', type=Path, required=True)
    i.add_argument('--dense', action='store_true', help='also build BGE-M3 meaning index via Ollama')
    i.add_argument('--batch', type=int, default=32)
    s = sub.add_parser('serve')
    s.add_argument('--data', type=Path, required=True)
    s.add_argument('--port', type=int, default=9601)
    a = ap.parse_args()
    if a.cmd == 'index':
        index(a.data, a.dense, a.batch)
    else:
        serve(a.data, a.port)


if __name__ == '__main__':
    sys.exit(main())
