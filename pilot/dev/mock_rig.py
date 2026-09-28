#!/usr/bin/env python3
"""DEV ONLY — a stand-in for the rig so the pilot can be exercised anywhere.

Serves the two things pilot_api.py calls on the rig:
  POST /retrieve             keyword scoring over ~1,900 real CBIC chunks from
                             eval/training_pairs/pairs_2000_20260422.jsonl
  POST /v1/chat/completions  a deterministic fake LLM: JSON for rewrite prompts,
                             a story answer quoting a real source sentence verbatim
                             (or NOT_FOUND) for answer prompts
Answers from this mock say nothing about real quality — it only proves the plumbing.

  python3 pilot/dev/mock_rig.py --port 9555
"""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request

ROOT = Path(__file__).resolve().parents[2]
CHUNKS = ROOT / 'eval' / 'training_pairs' / 'pairs_2000_20260422.jsonl'
STOP = set('the of and to a in for is on by with or as be at an this that from are under any which '
           'it its shall such said may has have was were not no if into our we my can what how does '
           'do i he she they their there than then also all other been'.split())
OFF_TOPIC = re.compile(r'income[- ]tax|\bMAT\b|115JB|companies act|RBI|FEMA|salary|TDS', re.I)

app = FastAPI()
docs = []
df = Counter()


def toks(s):
    return [w for w in re.findall(r'[a-z0-9]+', s.lower()) if w not in STOP and len(w) > 1]


def load():
    seen = set()
    for line in CHUNKS.open(encoding='utf-8'):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (r.get('doc_id'), (r.get('text') or '')[:80])
        if key in seen or not r.get('text'):
            continue
        seen.add(key)
        tf = Counter(toks((r.get('title') or '') + ' ' + r['text']))
        docs.append((r, tf))
        df.update(tf.keys())


@app.get('/health')
def health():
    return {'status': 'ok', 'mock': True, 'chunks': len(docs)}


@app.post('/retrieve')
async def retrieve(req: Request):
    body = await req.json()
    q = toks(body['question'])
    n = len(docs)
    scored = []
    for r, tf in docs:
        s = sum(tf[w] / (tf[w] + 1.5) * math.log(1 + n / (1 + df[w])) for w in q if w in tf)
        if s > 0:
            scored.append((s, r))
    scored.sort(key=lambda x: -x[0])
    hits = [{'chunk_id': r['chunk_id'], 'doc_id': r['doc_id'], 'linked_doc_ids': [],
             'section_ref': r.get('section_ref'), 'title': r.get('title'), 'text': r['text'],
             'score': round(s, 3), 'rerank_score': round(s, 3)}
            for s, r in scored[:body.get('k', 20)]]
    return {'hits': hits, 'router_category': None, 'timings': {'total_ms': 5}}


def _rewrite(q):
    refs = re.findall(r'\d{1,4}\s*/\s*\d{2,4}\s*[-–]?\s*[A-Za-z][A-Za-z .()]{1,30}', q)
    words = [w for w in toks(q) if not w.isdigit()][:12]
    hindi = bool(re.search(r'[ऀ-ॿ]', q))
    return {'issues': ['(mock) ' + ' '.join(words[:6])],
            'queries': [] if hindi else [' '.join(words)],
            'notification_refs': refs, 'in_scope': not OFF_TOPIC.search(q),
            'language': 'hi' if hindi else 'en'}


def _answer(user):
    q = re.search(r'QUESTION: (.*?)\n\nSOURCES', user, re.S).group(1)
    qt = set(toks(q))
    best = None
    for m in re.finditer(r'\[S(\d+)\] Doc: (.*?)\n---\n(.*?)\n---', user, re.S):
        idx, title, text = int(m.group(1)), m.group(2), m.group(3)
        for sent in re.split(r'(?<=[.;:])\s+', text):
            sent = sent.strip()
            if not (40 <= len(sent) <= 380) or '"' in sent:
                continue
            ov = len(qt & set(toks(sent)))
            if not best or ov > best[0]:
                best = (ov, idx, sent, title.split('|')[0].strip())
    if not best or best[0] < 2:
        return 'NOT_FOUND: none of the retrieved sources address this question.'
    ov, idx, sent, title = best
    return (f'**Answer:** (mock) The closest provision is in {title}.\n\n'
            f'**How we got here:** The source states *"{sent}"* [S{idx}]\n\n'
            f'**Conclusion:** (mock answer — for plumbing tests only.)')


@app.get('/v1/models')
def models():
    return {'object': 'list', 'data': [{'id': 'mock'}]}


@app.post('/api/embed')
async def embed(req: Request):
    """Ollama-style embeddings: hashed bag of words — similar texts get similar vectors."""
    body = await req.json()
    out = []
    for t in body['input']:
        v = [0.0] * 256
        for w in toks(t):
            v[hash(w) % 256] += 1.0
        out.append(v)
    return {'embeddings': out}


@app.post('/v1/chat/completions')
async def chat(req: Request):
    body = await req.json()
    system, user = body['messages'][0]['content'], body['messages'][1]['content']
    user = user.replace('\n/no_think', '')
    content = json.dumps(_rewrite(user)) if 'research librarian' in system else _answer(user)
    return {'choices': [{'message': {'role': 'assistant', 'content': content}}]}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=9555)
    a = ap.parse_args()
    load()
    uvicorn.run(app, host='127.0.0.1', port=a.port, log_level='warning')
