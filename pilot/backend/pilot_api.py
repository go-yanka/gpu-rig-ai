#!/usr/bin/env python3
"""CBIC pilot gateway — the service external testers talk to.

Sits in front of the existing cbic-rag-api (it only calls its POST /retrieve)
and adds:
  * issue-spotting query rewrite (incl. Hindi questions) + reciprocal-rank fusion
  * notification amendment status ("in force / amended by / rescinded by")
  * story-with-verbatim-quotes answer, every quote re-checked against its source
  * an explicit confidence level, and an honest "not found" instead of a guess
  * tester tokens, per-tester rate limit, query + structured feedback log, CSV export

It never modifies cbic-rag-api. Run on the rig next to it:

  CBIC_RAG_DIR=/opt/indian-legal-ai/rag/cbic_rag \
  UPSTREAM_URL=http://127.0.0.1:9500 \
  LLM_URL=http://127.0.0.1:9082 LLM_MODEL=qwen3-14b-q4_k_m.gguf \
  GRAPH_DB=/opt/indian-legal-ai/data/amendment_graph.sqlite \
  PILOT_TOKENS=/opt/cbic-auth/pilot_tokens.json PILOT_ADMIN_TOKEN=... \
  uvicorn pilot_api:app --host 0.0.0.0 --port 9600
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Literal, Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

for _p in os.environ.get('CBIC_RAG_DIR', '/opt/indian-legal-ai/rag/cbic_rag').split(os.pathsep):
    sys.path.insert(0, _p)
import amendment_graph as ag  # noqa: E402
from query_rewrite import rewrite, rrf  # noqa: E402
from storyformat import build_prompt, verify_quotes  # noqa: E402

HERE = Path(__file__).resolve().parent
APP_DIR = Path(os.environ.get('PILOT_APP_DIR', HERE.parent / 'app'))
TEST_PACK = Path(os.environ.get('PILOT_TEST_PACK', HERE.parent / 'test_pack.json'))

UPSTREAM_URL = os.environ.get('UPSTREAM_URL', 'http://127.0.0.1:9500')
COLLECTION = os.environ.get('COLLECTION') or None
RETRIEVE_K = int(os.environ.get('RETRIEVE_K', '20'))
ANSWER_K = int(os.environ.get('ANSWER_K', '8'))

LLM_URL = os.environ.get('LLM_URL', 'http://127.0.0.1:9082')
LLM_MODEL = os.environ.get('LLM_MODEL', 'qwen3-14b-q4_k_m.gguf')
LLM_KEY = os.environ.get('LLM_KEY', 'sk-anything')
# qwen3 only honours /no_think on /v1/chat/completions (JOURNAL 2026-04-26)
NO_THINK = os.environ.get('LLM_NO_THINK', '1') == '1'
USE_REWRITE = os.environ.get('USE_REWRITE', '1') == '1'

GRAPH_DB = os.environ.get('GRAPH_DB')
PILOT_DB = os.environ.get('PILOT_DB', str(HERE / 'pilot_log.sqlite'))
TOKENS_FILE = os.environ.get('PILOT_TOKENS')
ADMIN_TOKEN = os.environ.get('PILOT_ADMIN_TOKEN')
OPEN_ACCESS = os.environ.get('PILOT_OPEN') == '1'  # local development only
RATE_PER_MIN = int(os.environ.get('PILOT_RATE_PER_MIN', '6'))
MAX_CONCURRENT = int(os.environ.get('PILOT_MAX_CONCURRENT', '2'))

if not TOKENS_FILE and not OPEN_ACCESS:
    raise SystemExit('Set PILOT_TOKENS=<json file {name: token}> (or PILOT_OPEN=1 for local dev).')

ANSWER_RULES = """
ADDITIONAL RULES FOR THIS SERVICE:
7. Answer in the same language as the QUESTION (Hindi question -> Hindi answer), but keep
   every quote verbatim in the language of the source.
8. If the SOURCES do not contain the answer, reply with exactly one line:
   NOT_FOUND: <one sentence on what is missing>
   Do not answer from general knowledge.
9. If a NOTIFICATION STATUS block says a notification you rely on was amended, rescinded
   or superseded, say so in the answer.
"""

graph = ag.Graph(GRAPH_DB) if GRAPH_DB and os.path.exists(GRAPH_DB) else None
_ask_slots = threading.BoundedSemaphore(MAX_CONCURRENT)
_rate: Dict[str, List[float]] = {}
_rate_lock = threading.Lock()

app = FastAPI(title='cbic-pilot', version='0.1')


# ------------------------------------------------------------------ storage

def _db() -> sqlite3.Connection:
    con = sqlite3.connect(PILOT_DB, timeout=30)
    con.execute('PRAGMA journal_mode=WAL')
    return con


with _db() as _c:
    _c.executescript("""
    CREATE TABLE IF NOT EXISTS asks (
      id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tester TEXT, mission_id TEXT,
      question TEXT, status TEXT, confidence TEXT, answer TEXT, rewrites TEXT,
      sources TEXT, chains TEXT, verified INTEGER, suspicious INTEGER, total_ms REAL, error TEXT);
    CREATE TABLE IF NOT EXISTS feedback (
      id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, ask_id INTEGER, tester TEXT,
      answer_correct TEXT, refs_correct TEXT, correct_reference TEXT,
      would_rely INTEGER, comment TEXT);
    """)


# ------------------------------------------------------------------ auth

def _tokens() -> Dict[str, str]:
    if not TOKENS_FILE:
        return {}
    with open(TOKENS_FILE, encoding='utf-8') as f:
        return {tok: name for name, tok in json.load(f).items()}


def tester(request: Request) -> str:
    if OPEN_ACCESS:
        return 'dev'
    tok = request.headers.get('x-tester-token') or request.query_params.get('t')
    name = _tokens().get(tok or '')
    if not name:
        raise HTTPException(401, 'Missing or unknown tester token.')
    return name


def _check_rate(name: str) -> None:
    now = time.time()
    with _rate_lock:
        recent = [t for t in _rate.get(name, []) if now - t < 60]
        if len(recent) >= RATE_PER_MIN:
            raise HTTPException(429, f'Limit is {RATE_PER_MIN} questions a minute — please wait a moment.')
        recent.append(now)
        _rate[name] = recent


# ------------------------------------------------------------------ calls

def call_llm(system: str, user: str, max_tokens: int = 900) -> str:
    if NO_THINK:
        user = user + '\n/no_think'
    r = httpx.post(f'{LLM_URL}/v1/chat/completions', timeout=180,
                   headers={'Authorization': f'Bearer {LLM_KEY}'},
                   json={'model': LLM_MODEL, 'temperature': 0.0, 'max_tokens': max_tokens,
                         'messages': [{'role': 'system', 'content': system},
                                      {'role': 'user', 'content': user}]})
    r.raise_for_status()
    text = r.json()['choices'][0]['message']['content'] or ''
    return re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()


def upstream_retrieve(query: str) -> List[Dict]:
    body = {'question': query, 'k': RETRIEVE_K}
    if COLLECTION:
        body['collection'] = COLLECTION
    r = httpx.post(f'{UPSTREAM_URL}/retrieve', json=body, timeout=180)
    r.raise_for_status()
    return r.json().get('hits') or []


def _hit_key(h: Dict) -> str:
    return str(h.get('chunk_id') or f"{h.get('doc_id')}:{hash(h.get('text') or '')}")


def _chains(question: str, rw: Dict, top: List[Dict]) -> List[Dict]:
    if graph is None:
        return []
    asked: List[str] = []
    for text in [question] + list(rw.get('notification_refs') or []):
        asked += [r.key for r in ag.find_refs(text)]
        k = ag.parse_key(text)
        if k:
            asked.append(k)
    # notifications the user named are always reported; ones that merely appear in
    # the top sources only when there is an amendment / rescission to warn about
    from_sources = [k for k in (graph.key_for_doc(h.get('doc_id') or '') for h in top[:4]) if k]
    out, seen = [], set()
    for k, named in [(k, True) for k in asked] + [(k, False) for k in from_sources]:
        if k in seen:
            continue
        seen.add(k)
        ch = graph.chain(k)
        if ch['known'] and (named or ch['amended_by'] or ch['ended_by']):
            out.append(ch)
        if len(out) == 3:
            break
    return out


def _status_block(chains: List[Dict]) -> str:
    if not chains:
        return ''
    lines = ['NOTIFICATION STATUS (from the amendment index; each line is backed by corpus text):']
    for c in chains:
        amends = ', '.join(a['key'] for a in c['amended_by'][-5:]) or 'none found'
        lines.append(f"- {c['key']}: {c['status']}. Amended by: {amends}.")
    return '\n'.join(lines) + '\n\n'


def _confidence(verified: int, suspicious: int) -> str:
    if verified >= 2 and suspicious == 0:
        return 'high'
    if verified >= 1:
        return 'medium'
    return 'low'


# ------------------------------------------------------------------ API

class AskReq(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    mission_id: Optional[str] = None


class FeedbackReq(BaseModel):
    ask_id: int
    answer_correct: Literal['yes', 'partly', 'no', 'unsure']
    refs_correct: Literal['yes', 'partly', 'no', 'unsure']
    correct_reference: Optional[str] = Field(default=None, max_length=1000)
    would_rely: Optional[int] = Field(default=None, ge=1, le=5)
    comment: Optional[str] = Field(default=None, max_length=4000)


@app.get('/api/health')
def health():
    out = {'graph': graph is not None}
    for name, url in (('upstream', f'{UPSTREAM_URL}/health'), ('llm', f'{LLM_URL}/health')):
        try:
            out[name] = httpx.get(url, timeout=5).status_code == 200
        except httpx.HTTPError:
            out[name] = False
    return out


@app.get('/api/me')
def me(name: str = Depends(tester)):
    return {'tester': name}


@app.get('/api/test-pack')
def test_pack(name: str = Depends(tester)):
    return json.loads(TEST_PACK.read_text(encoding='utf-8'))


@app.get('/api/chain')
def chain(ref: str, name: str = Depends(tester)):
    if graph is None:
        raise HTTPException(503, 'Amendment index not built on this server.')
    key = ag.parse_key(ref)
    if not key:
        raise HTTPException(400, 'Write the notification like 50/2017-Customs or 35/2020-Central Tax.')
    return graph.chain(key)


@app.post('/api/ask')
def ask(req: AskReq, name: str = Depends(tester)):
    _check_rate(name)
    if not _ask_slots.acquire(timeout=120):
        raise HTTPException(503, 'The system is busy with other testers — please try again in a minute.')
    try:
        return _ask(req, name)
    finally:
        _ask_slots.release()


def _ask(req: AskReq, name: str) -> Dict:
    t0 = time.perf_counter()
    timings: Dict[str, float] = {}
    q = req.question.strip()

    rw = {'issues': [], 'queries': [], 'notification_refs': [], 'in_scope': True, 'language': 'en'}
    if USE_REWRITE:
        t = time.perf_counter()
        rw = rewrite(q, lambda s, u: call_llm(s, u, max_tokens=400))
        timings['rewrite_ms'] = round((time.perf_counter() - t) * 1000)

    if not rw['in_scope']:
        result = {'status': 'out_of_scope', 'confidence': 'none',
                  'answer_markdown': 'This question looks outside CBIC indirect-tax law '
                                     '(GST, Customs, Central Excise, Service Tax). '
                                     'This pilot only answers from CBIC documents.',
                  'sources': [], 'verified_quotes': [], 'suspicious_quotes': [], 'chains': []}
        return _finish(req, name, q, rw, result, timings, t0)

    t = time.perf_counter()
    queries = [q] + [x for x in rw['queries'] if x.lower() != q.lower()]
    with ThreadPoolExecutor(max_workers=len(queries)) as ex:
        lists = list(ex.map(upstream_retrieve, queries))
    timings['retrieve_ms'] = round((time.perf_counter() - t) * 1000)
    top = rrf(lists, key=_hit_key)[:ANSWER_K]
    chains = _chains(q, rw, top)

    if not top:
        result = {'status': 'not_found', 'confidence': 'none',
                  'answer_markdown': 'No CBIC document matched this question.',
                  'sources': [], 'verified_quotes': [], 'suspicious_quotes': [], 'chains': chains}
        return _finish(req, name, q, rw, result, timings, t0)

    for h in top:
        h.setdefault('page', None)
    system, user = build_prompt(q, top)
    t = time.perf_counter()
    answer = call_llm(system + ANSWER_RULES, _status_block(chains) + user)
    timings['answer_ms'] = round((time.perf_counter() - t) * 1000)

    v = verify_quotes(answer, top)
    cited = {x['source_index'] for x in v['verified']}
    sources = [{'index': i, 'doc_id': h.get('doc_id'), 'title': h.get('title'),
                'section_ref': h.get('section_ref'), 'text': (h.get('text') or '')[:1500],
                'linked_doc_ids': h.get('linked_doc_ids') or [],
                'found_by': h.get('hit_by', 1), 'quoted': i in cited}
               for i, h in enumerate(top, start=1)]
    if answer.strip().upper().startswith('NOT_FOUND'):
        result = {'status': 'not_found', 'confidence': 'none',
                  'answer_markdown': 'I could not find this in the CBIC documents I searched. '
                                     + answer.split(':', 1)[-1].strip()
                                     + '\n\nThe closest documents are listed below — please tell us '
                                       'the correct reference if you know it.',
                  'sources': sources, 'verified_quotes': [], 'suspicious_quotes': [], 'chains': chains}
    else:
        result = {'status': 'answered',
                  'confidence': _confidence(len(v['verified']), len(v['suspicious'])),
                  'answer_markdown': v['annotated_answer'], 'sources': sources,
                  'verified_quotes': v['verified'], 'suspicious_quotes': v['suspicious'],
                  'chains': chains}
    return _finish(req, name, q, rw, result, timings, t0)


def _finish(req: AskReq, name: str, q: str, rw: Dict, result: Dict, timings: Dict, t0: float) -> Dict:
    timings['total_ms'] = round((time.perf_counter() - t0) * 1000)
    with _db() as con:
        cur = con.execute(
            'INSERT INTO asks (ts, tester, mission_id, question, status, confidence, answer, rewrites,'
            ' sources, chains, verified, suspicious, total_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (time.time(), name, req.mission_id, q, result['status'], result['confidence'],
             result['answer_markdown'], json.dumps(rw, ensure_ascii=False),
             json.dumps([{k: s[k] for k in ('doc_id', 'title', 'quoted')} for s in result['sources']]),
             json.dumps([c['key'] for c in result['chains']]),
             len(result['verified_quotes']), len(result['suspicious_quotes']), timings['total_ms']))
        ask_id = cur.lastrowid
    return {'ask_id': ask_id, 'question': q, 'issues': rw.get('issues', []),
            'search_queries': rw.get('queries', []), 'timings': timings, **result}


@app.post('/api/feedback')
def feedback(req: FeedbackReq, name: str = Depends(tester)):
    with _db() as con:
        owner = con.execute('SELECT tester FROM asks WHERE id=?', (req.ask_id,)).fetchone()
        if not owner:
            raise HTTPException(404, 'Unknown question id.')
        con.execute('INSERT INTO feedback (ts, ask_id, tester, answer_correct, refs_correct,'
                    ' correct_reference, would_rely, comment) VALUES (?,?,?,?,?,?,?,?)',
                    (time.time(), req.ask_id, name, req.answer_correct, req.refs_correct,
                     req.correct_reference, req.would_rely, req.comment))
    return {'ok': True}


@app.get('/api/export.csv', response_class=PlainTextResponse)
def export(request: Request):
    if not ADMIN_TOKEN or request.headers.get('x-admin-token') != ADMIN_TOKEN:
        raise HTTPException(403, 'Admin token required.')
    with _db() as con:
        rows = con.execute(
            "SELECT a.id, datetime(a.ts, 'unixepoch'), a.tester, a.mission_id, a.question, a.status,"
            ' a.confidence, a.verified, a.suspicious, a.total_ms, a.chains, a.sources, a.answer,'
            ' f.answer_correct, f.refs_correct, f.correct_reference, f.would_rely, f.comment'
            ' FROM asks a LEFT JOIN feedback f ON f.ask_id = a.id ORDER BY a.id').fetchall()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['ask_id', 'time_utc', 'tester', 'mission_id', 'question', 'status', 'confidence',
                'verified_quotes', 'suspicious_quotes', 'total_ms', 'notification_chains', 'sources',
                'answer', 'answer_correct', 'refs_correct', 'correct_reference', 'would_rely', 'comment'])
    w.writerows(rows)
    return buf.getvalue()


@app.get('/')
def index():
    return FileResponse(APP_DIR / 'index.html')


app.mount('/static', StaticFiles(directory=APP_DIR), name='static')
