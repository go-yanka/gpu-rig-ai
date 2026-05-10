#!/usr/bin/env python3
"""FastAPI service for the cbic RAG.

Exposes:
  POST /query            -- { "question": "...", "k": 8, "filters": {...} } -> full payload
  POST /v1/chat/completions -- OpenAI-compatible endpoint for Open WebUI /
                               LiteLLM pass-through. Internally retrieves and
                               wraps the LLM, returning story-with-quotes as
                               `message.content` in markdown.
  GET  /v1/stats         -- corpus + query-log stats for dashboard
  GET  /v1/queries/recent?n=N  -- recent query summaries
  GET  /v1/queries/{id}  -- full record (answer + citations)
  POST /v1/queries/{id}/rate   -- { "rating": 1..5, "feedback"?: str }

Generation is delegated to the local LiteLLM / model gateway (env LITELLM_URL).
"""
from __future__ import annotations
import os, json, time, uuid, httpx, threading
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from retriever import retrieve, rerank
from storyformat import build_prompt, build_response_payload
from hyde import hyde
from router import route, filters_for
from colbert_rerank import rerank_colbert
import query_log

LITELLM_URL = os.environ.get('LITELLM_URL', 'http://127.0.0.1:4444')
LLM_MODEL   = os.environ.get('LLM_MODEL',   'qwen3-14b-hermes')
LLM_KEY     = os.environ.get('LITELLM_KEY', 'sk-anything')

QDRANT_URL  = os.environ.get('QDRANT_URL',  'http://127.0.0.1:6343')
QDRANT_COLL = os.environ.get('QDRANT_COLL', 'cbic_v1')

TOP_K_RETR  = int(os.environ.get('TOP_K_RETR', '24'))
TOP_K_RERANK= int(os.environ.get('TOP_K_RERANK', '8'))
USE_HYDE    = os.environ.get('USE_HYDE', '1') == '1'
USE_ROUTER  = os.environ.get('USE_ROUTER', '1') == '1'
# Rerank strategy: 'colbert' (default), 'ce' (bge-reranker-v2-m3), 'none'.
RERANK      = os.environ.get('RERANK', 'colbert').lower()

# Init query log DB at module load (idempotent, WAL mode).
try:
    query_log.init_db()
except Exception as _e:
    print(f'[query_log] init failed: {_e}')

app = FastAPI(title='cbic-rag', version='0.2')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])


class QueryReq(BaseModel):
    question: str
    k: int = TOP_K_RERANK
    filters: Optional[Dict[str, Any]] = None
    # H2 fix: evaluators can target a specific Qdrant collection (cbic_v1 / cbic_v2)
    # without mutating the process-wide QDRANT_COLL env var.
    collection: Optional[str] = None


def _call_llm(system: str, user: str, temperature: float = 0.0) -> str:
    with httpx.Client(timeout=120) as cli:
        r = cli.post(f'{LITELLM_URL}/v1/chat/completions',
                     headers={'Authorization': f'Bearer {LLM_KEY}',
                              'Content-Type': 'application/json'},
                     json={
                         'model': LLM_MODEL,
                         'messages': [
                             {'role': 'system', 'content': system},
                             {'role': 'user',   'content': user},
                         ],
                         'temperature': temperature,
                         'max_tokens': 900,
                     })
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content']


def _ms_since(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 2)


def _run_pipeline(req: QueryReq, client_ip: Optional[str] = None) -> Dict[str, Any]:
    """Run the RAG pipeline with per-stage timings + log to sqlite."""
    t_start = time.perf_counter()
    timings: Dict[str, Optional[float]] = {
        'router_ms': None, 'hyde_ms': None,
        'retrieve_ms': None, 'rerank_ms': None,
        'llm_ms': None, 'total_ms': None,
    }

    # 1) router
    filters = req.filters
    router_cat: Optional[str] = None
    if USE_ROUTER and not filters:
        t0 = time.perf_counter()
        router_cat = route(req.question)
        timings['router_ms'] = _ms_since(t0)
        if router_cat:
            filters = filters_for(router_cat)

    # 2) HyDE
    if USE_HYDE:
        t0 = time.perf_counter()
        search_text = hyde(req.question)
        timings['hyde_ms'] = _ms_since(t0)
    else:
        search_text = req.question

    # 3) retrieve
    t0 = time.perf_counter()
    hits = retrieve(search_text, k=TOP_K_RETR, filters=filters,
                    collection=req.collection)
    if not hits and filters:
        hits = retrieve(search_text, k=TOP_K_RETR, filters=None,
                        collection=req.collection)
    timings['retrieve_ms'] = _ms_since(t0)

    if not hits:
        payload = {'question': req.question,
                   'answer_markdown': '**Answer:** No matching documents in the CBIC corpus.\n\n'
                                      '**Conclusion:** Cannot answer.',
                   'verified_quotes': [], 'suspicious_quotes': [],
                   'citations': []}
        timings['total_ms'] = _ms_since(t_start)
        _log_and_attach(payload, req, router_cat, filters, hits, [], timings, client_ip)
        return payload

    # 4) rerank
    t0 = time.perf_counter()
    if RERANK == 'colbert':
        top = rerank_colbert(req.question, hits, top_n=req.k)
    elif RERANK == 'ce':
        top = rerank(req.question, hits, top_n=req.k)
    else:
        hits.sort(key=lambda c: c.get('score', 0), reverse=True)
        top = hits[:req.k]
    timings['rerank_ms'] = _ms_since(t0)

    # 5) LLM
    sys_p, usr_p = build_prompt(req.question, top)
    t0 = time.perf_counter()
    try:
        ans = _call_llm(sys_p, usr_p)
    except Exception as e:
        timings['llm_ms'] = _ms_since(t0)
        timings['total_ms'] = _ms_since(t_start)
        err_payload = {'question': req.question,
                       'answer_markdown': f'**Answer:** LLM gateway error.\n\n**Conclusion:** {e}',
                       'verified_quotes': [], 'suspicious_quotes': [],
                       'citations': []}
        _log_and_attach(err_payload, req, router_cat, filters, hits, top, timings, client_ip)
        raise HTTPException(502, f'LLM gateway error: {e}')
    timings['llm_ms'] = _ms_since(t0)

    payload = build_response_payload(req.question, top, ans)
    timings['total_ms'] = _ms_since(t_start)
    _log_and_attach(payload, req, router_cat, filters, hits, top, timings, client_ip)
    return payload


def _log_and_attach(payload: Dict[str, Any], req: QueryReq,
                    router_cat: Optional[str], filters: Optional[Dict[str, Any]],
                    hits: List[dict], top: List[dict],
                    timings: Dict[str, Optional[float]],
                    client_ip: Optional[str]) -> None:
    """Persist a query-log row and decorate the outgoing payload."""
    try:
        qid = query_log.log_query({
            'question': req.question,
            'k': req.k,
            'filters': filters,
            'router_category': router_cat,
            'retrieved_count': len(hits or []),
            'reranked_count': len(top or []),
            'citations': payload.get('citations', []),
            'verified_count': len(payload.get('verified_quotes') or []),
            'suspicious_count': len(payload.get('suspicious_quotes') or []),
            'answer_markdown': payload.get('answer_markdown'),
            't_total_ms':   timings.get('total_ms'),
            't_router_ms':  timings.get('router_ms'),
            't_hyde_ms':    timings.get('hyde_ms'),
            't_retrieve_ms':timings.get('retrieve_ms'),
            't_rerank_ms':  timings.get('rerank_ms'),
            't_llm_ms':     timings.get('llm_ms'),
            'client_ip': client_ip,
        })
        payload['query_id'] = qid
    except Exception as e:
        print(f'[query_log] log_query failed: {e}')
        payload['query_id'] = None
    payload['timings'] = timings
    payload['router_category'] = router_cat


@app.post('/query')
def query(req: QueryReq, request: Request):
    ip = None
    try:
        if request.client is not None:
            ip = request.client.host
    except Exception:
        pass
    return _run_pipeline(req, client_ip=ip)


# ---------- OpenAI-compatible pass-through (for Open WebUI) ----------

class ChatMsg(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    model: str
    messages: List[ChatMsg]
    stream: bool = False
    temperature: float = 0.0


def _render_markdown(payload: dict) -> str:
    md = payload['answer_markdown']
    if payload.get('suspicious_quotes'):
        md += '\n\n---\n> WARNING: Some quoted passages could not be verified verbatim '\
              'against the retrieved sources; they are marked inline.'
    md += '\n\n---\n### Sources\n'
    for c in payload['citations']:
        dl = ''
        if c.get('source_url'):
            dl = f" - [original]({c['source_url']})"
        src_tag = ''
        if c.get('download_source') and c['download_source'] != 'cbic_primary':
            src_tag = f" _(via {c['download_source']})_"
        md += (f"- **[S{c['index']}]** {c['title']} - "
               f"{c.get('subcategory','')} p.{c['page']}{src_tag}{dl}\n")
    return md


@app.post('/v1/chat/completions')
def chat(req: ChatReq):
    qs = [m.content for m in req.messages if m.role == 'user']
    if not qs:
        raise HTTPException(400, 'no user message')
    question = qs[-1]
    payload = _run_pipeline(QueryReq(question=question))
    md = _render_markdown(payload)
    now = int(time.time())
    return {
        'id': f'chatcmpl-{uuid.uuid4().hex[:12]}',
        'object': 'chat.completion',
        'created': now,
        'model': req.model,
        'choices': [{
            'index': 0,
            'message': {'role': 'assistant', 'content': md},
            'finish_reason': 'stop',
        }],
        'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
        'x_citations': payload['citations'],
        'x_verified_quotes': payload['verified_quotes'],
        'x_suspicious_quotes': payload['suspicious_quotes'],
        'x_query_id': payload.get('query_id'),
        'x_timings': payload.get('timings'),
    }


@app.get('/v1/models')
def list_models():
    return {'object': 'list',
            'data': [{'id': 'cbic-rag', 'object': 'model', 'owned_by': 'cbic',
                      'created': int(time.time())}]}


@app.get('/health')
def health():
    return {'status': 'ok'}


# ---------- Query log & corpus-stats endpoints --------------------------

# Cache for Qdrant category histogram (refreshed every 5 min).
_cat_cache: Dict[str, Any] = {'ts': 0, 'data': None, 'points': 0, 'docs': 0}
_cat_cache_lock = threading.Lock()
_CAT_CACHE_TTL = 300  # seconds
_CAT_SAMPLE_LIMIT = 5000


def _collection_stats() -> Dict[str, Any]:
    """Returns {points, docs, by_category}. Cached for 5 min."""
    with _cat_cache_lock:
        now = time.time()
        if _cat_cache['data'] is not None and (now - _cat_cache['ts']) < _CAT_CACHE_TTL:
            return {
                'points': _cat_cache['points'],
                'docs': _cat_cache['docs'],
                'by_category': _cat_cache['data'],
                'sampled': True,
                'sample_limit': _CAT_SAMPLE_LIMIT,
            }
    points = 0
    docs = 0
    by_cat: Dict[str, int] = {}
    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(url=QDRANT_URL, timeout=15)
        try:
            info = qc.get_collection(QDRANT_COLL)
            points = int(getattr(info, 'points_count', 0) or 0)
        except Exception as e:
            print(f'[stats] get_collection: {e}')
        try:
            next_page = None
            scanned = 0
            seen_docs = set()
            while scanned < _CAT_SAMPLE_LIMIT:
                batch_limit = min(512, _CAT_SAMPLE_LIMIT - scanned)
                pts, next_page = qc.scroll(
                    collection_name=QDRANT_COLL,
                    limit=batch_limit,
                    offset=next_page,
                    with_payload=['category', 'doc_id'],
                    with_vectors=False,
                )
                if not pts:
                    break
                for p in pts:
                    pl = p.payload or {}
                    cat = pl.get('category') or '(uncategorized)'
                    by_cat[cat] = by_cat.get(cat, 0) + 1
                    did = pl.get('doc_id')
                    if did:
                        seen_docs.add(did)
                scanned += len(pts)
                if next_page is None:
                    break
            docs = len(seen_docs)
        except Exception as e:
            print(f'[stats] scroll: {e}')
    except Exception as e:
        print(f'[stats] qdrant unavailable: {e}')

    with _cat_cache_lock:
        _cat_cache['ts'] = time.time()
        _cat_cache['data'] = by_cat
        _cat_cache['points'] = points
        _cat_cache['docs'] = docs
    return {
        'points': points,
        'docs': docs,
        'by_category': by_cat,
        'sampled': True,
        'sample_limit': _CAT_SAMPLE_LIMIT,
    }


@app.get('/v1/stats')
def v1_stats():
    try:
        qstats = query_log.stats()
    except Exception as e:
        qstats = {'error': str(e)}
    coll = _collection_stats()
    return {
        'ok': True,
        'collection': coll,
        'queries': qstats,
        'llm': {
            'model': LLM_MODEL,
            'endpoint': LITELLM_URL,
            'router_model': os.environ.get('ROUTER_MODEL', LLM_MODEL),
            'hyde_model': os.environ.get('HYDE_MODEL', LLM_MODEL),
            'use_router': USE_ROUTER,
            'use_hyde': USE_HYDE,
            'rerank': RERANK,
            'top_k_retr': TOP_K_RETR,
            'top_k_rerank': TOP_K_RERANK,
        },
        'qdrant': {'url': QDRANT_URL, 'collection': QDRANT_COLL},
    }


@app.get('/v1/queries/recent')
def v1_queries_recent(n: int = 20):
    return {'ok': True, 'items': query_log.recent(n)}


@app.get('/v1/queries/{qid}')
def v1_queries_by_id(qid: int):
    r = query_log.by_id(qid)
    if r is None:
        raise HTTPException(404, f'query {qid} not found')
    return {'ok': True, 'item': r}


class RateReq(BaseModel):
    rating: int
    feedback: Optional[str] = None


@app.post('/v1/queries/{qid}/rate')
def v1_queries_rate(qid: int, body: RateReq):
    try:
        ok = query_log.rate_query(qid, body.rating, body.feedback)
    except ValueError as ve:
        raise HTTPException(400, str(ve))
    if not ok:
        raise HTTPException(404, f'query {qid} not found')
    return {'ok': True, 'id': qid, 'rating': body.rating}


# --- Static UI ---------------------------------------------------------
app.mount('/ui', StaticFiles(directory='/opt/indian-legal-ai/rag/cbic_rag/static', html=True), name='ui')
# Admin/quality dashboard (separate mount so the main UI is preserved).
try:
    os.makedirs('/opt/indian-legal-ai/rag/cbic_rag/static_admin', exist_ok=True)
except Exception:
    pass
app.mount('/admin', StaticFiles(directory='/opt/indian-legal-ai/rag/cbic_rag/static_admin', html=True), name='admin')

# Shadow-cutover dashboard (Stage I). Mount static + attach router.
try:
    app.mount('/shadow_ui', StaticFiles(directory='static', html=True), name='shadow')
except Exception as _e:
    print(f'[shadow_ui] mount failed: {_e}')
try:
    from api_v2_shadow import attach as attach_shadow  # type: ignore
    attach_shadow(app)
except Exception as _e:
    print(f'[shadow] attach failed: {_e}')

@app.get('/')
def _root():
    return RedirectResponse('/ui/')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', '9500')))
