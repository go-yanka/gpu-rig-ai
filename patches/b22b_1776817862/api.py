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
import os, json, time, uuid, httpx, threading, asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, FileResponse, Response
from pydantic import BaseModel


# b13b14b15b16_v1 B16: faulthandler + serialize embedder calls (prevent SIGABRT in native code).
import faulthandler as _b16_faulthandler
try:
    _b16_faulthandler.enable()
except Exception:
    pass
import threading as _b16_threading
_B16_EMBED_LOCK = _b16_threading.Lock()
try:
    import embedder_direct as _b16_ed
    if hasattr(_b16_ed, '_Pool') and not getattr(_b16_ed._Pool, '_b16_patched', False):
        _b16_orig_embed = _b16_ed._Pool.embed
        def _b16_locked_embed(self, texts):
            with _B16_EMBED_LOCK:
                return _b16_orig_embed(self, texts)
        _b16_ed._Pool.embed = _b16_locked_embed
        _b16_ed._Pool._b16_patched = True
        print('[b16] embedder_direct._Pool.embed serialized with lock')
except Exception as _e:
    print(f'[b16] could not patch embedder_direct: {_e}')

# b22b_v1 sentinel
from retriever import retrieve, rerank, augment_section_aware
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


# ---- Phase 4: startup warmup (eliminates 15-25s cold-start penalty) ----
@app.on_event("startup")
def _phase4_warmup():
    import time as _t
    t0 = _t.time()
    try:
        import embedder as _emb
        _emb.get_bm25()
        print(f"[warmup] bm25 ok @{_t.time()-t0:.1f}s")
    except Exception as _e:
        print(f"[warmup] bm25 FAIL: {_e}")
    try:
        import embedder as _emb
        _emb.embed_query("warmup gst itc section 16")
        print(f"[warmup] embed_query ok @{_t.time()-t0:.1f}s")
    except Exception as _e:
        print(f"[warmup] embed_query FAIL: {_e}")
    try:
        from colbert_rerank import rerank_colbert as _rr
        _rr("warmup", [{"text": "warmup chunk gst"}], top_n=1)
        print(f"[warmup] colbert ok @{_t.time()-t0:.1f}s")
    except Exception as _e:
        print(f"[warmup] colbert FAIL: {_e}")
    try:
        with httpx.Client(timeout=30) as _cli:
            _cli.post(f"{LITELLM_URL}/v1/chat/completions",
                      headers={"Authorization": f"Bearer {LLM_KEY}",
                               "Content-Type": "application/json"},
                      json={"model": LLM_MODEL,
                            "messages": [{"role": "user", "content": "hi /no_think"}],
                            "max_tokens": 4, "temperature": 0.0,
                            "chat_template_kwargs": {"enable_thinking": False}})
        print(f"[warmup] llm ping ok @{_t.time()-t0:.1f}s")
    except Exception as _e:
        print(f"[warmup] llm ping FAIL: {_e}")
    print(f"[warmup] DONE in {_t.time()-t0:.1f}s")



class QueryReq(BaseModel):
    question: str
    k: int = TOP_K_RERANK
    filters: Optional[Dict[str, Any]] = None


def _call_llm(system: str, user: str, temperature: float = 0.0) -> str:
    # Phase 0: disable qwen3 thinking mode + cap tokens + enable prompt cache.
    sys_msg = system + " /no_think"
    user_msg = user + " /no_think"
    with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as cli:  # b8b9p5_v1
        r = cli.post(f"{LITELLM_URL}/v1/chat/completions",
                     headers={"Authorization": f"Bearer {LLM_KEY}",
                              "Content-Type": "application/json"},
                     json={
                         "model": LLM_MODEL,
                         "messages": [
                             {"role": "system", "content": sys_msg},
                             {"role": "user",   "content": user_msg},
                         ],
                         "temperature": temperature,
                         "top_p": 0.95,
                         "max_tokens": 6144,  # b8b9p5_v1
                         "cache_prompt": True,
                         "chat_template_kwargs": {"enable_thinking": False},
                     })
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def _ms_since(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 2)



# b22b_v1 B17: multi-sub-query decomposition for complex multi-part questions.
import functools as _b17_ft
import re as _b17_re

_B17_DECOMP_SYS = (
    "You split a complex legal question into focused sub-queries for retrieval. "
    "Output ONE sub-query per line, no numbering, no commentary, no preamble. "
    "Each sub-query focuses on a single legal concept, section, rule, or factual aspect."
)
_B17_DECOMP_USER = (
    "Split this complex legal question into 3-6 focused sub-queries for retrieval. "
    "Output ONE sub-query per line, no numbering, no commentary. Each sub-query "
    "should focus on a single legal concept, section, rule, or factual aspect.\n\n"
    "Question: {q}\n\nSub-queries:"
)

def _b17_is_multipart(question: str) -> bool:
    if not question:
        return False
    qmarks = question.count('?')
    if qmarks > 2:
        return True
    if len(question) > 500:
        return True
    numbered = len(_b17_re.findall(r'(?:^|\s)\(?\d+[\.\)]', question))
    if numbered > 1:
        return True
    return False

@_b17_ft.lru_cache(maxsize=256)
def _b17_decompose_cached(question: str) -> tuple:
    try:
        raw = _call_llm(_B17_DECOMP_SYS, _B17_DECOMP_USER.format(q=question),
                        temperature=0.1)
    except Exception as _e:
        print(f'[b17] decompose failed: {_e}')
        return tuple()
    lines = []
    for ln in (raw or '').splitlines():
        s = ln.strip()
        if not s:
            continue
        s = _b17_re.sub(r'^[\-\*\u2022]\s*', '', s)
        s = _b17_re.sub(r'^\(?\d+[\.\)]\s*', '', s)
        s = _b17_re.sub(r'^sub-?quer(?:y|ies)\s*[:\-]\s*', '', s, flags=_b17_re.I)
        if len(s) < 8:
            continue
        if s.lower().startswith(('here are', 'sub-queries', 'sub queries', 'output:')):
            continue
        lines.append(s)
        if len(lines) >= 6:
            break
    return tuple(lines)

def _b17_multi_retrieve(question: str, filters, k_sub: int = 6,
                       max_union: int = 40, timings=None) -> tuple:
    if not _b17_is_multipart(question):
        return [], 0, 0
    subs = _b17_decompose_cached(question)
    if not subs:
        return [], 0, 0
    seen = set()
    out = []
    for s in subs:
        try:
            extra = retrieve(s, k=k_sub, filters=filters, timings=None)
        except Exception:
            continue
        for c in extra:
            key = ((c.get('doc_id') or ''), c.get('chunk_index', c.get('char_start', '')))
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
            if len(out) >= max_union:
                break
        if len(out) >= max_union:
            break
    if timings is not None:
        timings['multi_subquery_count'] = len(subs)
        timings['subquery_union_size'] = len(out)
    return out, len(subs), len(out)




# b13b14b15b16_v1 B14: post-hoc answer validator (flag hallucinated sections/rules, placeholder leaks, repeats).
def _b14_post_validate_answer(answer_md, citations):
    import re as _re
    from collections import Counter as _Counter
    warnings = []
    if not isinstance(answer_md, str):
        return {'warnings': ['no_answer'], 'clean': False}
    if _re.search(r'<\s*exact\s*text\s*>|\[\s*exact\s*text\s*\]|<\s*n\s*>', answer_md, _re.I):
        warnings.append('placeholder_leak')
    cite_blob = ' '.join(
        (c.get('text_full') or c.get('text') or c.get('excerpt') or '') for c in (citations or [])
    ).lower()
    cited_sections = set(_re.findall(r'\b[Ss]ection\s+(\d+[A-Z]?(?:\(\d+\))?(?:\([a-z]\))?)', answer_md))
    unsupported_s = [s for s in cited_sections if f'section {s.lower()}' not in cite_blob]
    if unsupported_s:
        warnings.append('unsupported_sections:' + ','.join(sorted(unsupported_s)[:5]))
    cited_rules = set(_re.findall(r'\b[Rr]ule\s+(\d+[A-Z]?)', answer_md))
    unsupported_r = [r for r in cited_rules if f'rule {r.lower()}' not in cite_blob]
    if unsupported_r:
        warnings.append('unsupported_rules:' + ','.join(sorted(unsupported_r)[:5]))
    sents = _re.split(r'(?<=[.!?])\s+', answer_md)
    counts = _Counter(s.strip() for s in sents if len(s.strip()) > 30)
    if any(c >= 3 for c in counts.values()):
        warnings.append('repeated_sentence')
    return {'warnings': warnings, 'clean': len(warnings) == 0}


def _run_pipeline(req: QueryReq, client_ip: Optional[str] = None) -> Dict[str, Any]:
    """Run the RAG pipeline with per-stage timings + log to sqlite."""
    t_start = time.perf_counter()
    timings: Dict[str, Optional[float]] = {
        'router_ms': None, 'hyde_ms': None,
        'retrieve_ms': None, 'embed_query_ms': None, 'qdrant_ms': None,  # b2_subtimings
        'rerank_ms': None,
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
    _sub = {}  # b2_subtimings

    hits = retrieve(search_text, k=TOP_K_RETR, filters=filters, timings=_sub)

    timings['embed_query_ms'] = _sub.get('embed_query_ms')

    timings['qdrant_ms']      = _sub.get('qdrant_ms')
    # b11b12a8_v1: surface embed cache stats
    if 'embed_cache_stats' in _sub:
        timings['embed_cache_stats'] = _sub['embed_cache_stats']
    if 'embed_cache_hit' in _sub:
        timings['embed_cache_hit'] = _sub['embed_cache_hit']
    if not hits and filters:
        hits = retrieve(search_text, k=TOP_K_RETR, filters=None)
    # b22b_v1 B17: multi-sub-query union for complex questions
    timings['multi_subquery_count'] = 0
    timings['subquery_union_size'] = 0
    try:
        multi_hits, n_subs, union_sz = _b17_multi_retrieve(
            req.question, filters, k_sub=6, max_union=40, timings=timings)
        if multi_hits:
            seen = {((c.get('doc_id') or ''), c.get('chunk_index', c.get('char_start',''))) for c in hits}
            for c in multi_hits:
                k = ((c.get('doc_id') or ''), c.get('chunk_index', c.get('char_start','')))
                if k in seen:
                    continue
                seen.add(k); hits.append(c)
                if len(hits) >= 40:
                    break
    except Exception as _e:
        print(f'[b17] multi-retrieve err: {_e}')
    # b22b_v1 B19: section-aware augmentation (additive, capped at 40)
    try:
        hits = augment_section_aware(req.question, hits, filters=filters,
                                     k_per_ref=3, max_total=40, timings=timings)
    except Exception as _e:
        print(f'[b19] section-aware err: {_e}')
        timings.setdefault('section_aware_added', 0)
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
    # b13b14b15b16_v1 B14: attach quality flags.
    try:
        _v14 = _b14_post_validate_answer(payload.get('answer_markdown',''), payload.get('citations') or [])
        payload['quality_warnings'] = _v14['warnings']
        payload['quality_clean'] = _v14['clean']
    except Exception as _e:
        payload['quality_warnings'] = [f'validator_error:{_e}']
        payload['quality_clean'] = False
    # b13: surface diversity debug on timings
    try:
        if top and isinstance(top[-1], dict) and '_b13_debug' in top[-1]:
            _dbg = top[-1].pop('_b13_debug')
            timings['rerank_diversity_applied'] = _dbg.get('applied', False)
            timings['rerank_groups'] = _dbg.get('groups', {})
    except Exception:
        pass
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
async def query(req: QueryReq, request: Request):
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




from fastapi.responses import StreamingResponse


def _build_stream_prompt(question: str, top):
    """Reuse build_prompt; we only need the raw prompt strings."""
    return build_prompt(question, top)


async def _stream_from_llama(sys_p: str, usr_p: str):
    """Yield SSE data: frames of token deltas from llama-server streaming."""
    sys_msg = sys_p + ' /no_think'
    user_msg = usr_p + ' /no_think'
    payload = {
        'model': LLM_MODEL,
        'messages': [
            {'role': 'system', 'content': sys_msg},
            {'role': 'user',   'content': user_msg},
        ],
        'temperature': 0.0,
        'top_p': 0.95,
        'max_tokens': 6144,  # b8b9p5_v1
        'stream': True,
        'cache_prompt': True,
        'chat_template_kwargs': {'enable_thinking': False},
    }
    headers = {'Authorization': f'Bearer {LLM_KEY}',
               'Content-Type': 'application/json'}
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as cli:  # b8b9p5_v1
        async with cli.stream('POST', f'{LITELLM_URL}/v1/chat/completions',
                              json=payload, headers=headers) as resp:
            async for line in resp.aiter_lines():
                if not line or not line.startswith('data: '):
                    continue
                chunk = line[6:]
                if chunk.strip() == '[DONE]':
                    break
                try:
                    j = json.loads(chunk)
                    delta = j.get('choices', [{}])[0].get('delta', {}).get('content')
                    if delta:
                        yield delta
                except Exception:
                    continue


@app.post('/query_stream')
async def query_stream(req: QueryReq, request: Request):
    """SSE streaming endpoint. Frames:
    event: stage  data: {"stage":"retrieving|reranking|generating", "ms":123}
    event: token  data: {"text":"..."}
    event: final  data: {full payload with citations, verification, timings}
    """
    ip = request.client.host if request.client else None
    t_start = time.perf_counter()

    async def gen():
        # 1) retrieve + rerank in thread
        def _ret():
            t0 = time.perf_counter()
            category = None
            filt = req.filters
            if USE_ROUTER:
                try:
                    rr = route(req.question) or {}
                    category = rr.get('category')
                    if category and not filt:
                        filt = filters_for(category) or None
                except Exception:
                    pass
            hits = retrieve(req.question, k=TOP_K_RETR, filters=filt)
            t1 = time.perf_counter()
            if RERANK == 'colbert':
                top = rerank_colbert(req.question, hits, top_n=req.k or TOP_K_RERANK)
            else:
                top = rerank(req.question, hits, top_n=req.k or TOP_K_RERANK)
            t2 = time.perf_counter()
            return top, category, (t1-t0)*1000, (t2-t1)*1000

        top, category, ret_ms, rer_ms = await asyncio.to_thread(_ret)
        yield f"event: stage\ndata: {json.dumps({'stage':'retrieved','ret_ms':round(ret_ms,1),'rer_ms':round(rer_ms,1),'n':len(top),'category':category})}\n\n"

        # 2) build prompt + stream tokens
        sys_p, usr_p = build_prompt(req.question, top)
        answer_buf = []
        async for delta in _stream_from_llama(sys_p, usr_p):
            answer_buf.append(delta)
            yield f"event: token\ndata: {json.dumps({'text': delta})}\n\n"

        # 3) final payload (verification + citations)
        ans = ''.join(answer_buf)
        payload = build_response_payload(req.question, top, ans)
        try:
            _v14s = _b14_post_validate_answer(payload.get('answer_markdown',''), payload.get('citations') or [])
            payload['quality_warnings'] = _v14s['warnings']
            payload['quality_clean'] = _v14s['clean']
        except Exception:
            pass
        payload['timings'] = {
            'retrieve_ms': round(ret_ms, 1),
            'rerank_ms': round(rer_ms, 1),
            'total_ms': round((time.perf_counter() - t_start) * 1000, 1),
        }
        payload['router_category'] = category
        # persist to query_log (fire-and-forget-ish)
        try:
            qid = query_log.log_query(req.question, payload, client_ip=ip)
            payload['query_id'] = qid
        except Exception:
            pass
        yield f"event: final\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(gen(), media_type='text/event-stream')


@app.post('/v1/chat/completions')
async def chat(req: ChatReq):
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

# ---- B3: Server-side PDF proxy (b3_pdf_proxy) -----------------------------
import sqlite3 as _sqlite3  # b3
_MANIFEST = os.environ.get("CBIC_MANIFEST", "/opt/indian-legal-ai/data/scraped/cbic/_manifest.sqlite")

@app.get("/pdf/{doc_id}")
def serve_pdf(doc_id: str):
    """Serve a local PDF by doc_id. Looks up path_en in manifest.sqlite.
    Returns raw application/pdf; handles titles with spaces in Content-Disposition."""
    try:
        con = _sqlite3.connect(f"file:{_MANIFEST}?mode=ro", uri=True, timeout=5)
        con.row_factory = _sqlite3.Row
        r = con.execute("SELECT path_en, path_hi, title FROM docs WHERE doc_id=? LIMIT 1", (doc_id,)).fetchone()
        con.close()
    except Exception as e:
        raise HTTPException(500, f"manifest read failed: {e}")
    if r is None:
        raise HTTPException(404, f"doc_id not found: {doc_id}")
    path = r["path_en"] or r["path_hi"]
    if not path or not os.path.exists(path):
        raise HTTPException(404, f"pdf file missing on disk: {path}")
    title = (r["title"] or doc_id).strip() or doc_id
    # ASCII-safe filename for legacy Content-Disposition, plus RFC-5987 UTF-8 fallback
    safe = "".join(ch if (ch.isalnum() or ch in "-._") else "_" for ch in title)[:120] or "document"
    fname = safe + ".pdf"
    from urllib.parse import quote as _urlquote
    disp = f'inline; filename="{fname}"; filename*=UTF-8\'\'{_urlquote(title)}.pdf'
    return FileResponse(path, media_type="application/pdf", headers={"Content-Disposition": disp})



# ---- F2: PDF region snapshot (f2_snippet) ---------------------------------
# Returns a PNG crop of the page region where a quoted passage appears.

_FITZ_LOCK = threading.Lock()

def _open_pdf_by_doc(doc_id: str):
    con = _sqlite3.connect(f"file:{_MANIFEST}?mode=ro", uri=True, timeout=5)
    con.row_factory = _sqlite3.Row
    r = con.execute("SELECT path_en, path_hi, title FROM docs WHERE doc_id=? LIMIT 1", (doc_id,)).fetchone()
    con.close()
    if r is None: raise HTTPException(404, f"doc_id not found: {doc_id}")
    path = r["path_en"] or r["path_hi"]
    if not path or not os.path.exists(path): raise HTTPException(404, f"pdf missing: {path}")
    return path, (r["title"] or doc_id)


def _canon_for_search(t: str) -> str:
    import re as _re, unicodedata as _ud
    t = _ud.normalize("NFKC", t)
    t = t.replace("\u00a0", " ").replace("\u00ad", "")
    t = t.replace("\u2013", "-").replace("\u2014", "-")
    t = t.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    t = _re.sub(r"\s+", " ", t).strip()
    return t


@app.get("/pdf/{doc_id}/page/{page}/image")
def serve_pdf_page_png(doc_id: str, page: int, dpi: int = 110):
    """Render a whole PDF page as PNG. page is 1-based."""
    import fitz  # PyMuPDF
    path, _ = _open_pdf_by_doc(doc_id)
    dpi = max(60, min(250, int(dpi)))
    with _FITZ_LOCK:
        doc = fitz.open(path)
        try:
            if page < 1 or page > len(doc):
                raise HTTPException(404, f"page out of range 1..{len(doc)}")
            pg = doc[page-1]
            mat = fitz.Matrix(dpi/72.0, dpi/72.0)
            pix = pg.get_pixmap(matrix=mat, alpha=False)
            png = pix.tobytes("png")
        finally:
            doc.close()
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/pdf/{doc_id}/page/{page}/snippet")
def serve_pdf_snippet(doc_id: str, page: int, q: str = "", dpi: int = 150, pad: int = 12):
    """Render a cropped PNG around the first occurrence of `q` on the given page.
    If q is empty or not found, returns the whole page.
    """
    import fitz
    path, _ = _open_pdf_by_doc(doc_id)
    dpi = max(80, min(300, int(dpi)))
    q_canon = _canon_for_search(q or "")
    with _FITZ_LOCK:
        doc = fitz.open(path)
        try:
            if page < 1 or page > len(doc):
                raise HTTPException(404, f"page out of range 1..{len(doc)}")
            pg = doc[page-1]
            bbox = None
            if q_canon:
                # Try progressively shorter prefixes so we handle LLM prefix-glue
                tries = [q_canon, q_canon[:120], q_canon[:80], q_canon[:60], q_canon[:40]]
                seen = set()
                for t in tries:
                    if not t or t in seen: continue
                    seen.add(t)
                    hits = pg.search_for(t, quads=False)
                    if hits:
                        # Union all rects on this page
                        r = hits[0]
                        for rr in hits[1:]:
                            r = r | rr
                        bbox = fitz.Rect(r)
                        break
            if bbox is None:
                clip = pg.rect  # whole page fallback
            else:
                bbox = bbox + (-pad, -pad, pad, pad)  # pad the crop
                clip = bbox & pg.rect
            mat = fitz.Matrix(dpi/72.0, dpi/72.0)
            pix = pg.get_pixmap(matrix=mat, alpha=False, clip=clip)
            png = pix.tobytes("png")
        finally:
            doc.close()
    hdr = {"Cache-Control": "public, max-age=3600"}
    if bbox is None:
        hdr["X-Snippet-Match"] = "none"
    else:
        hdr["X-Snippet-Match"] = f"bbox={bbox.x0:.0f},{bbox.y0:.0f},{bbox.x1:.0f},{bbox.y1:.0f}"
    return Response(content=png, media_type="image/png", headers=hdr)


# esuite_v1 — /v1/meta + /v1/queries/{qid}/feedback
from pydantic import BaseModel as _EsuiteBaseModel

class _EsuiteFeedbackReq(_EsuiteBaseModel):
    kind: str
    citation_index: Optional[int] = None
    quote: Optional[str] = None
    reason: Optional[str] = None

@app.get('/v1/meta')
def _esuite_meta():
    import sqlite3 as _s3
    total_docs = None
    # Prefer Qdrant collection point count (spec pass criterion requires >100k)
    try:
        import httpx as _httpx
        _qurl = os.environ.get('QDRANT_URL', 'http://localhost:6343')
        _qcoll = os.environ.get('QDRANT_COLL', 'cbic_v1')
        r = _httpx.get(f"{_qurl}/collections/{_qcoll}", timeout=3.0)
        if r.status_code == 200:
            total_docs = r.json().get('result', {}).get('points_count')
    except Exception:
        pass
    if total_docs is None:
        try:
            con = _s3.connect(f"file:{_MANIFEST}?mode=ro", uri=True, timeout=3)
            total_docs = con.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
            con.close()
        except Exception:
            total_docs = None
    return {
        'llm_model': os.environ.get('LLM_MODEL', 'qwen3-14b-hermes'),
        'llm_url': os.environ.get('LITELLM_URL', os.environ.get('LLM_URL', '')),
        'embedder': 'BGE-M3 (Vulkan GPU5)',
        'fusion': 'RRF (dense + BM25)',
        'reranker': 'ColBERT (CPU, lazy)',
        'verifier': 'fuzzy (canon + label-strip + 6-gram 0.80)',
        'qdrant_coll': os.environ.get('QDRANT_COLL', 'cbic_v1'),
        'top_k_retr': int(os.environ.get('TOP_K_RETR', 12)),
        'top_k_rerank': int(os.environ.get('TOP_K_RERANK', 6)),
        'total_docs': total_docs,
        'corpus': 'CBIC (GST, Customs, Central Excise, Service Tax, Others)',
    }

@app.post('/v1/queries/{qid}/feedback')
def _esuite_feedback(qid: str, req: _EsuiteFeedbackReq):
    import json as _json, time as _time, pathlib as _pl
    rec = {'qid': qid, 'ts': _time.time(), **req.dict()}
    p = _pl.Path('/opt/indian-legal-ai/rag/cbic_rag/_feedback.jsonl')
    with p.open('a') as f:
        f.write(_json.dumps(rec) + '\n')
    return {'ok': True}

app.mount('/ui', StaticFiles(directory='/opt/indian-legal-ai/rag/cbic_rag/static', html=True), name='ui')
# Admin/quality dashboard (separate mount so the main UI is preserved).
try:
    os.makedirs('/opt/indian-legal-ai/rag/cbic_rag/static_admin', exist_ok=True)
except Exception:
    pass
app.mount('/admin', StaticFiles(directory='/opt/indian-legal-ai/rag/cbic_rag/static_admin', html=True), name='admin')

@app.get('/')
def _root():
    return RedirectResponse('/ui/')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', '9500')))
