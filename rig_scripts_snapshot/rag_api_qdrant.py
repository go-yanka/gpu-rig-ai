#!/usr/bin/env python3
"""
Indian Legal AI — Production RAG API (Qdrant + fastembed)
Port: 7000 (same as the POC API — drop-in replacement)

  GET  /          → Chat UI
  POST /ask       → {question} → {answer, sources}
  GET  /health    → status
  GET  /docs      → Swagger
"""

import json, os, sys, urllib.request, time
from concurrent.futures import ThreadPoolExecutor
import re
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

QDRANT_URL = "http://localhost:6333"
COLLECTION = "indian_legal_full"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
GPU_PORTS  = [9080, 9081, 9082, 9083, 9084, 9086]

# -- Retrieval tuning --
DEFAULT_TOP_K       = 12
OVERFETCH_FACTOR    = 3
MIN_RAW_SCORE       = 0.35
MIN_CONFIDENCE      = 0.45
TIER_BOOST          = {1: 0.15, 2: 0.08, 3: 0.00}
MAX_CHUNKS_PER_SRC  = 3
RERANK_URL          = "http://localhost:9096/v1/rerank"
RERANK_POOL         = 40    # rerank top-N before per-source cap
RERANK_TIMEOUT      = 15
DECOMPOSER_URL      = "http://localhost:9081/v1/chat/completions"
DECOMPOSER_TIMEOUT  = 15
MAX_SUB_QUERIES     = 3
FORCE_HIT_SCORE     = 0.95   # synthetic score for keyword-anchored hits
MAX_FORCED_HITS     = 8      # cap total keyword-anchored hits merged in
LLM_MAX_TOKENS      = -1    # -1 = unlimited, bounded only by 16K context window
DATASET_BOOST = {
    'gst':     ['gst_acts','gst_rates','gst_wiki','gst_council','gst_faqs'],
    'ipc':     ['indian_laws','ipc_insights'],
    'incometax':['indian_laws'],
    'constitution':['indian_laws'],
    'companies':['indian_laws'],
    'ni':      ['indian_laws'],
}
TOPIC_TRIGGERS = {
    'gst':     ['gst','cgst','igst','input tax credit','online money gaming',
                'rule 31b','amendment act 2023','cess','hsn','oidar'],
    'ipc':     ['ipc','penal code','section 302','section 378','section 420',
                'murder','theft','cheating'],
    'incometax':['income tax','80c','80d','itr','tds','hra','section 139'],
    'constitution':['constitution','article 14','article 19','article 21',
                   'fundamental right','directive principle'],
    'companies':['companies act','director','board resolution'],
    'ni':      ['cheque','section 138','ni act','negotiable instrument'],
}

SYSTEM = """You are an expert Indian legal and tax advisor with deep knowledge of:
- Income Tax Act 1961 and all amendments
- GST (CGST Act 2017, IGST, SGST)
- Indian Penal Code 1860
- Constitution of India
- Companies Act 2013
- FEMA 1999
- Negotiable Instruments Act 1881
- Labour Laws

CITATION RULES (mandatory):
1. Use ONLY the provided LEGAL CONTEXT. Do not invent sections, rules, or case names.
2. Every claim must cite the Act name and Section/Rule number inline, e.g. "Section 138 NI Act".
3. If a case name appears in context (e.g. "Bachan Singh v. State of Punjab"), cite it when relevant.
4. For comparison questions, organize the answer so that each statute/section is addressed distinctly.
5. If the context is insufficient for any part of the question, state EXPLICITLY which part is not covered
   and name the specific section/rule/notification that would be needed.
6. Never output inner monologue, "Wait, let me check", or reasoning traces. Only the final answer."""

# ── Load embedder + Qdrant ──────────────────────────────────────────────────
print("Loading embedder...", flush=True)
from fastembed import TextEmbedding
embedder = TextEmbedding(model_name=EMBED_MODEL)

from qdrant_client import QdrantClient
client = QdrantClient(url=QDRANT_URL, timeout=60)

# Verify collection exists
try:
    info = client.get_collection(COLLECTION)
    print(f"Connected to Qdrant: '{COLLECTION}' has {info.points_count:,} points", flush=True)
except Exception as e:
    print(f"FATAL: Qdrant collection '{COLLECTION}' not found: {e}", flush=True)
    print("Run build_rag_index_v2.py first.", flush=True)
    sys.exit(1)

LLM_URL = None
for port in GPU_PORTS:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
            data = json.loads(r.read())
            if data.get("status") in ("ok", "no slot available"):
                LLM_URL = f"http://127.0.0.1:{port}/v1/chat/completions"
                print(f"LLM on port {port}", flush=True)
                break
    except:
        continue

app = FastAPI(title="Indian Legal AI", version="2.0-qdrant")

CHAT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Indian Legal AI</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0; height: 100vh; display: flex; flex-direction: column; }
  .header { background: linear-gradient(135deg, #1a1f2e 0%, #141928 100%); border-bottom: 1px solid #2d3748; padding: 14px 24px; display: flex; align-items: center; gap: 14px; flex-shrink: 0; }
  .header-icon { width: 40px; height: 40px; background: linear-gradient(135deg, #f59e0b, #d97706); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px; flex-shrink: 0; }
  .header-text h1 { font-size: 18px; font-weight: 700; color: #f1f5f9; }
  .header-text p { font-size: 12px; color: #94a3b8; margin-top: 1px; }
  .badge { margin-left: auto; background: #1e3a5f; color: #60a5fa; font-size: 11px; padding: 4px 10px; border-radius: 20px; border: 1px solid #2563eb33; }
  .chat-area { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 20px; scroll-behavior: smooth; }
  .welcome { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; gap: 28px; text-align: center; padding: 20px; }
  .welcome-icon { font-size: 56px; }
  .welcome h2 { font-size: 24px; font-weight: 700; color: #f1f5f9; }
  .welcome p { color: #94a3b8; font-size: 15px; max-width: 480px; line-height: 1.6; }
  .suggestions { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; max-width: 600px; width: 100%; }
  .suggestion-btn { background: #1e2535; border: 1px solid #2d3748; color: #cbd5e1; padding: 12px 16px; border-radius: 10px; cursor: pointer; font-size: 13px; text-align: left; transition: all 0.2s; line-height: 1.4; }
  .suggestion-btn:hover { background: #252e42; border-color: #f59e0b55; color: #f1f5f9; }
  .msg { display: flex; gap: 12px; max-width: 820px; width: 100%; }
  .msg.user { align-self: flex-end; flex-direction: row-reverse; }
  .msg.bot { align-self: flex-start; }
  .avatar { width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 16px; flex-shrink: 0; margin-top: 2px; }
  .msg.user .avatar { background: #2563eb; }
  .msg.bot .avatar { background: #92400e; }
  .bubble { padding: 14px 18px; border-radius: 16px; max-width: calc(100% - 50px); line-height: 1.65; font-size: 14px; }
  .msg.user .bubble { background: #1d4ed8; color: #eff6ff; border-bottom-right-radius: 4px; }
  .msg.bot .bubble { background: #1e2535; border: 1px solid #2d3748; color: #e2e8f0; border-bottom-left-radius: 4px; }
  .bubble strong { color: #f1f5f9; font-weight: 600; }
  .bubble em { color: #fbbf24; font-style: normal; }
  .bubble code { background: #0f1117; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 12px; color: #7dd3fc; }
  .sources { margin-top: 10px; padding-top: 10px; border-top: 1px solid #2d3748; display: flex; flex-wrap: wrap; gap: 6px; }
  .source-pill { background: #0f1117; border: 1px solid #334155; color: #94a3b8; font-size: 11px; padding: 3px 10px; border-radius: 20px; }
  .source-pill .score { color: #f59e0b; }
  .typing .bubble { padding: 16px 20px; }
  .dots { display: flex; gap: 5px; }
  .dots span { width: 7px; height: 7px; background: #60a5fa; border-radius: 50%; animation: bounce 1.2s infinite; }
  .dots span:nth-child(2) { animation-delay: 0.2s; }
  .dots span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce { 0%, 80%, 100% { transform: translateY(0); opacity: 0.4; } 40% { transform: translateY(-6px); opacity: 1; } }
  .input-bar { background: #141928; border-top: 1px solid #2d3748; padding: 16px 24px; flex-shrink: 0; }
  .input-row { display: flex; gap: 10px; max-width: 860px; margin: 0 auto; align-items: flex-end; }
  .input-box { flex: 1; background: #1e2535; border: 1px solid #2d3748; border-radius: 12px; padding: 12px 16px; color: #e2e8f0; font-size: 14px; font-family: inherit; resize: none; outline: none; min-height: 48px; max-height: 140px; overflow-y: auto; line-height: 1.5; transition: border-color 0.2s; }
  .input-box:focus { border-color: #f59e0b66; }
  .input-box::placeholder { color: #475569; }
  .send-btn { width: 48px; height: 48px; flex-shrink: 0; background: linear-gradient(135deg, #f59e0b, #d97706); border: none; border-radius: 12px; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 18px; transition: opacity 0.2s, transform 0.1s; }
  .send-btn:hover:not(:disabled) { opacity: 0.9; transform: scale(1.05); }
  .send-btn:active:not(:disabled) { transform: scale(0.97); }
  .send-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .hint { text-align: center; font-size: 11px; color: #475569; margin-top: 8px; }
  .chat-area::-webkit-scrollbar { width: 6px; }
  .chat-area::-webkit-scrollbar-track { background: transparent; }
  .chat-area::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 3px; }
</style>
</head>
<body>
<div class="header">
  <div class="header-icon">⚖️</div>
  <div class="header-text">
    <h1>Indian Legal AI</h1>
    <p id="subtitle">Income Tax · GST · IPC · Constitution · FEMA · Companies Act · NI Act</p>
  </div>
  <span class="badge" id="status-badge">● Loading</span>
</div>

<div class="chat-area" id="chat">
  <div class="welcome" id="welcome">
    <div class="welcome-icon">⚖️</div>
    <h2>Ask me any Indian Legal Question</h2>
    <p>I search relevant sections of Indian law and give you cited answers from the actual legislation.</p>
    <div class="suggestions">
      <button class="suggestion-btn" onclick="ask('What is the maximum deduction under Section 80C?')">💰 80C deduction limit?</button>
      <button class="suggestion-btn" onclick="ask('Can I claim HRA and home loan interest deduction together?')">🏠 HRA + home loan together?</button>
      <button class="suggestion-btn" onclick="ask('What is the GST registration threshold for a service business?')">🧾 GST registration threshold?</button>
      <button class="suggestion-btn" onclick="ask('What is the punishment for cheque bounce under NI Act?')">⚠️ Cheque bounce punishment?</button>
      <button class="suggestion-btn" onclick="ask('What are the new tax regime slabs for FY 2024-25?')">📊 New tax regime slabs 2024-25?</button>
      <button class="suggestion-btn" onclick="ask('What is Input Tax Credit eligibility under GST?')">🔄 What is Input Tax Credit?</button>
    </div>
  </div>
</div>

<div class="input-bar">
  <div class="input-row">
    <textarea class="input-box" id="q" placeholder="Ask about Income Tax, GST, IPC, Constitution..." rows="1" onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
    <button class="send-btn" id="send-btn" onclick="sendQuestion()">➤</button>
  </div>
  <div class="hint">Press Enter to send · Shift+Enter for new line</div>
</div>

<script>
let busy = false;

fetch('/health').then(r => r.json()).then(h => {
  document.getElementById('status-badge').textContent = `● ${h.documents.toLocaleString()} law passages`;
});

function autoResize(el) { el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 140) + 'px'; }
function handleKey(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuestion(); } }
function ask(text) { document.getElementById('q').value = text; sendQuestion(); }
function scrollDown() { const c = document.getElementById('chat'); c.scrollTop = c.scrollHeight; }
function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function addMsg(role, html, sources) {
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.remove();
  const chat = document.getElementById('chat');
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  const avatar = role === 'user' ? '👤' : '⚖️';
  let sourcesHtml = '';
  if (sources && sources.length) {
    sourcesHtml = `<div class="sources">` + sources.slice(0, 4).map(s =>
      `<span class="source-pill">📖 ${escHtml(s.source)} <span class="score">${s.score}</span></span>`
    ).join('') + `</div>`;
  }
  div.innerHTML = `<div class="avatar">${avatar}</div><div class="bubble">${html}${sourcesHtml}</div>`;
  chat.appendChild(div);
  scrollDown();
  return div;
}

function formatAnswer(text) {
  return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/Section (\d+[\w\(\)]*)/g, '<strong>Section $1</strong>')
    .replace(/Article (\d+[\w\(\)]*)/g, '<strong>Article $1</strong>')
    .replace(/Rs\. ([\d,]+)/g, '<strong>Rs. $1</strong>')
    .replace(/\n\n/g, '<br><br>')
    .replace(/\n([•\-\*]) /g, '<br>• ')
    .replace(/\n/g, '<br>');
}

async function sendQuestion() {
  if (busy) return;
  const inp = document.getElementById('q');
  const question = inp.value.trim();
  if (!question) return;
  inp.value = ''; inp.style.height = 'auto';
  busy = true;
  document.getElementById('send-btn').disabled = true;
  addMsg('user', escHtml(question));
  const typingDiv = document.createElement('div');
  typingDiv.className = 'msg bot typing';
  typingDiv.innerHTML = `<div class="avatar">⚖️</div><div class="bubble"><div class="dots"><span></span><span></span><span></span></div></div>`;
  document.getElementById('chat').appendChild(typingDiv);
  scrollDown();
  try {
    const res = await fetch('/ask', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question})
    });
    typingDiv.remove();
    if (!res.ok) {
      const err = await res.json().catch(() => ({detail: 'Unknown error'}));
      addMsg('bot', `<span style="color:#f87171">⚠️ Error: ${escHtml(err.detail || res.statusText)}</span>`);
    } else {
      const data = await res.json();
      addMsg('bot', formatAnswer(data.answer || '(no answer)'), data.sources);
    }
  } catch (e) {
    typingDiv.remove();
    addMsg('bot', `<span style="color:#f87171">⚠️ Could not reach server: ${escHtml(e.message)}</span>`);
  }
  busy = false;
  document.getElementById('send-btn').disabled = false;
  document.getElementById('q').focus();
}
</script>
</body>
</html>"""


class Question(BaseModel):
    question: str
    top_k: int = DEFAULT_TOP_K

def _find_llm():
    for port in GPU_PORTS:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                data = json.loads(r.read())
                if data.get("status") in ("ok", "no slot available"):
                    return f"http://127.0.0.1:{port}/v1/chat/completions"
        except:
            continue
    return None


# Keyword patterns for anchored retrieval
KEYWORD_RE = re.compile(
    r"("
    r"(?:Section|SECTION|Sec\.?)\s+\d+[A-Z]*"            # Section 138 / Section 80C
    r"|(?:Rule|RULE)\s+\d+[A-Z]*"                         # Rule 31B
    r"|(?:Article|ARTICLE)\s+\d+[A-Z]*"                   # Article 14
    r"|(?:Schedule|SCHEDULE)\s+[IVX]+"                     # Schedule III
    r"|(?:Amendment\s+Act[,]?\s*\d{4})"                  # Amendment Act 2023
    r")",
    re.IGNORECASE,
)
# Multi-word legal phrases to anchor on if they appear verbatim
LEGAL_PHRASES = [
    "online money gaming", "input tax credit", "place of supply",
    "reverse charge", "composition scheme", "registered person",
    "actionable claim", "specified actionable claim", "face value",
    "gross total income", "capital gains", "house rent allowance",
    "new tax regime", "old tax regime",
    "fundamental right", "directive principle",
    "cheque bounce", "dishonour of cheque",
]

def _extract_keywords(question: str):
    """Return list of (field, literal) tuples for Qdrant text-match."""
    q = question.strip()
    ql = q.lower()
    out = []
    # Statutory identifiers -> search both source and text
    for m in KEYWORD_RE.findall(q):
        token = m.strip()
        out.append(("text", token))
    # Legal phrases
    for phrase in LEGAL_PHRASES:
        if phrase in ql:
            out.append(("text", phrase))
    # Specific year-Act coupling: "2023 amendment" / "amendment act 2023"
    if "amendment" in ql and re.search(r"\b(19|20)\d{2}\b", ql):
        yr = re.search(r"\b(19|20)\d{2}\b", ql).group(0)
        out.append(("source", f"Amendment Act {yr}"))
    # Dedupe preserving order
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq[:8]   # cap

def _detect_topics(question: str):
    """Return set of topic tags triggered by question keywords."""
    ql = question.lower()
    hit = set()
    for topic, trigs in TOPIC_TRIGGERS.items():
        for t in trigs:
            if t in ql:
                hit.add(topic)
                break
    return hit

def _keyword_hits(keywords, topics, want=MAX_FORCED_HITS):
    """Scroll Qdrant for keyword matches; return hit-shaped dicts."""
    import urllib.request
    prefer_datasets = set()
    for tp in topics:
        prefer_datasets.update(DATASET_BOOST.get(tp, []))
    merged = {}
    for field, literal in keywords:
        body = {
            "filter": {"must": [{"key": field, "match": {"text": literal}}]},
            "limit": 20,
            "with_payload": True,
        }
        req = urllib.request.Request(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                resp = json.loads(r.read())
        except Exception:
            continue
        for p in resp.get("result", {}).get("points", []) or []:
            pid = p.get("id")
            if pid in merged:
                continue
            pl = p.get("payload", {}) or {}
            ds = pl.get("dataset", "")
            # Prefer chunks from the datasets that matched the topic
            boost = 0.05 if ds in prefer_datasets else 0.0
            merged[pid] = {
                "text":    pl.get("text", ""),
                "source":  pl.get("source", ""),
                "dataset": ds,
                "domain":  pl.get("domain", ""),
                "tier":    int(pl.get("tier", 3) or 3),
                "score":   FORCE_HIT_SCORE + boost,
                "_keyword": literal,
            }
            if len(merged) >= want:
                break
        if len(merged) >= want:
            break
    return list(merged.values())


def _rerank(question: str, hits: list) -> list:
    """Send (query, docs) to BGE reranker; attach .rerank_score to each hit.
       On any failure, leave hits untouched."""
    if not hits:
        return hits
    try:
        body = json.dumps({
            "model": "bge",
            "query": question,
            "documents": [h["text"][:1500] for h in hits],
        }).encode()
        req = urllib.request.Request(RERANK_URL, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=RERANK_TIMEOUT) as r:
            resp = json.loads(r.read())
        for item in resp.get("results", []):
            idx = item.get("index")
            sc  = float(item.get("relevance_score", 0.0))
            if isinstance(idx, int) and 0 <= idx < len(hits):
                hits[idx]["rerank_score"] = sc
        # Normalize: BGE reranker returns roughly -15..+15. Min-max scale to 0..1.
        scored = [h for h in hits if "rerank_score" in h]
        if scored:
            lo = min(h["rerank_score"] for h in scored)
            hi = max(h["rerank_score"] for h in scored)
            rng = max(hi - lo, 1e-6)
            for h in hits:
                rs = h.get("rerank_score", lo)
                h["rerank_norm"] = (rs - lo) / rng
        return hits
    except Exception as e:
        print(f"[rerank] failed, falling back: {e}", flush=True)
        return hits


_MULTI_ACT_RE = re.compile(
    r"(?:IPC|Penal Code|NI Act|Negotiable Instrument|Income[- ]Tax|GST|CGST|IGST|"
    r"Constitution|Art(?:icle)?\.?|Companies Act|FEMA|CrPC|CPC|Evidence Act)",
    re.I,
)
_CONTRAST_RE = re.compile(
    r"\b(?:vs\.?|versus|compare|compared to|difference between|differ|"
    r"and also|as well as|whereas|both|either .*? or|remedies|remedy)\b",
    re.I,
)

def _should_decompose(question: str) -> bool:
    """Heuristic: decompose if question touches 2+ statutes OR uses contrast words."""
    acts = set(m.group(0).upper() for m in _MULTI_ACT_RE.finditer(question))
    if len(acts) >= 2:
        return True
    if _CONTRAST_RE.search(question):
        return True
    # Too many distinct section anchors also suggests multi-part
    anchors = re.findall(r"(?:Section|Rule|Article)\s+\d+[A-Z]*", question, re.I)
    return len(set(anchors)) >= 3


_DECOMP_SYS = (
    "You break Indian legal research questions into 2 or 3 focused sub-questions. "
    "Each sub-question must be self-contained, cite the specific statute/section "
    "if mentioned, and be answerable by looking up one topic. "
    "Output ONLY a JSON array of strings. No prose, no keys, no wrapping."
)
_DECOMP_EXAMPLES = [
    {
        "q": "Compare remedies under Section 138 NI Act vs Section 420 IPC for the same dishonoured cheque",
        "a": [
            "What are the ingredients, punishment, and procedural requirements of Section 138 of the Negotiable Instruments Act for cheque dishonour?",
            "What are the ingredients and punishment under Section 420 of the Indian Penal Code for cheating?",
            "Can both Section 138 NI Act and Section 420 IPC be prosecuted simultaneously for the same cheque? Are they compoundable?"
        ],
    },
    {
        "q": "What is the GST rate on online money gaming after the 2023 amendment, and how is the taxable value determined under Rule 31B?",
        "a": [
            "What does the CGST Amendment Act 2023 provide about the GST treatment and rate for online money gaming?",
            "How is taxable value determined for online gaming under Rule 31B of the CGST Rules?"
        ],
    },
    {
        "q": "Explain right to privacy under Article 21 with reference to Puttaswamy judgment",
        "a": [
            "What does Article 21 of the Indian Constitution guarantee and how has the right to privacy been read into it?",
            "What was the ratio of Justice K.S. Puttaswamy v. Union of India on the right to privacy?"
        ],
    },
]

def _decompose(question: str) -> list:
    """Call 1.5B decomposer to split into sub-questions. Fallback = [question]."""
    if not _should_decompose(question):
        return [question]
    try:
        few_shot = []
        for ex in _DECOMP_EXAMPLES:
            few_shot.append({"role": "user", "content": ex["q"]})
            few_shot.append({"role": "assistant", "content": json.dumps(ex["a"])})
        body = json.dumps({
            "messages": [{"role": "system", "content": _DECOMP_SYS}] + few_shot +
                        [{"role": "user", "content": question}],
            "temperature": 0.0,
            "max_tokens": 400,
            "stream": False,
        }).encode()
        req = urllib.request.Request(DECOMPOSER_URL, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=DECOMPOSER_TIMEOUT) as r:
            data = json.loads(r.read())
        raw = data["choices"][0]["message"].get("content", "").strip()
        # strip common wrappers
        raw = raw.strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
        # extract first JSON array
        m = re.search(r"\[.*?\]", raw, re.S)
        if not m:
            print(f"[decompose] no JSON array in reply: {raw[:200]}", flush=True)
            return [question]
        subs = json.loads(m.group(0))
        subs = [s.strip() for s in subs if isinstance(s, str) and len(s.strip()) > 10]
        if not subs:
            return [question]
        # ALWAYS include the original question as first sub-query so we don't lose context
        subs = [question] + subs[:MAX_SUB_QUERIES - 1]
        print(f"[decompose] {len(subs)} sub-queries for: {question[:80]}", flush=True)
        for s in subs:
            print(f"  - {s[:100]}", flush=True)
        return subs
    except Exception as e:
        print(f"[decompose] failed, using original: {e}", flush=True)
        return [question]


def _search_merge(question: str, top_k: int = DEFAULT_TOP_K):
    """Decompose question, run _search on each sub-query in parallel,
       merge unique hits, and return top_k after re-ranking the merged pool."""
    sub_queries = _decompose(question)
    if len(sub_queries) == 1:
        return _search(question, top_k)

    # Parallel search. _search itself already does vec+kw+rerank for one query.
    # We grab bigger top_k per sub-query, then merge and trim.
    per_sub_k = max(top_k, 8)
    with ThreadPoolExecutor(max_workers=len(sub_queries)) as ex:
        results_lists = list(ex.map(lambda q: _search(q, per_sub_k), sub_queries))

    # Merge unique by (source,text[:120]); keep best score per key
    merged = {}
    for hits in results_lists:
        for h in hits:
            key = (h["source"], h["text"][:120])
            if key not in merged or h["score"] > merged[key]["score"]:
                merged[key] = h
    merged_list = list(merged.values())
    merged_list.sort(key=lambda h: h["score"], reverse=True)

    # Second-stage rerank on the merged list against ORIGINAL question (synthesizer intent)
    merged_list = _rerank(question, merged_list[:RERANK_POOL])
    adj_vals = [h.get("rerank_norm", h["score"]) for h in merged_list] or [0.0]
    merged_list.sort(key=lambda h: h.get("rerank_norm", h["score"]), reverse=True)

    # per-source diversity cap
    per_src = {}
    kept = []
    for h in merged_list:
        key = h["source"].split("[")[0].strip()[:40] or h["dataset"]
        if per_src.get(key, 0) >= MAX_CHUNKS_PER_SRC:
            continue
        per_src[key] = per_src.get(key, 0) + 1
        kept.append(h)
        if len(kept) >= top_k:
            break

    print(f"[search_merge] subs={len(sub_queries)} merged_uniq={len(merged_list)} kept={len(kept)}", flush=True)
    return kept

def _search(question: str, top_k: int = DEFAULT_TOP_K):
    vec = next(iter(embedder.embed([question])))
    overfetch = max(top_k * OVERFETCH_FACTOR, 30)
    results = client.query_points(
        collection_name=COLLECTION,
        query=vec.tolist(),
        limit=overfetch,
        with_payload=True,
    ).points

    topics = _detect_topics(question)
    prefer_ds = set()
    for tp in topics:
        prefer_ds.update(DATASET_BOOST.get(tp, []))

    # --- vector hits -------------------------------------------------------
    raw = []
    for r in results:
        sc = float(r.score)
        if sc < MIN_RAW_SCORE:
            continue
        pl = r.payload or {}
        tier = int(pl.get("tier", 3) or 3)
        ds = pl.get("dataset", "")
        adj = sc + TIER_BOOST.get(tier, 0.0)
        if ds in prefer_ds:
            adj += 0.10        # dataset-topic boost
        raw.append({
            "text":     pl.get("text", ""),
            "source":   pl.get("source", ""),
            "dataset":  ds,
            "domain":   pl.get("domain", ""),
            "tier":     tier,
            "score":    sc,
            "adjusted": adj,
            "_keyword": None,
        })

    # --- keyword-anchored hits --------------------------------------------
    kw = _extract_keywords(question)
    forced = _keyword_hits(kw, topics) if kw else []
    for f in forced:
        f["adjusted"] = f["score"] + TIER_BOOST.get(f["tier"], 0.0)
        raw.append(f)

    # dedupe by (source,text-prefix)
    seen = set()
    uniq = []
    for h in raw:
        key = (h["source"], h["text"][:120])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(h)

    # --- rerank top RERANK_POOL by adjusted, then send to BGE reranker ---
    uniq.sort(key=lambda h: h["adjusted"], reverse=True)
    pool = uniq[:RERANK_POOL]
    pool = _rerank(question, pool)
    # Final score: 0.7 * rerank_norm + 0.3 * adjusted-normalized
    adj_vals = [h["adjusted"] for h in pool] or [0.0]
    a_lo, a_hi = min(adj_vals), max(adj_vals)
    a_rng = max(a_hi - a_lo, 1e-6)
    for h in pool:
        adj_n = (h["adjusted"] - a_lo) / a_rng
        rr_n  = h.get("rerank_norm", adj_n)   # fallback if rerank failed
        h["final"] = 0.7 * rr_n + 0.3 * adj_n
    pool.sort(key=lambda h: h["final"], reverse=True)
    uniq = pool + [h for h in uniq if h not in pool]

    per_src = {}
    kept = []
    for h in uniq:
        key = h["source"].split("[")[0].strip()[:40] or h["dataset"]
        if per_src.get(key, 0) >= MAX_CHUNKS_PER_SRC:
            continue
        per_src[key] = per_src.get(key, 0) + 1
        kept.append(h)
        if len(kept) >= top_k:
            break

    # diagnostic log
    rr_ok = any("rerank_score" in h for h in pool)
    print(f"[search] topics={topics} kw={kw} vec={len(raw)-len(forced)} kw_hits={len(forced)} pool={len(pool)} kept={len(kept)} rerank_ok={rr_ok}", flush=True)

    return [
        {
            "text":    h["text"],
            "source":  h["source"],
            "dataset": h["dataset"],
            "domain":  h["domain"],
            "tier":    h["tier"],
            "score":   round(h["score"], 4),
            "via":     "keyword" if h.get("_keyword") else "vector",
        }
        for h in kept
    ]

def _llm(url: str, question: str, context: str) -> str:
    prompt = (
        f"LEGAL CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer in 2-5 short paragraphs with specific Act and Section citations. "
        "If the context only partially answers, state exactly what is covered "
        "and what is missing. Do NOT output inner monologue or 'Wait, I need to check' "
        "style reasoning. Do NOT invent section numbers."
    )
    body = json.dumps({
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": LLM_MAX_TOKENS,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    msg = data["choices"][0]["message"]
    return msg.get("content") or msg.get("reasoning_content", "")

@app.get("/", response_class=HTMLResponse)
def chat_ui():
    return HTMLResponse(content=CHAT_HTML)

@app.post("/ask")
async def ask(q: Question):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    hits = _search_merge(q.question, q.top_k or DEFAULT_TOP_K)
    if not hits:
        return {"answer": "No relevant legal sections found in the indexed corpus.",
                "sources": [], "context_chunks": 0, "confidence": "none", "top_score": 0.0}

    top_score = max(h["score"] for h in hits)
    low_conf = top_score < MIN_CONFIDENCE

    parts = []
    if low_conf:
        parts.append(
            "NOTE: Retrieved context has low semantic similarity to the question "
            f"(best match {top_score:.2f}). If the passages below do not clearly "
            "answer the question, state that the corpus does not contain a definitive answer."
        )
    parts.extend(
        f"[{h['source']} | tier {h['tier']} | score {h['score']:.2f}]\n{h['text']}"
        for h in hits
    )
    context = "\n\n---\n\n".join(parts)
    sources = [
        {"source": h["source"], "domain": h["domain"], "tier": h["tier"], "score": h["score"], "via": h.get("via","vector")}
        for h in hits
    ]

    llm_url = LLM_URL or _find_llm()
    if not llm_url:
        raise HTTPException(status_code=503, detail="No LLM available")

    try:
        answer = _llm(llm_url, q.question, context)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    return {
        "answer": answer,
        "sources": sources,
        "context_chunks": len(hits),
        "top_score": round(top_score, 4),
        "confidence": "low" if low_conf else "ok",
    }


@app.get("/health")
def health():
    llm_url = LLM_URL or _find_llm()
    try:
        info = client.get_collection(COLLECTION)
        doc_count = info.points_count
    except:
        doc_count = 0
    return {
        "status":    "ok",
        "documents": doc_count,
        "llm":       llm_url or "not found",
        "retrieval": "qdrant+fastembed",
        "embed_model": EMBED_MODEL
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000, log_level="warning")
