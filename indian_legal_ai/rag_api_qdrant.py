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
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

QDRANT_URL = "http://localhost:6333"
COLLECTION = "indian_legal_full"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
GPU_PORTS  = [9080, 9081, 9082, 9083, 9084, 9086]

SYSTEM = """You are an expert Indian legal and tax advisor with deep knowledge of:
- Income Tax Act 1961 and all amendments
- GST (CGST Act 2017, IGST, SGST)
- Indian Penal Code 1860
- Constitution of India
- Companies Act 2013
- FEMA 1999
- Negotiable Instruments Act 1881
- Labour Laws

Use ONLY the provided legal context to answer. Always cite the Act and Section number.
If the context is insufficient, say so clearly. Never invent section numbers."""

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
      body: JSON.stringify({question, top_k: 5})
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
    top_k: int = 5

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

def _search(question: str, top_k: int = 5):
    vec = next(iter(embedder.embed([question])))
    from qdrant_client.http import models as qm
    results = client.search(
        collection_name=COLLECTION,
        query_vector=vec.tolist(),
        limit=top_k,
        with_payload=True
    )
    return [
        {
            "text":   r.payload.get("text", ""),
            "source": r.payload.get("source", ""),
            "domain": r.payload.get("domain", ""),
            "score":  round(float(r.score), 4)
        }
        for r in results
    ]

def _llm(url: str, question: str, context: str) -> str:
    prompt = f"LEGAL CONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer with specific Act and Section citations:"
    body = json.dumps({
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 5000,
        "stream": False
    }).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=180) as r:
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

    hits = _search(q.question, q.top_k)
    if not hits:
        return {"answer": "No relevant legal sections found.", "sources": [], "context_chunks": 0}

    context = "\n\n---\n\n".join([f"[{h['source']}]\n{h['text']}" for h in hits])
    sources = [{"source": h["source"], "domain": h["domain"], "score": h["score"]} for h in hits]

    llm_url = LLM_URL or _find_llm()
    if not llm_url:
        raise HTTPException(status_code=503, detail="No LLM available")

    try:
        answer = _llm(llm_url, q.question, context)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    return {"answer": answer, "sources": sources, "context_chunks": len(hits)}

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
