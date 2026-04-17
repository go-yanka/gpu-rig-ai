#!/usr/bin/env python3
"""
Indian Legal AI — OpenAI-Compatible API Wrapper
Port: 7001

Makes the Indian Legal RAG system look like an OpenAI model so it can be
added directly to Open WebUI as a custom model provider.

Add to Open WebUI:
  Settings → Connections → Add OpenAI Connection
  URL:   http://192.168.1.107:7001
  Key:   indian-legal
  Model: indian-legal-ai (appears in model list automatically)

Usage: chat normally in Open WebUI — it automatically does RAG retrieval
       and answers with citations from Indian law.
"""

import json, os, pickle, sys, urllib.request, time, uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List

WORK_DIR   = "/opt/indian-legal-ai"
INDEX_PATH = f"{WORK_DIR}/rag/tfidf_index.pkl"
GPU_PORTS  = [9080, 9081, 9082, 9083, 9084, 9086]
MODEL_ID   = "indian-legal-ai"

SYSTEM_LEGAL = """You are an expert Indian legal and tax advisor. You have deep knowledge of:
- Income Tax Act 1961 (all sections, deductions, TDS, capital gains)
- GST / CGST Act 2017 (rates, registration, ITC, returns, e-invoicing)
- Indian Penal Code 1860 (all major offences and punishments)
- Constitution of India (Fundamental Rights, DPSPs, Amendments)
- Companies Act 2013 (incorporation, compliance, directors)
- FEMA 1999 (foreign exchange, remittances)
- Negotiable Instruments Act 1881 (cheque bounce, dishonour)
- Labour Laws (PF, ESI, Gratuity, Shops & Establishments)

INSTRUCTIONS:
- Use ONLY the retrieved legal context below to answer
- ALWAYS cite the specific Act name and Section number
- Give practical, actionable answers
- If the context doesn't cover the question, say so clearly
- Never invent or guess section numbers"""

# ── Load TF-IDF index at startup ───────────────────────────────────────────
print("Loading TF-IDF index...", flush=True)
try:
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    with open(INDEX_PATH, "rb") as f:
        idx = pickle.load(f)
    vectorizer   = idx["vectorizer"]
    tfidf_matrix = idx["tfidf_matrix"]
    documents    = idx["documents"]
    print(f"Loaded {len(documents)} legal sections", flush=True)
except FileNotFoundError:
    print(f"WARN: TF-IDF index not found at {INDEX_PATH}", flush=True)
    print("Run poc_setup.py first to build the index.", flush=True)
    vectorizer = tfidf_matrix = documents = None
except Exception as e:
    print(f"FATAL: Could not load index: {e}", flush=True)
    vectorizer = tfidf_matrix = documents = None

# ── Helpers ────────────────────────────────────────────────────────────────
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
    if vectorizer is None or tfidf_matrix is None:
        return []
    q_vec  = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "text":   documents[i]["text"],
            "source": documents[i]["source"],
            "domain": documents[i]["domain"],
            "score":  round(float(scores[i]), 4)
        }
        for i in top_idx if scores[i] > 0.01
    ]

def _build_answer(question: str, conversation_history: list = None) -> str:
    """Full RAG pipeline: retrieve → augment → generate."""
    # 1. Retrieve relevant legal sections
    hits = _search(question, top_k=5)

    # 2. Build augmented system prompt
    if hits:
        law_context = "\n\n---\n\n".join([
            f"[{h['source']}]\n{h['text']}"
            for h in hits
        ])
        system_content = f"{SYSTEM_LEGAL}\n\nRELEVANT LAW SECTIONS:\n{law_context}"
    else:
        system_content = SYSTEM_LEGAL + "\n\n[No specific legal sections found for this query. Answer from general knowledge and advise consulting a lawyer.]"

    # 3. Build messages for LLM
    messages = [{"role": "system", "content": system_content}]

    # Include conversation history (last 4 turns for context)
    if conversation_history:
        for msg in conversation_history[-8:]:
            if msg.get("role") in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})
    else:
        messages.append({"role": "user", "content": question})

    # 4. Find and call LLM
    llm_url = _find_llm()
    if not llm_url:
        return "Error: No LLM available. Please ensure a model is loaded on the rig."

    body = json.dumps({
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 5000,
        "stream": False
    }).encode()

    req = urllib.request.Request(llm_url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())

    # Qwen3 thinking: actual answer in "content", thinking in "reasoning_content"
    msg = data["choices"][0]["message"]
    answer = msg.get("content") or msg.get("reasoning_content", "")

    # Append source citations at the end
    if hits:
        citations = "\n\n---\n📚 **Sources:**\n" + "\n".join([
            f"- {h['source']} (relevance: {h['score']:.2f})"
            for h in hits[:3]
        ])
        answer += citations

    return answer

# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(title="Indian Legal AI — OpenAI API", version="1.0")

# ── OpenAI-compatible endpoints ────────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
    """Return model list — Open WebUI reads this to show available models."""
    return {
        "object": "list",
        "data": [
            {
                "id":       MODEL_ID,
                "object":   "model",
                "created":  int(time.time()),
                "owned_by": "indian-legal-ai",
                "name":     "Indian Legal AI (RAG)",
                "description": "RAG-based Indian legal assistant. Answers questions about Income Tax, GST, IPC, Constitution, FEMA, Companies Act with accurate citations."
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint.
    Intercepts the user's question, runs RAG retrieval, passes enriched context to LLM.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # Extract the latest user question
    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")

    question = user_messages[-1].get("content", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    # History = all messages except the last user message
    history = messages[:-1] if len(messages) > 1 else []

    # Run RAG pipeline
    try:
        answer = _build_answer(question, history)
    except Exception as e:
        answer = f"I encountered an error processing your legal question: {str(e)}\n\nPlease try again or rephrase your question."

    completion_id = f"chatcmpl-legal-{uuid.uuid4().hex[:12]}"
    now = int(time.time())

    # Handle streaming (Open WebUI may request it)
    stream = body.get("stream", False)

    if stream:
        def event_stream():
            # Send the full answer as a single streamed chunk
            chunk = {
                "id":      completion_id,
                "object":  "chat.completion.chunk",
                "created": now,
                "model":   MODEL_ID,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": answer},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            # Send final chunk
            final = {
                "id":      completion_id,
                "object":  "chat.completion.chunk",
                "created": now,
                "model":   MODEL_ID,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming response
    return {
        "id":      completion_id,
        "object":  "chat.completion",
        "created": now,
        "model":   MODEL_ID,
        "choices": [{
            "index":         0,
            "message": {
                "role":    "assistant",
                "content": answer
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens":     len(question.split()),
            "completion_tokens": len(answer.split()),
            "total_tokens":      len(question.split()) + len(answer.split())
        }
    }

# ── Health + info ──────────────────────────────────────────────────────────

@app.get("/health")
def health():
    llm = _find_llm()
    docs = len(documents) if documents else 0
    return {
        "status":    "ok",
        "documents": docs,
        "llm":       llm or "not found",
        "retrieval": "tfidf",
        "model_id":  MODEL_ID
    }

@app.get("/")
def root():
    return {
        "service": "Indian Legal AI — OpenAI API",
        "usage":   "POST /v1/chat/completions",
        "models":  "GET /v1/models",
        "add_to_open_webui": {
            "url":   "http://192.168.1.107:7001",
            "key":   "indian-legal",
            "model": MODEL_ID
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7001, log_level="warning")
