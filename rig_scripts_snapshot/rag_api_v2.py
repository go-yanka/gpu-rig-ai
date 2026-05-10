#!/usr/bin/env python3
"""Indian Legal AI V2 API — hybrid retrieval + grounded answer + strict verifier.

Port: 8090
Pipeline: dense(BGE 384d) + sparse(BM25) -> RRF -> rerank(bge-reranker-v2-m3)
         -> Qwen3-14B answer with forced-citation prompt -> strict verifier.

Endpoints:
  GET  /          -> minimal chat UI
  POST /ask       -> {q, top_k?, status?} -> {answer, sources, verified, refused}
  POST /retrieve  -> {q, top_k?, status?} -> retrieval-only, no LLM
  GET  /health    -> service health
"""
import os, sys, re, json, time, hashlib, urllib.request, logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List

# ---- Config ----
QDRANT = "http://localhost:6333"
COLL = "indian_legal_t1_v2"
EMBED_URL = "http://localhost:9092/v1/embeddings"
RERANK_URL = "http://localhost:9096/v1/rerank"
LLM_URL = "http://localhost:9086/v1/chat/completions"
LLM_MODEL = "meta-llama-3.1-8b-instruct.Q4_K_M.gguf"
IDF_PATH = "/opt/indian-legal-ai/rag/bm25_idf_v2.json"
LEGAL_PREFIX = "Indian statutory law query: "

TOP_K_DEFAULT = 5
RRF_LIMIT = 25
RERANK_POOL = 20

# ---- Tokenizer / BM25 (matches ingest_t1_v2.py) ----
STOPWORDS = set("""a an the of in on at by for to from with and or but not is are was were be been
being have has had do does did can could would should shall will may might must this that these
those it its as if then than there here which who whom whose what when where why how""".split())
TOK_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{1,}")

def tokenize(text):
    return [t for t in TOK_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]

def token_to_idx(tok):
    h = hashlib.md5(tok.encode()).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF

# ---- IDF load ----
try:
    with open(IDF_PATH) as f:
        IDF = json.load(f)["idf"]
except Exception as e:
    logging.warning(f"IDF load failed: {e}; BM25 disabled")
    IDF = {}

# ---- HTTP helper ----
def http_json(url, body=None, method="POST", timeout=120):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode() if body else None, method=method
    )
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def embed(text):
    return http_json(EMBED_URL, {"input": [text], "model": "bge"})["data"][0]["embedding"]

def bm25_sparse(text):
    toks = tokenize(text)
    tf = Counter(toks)
    idx, val = [], []
    for t, f in tf.items():
        if t not in IDF:
            continue
        w = IDF[t] * f
        idx.append(token_to_idx(t))
        val.append(float(w))
    return {"indices": idx, "values": val}

def rrf_fuse(runs, k=60):
    scores, meta = {}, {}
    for run in runs:
        for rank, p in enumerate(run):
            pid = p["id"]
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
            meta[pid] = p
    ordered = sorted(scores.items(), key=lambda x: -x[1])
    out = []
    for pid, s in ordered:
        m = dict(meta[pid])
        m["rrf"] = s
        out.append(m)
    return out

def build_filter(status=None, act=None, year_min=None, dataset=None):
    must = []
    if status:
        must.append({"key": "status", "match": {"value": status}})
    if act:
        must.append({"key": "act_name", "match": {"value": act}})
    if dataset:
        must.append({"key": "dataset", "match": {"value": dataset}})
    if year_min:
        must.append({"key": "act_year", "range": {"gte": int(year_min)}})
    return {"must": must} if must else None

# Act auto-boost: map query mentions -> act_name substrings expected in payload.
# If a candidate's payload.act_name contains the substring (case-insensitive),
# the candidate gets ACT_MATCH_BOOST added to its rerank_score.
ACT_ALIASES = [
    (re.compile(r"\b(indian\s+trusts?\s+act|trusts?\s+act,?\s*1882)\b", re.I), "indian trusts 1882"),
    (re.compile(r"\b(income[- ]?tax\s+act,?\s*2025|it\s+act\s*2025)\b", re.I), "income tax act, 2025"),
    (re.compile(r"\b(income[- ]?tax\s+act,?\s*1961|it\s+act\s*1961)\b", re.I), "income tax act, 1961"),
    (re.compile(r"\b(BNS|bharatiya\s+nyaya\s+sanhita)\b", re.I), "bharatiya nyaya sanhita"),
    (re.compile(r"\b(BNSS|bharatiya\s+nagarik\s+suraksha)\b", re.I), "bharatiya nagarik suraksha"),
    (re.compile(r"\b(BSA|bharatiya\s+sakshya)\b", re.I), "bharatiya sakshya"),
    (re.compile(r"\b(IPC|indian\s+penal\s+code)\b", re.I), "indian penal code"),
    (re.compile(r"\b(CrPC|code\s+of\s+criminal\s+procedure)\b", re.I), "criminal procedure"),
    (re.compile(r"\b(evidence\s+act,?\s*1872)\b", re.I), "evidence act"),
    (re.compile(r"\b(CGST|central\s+goods\s+and\s+services\s+tax)\b", re.I), "cgst"),
    (re.compile(r"\b(IGST|integrated\s+goods\s+and\s+services\s+tax)\b", re.I), "igst"),
    (re.compile(r"\bcompanies\s+act,?\s*2013\b", re.I), "companies act, 2013"),
    (re.compile(r"\bcontract\s+act,?\s*1872\b", re.I), "contract act"),
    (re.compile(r"\bRTI\s+act\b|right\s+to\s+information", re.I), "right to information"),
    (re.compile(r"\bconsumer\s+protection\s+act", re.I), "consumer protection"),
    (re.compile(r"\barbitration\s+and\s+conciliation", re.I), "arbitration"),
    (re.compile(r"\bFEMA\b|foreign\s+exchange\s+management", re.I), "fema"),
    (re.compile(r"\bNI\s+act\b|negotiable\s+instruments", re.I), "negotiable instruments"),
    (re.compile(r"\bIBC\b|insolvency\s+and\s+bankruptcy", re.I), "insolvency and bankruptcy"),
    (re.compile(r"\bconstitution\s+of\s+india\b", re.I), "constitution"),
    (re.compile(r"\bdpdp|digital\s+personal\s+data\s+protection", re.I), "digital personal data"),
    (re.compile(r"\bcode\s+on\s+wages\b", re.I), "wages"),
]
ACT_MATCH_BOOST = 1.5  # added to rerank_score when query's named act matches candidate act_name

def detect_query_acts(query):
    """Return list of act_name substrings the query refers to explicitly."""
    hits = []
    for rx, key in ACT_ALIASES:
        if rx.search(query):
            hits.append(key)
    return hits

def retrieve(query, top_k=TOP_K_DEFAULT, status=None, act=None, year_min=None, dataset=None):
    flt = build_filter(status, act, year_min, dataset)
    query_acts = detect_query_acts(query)

    def dense_call():
        body = {"query": embed(query), "using": "dense", "limit": RRF_LIMIT, "with_payload": True}
        if flt: body["filter"] = flt
        return http_json(f"{QDRANT}/collections/{COLL}/points/query", body)["result"]["points"]

    def sparse_call():
        sp = bm25_sparse(query)
        if not sp["indices"]:
            return []
        body = {"query": sp, "using": "bm25", "limit": RRF_LIMIT, "with_payload": True}
        if flt: body["filter"] = flt
        return http_json(f"{QDRANT}/collections/{COLL}/points/query", body)["result"]["points"]

    with ThreadPoolExecutor(max_workers=2) as ex:
        fd = ex.submit(dense_call)
        fs = ex.submit(sparse_call)
        dense_res = fd.result()
        sparse_res = fs.result()

    fused = rrf_fuse([dense_res, sparse_res])[:RERANK_POOL]
    if not fused:
        return []
    # rerank with legal prefix + contextual header so act/section identity is scored
    def _rerank_doc(c):
        p = c.get("payload", {})
        head = f"[{p.get('act_name','?')}"
        if p.get("chapter_no"): head += f" Ch.{p['chapter_no']}"
        if p.get("section_no"): head += f" Sec.{p['section_no']}"
        head += f" | {(p.get('status','?') or '?').upper()}]"
        return head + "\n" + (p.get("text", "") or "")[:1400]
    docs = [_rerank_doc(c) for c in fused]
    r = http_json(RERANK_URL, {"model": "bge-reranker", "query": LEGAL_PREFIX + query, "documents": docs})
    scored = []
    for item in r["results"]:
        c = dict(fused[item["index"]])
        base = item["relevance_score"]
        # act auto-boost: if query named an act and this chunk's act_name matches
        boost = 0.0
        if query_acts:
            an = (c.get("payload", {}).get("act_name") or "").lower()
            for k in query_acts:
                if k in an:
                    boost = ACT_MATCH_BOOST
                    break
        c["rerank_score"] = base + boost
        c["rerank_raw"] = base
        c["act_boost"] = boost
        scored.append(c)
    scored.sort(key=lambda x: -x["rerank_score"])
    return scored[:top_k]

# ---- Citation verifier ----
CITE_RE = re.compile(
    r"(?:Section|Sec\.?|§|Article|Art\.?|Rule)\s*(\d+[A-Z]*)",
    re.I,
)

def build_citation_index(chunks):
    """Set of section_no strings appearing in retrieved context."""
    idx = set()
    for c in chunks:
        p = c.get("payload", {})
        sec = (p.get("section_no") or "").upper().strip()
        if sec:
            idx.add(sec)
        # also pull section numbers from the text itself
        for m in CITE_RE.findall(p.get("text", "")):
            idx.add(m.upper())
    return idx

def verify_citations_strict(answer, cite_idx):
    """Strip sentences whose claimed citation is not in the retrieval context."""
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    kept, stripped = [], []
    for s in sentences:
        m = CITE_RE.findall(s)
        if not m:
            kept.append(s)
            continue
        ok = any(sec.upper() in cite_idx for sec in m)
        if ok:
            kept.append(s)
        else:
            stripped.append(s)
    return " ".join(kept), stripped

# ---- LLM ----
SYSTEM_PROMPT = """You are an Indian statutory-law research assistant. Answer ONLY from the provided context chunks.

Rules:
1. Cite every factual claim inline as [Act-name Section-X] using the act_name and section_no from the context.
2. If the context does not contain an answer, say "I do not have a statutory source for this in my corpus." and stop.
3. Distinguish CURRENT law from LEGACY law when both appear. Prefer CURRENT.
4. Do NOT invent section numbers or act names. Only use what appears in the context.
5. Be concise (<= 6 sentences)."""

def build_user_prompt(q, chunks):
    lines = []
    for i, c in enumerate(chunks):
        p = c.get("payload", {})
        head = f"[{i+1}] {p.get('act_name','?')}"
        if p.get("chapter_no"): head += f" Ch.{p['chapter_no']}"
        if p.get("section_no"): head += f" Section {p['section_no']}"
        head += f" [{(p.get('status','?') or '?').upper()}]"
        lines.append(head)
        lines.append(p.get("text", "")[:1200])
        lines.append("")
    ctx = "\n".join(lines)
    return f"Question: {q}\n\nContext:\n{ctx}\n\nAnswer:"

def call_llm(q, chunks, max_tokens=700, temperature=0.1):
    body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(q, chunks)},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = http_json(LLM_URL, body, timeout=180)
    return r["choices"][0]["message"]["content"].strip()

# ---- FastAPI ----
app = FastAPI(title="Indian Legal AI V2", version="2.0")

class AskReq(BaseModel):
    q: str
    top_k: Optional[int] = TOP_K_DEFAULT
    status: Optional[str] = None
    act: Optional[str] = None
    dataset: Optional[str] = None
    year_min: Optional[int] = None
    with_answer: Optional[bool] = True


# ---- Category labels (dataset -> human name) ----
CATEGORY_LABELS = {
    "t1_constitution": "Constitution",
    "t1_criminal_codes_2023": "Criminal Law — New Codes (BNS/BNSS/BSA)",
    "t1_old_criminal": "Criminal Law — Old (IPC/CrPC/Evidence)",
    "t1_bridges": "IPC↔BNS Concordance",
    "t1_criminal_special": "Criminal — Special Acts",
    "t1_civil_procedure": "Civil Procedure (CPC)",
    "t1_income_tax": "Income Tax",
    "t1_gst_circulars": "GST (Acts, Rules, Circulars)",
    "t1_customs_excise": "Customs & Excise",
    "t1_finance_acts": "Finance Acts",
    "t1_banking": "Banking",
    "t1_companies_sebi": "Companies & SEBI",
    "t1_ibc": "Insolvency & Bankruptcy",
    "t1_fema_rbi": "FEMA & RBI",
    "t1_fema_notifications": "FEMA Notifications",
    "t1_commercial_acts": "Commercial Acts",
    "t1_labour_codes": "Labour — New Codes (2019–2020)",
    "t1_pre_codified_labour": "Labour — Pre-Codification",
    "t1_workplace": "Workplace (POSH etc.)",
    "t1_personal_law": "Personal Law",
    "t1_ip_acts": "Intellectual Property",
    "t1_adr": "ADR & Mediation",
    "t1_environment": "Environment",
    "t1_data_privacy": "Data Privacy (DPDP/IT)",
    "t1_real_estate": "Real Estate (RERA)",
    "t1_disaster_mgmt": "Disaster Management",
    "t1_telecom": "Telecom",
    "t1_interpretation": "Interpretation & General Clauses",
    "t1_other_bare_acts": "Other Bare Acts",
}

@app.get("/health")
def health():
    out = {"ok": True, "collection": COLL, "idf_terms": len(IDF)}
    try:
        c = http_json(f"{QDRANT}/collections/{COLL}/points/count", {"exact": True})["result"]["count"]
        out["points"] = c
    except Exception as e:
        out["qdrant_error"] = str(e)
    return out

@app.post("/retrieve")
def ep_retrieve(req: AskReq):
    hits = retrieve(req.q, top_k=req.top_k, status=req.status, act=req.act,
                    year_min=req.year_min, dataset=req.dataset)
    return {"q": req.q, "hits": [
        {"id": h["id"], "rerank": h.get("rerank_score"), "rrf": h.get("rrf"),
         "payload": h.get("payload", {})} for h in hits
    ]}


@app.get("/meta")
def ep_meta():
    """Categories (with human labels + counts) and top act_names for UI dropdowns."""
    try:
        ds = http_json(f"{QDRANT}/collections/{COLL}/facet",
                       {"key": "dataset", "limit": 50, "exact": True})["result"]["hits"]
        acts = http_json(f"{QDRANT}/collections/{COLL}/facet",
                         {"key": "act_name", "limit": 200, "exact": True})["result"]["hits"]
    except Exception as e:
        return {"error": str(e), "categories": [], "acts": []}
    cats = []
    for h in ds:
        v = h["value"]
        cats.append({
            "value": v,
            "label": CATEGORY_LABELS.get(v, v.replace("t1_", "").replace("_", " ").title()),
            "count": h["count"],
        })
    cats.sort(key=lambda c: -c["count"])
    acts_out = [{"value": h["value"], "count": h["count"]} for h in acts]
    return {"categories": cats, "acts": acts_out}


@app.get("/lookup")
def ep_lookup(act: str, section: Optional[str] = None, chapter: Optional[str] = None,
              status: Optional[str] = None):
    """Direct section lookup: filter by exact act_name + optional section/chapter/status."""
    must = [{"key": "act_name", "match": {"value": act}}]
    if section: must.append({"key": "section_no", "match": {"value": section}})
    if chapter: must.append({"key": "chapter_no", "match": {"value": chapter}})
    if status: must.append({"key": "status", "match": {"value": status}})
    body = {"filter": {"must": must}, "limit": 50, "with_payload": True}
    try:
        r = http_json(f"{QDRANT}/collections/{COLL}/points/scroll", body)["result"]
    except Exception as e:
        raise HTTPException(502, f"qdrant error: {e}")
    return {"act": act, "section": section, "chapter": chapter, "status": status,
            "hits": [{"id": p["id"], "payload": p["payload"]} for p in r.get("points", [])]}


@app.post("/ask")
def ep_ask(req: AskReq):
    t0 = time.time()
    hits = retrieve(req.q, top_k=req.top_k, status=req.status, act=req.act,
                    year_min=req.year_min, dataset=req.dataset)
    if not hits:
        return {"q": req.q, "answer": "I do not have a statutory source for this in my corpus.",
                "sources": [], "refused": True, "verified": True, "stripped": [],
                "elapsed_ms": int((time.time()-t0)*1000)}
    if not req.with_answer:
        return {"q": req.q, "hits": hits}
    try:
        raw = call_llm(req.q, hits)
    except Exception as e:
        raise HTTPException(502, f"LLM error: {e}")
    cite_idx = build_citation_index(hits)
    final, stripped = verify_citations_strict(raw, cite_idx)
    sources = []
    for h in hits:
        p = h.get("payload", {})
        sources.append({
            "act": p.get("act_name"),
            "chapter": p.get("chapter_no"),
            "section": p.get("section_no"),
            "status": p.get("status"),
            "dataset": p.get("dataset"),
            "file": p.get("file"),
            "text": p.get("text", "")[:500],
            "rerank": h.get("rerank_score"),
        })
    return {
        "q": req.q,
        "answer": final.strip() or "I do not have a verifiable statutory source for this in my corpus.",
        "raw_answer": raw,
        "sources": sources,
        "verified": True,
        "stripped": stripped,
        "refused": False,
        "elapsed_ms": int((time.time()-t0)*1000),
    }

UI_HTML = r"""<!doctype html><html><head><meta charset='utf-8'>
<title>Indian Legal AI</title>
<style>
 *{box-sizing:border-box}
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0e1014;color:#e6e6e6;margin:0;padding:0;font-size:14px;line-height:1.5}
 .wrap{max-width:1280px;margin:0 auto;padding:20px}
 header{display:flex;align-items:baseline;justify-content:space-between;border-bottom:1px solid #22252d;padding-bottom:14px;margin-bottom:18px}
 header h1{font-size:20px;margin:0;color:#9cdcfe;font-weight:600}
 header .sub{font-size:12px;color:#7b8490;margin-left:10px}
 .tabs{display:flex;gap:4px;margin-bottom:18px;border-bottom:1px solid #22252d}
 .tab{padding:10px 18px;cursor:pointer;color:#9aa4b2;border-bottom:2px solid transparent;user-select:none;font-size:13px}
 .tab.on{color:#9cdcfe;border-bottom-color:#1f6feb}
 .panel{display:none}
 .panel.on{display:block}

 /* Ask panel */
 .ask-main{display:grid;grid-template-columns:1fr 320px;gap:14px;margin-bottom:12px}
 textarea#q{width:100%;background:#181a20;color:#e6e6e6;border:1px solid #2a2d35;border-radius:8px;padding:14px;font-family:inherit;font-size:15px;min-height:90px;resize:vertical}
 textarea#q:focus{outline:none;border-color:#1f6feb}
 .filters{background:#12141a;border:1px solid #22252d;border-radius:8px;padding:14px}
 .filters label{display:block;font-size:11px;color:#7b8490;text-transform:uppercase;letter-spacing:.5px;margin:0 0 4px 0}
 .filters .grp{margin-bottom:12px}
 .filters select,.filters input{width:100%;background:#181a20;color:#e6e6e6;border:1px solid #2a2d35;border-radius:6px;padding:8px 10px;font-size:13px;font-family:inherit}
 .filters select:focus,.filters input:focus{outline:none;border-color:#1f6feb}
 .status-toggle{display:flex;gap:0;border:1px solid #2a2d35;border-radius:6px;overflow:hidden}
 .status-toggle button{flex:1;background:#181a20;color:#9aa4b2;border:none;padding:7px 0;font-size:12px;cursor:pointer;border-right:1px solid #2a2d35}
 .status-toggle button:last-child{border-right:none}
 .status-toggle button.on{background:#1f6feb;color:#fff}
 .chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}
 .chip{background:#181a20;border:1px solid #2a2d35;color:#9cdcfe;border-radius:14px;padding:5px 12px;font-size:12px;cursor:pointer;user-select:none}
 .chip:hover{background:#1f2430}
 .bar{display:flex;gap:10px;align-items:center;margin-bottom:18px}
 .bar .go{background:#1f6feb;color:#fff;border:none;border-radius:6px;padding:10px 22px;font-size:14px;font-weight:500;cursor:pointer}
 .bar .go:hover{background:#2d7af0}
 .bar .go:disabled{opacity:.5;cursor:not-allowed}
 .bar .hint{color:#7b8490;font-size:12px}
 .bar .devlink{margin-left:auto;color:#7b8490;font-size:12px;cursor:pointer;text-decoration:underline}
 .bar .devlink:hover{color:#9cdcfe}

 .results{display:grid;grid-template-columns:1.1fr 1fr;gap:14px}
 .card{background:#12141a;border:1px solid #22252d;border-radius:8px;padding:16px;max-height:75vh;overflow:auto}
 .card h2{font-size:12px;margin:0 0 12px;color:#7ec699;letter-spacing:.8px;text-transform:uppercase;font-weight:600}
 .answer{font-size:14px;line-height:1.65;white-space:pre-wrap}
 .answer .cite{color:#9cdcfe;background:#1a2740;padding:1px 5px;border-radius:3px;font-size:12px;cursor:pointer}
 .answer .cite:hover{background:#223b5f}
 .src{border-bottom:1px solid #1d2027;padding:12px 0}
 .src:last-child{border-bottom:none}
 .src h3{margin:0 0 4px;font-size:13px;color:#dcdcaa;font-weight:500}
 .src h3 .num{color:#7b8490;margin-right:6px;font-weight:400}
 .src .sec{color:#e6e6e6;font-weight:600}
 .badge{display:inline-block;padding:1px 7px;border-radius:3px;font-size:10px;margin-left:6px;font-weight:600;letter-spacing:.3px}
 .cur{background:#1b4332;color:#95d5b2}
 .leg{background:#4a1a1a;color:#f5a97f}
 .src .meta{font-size:11px;color:#7b8490;margin:4px 0 6px}
 .src .txt{font-size:12.5px;color:#c8c8c8;white-space:pre-wrap;line-height:1.55}
 .stripped{color:#f5a97f;font-size:11px;margin-top:12px;border-top:1px solid #2a2d35;padding-top:10px}
 .runmeta{color:#7b8490;font-size:11px;margin-bottom:10px}
 .empty{color:#7b8490;font-style:italic}
 .dev{display:none;background:#0a0c10;border:1px solid #22252d;border-radius:6px;padding:10px;font-size:11px;color:#7b8490;margin-top:10px;font-family:ui-monospace,Menlo,Consolas,monospace}
 .dev.on{display:block}

 /* Lookup panel */
 .lookup-form{display:grid;grid-template-columns:1fr 140px 140px 140px auto;gap:10px;margin-bottom:18px;align-items:end}
 .lookup-form label{display:block;font-size:11px;color:#7b8490;text-transform:uppercase;margin-bottom:4px;letter-spacing:.5px}
 .lookup-form select,.lookup-form input{width:100%;background:#181a20;color:#e6e6e6;border:1px solid #2a2d35;border-radius:6px;padding:8px 10px;font-size:13px}

 /* Browse panel */
 .browse-form{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:18px}
 .browse-form select{width:100%;background:#181a20;color:#e6e6e6;border:1px solid #2a2d35;border-radius:6px;padding:8px 10px;font-size:13px}
 .browse-results{font-size:13px}
 .browse-item{border-bottom:1px solid #1d2027;padding:10px 0;cursor:pointer}
 .browse-item:hover{background:#14161c}
 .browse-item .hd{color:#dcdcaa;font-weight:500}
 .browse-item .tx{color:#bbb;font-size:12px;margin-top:3px;line-height:1.5}
</style></head><body>
<div class='wrap'>

<header>
  <div><h1>Indian Legal AI</h1><span class='sub'>Tier-1 statutes · hybrid retrieval · grounded answers · strict citation verifier</span></div>
  <div id='healthBadge' class='sub'>·</div>
</header>

<div class='tabs'>
  <div class='tab on' data-tab='ask'>Ask a question</div>
  <div class='tab' data-tab='lookup'>Look up section</div>
  <div class='tab' data-tab='browse'>Browse acts</div>
</div>

<!-- ========== ASK ========== -->
<div class='panel on' id='panel-ask'>

  <div class='chips' id='chips'>
    <span class='chip' data-tpl='What is the definition of {} under Indian law?'>Definition of…</span>
    <span class='chip' data-tpl='What is the punishment for {} under the current criminal code?'>Punishment for…</span>
    <span class='chip' data-tpl='What is the procedure for {} under Indian law?'>Procedure for…</span>
    <span class='chip' data-tpl='Compare the old and new provisions for {}.'>Old vs new for…</span>
    <span class='chip' data-tpl='What amendments have been made to {}?'>Amendments to…</span>
    <span class='chip' data-tpl='Which acts cross-reference {}?'>Cross-references for…</span>
  </div>

  <div class='ask-main'>
    <textarea id='q' placeholder='Ask a statutory question in plain English. e.g. "What is the punishment for theft under the current Indian criminal code?"'></textarea>
    <div class='filters'>
      <div class='grp'>
        <label>Category</label>
        <select id='fCat'><option value=''>Any category</option></select>
      </div>
      <div class='grp'>
        <label>Status</label>
        <div class='status-toggle' id='fStatus'>
          <button data-v='' class='on'>Any</button>
          <button data-v='current'>Current</button>
          <button data-v='legacy'>Legacy</button>
        </div>
      </div>
      <div class='grp'>
        <label>Specific act (optional)</label>
        <input id='fAct' list='actList' placeholder='e.g. bharatiya nyaya sanhita'>
        <datalist id='actList'></datalist>
      </div>
    </div>
  </div>

  <div class='bar'>
    <button class='go' id='goAsk'>Ask</button>
    <span class='hint'>Ctrl+Enter to submit</span>
    <span class='devlink' id='toggleDev'>Show retrieval details</span>
  </div>

  <div class='runmeta' id='runmeta'></div>

  <div class='results'>
    <div class='card'>
      <h2>Answer</h2>
      <div id='ans' class='empty'>Your answer will appear here, with inline citations to the source sections.</div>
      <div class='dev' id='devPanel'></div>
    </div>
    <div class='card'>
      <h2>Sources</h2>
      <div id='src' class='empty'>Retrieved sections will appear here.</div>
    </div>
  </div>
</div>

<!-- ========== LOOKUP ========== -->
<div class='panel' id='panel-lookup'>
  <div class='lookup-form'>
    <div>
      <label>Act name</label>
      <input id='luAct' list='actList' placeholder='e.g. bharatiya nyaya sanhita'>
    </div>
    <div>
      <label>Section #</label>
      <input id='luSec' placeholder='e.g. 303'>
    </div>
    <div>
      <label>Chapter (optional)</label>
      <input id='luCh' placeholder='e.g. XVII'>
    </div>
    <div>
      <label>Status</label>
      <select id='luStatus'><option value=''>Any</option><option value='current'>Current</option><option value='legacy'>Legacy</option></select>
    </div>
    <div>
      <button class='go' id='goLookup'>Look up</button>
    </div>
  </div>
  <div class='card'>
    <h2>Matching chunks</h2>
    <div id='luResults' class='empty'>Pick an act and (optionally) a section to see exact chunks.</div>
  </div>
</div>

<!-- ========== BROWSE ========== -->
<div class='panel' id='panel-browse'>
  <div class='browse-form'>
    <select id='brCat'><option value=''>Any category</option></select>
    <select id='brAct'><option value=''>Any act (pick a category first for best results)</option></select>
    <select id='brStatus'><option value=''>Any status</option><option value='current'>Current</option><option value='legacy'>Legacy</option></select>
  </div>
  <div class='card'>
    <h2>Sections in this act</h2>
    <div id='brResults' class='browse-results empty'>Pick an act to browse its sections.</div>
  </div>
</div>

</div>
<script>
const $ = id=>document.getElementById(id);
const esc = s=>(s==null?'':String(s)).replace(/</g,'&lt;').replace(/>/g,'&gt;');

let META = {categories:[], acts:[]};
let STATUS = '';
let CITE_INDEX = new Set();

async function loadMeta(){
  try{
    const r = await fetch('/meta'); const d = await r.json();
    META = d;
    const cat = $('fCat'), brCat = $('brCat');
    for(const c of d.categories){
      const opt = `<option value='${esc(c.value)}'>${esc(c.label)} (${c.count})</option>`;
      cat.insertAdjacentHTML('beforeend', opt);
      brCat.insertAdjacentHTML('beforeend', opt);
    }
    const dl = $('actList');
    for(const a of d.acts){
      dl.insertAdjacentHTML('beforeend', `<option value='${esc(a.value)}'>${esc(a.value)} (${a.count})</option>`);
    }
  }catch(e){console.warn('meta load failed',e);}
  try{
    const h = await (await fetch('/health')).json();
    $('healthBadge').textContent = `${h.points||'?'} chunks · ${h.idf_terms||'?'} BM25 terms`;
  }catch(e){}
}

// tab switching
document.querySelectorAll('.tab').forEach(t=>{
  t.onclick=()=>{
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('on'));
    document.querySelectorAll('.panel').forEach(x=>x.classList.remove('on'));
    t.classList.add('on');
    $('panel-'+t.dataset.tab).classList.add('on');
  };
});

// status toggle
$('fStatus').querySelectorAll('button').forEach(b=>{
  b.onclick=()=>{
    $('fStatus').querySelectorAll('button').forEach(x=>x.classList.remove('on'));
    b.classList.add('on');
    STATUS = b.dataset.v;
  };
});

// chips
$('chips').querySelectorAll('.chip').forEach(c=>{
  c.onclick=()=>{
    const tpl = c.dataset.tpl;
    const ta = $('q');
    const cur = ta.value.trim();
    if(!cur) ta.value = tpl.replace('{}','');
    else ta.value = tpl.replace('{}', cur);
    ta.focus();
    // put cursor where the {} was
    const pos = tpl.indexOf('{}');
    if(pos>=0 && !cur){ ta.setSelectionRange(pos, pos); }
  };
});

// dev toggle
$('toggleDev').onclick=()=>{
  const dp = $('devPanel'); dp.classList.toggle('on');
  $('toggleDev').textContent = dp.classList.contains('on')?'Hide retrieval details':'Show retrieval details';
};

function bstatus(s){s=(s||'').toLowerCase();return s=='current'?'<span class="badge cur">CURRENT</span>':(s=='legacy'?'<span class="badge leg">LEGACY</span>':'');}

function renderSources(sources){
  if(!sources||!sources.length) return "<div class='empty'>No sources retrieved.</div>";
  return sources.map((s,i)=>{
    const catLabel = (META.categories.find(c=>c.value===s.dataset)||{}).label || s.dataset || '';
    return `<div class='src' id='src-${i+1}'>
      <h3><span class='num'>[${i+1}]</span> ${esc(s.act||'?')}
        ${s.chapter?('Ch. '+esc(s.chapter)):''}
        ${s.section?'<span class="sec">§ '+esc(s.section)+'</span>':''}
        ${bstatus(s.status)}</h3>
      <div class='meta'>${esc(catLabel)}${s.rerank?(' · rerank '+(+s.rerank).toFixed(3)):''}</div>
      <div class='txt'>${esc(s.text||'')}</div>
    </div>`;
  }).join('');
}

function linkifyCitations(answer, sources){
  // highlight Section/§/Article/Rule N tokens; click scrolls to matching source
  const secSet = new Set();
  sources.forEach((s,i)=>{ if(s.section) secSet.add(String(s.section).toUpperCase()); });
  return esc(answer).replace(/(Section|Sec\.?|§|Article|Art\.?|Rule)\s*(\d+[A-Z]*)/gi, (m,a,b)=>{
    const key = b.toUpperCase();
    const idx = sources.findIndex(s=>String(s.section||'').toUpperCase()===key);
    if(idx>=0) return `<span class='cite' onclick="document.getElementById('src-${idx+1}').scrollIntoView({behavior:'smooth',block:'center'})">${m}</span>`;
    return m;
  });
}

async function ask(){
  const q = $('q').value.trim(); if(!q) return;
  const btn = $('goAsk'); btn.disabled=true; btn.textContent='Thinking…';
  $('ans').innerHTML = '<span class="empty">Retrieving + ranking + answering…</span>';
  $('src').innerHTML = '<span class="empty">…</span>';
  const body = {q, with_answer:true};
  if(STATUS) body.status = STATUS;
  if($('fCat').value) body.dataset = $('fCat').value;
  if($('fAct').value.trim()) body.act = $('fAct').value.trim();
  try{
    const t0 = Date.now();
    const res = await fetch('/ask',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});
    const d = await res.json();
    const dt = Date.now()-t0;
    $('runmeta').textContent = `elapsed ${dt}ms · server ${d.elapsed_ms||'-'}ms · ${(d.sources||[]).length} sources · ${d.refused?'refused (no corpus match)':'verified'}${d.stripped&&d.stripped.length?(' · '+d.stripped.length+' unverified sentence(s) stripped'):''}`;
    $('src').innerHTML = renderSources(d.sources||[]);
    $('ans').innerHTML = linkifyCitations(d.answer||'', d.sources||[]) +
      ((d.stripped&&d.stripped.length)?`<div class='stripped'><b>Stripped (unverified citations):</b><br>${d.stripped.map(esc).join('<br>')}</div>`:'');
    $('devPanel').innerHTML = `<b>Request:</b> ${esc(JSON.stringify(body))}<br><br><b>Raw LLM:</b><br>${esc(d.raw_answer||'')}`;
  }catch(e){
    $('ans').innerHTML = '<span style="color:#f5a97f">Error: '+esc(e.message)+'</span>';
  }
  btn.disabled=false; btn.textContent='Ask';
}

$('goAsk').onclick = ask;
$('q').addEventListener('keydown', e=>{ if(e.ctrlKey && e.key==='Enter') ask(); });

// ---- Lookup ----
async function doLookup(){
  const act = $('luAct').value.trim(); if(!act){ alert('Pick an act'); return; }
  const params = new URLSearchParams({act});
  if($('luSec').value.trim()) params.set('section', $('luSec').value.trim());
  if($('luCh').value.trim()) params.set('chapter', $('luCh').value.trim());
  if($('luStatus').value) params.set('status', $('luStatus').value);
  $('luResults').innerHTML = '<span class="empty">Looking up…</span>';
  try{
    const r = await fetch('/lookup?'+params.toString());
    const d = await r.json();
    if(!d.hits || !d.hits.length){ $('luResults').innerHTML = '<span class="empty">No matches.</span>'; return; }
    $('luResults').innerHTML = d.hits.map((h,i)=>{
      const p = h.payload||{};
      return `<div class='src'>
        <h3><span class='num'>[${i+1}]</span> ${esc(p.act_name||'?')}
          ${p.chapter_no?('Ch. '+esc(p.chapter_no)):''}
          ${p.section_no?'<span class="sec">§ '+esc(p.section_no)+'</span>':''}
          ${bstatus(p.status)}</h3>
        <div class='meta'>${esc(p.dataset||'')} · ${esc(p.file||'')}</div>
        <div class='txt'>${esc(p.text||'')}</div></div>`;
    }).join('');
  }catch(e){ $('luResults').innerHTML = 'Error: '+esc(e.message); }
}
$('goLookup').onclick = doLookup;

// ---- Browse ----
async function browseAct(){
  const act = $('brAct').value;
  if(!act){ $('brResults').innerHTML = '<span class="empty">Pick an act.</span>'; return; }
  const params = new URLSearchParams({act});
  if($('brStatus').value) params.set('status', $('brStatus').value);
  $('brResults').innerHTML = '<span class="empty">Loading…</span>';
  try{
    const r = await fetch('/lookup?'+params.toString());
    const d = await r.json();
    if(!d.hits || !d.hits.length){ $('brResults').innerHTML = '<span class="empty">No chunks.</span>'; return; }
    // sort by section_no if present
    d.hits.sort((a,b)=>{
      const sa = String(a.payload?.section_no||''), sb = String(b.payload?.section_no||'');
      const na = parseInt(sa)||0, nb = parseInt(sb)||0;
      return na-nb || sa.localeCompare(sb);
    });
    $('brResults').innerHTML = d.hits.map(h=>{
      const p = h.payload||{};
      return `<div class='browse-item'>
        <div class='hd'>${p.chapter_no?('Ch. '+esc(p.chapter_no)+' '):''}${p.section_no?'§ '+esc(p.section_no):'(no section)'} ${bstatus(p.status)}</div>
        <div class='tx'>${esc((p.text||'').slice(0,280))}${(p.text||'').length>280?'…':''}</div>
      </div>`;
    }).join('');
  }catch(e){ $('brResults').innerHTML = 'Error: '+esc(e.message); }
}
$('brAct').onchange = browseAct;
$('brStatus').onchange = browseAct;
$('brCat').onchange = ()=>{
  // filter act dropdown to those most likely in this category — simple heuristic: all acts remain; user may refine
  // (a future improvement would fetch act_name facets filtered by dataset)
  const cat = $('brCat').value;
  const sel = $('brAct');
  sel.innerHTML = "<option value=''>Any act</option>";
  for(const a of META.acts){
    sel.insertAdjacentHTML('beforeend', `<option value='${esc(a.value)}'>${esc(a.value)} (${a.count})</option>`);
  }
};

loadMeta();
</script></body></html>"""

@app.get("/", response_class=HTMLResponse)
def ui():
    return UI_HTML

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090, log_level="info")
