#!/usr/bin/env python3
"""
Indian Legal AI — Qdrant RAG Index Builder
============================================
Reads all JSONL datasets from /opt/indian-legal-ai/datasets/,
chunks the text, embeds with fastembed (ONNX, no PyTorch),
and upserts into a Qdrant collection named "indian_legal_full".

Also writes a production Qdrant-backed RAG API to:
    /opt/indian-legal-ai/scripts/rag_api_qdrant.py

Usage:
    python3 /opt/indian-legal-ai/scripts/build_rag_index.py

Features:
- Auto-installs fastembed if missing
- Checkpoint file — resumes after interruption
- Atomic chunk counting so duplicates are never upserted
- Detailed timestamped logging every 1000 chunks
- Replaces the Qdrant collection cleanly at the start of each run
  (delete + recreate) so the index is always consistent

Requirements (all pip-installable, no PyTorch):
    pip install fastembed qdrant-client tqdm
"""

import json
import os
import sys
import subprocess
import datetime
import time
import traceback
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
WORK_DIR        = "/opt/indian-legal-ai"
DATASETS_DIR    = f"{WORK_DIR}/datasets"
RAG_DIR         = f"{WORK_DIR}/rag"
SCRIPTS_DIR     = f"{WORK_DIR}/scripts"
CHECKPOINT_FILE = f"{RAG_DIR}/build_checkpoint.json"
LOGS_DIR        = f"{WORK_DIR}/logs"

COLLECTION_NAME = "indian_legal_full"
EMBED_MODEL     = "BAAI/bge-small-en-v1.5"
EMBED_DIM       = 384

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

CHUNK_SIZE    = 500   # characters
CHUNK_OVERLAP = 50    # characters
MIN_CHUNK_LEN = 50    # skip chunks shorter than this

EMBED_BATCH_SIZE  = 64   # texts per fastembed call
UPSERT_BATCH_SIZE = 100  # points per Qdrant upsert call

LOG_EVERY = 1000  # log progress every N chunks


# ── Logging ───────────────────────────────────────────────────────────────────
def log(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def log_sep(title: str):
    log("")
    log("=" * 60)
    log(f"  {title}")
    log("=" * 60)


# ── Dependency bootstrap ──────────────────────────────────────────────────────
def ensure_fastembed():
    """Install fastembed if not already installed."""
    try:
        import fastembed  # noqa: F401
        log("fastembed already installed")
    except ImportError:
        log("Installing fastembed (ONNX-based, no PyTorch)...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "fastembed", "-q"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            log(f"pip install failed:\n{result.stderr}")
            sys.exit(1)
        log("fastembed installed successfully")

def ensure_qdrant_client():
    """Install qdrant-client if not already installed."""
    try:
        import qdrant_client  # noqa: F401
        log("qdrant-client already installed")
    except ImportError:
        log("Installing qdrant-client...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "qdrant-client", "-q"],
            check=True
        )
        log("qdrant-client installed")


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def load_checkpoint() -> dict:
    """Load progress checkpoint, or return empty state."""
    if os.path.isfile(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                cp = json.load(f)
            log(f"Checkpoint loaded: {cp.get('total_upserted', 0):,} chunks already done")
            return cp
        except Exception as e:
            log(f"Could not read checkpoint ({e}), starting fresh")
    return {
        "completed_files": [],   # list of JSONL paths fully processed
        "total_upserted": 0,
        "started_at": datetime.datetime.now().isoformat(),
    }

def save_checkpoint(cp: dict):
    """Persist checkpoint to disk."""
    os.makedirs(RAG_DIR, exist_ok=True)
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cp, f, indent=2)
    os.replace(tmp, CHECKPOINT_FILE)


# ── Text chunker ──────────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks of at most `size` characters.
    Tries to split on sentence boundaries ('. ') to preserve readability.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]

        # If not at end of string, try to split at the last sentence boundary
        if end < len(text):
            last_period = chunk.rfind(". ")
            if last_period > size // 2:  # only if the boundary isn't too early
                chunk = chunk[: last_period + 1]
                end = start + last_period + 1

        chunks.append(chunk.strip())
        start = end - overlap  # overlap with next chunk
        if start >= len(text):
            break

    return [c for c in chunks if len(c) >= MIN_CHUNK_LEN]


# ── Dataset loader ────────────────────────────────────────────────────────────
def iter_all_jsonl(datasets_dir: str):
    """
    Walk all subdirectories of datasets_dir, yield dicts from JSONL files.
    Each dict gets a 'dataset_name' key from its parent directory name.
    """
    base = Path(datasets_dir)
    jsonl_files = sorted(base.rglob("*.jsonl"))
    log(f"Found {len(jsonl_files)} JSONL files across all datasets")
    for path in jsonl_files:
        log(f"  {path.relative_to(base)}")
    return jsonl_files


def load_jsonl_rows(path: Path) -> list[dict]:
    """Load all rows from a JSONL file. Returns list of dicts."""
    rows = []
    dataset_name = path.parent.name
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    row.setdefault("dataset", dataset_name)
                    row.setdefault("source",  str(path))
                    row.setdefault("domain",  "Indian Law")
                    rows.append(row)
                except json.JSONDecodeError:
                    pass  # skip malformed lines silently
    except Exception as e:
        log(f"  ERROR reading {path}: {e}")
    return rows


def extract_text(row: dict) -> str:
    """
    Extract the best available text representation from a dataset row.
    Handles the varying schemas used by different legal datasets.
    """
    # If there's already a 'text' field, prefer it
    if row.get("text") and len(str(row["text"])) >= MIN_CHUNK_LEN:
        return str(row["text"])

    # Build text from Q&A fields
    parts = []
    for q_key in ("question", "instruction", "prompt", "input"):
        if row.get(q_key):
            parts.append(str(row[q_key]).strip())
    for a_key in ("answer", "output", "response", "completion"):
        if row.get(a_key):
            parts.append(str(row[a_key]).strip())

    if parts:
        return "\n".join(parts)

    # Last resort: join all string values
    return " ".join(str(v) for v in row.values() if isinstance(v, str))


# ── Main indexing pipeline ────────────────────────────────────────────────────
def main():
    log_sep("Indian Legal AI — Qdrant RAG Index Builder")
    log(f"Collection : {COLLECTION_NAME}")
    log(f"Embed model: {EMBED_MODEL} ({EMBED_DIM}d)")
    log(f"Qdrant     : {QDRANT_HOST}:{QDRANT_PORT}")
    log(f"Datasets   : {DATASETS_DIR}")

    os.makedirs(RAG_DIR,     exist_ok=True)
    os.makedirs(LOGS_DIR,    exist_ok=True)
    os.makedirs(SCRIPTS_DIR, exist_ok=True)

    # ── Step 1: Ensure dependencies ──────────────────────────────────────────
    log_sep("Step 1: Dependencies")
    ensure_fastembed()
    ensure_qdrant_client()

    # Import after installation
    from fastembed import TextEmbedding
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, OptimizersConfigDiff
    )

    # ── Step 2: Connect to Qdrant ────────────────────────────────────────────
    log_sep("Step 2: Qdrant connection")
    try:
        qclient = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
        info = qclient.get_collections()
        existing = [c.name for c in info.collections]
        log(f"Connected to Qdrant. Existing collections: {existing}")
    except Exception as e:
        log(f"FATAL: Cannot connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}: {e}")
        log("Is Docker running? Try: docker start qdrant  OR  docker run -d -p 6333:6333 qdrant/qdrant")
        sys.exit(1)

    # ── Step 3: Create / recreate collection ─────────────────────────────────
    log_sep("Step 3: Qdrant collection setup")
    if COLLECTION_NAME in existing:
        log(f"Deleting existing collection '{COLLECTION_NAME}' for a clean rebuild...")
        qclient.delete_collection(COLLECTION_NAME)
        time.sleep(1)

    qclient.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20_000  # build HNSW index after 20K vectors
        ),
    )
    log(f"Collection '{COLLECTION_NAME}' created (dim={EMBED_DIM}, metric=Cosine)")

    # ── Step 4: Load embedding model ─────────────────────────────────────────
    log_sep("Step 4: Embedding model")
    log(f"Loading {EMBED_MODEL} via fastembed (downloads ONNX model ~60MB first run)...")
    try:
        embedder = TextEmbedding(model_name=EMBED_MODEL)
        # Warm up
        _ = list(embedder.embed(["warmup"]))
        log("Embedding model ready")
    except Exception as e:
        log(f"FATAL: Could not load embedding model: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ── Step 5: Load checkpoint ───────────────────────────────────────────────
    log_sep("Step 5: Checkpoint and dataset scan")
    checkpoint = load_checkpoint()
    completed_files = set(checkpoint.get("completed_files", []))
    total_upserted  = checkpoint.get("total_upserted", 0)
    point_id        = total_upserted  # use sequential IDs; safe to resume

    # ── Step 6: Collect all JSONL files ──────────────────────────────────────
    jsonl_files = iter_all_jsonl(DATASETS_DIR)
    if not jsonl_files:
        log("ERROR: No JSONL files found. Run download_datasets.py and scrape_gst_faqs.py first.")
        sys.exit(1)

    # ── Step 7: Process each file ─────────────────────────────────────────────
    log_sep("Step 6: Embedding and indexing")

    chunk_buffer_texts    = []  # text strings for embedding
    chunk_buffer_payloads = []  # metadata dicts

    def flush_buffer():
        """Embed and upsert everything currently in the buffer."""
        nonlocal total_upserted, point_id
        if not chunk_buffer_texts:
            return

        # Embed in batches
        all_embeddings = []
        for batch_start in range(0, len(chunk_buffer_texts), EMBED_BATCH_SIZE):
            batch = chunk_buffer_texts[batch_start: batch_start + EMBED_BATCH_SIZE]
            try:
                vecs = list(embedder.embed(batch))
                all_embeddings.extend(vecs)
            except Exception as e:
                log(f"  Embedding error on batch starting at {batch_start}: {e}")
                # Fill with zeros so IDs remain consistent
                all_embeddings.extend([[0.0] * EMBED_DIM] * len(batch))

        # Upsert in batches
        for upsert_start in range(0, len(all_embeddings), UPSERT_BATCH_SIZE):
            batch_vecs     = all_embeddings   [upsert_start: upsert_start + UPSERT_BATCH_SIZE]
            batch_payloads = chunk_buffer_payloads[upsert_start: upsert_start + UPSERT_BATCH_SIZE]
            points = [
                PointStruct(
                    id=point_id + i,
                    vector=vec.tolist() if hasattr(vec, "tolist") else list(vec),
                    payload=payload,
                )
                for i, (vec, payload) in enumerate(zip(batch_vecs, batch_payloads))
            ]
            try:
                qclient.upsert(collection_name=COLLECTION_NAME, points=points)
                point_id      += len(points)
                total_upserted += len(points)
            except Exception as e:
                log(f"  Qdrant upsert error: {e} — will retry once")
                time.sleep(3)
                try:
                    qclient.upsert(collection_name=COLLECTION_NAME, points=points)
                    point_id      += len(points)
                    total_upserted += len(points)
                except Exception as e2:
                    log(f"  Retry failed: {e2} — skipping batch of {len(points)} points")

        chunk_buffer_texts.clear()
        chunk_buffer_payloads.clear()

    for jsonl_path in jsonl_files:
        file_key = str(jsonl_path)
        if file_key in completed_files:
            log(f"SKIP (already indexed): {jsonl_path.name}")
            continue

        log(f"\nProcessing: {jsonl_path}")
        rows = load_jsonl_rows(jsonl_path)
        log(f"  Loaded {len(rows):,} rows")

        chunks_from_file = 0
        for row in rows:
            raw_text = extract_text(row)
            chunks   = chunk_text(raw_text)

            for chunk in chunks:
                payload = {
                    "text":    chunk,
                    "source":  row.get("source", str(jsonl_path)),
                    "domain":  row.get("domain",  "Indian Law"),
                    "dataset": row.get("dataset", jsonl_path.parent.name),
                    # optional fields — include if present
                    **({"question": row["question"]} if row.get("question") else {}),
                }
                chunk_buffer_texts.append(chunk)
                chunk_buffer_payloads.append(payload)
                chunks_from_file += 1

                # Flush every UPSERT_BATCH_SIZE worth of embeddings
                if len(chunk_buffer_texts) >= EMBED_BATCH_SIZE * 4:
                    flush_buffer()

                    if total_upserted % LOG_EVERY < EMBED_BATCH_SIZE * 4:
                        log(f"  Progress: {total_upserted:,} total chunks upserted")
                        save_checkpoint({
                            "completed_files": list(completed_files),
                            "total_upserted":  total_upserted,
                            "last_updated":    datetime.datetime.now().isoformat(),
                        })

        # Flush remaining buffer for this file
        flush_buffer()

        completed_files.add(file_key)
        save_checkpoint({
            "completed_files": list(completed_files),
            "total_upserted":  total_upserted,
            "last_updated":    datetime.datetime.now().isoformat(),
        })
        log(f"  File complete: {chunks_from_file:,} chunks from {jsonl_path.name}")

    # Final flush
    flush_buffer()

    # ── Step 8: Final stats ───────────────────────────────────────────────────
    log_sep("Step 7: Index statistics")
    try:
        coll_info = qclient.get_collection(COLLECTION_NAME)
        count = coll_info.points_count
        log(f"Collection '{COLLECTION_NAME}': {count:,} vectors indexed")
    except Exception as e:
        log(f"Could not fetch collection info: {e}")
    log(f"Total upserted this run: {total_upserted:,}")

    # ── Step 9: Write Qdrant RAG API ──────────────────────────────────────────
    log_sep("Step 8: Writing Qdrant RAG API")
    write_qdrant_rag_api()

    log_sep("DONE")
    log(f"Qdrant collection '{COLLECTION_NAME}' is ready.")
    log(f"To start the new RAG API:")
    log(f"  pkill -f rag_api.py")
    log(f"  python3 {SCRIPTS_DIR}/rag_api_qdrant.py &")
    log(f"  # Or: uvicorn rag_api_qdrant:app --host 0.0.0.0 --port 7000")


# ── Qdrant RAG API writer ─────────────────────────────────────────────────────
def write_qdrant_rag_api():
    """Write the production Qdrant-backed RAG API to disk."""
    api_path = os.path.join(SCRIPTS_DIR, "rag_api_qdrant.py")
    log(f"Writing Qdrant RAG API to: {api_path}")

    code = '''#!/usr/bin/env python3
"""
Indian Legal AI — Qdrant RAG API (Production)
===============================================
Port   : 7000
Endpoints:
  POST /ask    {"question": "...", "top_k": 5}
               → {"answer": "...", "sources": [...], "context_chunks": N}
  GET  /health → {"status": "ok", "vectors": N, "llm": "http://..."}
  GET  /        → service info
  GET  /docs    → Swagger UI (via FastAPI)

Retrieval: fastembed (BAAI/bge-small-en-v1.5) → Qdrant cosine search
Generation: Qwen3 / any OpenAI-compatible llama-server on ports 9080-9086
            Handles Qwen3 thinking mode (content + reasoning_content split)

Requirements:
    pip install fastembed qdrant-client fastapi uvicorn

Usage:
    python3 /opt/indian-legal-ai/scripts/rag_api_qdrant.py
"""

import json
import os
import sys
import subprocess
import urllib.request
import urllib.error
import datetime
from typing import Optional

# ── Auto-install fastembed if missing ─────────────────────────────────────────
try:
    from fastembed import TextEmbedding
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "fastembed", "-q"], check=True)
    from fastembed import TextEmbedding

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "qdrant-client", "-q"], check=True)
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────
COLLECTION_NAME = "indian_legal_full"
EMBED_MODEL     = "BAAI/bge-small-en-v1.5"
QDRANT_HOST     = "localhost"
QDRANT_PORT     = 6333
GPU_PORTS       = [9080, 9081, 9082, 9083, 9084, 9086]
MAX_TOKENS      = 5000   # Qwen3 thinking mode needs room

SYSTEM_PROMPT = """You are an expert Indian legal and tax advisor with deep knowledge of:
- Income Tax Act 1961 (deductions, slabs, TDS, capital gains, advance tax)
- GST / CGST Act 2017 (registration, ITC, returns, rates, composition, penalties)
- Indian Penal Code 1860 and criminal procedure
- Constitution of India (fundamental rights, DPSPs)
- Companies Act 2013
- FEMA 1999
- Negotiable Instruments Act 1881
- Labour laws (PF, ESI, Maternity Benefit, Minimum Wages)

Instructions:
- Use ONLY the legal context provided below. Do not invent facts.
- Cite the specific Act and Section number in your answer.
- If the context does not address the question, state clearly that a specialist
  should be consulted for that specific point.
- Be precise, structured, and cite sources."""


# ── Logging ───────────────────────────────────────────────────────────────────
def _log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Startup: load model and connect ──────────────────────────────────────────
_log(f"Loading embedding model {EMBED_MODEL}...")
try:
    _embedder = TextEmbedding(model_name=EMBED_MODEL)
    _ = list(_embedder.embed(["warmup"]))
    _log("Embedding model ready")
except Exception as e:
    _log(f"FATAL: Could not load embedding model: {e}")
    sys.exit(1)

_log(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
try:
    _qclient = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10)
    _coll    = _qclient.get_collection(COLLECTION_NAME)
    _log(f"Qdrant OK — {_coll.points_count:,} vectors in '{COLLECTION_NAME}'")
except Exception as e:
    _log(f"FATAL: Cannot connect to Qdrant: {e}")
    sys.exit(1)

# Probe for a live LLM at startup
_LLM_URL: Optional[str] = None
for _port in GPU_PORTS:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{_port}/health", timeout=3) as _r:
            _data = json.loads(_r.read())
            if _data.get("status") in ("ok", "no slot available", "loading model"):
                _LLM_URL = f"http://127.0.0.1:{_port}/v1/chat/completions"
                _log(f"LLM found on port {_port}")
                break
    except Exception:
        continue

if not _LLM_URL:
    _log("WARN: No LLM found at startup — will retry on each request")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Indian Legal AI (Qdrant)",
    description=(
        "Production RAG-based Indian legal assistant. "
        "Retrieval: fastembed + Qdrant vector search. "
        "Generation: Qwen3 / LLaMA on local GPU. "
        "POST /ask with {question: str} to get a cited legal answer."
    ),
    version="2.0-qdrant",
)

class Question(BaseModel):
    question: str
    top_k: int = 5
    domain_filter: Optional[str] = None   # e.g. "GST", "Income Tax"


# ── Helper: find a live LLM ───────────────────────────────────────────────────
def _find_llm() -> Optional[str]:
    for port in GPU_PORTS:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=3
            ) as r:
                data = json.loads(r.read())
                if data.get("status") in ("ok", "no slot available", "loading model"):
                    return f"http://127.0.0.1:{port}/v1/chat/completions"
        except Exception:
            continue
    return None


# ── Helper: embed a query ─────────────────────────────────────────────────────
def _embed_query(text: str) -> list[float]:
    vecs = list(_embedder.embed([text]))
    vec = vecs[0]
    return vec.tolist() if hasattr(vec, "tolist") else list(vec)


# ── Helper: Qdrant vector search ──────────────────────────────────────────────
def _search(question: str, top_k: int = 5, domain: Optional[str] = None) -> list[dict]:
    query_vec = _embed_query(question)

    search_kwargs = dict(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=top_k,
        with_payload=True,
    )

    # Optional domain filter (e.g. only GST chunks)
    if domain:
        from qdrant_client.models import FieldCondition, MatchValue, Filter
        search_kwargs["query_filter"] = Filter(
            must=[FieldCondition(key="domain", match=MatchValue(value=domain))]
        )

    results = _qclient.search(**search_kwargs)

    hits = []
    for r in results:
        payload = r.payload or {}
        hits.append({
            "text":    payload.get("text", ""),
            "source":  payload.get("source", "unknown"),
            "domain":  payload.get("domain", "Indian Law"),
            "dataset": payload.get("dataset", ""),
            "score":   round(float(r.score), 4),
        })
    return hits


# ── Helper: call LLM ──────────────────────────────────────────────────────────
def _call_llm(llm_url: str, question: str, context: str) -> str:
    prompt = (
        f"LEGAL CONTEXT:\\n{context}\\n\\n"
        f"QUESTION: {question}\\n\\n"
        f"Provide a detailed answer with specific Act and Section citations:"
    )
    body = json.dumps({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }).encode()

    req = urllib.request.Request(llm_url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")

    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())

    msg = data["choices"][0]["message"]

    # Qwen3 thinking mode: real answer is in "content",
    # chain-of-thought is in "reasoning_content" (separate field).
    # Use "content" if non-empty, else fall back to "reasoning_content".
    answer = msg.get("content", "").strip()
    if not answer:
        answer = msg.get("reasoning_content", "").strip()
    return answer


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/ask")
async def ask(q: Question):
    question = q.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question cannot be empty")

    # 1. Vector search
    try:
        hits = _search(question, top_k=q.top_k, domain=q.domain_filter)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant search error: {e}")

    if not hits:
        return {
            "answer": "No relevant legal content found for this query.",
            "sources": [],
            "context_chunks": 0,
        }

    # 2. Build context string (top-5 chunks, separator-delimited)
    context = "\\n\\n---\\n\\n".join(
        f"[{h[\'source\']}]\\n{h[\'text\']}"
        for h in hits
    )
    sources = [
        {
            "source":  h["source"],
            "domain":  h["domain"],
            "dataset": h["dataset"],
            "score":   h["score"],
        }
        for h in hits
    ]

    # 3. Find LLM
    llm_url = _LLM_URL or _find_llm()
    if not llm_url:
        raise HTTPException(
            status_code=503,
            detail="No LLM available — load a model via llama-server on port 9080-9086"
        )

    # 4. Generate answer
    try:
        answer = _call_llm(llm_url, question, context)
    except urllib.error.URLError as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    return {
        "answer":         answer,
        "sources":        sources,
        "context_chunks": len(hits),
    }


@app.get("/health")
def health():
    llm_url = _LLM_URL or _find_llm()
    try:
        coll = _qclient.get_collection(COLLECTION_NAME)
        vec_count = coll.points_count
    except Exception:
        vec_count = -1

    return {
        "status":     "ok",
        "collection": COLLECTION_NAME,
        "vectors":    vec_count,
        "llm":        llm_url or "not found",
        "retrieval":  "qdrant+fastembed",
        "embed_model": EMBED_MODEL,
    }


@app.get("/")
def root():
    return {
        "service":    "Indian Legal AI (Production — Qdrant)",
        "version":    "2.0",
        "usage":      "POST /ask with {question: str, top_k: int, domain_filter: str|null}",
        "docs":       "/docs",
        "coverage":   "Income Tax, GST, IPC, Constitution, FEMA, Companies Act, NI Act",
        "retrieval":  f"Qdrant '{COLLECTION_NAME}' + {EMBED_MODEL}",
    }


if __name__ == "__main__":
    import uvicorn
    _log("Starting Qdrant RAG API on port 7000...")
    uvicorn.run(app, host="0.0.0.0", port=7000, log_level="warning")
'''

    with open(api_path, "w", encoding="utf-8") as f:
        f.write(code)
    os.chmod(api_path, 0o755)
    log(f"Qdrant RAG API written: {api_path}")
    log("")
    log("To start it:")
    log(f"  pkill -f rag_api.py 2>/dev/null; python3 {api_path} &")
    log(f"  curl http://localhost:7000/health")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
