#!/usr/bin/env python3
"""
Streaming Qdrant index builder — memory-efficient.
Reads JSONL files line-by-line, chunks, embeds in small batches, upserts to Qdrant.
Never holds more than BATCH_SIZE chunks in memory.

Uses fastembed BAAI/bge-small-en-v1.5 (384-dim ONNX, CPU).
"""

import os, json, sys, time, datetime, glob, hashlib, gc

WORK_DIR    = "/opt/indian-legal-ai"
DATA_DIR    = f"{WORK_DIR}/datasets"
RAG_DIR     = f"{WORK_DIR}/rag"
COLLECTION  = "indian_legal_full"
QDRANT_URL  = "http://localhost:6333"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM   = 384

CHUNK_SIZE   = 500
CHUNK_OVERLAP = 50
BATCH_SIZE   = 64         # number of chunks to embed at once
UPSERT_BATCH = 128        # number of points to upsert at once
MIN_CHUNK_LEN = 60

os.makedirs(RAG_DIR, exist_ok=True)

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Install fastembed if needed ─────────────────────────────────────────────
try:
    from fastembed import TextEmbedding
except ImportError:
    log("Installing fastembed...")
    os.system(f"{sys.executable} -m pip install -q fastembed")
    from fastembed import TextEmbedding

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# ── Setup ───────────────────────────────────────────────────────────────────
log("=" * 56)
log(" Qdrant Index Builder — Streaming")
log("=" * 56)

log("Loading fastembed model (~60MB first time)...")
embedder = TextEmbedding(model_name=EMBED_MODEL)
log(f"Embedder ready: {EMBED_MODEL} ({EMBED_DIM}-dim)")

client = QdrantClient(url=QDRANT_URL, timeout=60)

# Recreate collection (drop if exists)
try:
    client.delete_collection(COLLECTION)
    log(f"Deleted old collection '{COLLECTION}'")
except Exception:
    pass

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=qm.VectorParams(size=EMBED_DIM, distance=qm.Distance.COSINE),
)
log(f"Created collection '{COLLECTION}'")

# ── Chunker ─────────────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    text = (text or "").strip()
    if len(text) <= size:
        if len(text) >= MIN_CHUNK_LEN:
            yield text
        return
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # Try to break at sentence boundary
        if end < len(text):
            for sep in [". ", "! ", "? ", "\n"]:
                cut = text.rfind(sep, start + size // 2, end)
                if cut > start:
                    end = cut + len(sep)
                    break
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_LEN:
            yield chunk
        if end >= len(text):
            break
        start = end - overlap

# ── Streaming upsert ────────────────────────────────────────────────────────
def flush_batch(buffer, point_id_counter):
    """Embed buffered chunks and upsert to Qdrant."""
    if not buffer:
        return point_id_counter
    texts = [b["text"] for b in buffer]
    vectors = list(embedder.embed(texts))
    points = [
        qm.PointStruct(
            id=point_id_counter + i,
            vector=vec.tolist(),
            payload=buffer[i]
        )
        for i, vec in enumerate(vectors)
    ]
    # Upsert in batches
    for i in range(0, len(points), UPSERT_BATCH):
        sub = points[i:i+UPSERT_BATCH]
        for attempt in range(3):
            try:
                client.upsert(collection_name=COLLECTION, points=sub)
                break
            except Exception as e:
                log(f"  upsert retry {attempt+1}: {e}")
                time.sleep(2)
    return point_id_counter + len(points)

# ── Process each JSONL file ─────────────────────────────────────────────────
files = []
for root, _, filenames in os.walk(DATA_DIR):
    for f in filenames:
        if f.endswith(".jsonl"):
            files.append(os.path.join(root, f))

log(f"Found {len(files)} JSONL files to process")
for f in files:
    log(f"  {f}")

point_id    = 0
total_chunks = 0
total_rows   = 0
t_start = time.time()

for filepath in files:
    dataset_name = os.path.basename(os.path.dirname(filepath))
    log("")
    log(f"━━ Processing {dataset_name} ━━")

    buffer = []
    rows_this_file = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = row.get("text") or ""
            source = row.get("source") or dataset_name
            domain = row.get("domain") or dataset_name

            rows_this_file += 1
            for chunk in chunk_text(text):
                buffer.append({
                    "text":    chunk,
                    "source":  source,
                    "domain":  domain,
                    "dataset": dataset_name
                })
                if len(buffer) >= BATCH_SIZE:
                    point_id = flush_batch(buffer, point_id)
                    total_chunks += len(buffer)
                    buffer = []
                    if total_chunks % 1000 == 0:
                        rate = total_chunks / (time.time() - t_start)
                        log(f"  {total_chunks:,} chunks indexed ({rate:.0f} chunks/sec)")

    # Flush remaining
    if buffer:
        point_id = flush_batch(buffer, point_id)
        total_chunks += len(buffer)

    total_rows += rows_this_file
    log(f"  [DONE] {dataset_name}: {rows_this_file:,} rows → index now at {total_chunks:,} chunks")
    gc.collect()

# ── Final stats ─────────────────────────────────────────────────────────────
elapsed = time.time() - t_start
log("")
log("=" * 56)
log(" BUILD COMPLETE")
log("=" * 56)
log(f"  Total rows processed: {total_rows:,}")
log(f"  Total chunks indexed: {total_chunks:,}")
log(f"  Collection:           {COLLECTION}")
log(f"  Time:                 {elapsed/60:.1f} minutes")

# Verify collection info
info = client.get_collection(COLLECTION)
log(f"  Qdrant points:        {info.points_count:,}")
log("")
log("Next: start rag_api_qdrant.py to serve this index")
