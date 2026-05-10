#!/usr/bin/env python3
"""
Build Qdrant index from the CURATED corpus only.

Reads /opt/indian-legal-ai/datasets/_curated/*.jsonl
Each row already has: text, source, dataset, tier
Includes only Tiers 1 and 2 (statutory + case law) — Tier 3 Q&A is reserved
for future fine-tuning, not retrieval.

Streaming / memory-safe: chunks → batch embed → batch upsert, never holds
more than BATCH_SIZE chunks in memory.
"""

import os, json, sys, time, datetime, gc

WORK_DIR    = "/opt/indian-legal-ai"
CURATED_DIR = f"{WORK_DIR}/datasets/_curated"
COLLECTION  = "indian_legal_full"
QDRANT_URL  = "http://localhost:6333"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM   = 384

INCLUDE_TIERS = {1, 2}   # statutory + case law for RAG

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
MIN_CHUNK_LEN = 60
BATCH_SIZE    = 64
UPSERT_BATCH  = 128

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

try:
    from fastembed import TextEmbedding
except ImportError:
    log("Installing fastembed...")
    os.system(f"{sys.executable} -m pip install -q fastembed")
    from fastembed import TextEmbedding

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

log("=" * 64)
log(" Qdrant Index Builder — Curated Corpus (Tiers 1+2)")
log("=" * 64)

log("Loading fastembed BAAI/bge-small-en-v1.5...")
embedder = TextEmbedding(model_name=EMBED_MODEL)
log(f"Embedder ready ({EMBED_DIM}-dim)")

client = QdrantClient(url=QDRANT_URL, timeout=60)

# Recreate collection
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

# Create payload indexes for filtering by tier / dataset
for field in ("tier", "dataset"):
    try:
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name=field,
            field_schema=qm.PayloadSchemaType.KEYWORD if field == "dataset"
                        else qm.PayloadSchemaType.INTEGER,
        )
    except Exception as e:
        log(f"  (index on {field} skipped: {e})")

# ── Chunker ─────────────────────────────────────────────────────────────────
def chunk_text(text: str):
    text = (text or "").strip()
    if len(text) <= CHUNK_SIZE:
        if len(text) >= MIN_CHUNK_LEN:
            yield text
        return
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            for sep in [". ", "! ", "? ", "\n"]:
                cut = text.rfind(sep, start + CHUNK_SIZE // 2, end)
                if cut > start:
                    end = cut + len(sep)
                    break
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_LEN:
            yield chunk
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP

# ── Flush helper ────────────────────────────────────────────────────────────
def flush_batch(buffer, point_id_counter):
    if not buffer:
        return point_id_counter
    texts = [b["text"] for b in buffer]
    vectors = list(embedder.embed(texts))
    points = [
        qm.PointStruct(
            id=point_id_counter + i,
            vector=vec.tolist(),
            payload=buffer[i],
        )
        for i, vec in enumerate(vectors)
    ]
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

# ── Process each curated file ───────────────────────────────────────────────
files = sorted([
    os.path.join(CURATED_DIR, f)
    for f in os.listdir(CURATED_DIR)
    if f.endswith(".jsonl")
])
log(f"Found {len(files)} curated files")

point_id     = 0
total_chunks = 0
total_rows   = 0
skipped_tier = 0
t_start = time.time()

for filepath in files:
    dataset_name = os.path.basename(filepath).replace(".jsonl", "")
    buffer = []
    rows_this_file = 0
    chunks_this_file = 0

    # Peek first line to find tier
    with open(filepath, "r", encoding="utf-8") as f:
        first = f.readline().strip()
        if not first:
            continue
        try:
            first_tier = json.loads(first).get("tier", 99)
        except Exception:
            first_tier = 99
    if first_tier not in INCLUDE_TIERS:
        log(f"  [SKIP tier {first_tier}] {dataset_name}")
        skipped_tier += 1
        continue

    log("")
    log(f"━━ {dataset_name} (tier {first_tier}) ━━")

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            tier = row.get("tier", 99)
            if tier not in INCLUDE_TIERS:
                continue

            text   = row.get("text") or ""
            source = row.get("source") or dataset_name
            rows_this_file += 1

            for chunk in chunk_text(text):
                buffer.append({
                    "text":    chunk,
                    "source":  source,
                    "dataset": dataset_name,
                    "tier":    tier,
                })
                if len(buffer) >= BATCH_SIZE:
                    point_id = flush_batch(buffer, point_id)
                    total_chunks    += len(buffer)
                    chunks_this_file += len(buffer)
                    buffer = []
                    if total_chunks % 2000 == 0:
                        rate = total_chunks / (time.time() - t_start)
                        log(f"  {total_chunks:,} chunks indexed ({rate:.0f}/s)")

    if buffer:
        point_id = flush_batch(buffer, point_id)
        total_chunks    += len(buffer)
        chunks_this_file += len(buffer)

    total_rows += rows_this_file
    log(f"  [DONE] {dataset_name}: {rows_this_file:,} rows → {chunks_this_file:,} chunks")
    gc.collect()

elapsed = time.time() - t_start
log("")
log("=" * 64)
log(" BUILD COMPLETE")
log("=" * 64)
log(f"  Rows processed:   {total_rows:,}")
log(f"  Chunks indexed:   {total_chunks:,}")
log(f"  Files skipped:    {skipped_tier}  (tier not in {sorted(INCLUDE_TIERS)})")
log(f"  Collection:       {COLLECTION}")
log(f"  Time:             {elapsed/60:.1f} min")

info = client.get_collection(COLLECTION)
log(f"  Qdrant points:    {info.points_count:,}")
log("")
log("Next: switch port 7000 to rag_api_qdrant.py")
