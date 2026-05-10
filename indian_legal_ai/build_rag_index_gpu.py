#!/usr/bin/env python3
"""
Build Qdrant index from CURATED corpus using GPU embeddings.

Uses a local llama-server instance (BGE-small-en-v1.5 GGUF) running with
--embeddings on an idle GPU. ~3x faster than CPU fastembed.

Prereqs:
  1. BGE server on http://localhost:9090  (started separately on GPU)
  2. Qdrant on http://localhost:6333
  3. Curated JSONL in /opt/indian-legal-ai/datasets/_curated/

Indexes only Tiers 1+2 (statutory + case law). Tier 3 held for fine-tuning.
"""

import os, json, sys, time, datetime, gc, urllib.request

WORK_DIR    = "/opt/indian-legal-ai"
CURATED_DIR = f"{WORK_DIR}/datasets/_curated"
COLLECTION  = "indian_legal_full"
QDRANT_URL  = "http://localhost:6333"
EMBED_URL   = "http://localhost:9090/v1/embeddings"
EMBED_DIM   = 384

INCLUDE_TIERS = {1, 2}

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
MIN_CHUNK_LEN = 60
BATCH_SIZE    = 64       # must be <= --parallel on BGE server
UPSERT_BATCH  = 128
MAX_CHARS     = 1800     # ~450 tokens — stays under BGE's 512 per-slot limit
EMBED_TIMEOUT = 30
EMBED_RETRIES = 3

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Verify BGE server ───────────────────────────────────────────────────────
try:
    with urllib.request.urlopen("http://localhost:9090/health", timeout=5) as r:
        if b"ok" not in r.read():
            raise RuntimeError("BGE server not healthy")
except Exception as e:
    log(f"FATAL: BGE server on :9090 not reachable: {e}")
    log("Start it first:")
    log("  GGML_VK_VISIBLE_DEVICES=1 llama-server-rocm --model bge-small-en-v1.5-q8_0.gguf \\")
    log("    --host 0.0.0.0 --port 9090 --embeddings -c 32768 --parallel 64 -ngl 99")
    sys.exit(1)

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

log("=" * 64)
log(" Qdrant Index Builder — Curated Corpus on GPU (via llama-server BGE)")
log("=" * 64)

client = QdrantClient(url=QDRANT_URL, timeout=60)

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

for field, schema in [("dataset", qm.PayloadSchemaType.KEYWORD),
                      ("tier",    qm.PayloadSchemaType.INTEGER)]:
    try:
        client.create_payload_index(collection_name=COLLECTION, field_name=field, field_schema=schema)
    except Exception as e:
        log(f"  (payload index on {field} skipped: {e})")

# ── GPU embed helper ────────────────────────────────────────────────────────
def embed_batch(texts):
    """Call llama-server /v1/embeddings. Returns list of vectors."""
    body = json.dumps({"input": texts}).encode()
    req = urllib.request.Request(EMBED_URL, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    last_err = None
    for attempt in range(EMBED_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT) as r:
                d = json.loads(r.read())
            # Reorder by index just in case
            out = [None] * len(texts)
            for item in d["data"]:
                out[item["index"]] = item["embedding"]
            return out
        except Exception as e:
            last_err = e
            time.sleep(1 + attempt)
    raise RuntimeError(f"embed failed after {EMBED_RETRIES} retries: {last_err}")

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
            # Safety clamp to stay below BGE 512-token per-slot limit
            if len(chunk) > MAX_CHARS:
                chunk = chunk[:MAX_CHARS]
            yield chunk
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP

# ── Flush ───────────────────────────────────────────────────────────────────
def flush_batch(buffer, point_id_counter):
    if not buffer:
        return point_id_counter
    texts   = [b["text"] for b in buffer]
    vectors = embed_batch(texts)
    points = [
        qm.PointStruct(id=point_id_counter + i, vector=vec, payload=buffer[i])
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

# ── Process curated files ───────────────────────────────────────────────────
files = sorted(os.path.join(CURATED_DIR, f)
               for f in os.listdir(CURATED_DIR) if f.endswith(".jsonl"))
log(f"Found {len(files)} curated files")

point_id     = 0
total_chunks = 0
total_rows   = 0
skipped_tier = 0
t_start = time.time()

for filepath in files:
    dataset_name = os.path.basename(filepath).replace(".jsonl", "")

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

    buffer = []
    rows_this_file = 0
    chunks_this_file = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("tier", 99) not in INCLUDE_TIERS:
                continue

            text   = row.get("text") or ""
            source = row.get("source") or dataset_name
            tier   = row.get("tier", first_tier)
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
                    if total_chunks % 5000 == 0:
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
log(f"  Throughput:       {total_chunks/max(elapsed,1):.0f} chunks/sec")

info = client.get_collection(COLLECTION)
log(f"  Qdrant points:    {info.points_count:,}")
