#!/usr/bin/env python3
"""
Tiered variant of build_rag_worker.py.

Same behaviour as build_rag_worker, but the tier filter is a CLI argument
instead of a hard-coded {1, 2} set. This lets us index Tier 3 Q&A data
into the same Qdrant collection during the overnight build.

Invocation:
  python build_rag_worker_tiered.py \\
      --endpoint http://localhost:9090/v1/embeddings \\
      --id-base 300000000 \\
      --worker-id 5001 \\
      --files prarabdha_sft \\
      --tiers 1,2,3

Each worker uses a disjoint id-base so points never collide.
"""

import argparse, os, json, time, datetime, gc, urllib.request

CURATED_DIR = "/opt/indian-legal-ai/datasets/_curated"
COLLECTION  = "indian_legal_full"
QDRANT_URL  = "http://localhost:6333"
EMBED_DIM   = 384

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
MIN_CHUNK_LEN = 60
MAX_CHARS     = 1800
BATCH_SIZE    = 64
UPSERT_BATCH  = 128
EMBED_TIMEOUT = 30
EMBED_RETRIES = 3


def log(worker_id, msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}][W{worker_id}] {msg}", flush=True)


def embed_batch(endpoint, texts):
    body = json.dumps({"input": texts}).encode()
    req = urllib.request.Request(endpoint, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    last_err = None
    for attempt in range(EMBED_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT) as r:
                d = json.loads(r.read())
            out = [None] * len(texts)
            for item in d["data"]:
                out[item["index"]] = item["embedding"]
            return out
        except Exception as e:
            last_err = e
            time.sleep(1 + attempt)
    raise RuntimeError(f"embed failed: {last_err}")


def chunk_text(text):
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
            if len(chunk) > MAX_CHARS:
                chunk = chunk[:MAX_CHARS]
            yield chunk
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--id-base",  type=int, required=True)
    ap.add_argument("--files",    required=True, help="comma-separated dataset names (no .jsonl)")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--tiers", default="1,2,3",
                    help="comma-separated tier ids to include (default 1,2,3)")
    args = ap.parse_args()

    include_tiers = {int(t.strip()) for t in args.tiers.split(",") if t.strip()}

    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm
    client = QdrantClient(url=QDRANT_URL, timeout=60)

    wid = args.worker_id
    log(wid, f"endpoint={args.endpoint}  id-base={args.id_base:,}  files={args.files}  tiers={sorted(include_tiers)}")

    point_id = args.id_base
    total_chunks = 0
    total_rows   = 0
    t_start = time.time()

    for name in args.files.split(","):
        name = name.strip()
        filepath = os.path.join(CURATED_DIR, f"{name}.jsonl")
        if not os.path.exists(filepath):
            log(wid, f"  SKIP missing: {filepath}")
            continue

        log(wid, f"---- {name} ----")
        buffer = []
        rows_this = 0
        chunks_this = 0

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
                if tier not in include_tiers:
                    continue

                text   = row.get("text") or ""
                source = row.get("source") or name
                rows_this += 1

                for ch in chunk_text(text):
                    buffer.append({
                        "text":    ch,
                        "source":  source,
                        "dataset": name,
                        "tier":    tier,
                    })
                    if len(buffer) >= BATCH_SIZE:
                        vecs = embed_batch(args.endpoint, [b["text"] for b in buffer])
                        pts = [qm.PointStruct(id=point_id + i, vector=vecs[i], payload=buffer[i])
                               for i in range(len(buffer))]
                        for i in range(0, len(pts), UPSERT_BATCH):
                            sub = pts[i:i+UPSERT_BATCH]
                            for attempt in range(3):
                                try:
                                    client.upsert(collection_name=COLLECTION, points=sub)
                                    break
                                except Exception as e:
                                    log(wid, f"  upsert retry {attempt+1}: {e}")
                                    time.sleep(2)
                        point_id += len(buffer)
                        total_chunks += len(buffer)
                        chunks_this  += len(buffer)
                        buffer = []

        if buffer:
            vecs = embed_batch(args.endpoint, [b["text"] for b in buffer])
            pts = [qm.PointStruct(id=point_id + i, vector=vecs[i], payload=buffer[i])
                   for i in range(len(buffer))]
            for i in range(0, len(pts), UPSERT_BATCH):
                sub = pts[i:i+UPSERT_BATCH]
                for attempt in range(3):
                    try:
                        client.upsert(collection_name=COLLECTION, points=sub)
                        break
                    except Exception as e:
                        log(wid, f"  upsert retry {attempt+1}: {e}")
                        time.sleep(2)
            point_id += len(buffer)
            total_chunks += len(buffer)
            chunks_this  += len(buffer)

        total_rows += rows_this
        log(wid, f"  [DONE] {name}: {rows_this:,} rows -> {chunks_this:,} chunks")
        gc.collect()

    elapsed = time.time() - t_start
    log(wid, f"WORKER FINISHED  rows={total_rows:,}  chunks={total_chunks:,}  time={elapsed/60:.1f}min  rate={total_chunks/max(elapsed,1):.0f}/s")


if __name__ == "__main__":
    main()
