"""
Hard-negative miner for BGE-M3 fine-tuning.

For each training question:
  1. Embed with BGE-M3 (CPU).
  2. Search Qdrant top-20 by dense cosine.
  3. Hard negatives = top-20 MINUS the gold/positive chunk, up to 10.

Resume-safe: skips questions already present in the output file.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import httpx

QDRANT = "http://192.168.1.107:6343"
COLLECTION = "cbic_v1"
VECTOR_NAME = "dense"

PAIR_FILES = [
    ("pairs_2000_20260422", "D:/_gpu_rig_ai/eval/training_pairs/pairs_2000_20260422.jsonl"),
    ("pairs_opus_highcomplex", "D:/_gpu_rig_ai/eval/training_pairs/pairs_opus_highcomplex.jsonl"),
    ("pairs_claude_opus", "D:/_gpu_rig_ai/eval/training_pairs/pairs_claude_opus.jsonl"),
]

OUT_PATH = Path("D:/_gpu_rig_ai/training/hard_negatives.jsonl")
LOG_PATH = Path("D:/_gpu_rig_ai/training/mine_hard_negatives.log")

TOP_K = 20
MAX_HARD_NEGS = 10
BATCH_SIZE = 32


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def iter_records(path: str) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        log(f"WARN: missing file {path}")
        return
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                log(f"WARN: bad JSON {path}:{i} {e}")


def extract_questions(rec: dict) -> list[str]:
    out = []
    for q in rec.get("questions", []) or []:
        if not isinstance(q, dict):
            continue
        text = q.get("q") or q.get("question")
        if text and isinstance(text, str):
            out.append(text.strip())
    return out


def load_all_questions() -> list[dict]:
    """Returns list of {question, positive_chunk_id, source_file}."""
    seen = {}
    total_in = 0
    for src, path in PAIR_FILES:
        count = 0
        for rec in iter_records(path):
            cid = rec.get("chunk_id")
            if cid is None:
                continue
            for qt in extract_questions(rec):
                total_in += 1
                key = qt
                if key in seen:
                    continue
                seen[key] = {
                    "question": qt,
                    "positive_chunk_id": cid,
                    "source_file": src,
                }
                count += 1
        log(f"Loaded {count} unique questions from {src} ({path})")
    log(f"Total questions (pre-dedup): {total_in}, unique: {len(seen)}")
    return list(seen.values())


def load_done() -> set[str]:
    if not OUT_PATH.exists():
        return set()
    done = set()
    with OUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("question"):
                    done.add(rec["question"])
            except Exception:
                pass
    return done


def qdrant_search(client: httpx.Client, vec: list[float]) -> list[dict]:
    r = client.post(
        f"{QDRANT}/collections/{COLLECTION}/points/search",
        json={
            "vector": {"name": VECTOR_NAME, "vector": vec},
            "limit": TOP_K,
            "with_payload": True,
        },
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json()["result"]


def main() -> None:
    log("=== mine_hard_negatives start ===")
    log(f"Python {sys.version.split()[0]} PID {os.getpid()}")

    # Verify Qdrant + vector name
    with httpx.Client() as c:
        info = c.get(f"{QDRANT}/collections/{COLLECTION}", timeout=10.0).json()
        vectors = info["result"]["config"]["params"].get("vectors", {})
        if VECTOR_NAME not in vectors:
            log(f"FATAL: vector '{VECTOR_NAME}' not in {list(vectors.keys())}")
            sys.exit(1)
        log(f"Qdrant OK. dim={vectors[VECTOR_NAME]['size']} points={info['result']['points_count']}")

    questions = load_all_questions()
    done = load_done()
    log(f"Resuming: {len(done)} already mined, {len(questions) - len(done)} remaining")

    todo = [q for q in questions if q["question"] not in done]
    if not todo:
        log("Nothing to do.")
        return

    # Load model
    log("Loading BGE-M3 (CPU)...")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    log(f"Model loaded in {time.time()-t0:.1f}s")

    # Stats counters
    pos_in_top20 = 0
    total_mined = 0
    hard_neg_counts: list[int] = []

    out_f = OUT_PATH.open("a", encoding="utf-8")
    client = httpx.Client()

    t_start = time.time()
    n = len(todo)
    for batch_idx in range(0, n, BATCH_SIZE):
        batch = todo[batch_idx : batch_idx + BATCH_SIZE]
        texts = [b["question"] for b in batch]
        t_embed = time.time()
        vecs = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        embed_dt = time.time() - t_embed

        for item, vec in zip(batch, vecs):
            try:
                hits = qdrant_search(client, vec.tolist())
            except Exception as e:
                log(f"WARN: search failed for q='{item['question'][:60]}': {e}")
                continue

            pos_id = item["positive_chunk_id"]
            top_ids = [h["id"] for h in hits]
            if pos_id in top_ids:
                pos_in_top20 += 1

            hard_negs = []
            for rank, h in enumerate(hits, 1):
                if h["id"] == pos_id:
                    continue
                payload = h.get("payload", {}) or {}
                txt = payload.get("text", "") or ""
                hard_negs.append({
                    "chunk_id": h["id"],
                    "rank": rank,
                    "score": round(float(h["score"]), 6),
                    "section_ref": payload.get("section_ref", ""),
                    "parent_act": payload.get("parent_act", ""),
                    "title": payload.get("title", ""),
                    "text_snippet": txt[:200],
                })
                if len(hard_negs) >= MAX_HARD_NEGS:
                    break

            rec = {
                "question": item["question"],
                "positive_chunk_id": pos_id,
                "source_file": item["source_file"],
                "hard_negs": hard_negs,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            total_mined += 1
            hard_neg_counts.append(len(hard_negs))

        done_so_far = batch_idx + len(batch)
        if (batch_idx // BATCH_SIZE) % 5 == 0 or done_so_far >= n:
            elapsed = time.time() - t_start
            rate = done_so_far / elapsed if elapsed > 0 else 0
            eta = (n - done_so_far) / rate if rate > 0 else 0
            log(
                f"batch {batch_idx//BATCH_SIZE + 1}/{(n+BATCH_SIZE-1)//BATCH_SIZE} "
                f"done={done_so_far}/{n} "
                f"embed_dt={embed_dt:.1f}s "
                f"rate={rate:.2f} q/s "
                f"eta={eta/60:.1f}min "
                f"pos_in_top20={pos_in_top20}/{total_mined} "
                f"({100*pos_in_top20/max(1,total_mined):.1f}%)"
            )

    out_f.close()
    client.close()

    if hard_neg_counts:
        hard_neg_counts.sort()
        median = hard_neg_counts[len(hard_neg_counts)//2]
    else:
        median = 0
    log("=== DONE ===")
    log(f"Total mined this run: {total_mined}")
    log(f"Median hard-negs/question: {median}")
    log(f"Positive in top-20: {pos_in_top20}/{total_mined} "
        f"({100*pos_in_top20/max(1,total_mined):.2f}%)")
    log(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
