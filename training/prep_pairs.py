"""
Prepare BGE-M3 fine-tune data.

Supports THREE input modes (auto-detected or forced via --mode):

1. `raw` mode -- legacy: reads one or more Gemini/Claude raw pair JSONL files
   (per-chunk records with an embedded `questions` list). Accepts both
   question key variants:
       {"q": "..."}                       (Gemini, Claude Opus)
       {"question": "...", "complexity": "HIGH", ...}  (Sonnet highcomplex)

2. `curated` mode -- reads `curated_pairs.jsonl` produced by the curator.
   One line per training pair:
       {
         "question": "...",
         "positive_chunk_id": <int|str>,
         "chunk_text": "...",
         "category": "...",
         "complexity": "...",
         "qa_scores": {...},
         "source": "..."
       }

3. `hardneg` mode -- given a curated file AND a hard-negatives file, emit
   triplet-format training rows for MNRL with explicit HN.
   Hard-negs schema (one line per question):
       {
         "question": "...",
         "positive_chunk_id": ...,
         "hard_negs": [{"chunk_id": ..., "text_snippet": "...", ...}, ...]
       }

Output: train.jsonl + val.jsonl (split by positive_chunk_id, no leakage).

Each output line in pair mode:
  {"query": "...", "positive": "...", "chunk_id": ..., "category": "...", ...}

Each output line in triplet mode (when --hard-negatives given):
  {"query": "...", "positive": "...", "negative": "...", "chunk_id": ...}

Usage:
  # Curator output (preferred for v1 RunPod run):
  python prep_pairs.py --mode curated \
      --in curated_pairs.jsonl \
      --hard-negatives hard_negatives.jsonl \
      --out-dir .

  # Legacy raw pair files (can pass multiple):
  python prep_pairs.py --mode raw \
      --in ../eval/training_pairs/pairs_2000_20260422.jsonl \
           ../eval/training_pairs/pairs_opus_highcomplex.jsonl \
           ../eval/training_pairs/pairs_claude_opus.jsonl \
      --out-dir .
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    recs = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [warn] bad JSON in {path.name} line {i}: {e}")
    return recs


def _extract_question(qobj: dict) -> str:
    """Accept both {'q': ...} and {'question': ...} shapes."""
    q = qobj.get("q") or qobj.get("question") or ""
    return q.strip() if isinstance(q, str) else ""


def flatten_raw(records: list[dict]) -> list[dict]:
    """Raw Gemini / Claude / Sonnet pair files -> flat (query, positive) rows."""
    rows = []
    seen: set[str] = set()
    for r in records:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        chunk_id = r.get("chunk_id")
        for qobj in r.get("questions") or []:
            if not isinstance(qobj, dict):
                continue
            q = _extract_question(qobj)
            if not q or len(q) < 5:
                continue
            key = hashlib.sha1(q.lower().encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "query": q,
                "positive": text,
                "chunk_id": chunk_id,
                "doc_id": r.get("doc_id"),
                "category": r.get("category"),
                "subcategory": r.get("subcategory"),
                "section_ref": r.get("section_ref"),
                "complexity": (qobj.get("complexity") or "").lower() or None,
                "source": r.get("generator") or "raw",
            })
    return rows


def flatten_curated(records: list[dict]) -> list[dict]:
    """curated_pairs.jsonl -> flat rows."""
    rows = []
    seen: set[str] = set()
    for r in records:
        q = (r.get("question") or "").strip()
        text = (r.get("chunk_text") or "").strip()
        cid = r.get("positive_chunk_id")
        if not q or not text or cid is None:
            continue
        key = hashlib.sha1(q.lower().encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "query": q,
            "positive": text,
            "chunk_id": cid,
            "category": r.get("category"),
            "complexity": (r.get("complexity") or "").lower() or None,
            "qa_scores": r.get("qa_scores"),
            "source": r.get("source") or "curated",
        })
    return rows


def attach_hard_negatives(rows: list[dict], hn_path: Path) -> list[dict]:
    """
    For each row, look up hard negatives by (question, positive_chunk_id) and
    emit one triplet per negative. If no HN found for a row, keep the pair form
    (MNRL supports mixed batches via separate collations -- we simply leave
    the pair; training code handles both shapes).
    """
    hn_records = load_jsonl(hn_path)
    # index by question text hash + positive chunk id
    idx: dict[tuple[str, str], list[dict]] = {}
    for h in hn_records:
        q = (h.get("question") or "").strip()
        pid = str(h.get("positive_chunk_id"))
        if not q:
            continue
        idx.setdefault((q, pid), []).extend(h.get("hard_negs") or [])

    out = []
    n_triplets = 0
    for r in rows:
        key = (r["query"], str(r["chunk_id"]))
        hns = idx.get(key) or []
        emitted = False
        for hn in hns:
            neg_text = (hn.get("text_snippet") or hn.get("text") or "").strip()
            if not neg_text:
                continue
            out.append({**r, "negative": neg_text, "negative_chunk_id": hn.get("chunk_id")})
            emitted = True
            n_triplets += 1
        if not emitted:
            out.append(r)
    print(f"[hn] attached {n_triplets} triplets across {len(rows)} rows "
          f"from {hn_path} ({len(hn_records)} HN records)")
    return out


def split_by_chunk(rows: list[dict], val_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    by_chunk: dict[str, list[dict]] = {}
    for row in rows:
        by_chunk.setdefault(str(row["chunk_id"]), []).append(row)
    chunk_ids = sorted(by_chunk.keys())
    rng = random.Random(seed)
    rng.shuffle(chunk_ids)
    n_val = max(1, int(round(len(chunk_ids) * val_frac))) if len(chunk_ids) > 1 else 0
    val_ids = set(chunk_ids[:n_val])
    train, val = [], []
    for cid, items in by_chunk.items():
        (val if cid in val_ids else train).extend(items)
    return train, val


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stats(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    ql = [len(r["query"]) for r in rows]
    pl = [len(r["positive"]) for r in rows]
    cats: dict[str, int] = {}
    comps: dict[str, int] = {}
    n_trip = sum(1 for r in rows if r.get("negative"))
    for r in rows:
        cats[r.get("category") or "?"] = cats.get(r.get("category") or "?", 0) + 1
        comps[r.get("complexity") or "?"] = comps.get(r.get("complexity") or "?", 0) + 1
    return {
        "n": len(rows),
        "n_triplets": n_trip,
        "n_pairs_only": len(rows) - n_trip,
        "unique_chunks": len({r["chunk_id"] for r in rows}),
        "query_len_chars": {
            "mean": round(statistics.mean(ql), 1),
            "median": statistics.median(ql),
            "min": min(ql),
            "max": max(ql),
        },
        "positive_len_chars": {
            "mean": round(statistics.mean(pl), 1),
            "median": statistics.median(pl),
            "min": min(pl),
            "max": max(pl),
        },
        "by_category": dict(sorted(cats.items(), key=lambda kv: -kv[1])),
        "by_complexity": dict(sorted(comps.items(), key=lambda kv: -kv[1])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True,
                    help="input JSONL file(s). Mode 'raw' accepts multiple; "
                         "mode 'curated' uses the first.")
    ap.add_argument("--mode", choices=["auto", "raw", "curated"], default="auto")
    ap.add_argument("--hard-negatives", default=os.environ.get("HARD_NEGATIVES"),
                    help="optional path to hard_negatives.jsonl "
                         "(schema: {question, positive_chunk_id, hard_negs:[...]})")
    ap.add_argument("--out-dir", default=os.environ.get("PREP_OUT_DIR", "."))
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect mode from first file's first record shape
    paths = [Path(p) for p in args.inp]
    mode = args.mode
    if mode == "auto":
        probe = load_jsonl(paths[0])[:1]
        if probe and "chunk_text" in probe[0] and "positive_chunk_id" in probe[0]:
            mode = "curated"
        else:
            mode = "raw"
        print(f"[auto] detected mode={mode}")

    all_rows: list[dict] = []
    per_file_counts = {}
    if mode == "raw":
        for p in paths:
            recs = load_jsonl(p)
            rows = flatten_raw(recs)
            per_file_counts[p.name] = {"records": len(recs), "pairs": len(rows)}
            all_rows.extend(rows)
    else:  # curated
        recs = load_jsonl(paths[0])
        all_rows = flatten_curated(recs)
        per_file_counts[paths[0].name] = {"records": len(recs), "pairs": len(all_rows)}

    # Deduplicate across files by query hash
    seen = set()
    dedup = []
    for r in all_rows:
        k = hashlib.sha1(r["query"].lower().encode("utf-8")).hexdigest()
        if k in seen:
            continue
        seen.add(k)
        dedup.append(r)
    all_rows = dedup

    if args.hard_negatives:
        hn_path = Path(args.hard_negatives)
        if hn_path.exists():
            all_rows = attach_hard_negatives(all_rows, hn_path)
        else:
            print(f"[warn] --hard-negatives {hn_path} does not exist; skipping")

    train, val = split_by_chunk(all_rows, args.val_frac, args.seed)

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)

    report = {
        "mode": mode,
        "inputs": per_file_counts,
        "hard_negatives": args.hard_negatives,
        "pairs_total_after_dedup": len(all_rows),
        "train": stats(train),
        "val": stats(val),
    }
    (out_dir / "stats.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_dir/'train.jsonl'} ({len(train)}) + "
          f"{out_dir/'val.jsonl'} ({len(val)})")


if __name__ == "__main__":
    main()
