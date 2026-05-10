#!/usr/bin/env python3
"""Curate a balanced training set from generated Q/A pair pools + QA scores.

Usage:
    python curate_training_set.py [--n 5000] [--seed 42]
                                  [--include-unqa] [--strict-realistic]
                                  [--out PATH]

Inputs (hardcoded defaults, pointed at D:/_gpu_rig_ai/eval/training_pairs/):
    pairs_2000_20260422.jsonl      (Gemini, questions[].q)
    pairs_opus_highcomplex.jsonl   (Sonnet HIGH, questions[].q)
    pairs_claude_opus.jsonl        (Claude Opus, questions[].question)
    qa_gemini.jsonl / qa_opus_highcomplex.jsonl / qa_claude_opus.jsonl
      (one per question, joined by (chunk_id, question_text))

Output: curated_pairs.jsonl — one JSON per line, schema documented in the task.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

TP_DIR = Path("D:/_gpu_rig_ai/eval/training_pairs")

SOURCES = [
    # (source_tag, pairs_file, qa_file)
    ("gemini",        TP_DIR / "pairs_2000_20260422.jsonl",     TP_DIR / "qa_gemini.jsonl"),
    ("sonnet_high",   TP_DIR / "pairs_opus_highcomplex.jsonl",  TP_DIR / "qa_opus_highcomplex.jsonl"),
    ("claude_opus",   TP_DIR / "pairs_claude_opus.jsonl",       TP_DIR / "qa_claude_opus.jsonl"),
]

DEFAULT_OUT = Path("D:/_gpu_rig_ai/training/curated_pairs.jsonl")

# Target complexity mix
TARGET_MIX = {"low": 0.35, "medium": 0.45, "high": 0.20}
CAT_MAX = 0.40
CAT_MIN = 0.05


def q_text(q):
    """Normalize question text across generator schemas."""
    if isinstance(q, str):
        return q
    return (q.get("q") or q.get("question") or "").strip()


def q_complexity(q, record):
    """Best-effort complexity label from the question or its parent record."""
    # Sonnet HIGH explicit
    gen = (record.get("generator") or "").lower()
    if "highcomplex" in gen:
        return "high"
    # Claude Opus uses difficulty=intermediate -> medium
    diff = (q.get("difficulty") if isinstance(q, dict) else None) or ""
    diff = diff.lower()
    if diff in ("easy", "low", "beginner"):
        return "low"
    if diff in ("medium", "intermediate"):
        return "medium"
    if diff in ("hard", "high", "advanced", "expert"):
        return "high"
    return None  # unknown -> will be filled from QA grading if present


def load_pairs():
    """Return list of triple-dicts: one per question across all source files."""
    triples = []
    for src, pfile, _qfile in SOURCES:
        if not pfile.exists():
            print(f"[warn] missing pairs file: {pfile}", file=sys.stderr)
            continue
        with pfile.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk_id = rec.get("chunk_id")
                chunk_text = (rec.get("text") or "")[:2000]
                category = rec.get("category") or "unknown"
                section_ref = rec.get("section_ref") or ""
                for q in rec.get("questions", []) or []:
                    qt = q_text(q)
                    if not qt:
                        continue
                    triples.append({
                        "question": qt,
                        "positive_chunk_id": chunk_id,
                        "chunk_text": chunk_text,
                        "category": category,
                        "section_ref": section_ref,
                        "complexity_hint": q_complexity(q, rec),
                        "source": src,
                    })
    return triples


def load_qa():
    """Return dict keyed by (chunk_id, question_text) -> grading dict."""
    qa = {}
    for _src, _p, qfile in SOURCES:
        if not qfile.exists():
            print(f"[warn] missing QA file (will mark unqa): {qfile}", file=sys.stderr)
            continue
        with qfile.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (rec.get("chunk_id"), (rec.get("question") or "").strip())
                g = rec.get("grading") or {}
                qa[key] = g
    return qa


def passes_hard_filter(g, strict_realistic):
    if not g:
        return False  # no QA
    if int(g.get("answerable", 0)) != 1:
        return False
    if int(g.get("specific", 0)) != 1:
        return False
    if strict_realistic and int(g.get("realistic", 0)) != 1:
        return False
    return True


def bucketize(pairs):
    """Group pairs by (category, complexity)."""
    buckets = defaultdict(list)
    for p in pairs:
        buckets[(p["category"], p["complexity"])].append(p)
    return buckets


def compute_category_quotas(pairs, n_target):
    """Proportional to observed distribution, clamped to [CAT_MIN, CAT_MAX]."""
    counts = Counter(p["category"] for p in pairs)
    total = sum(counts.values()) or 1
    raw = {c: v / total for c, v in counts.items()}
    # Clamp
    clamped = {c: min(CAT_MAX, max(CAT_MIN, f)) for c, f in raw.items()}
    # Renormalize
    s = sum(clamped.values()) or 1
    quotas = {c: int(round(n_target * f / s)) for c, f in clamped.items()}
    return quotas, raw


def select(pairs, n_target, seed):
    rng = random.Random(seed)

    # Dedup by chunk_id: when we later pick, we must not pick >1 question per chunk.
    # Strategy: select per (category, complexity) bucket, tracking used chunk_ids.

    cat_quota, cat_raw = compute_category_quotas(pairs, n_target)

    # Complexity targets: honor TARGET_MIX unless HIGH is scarce
    by_complexity = Counter(p["complexity"] for p in pairs)
    hi_avail = by_complexity.get("high", 0)
    hi_target_natural = int(n_target * TARGET_MIX["high"])
    if hi_avail < hi_target_natural:
        # Use all high, redistribute the diff between low/medium proportionally
        hi_target = hi_avail
        remain = n_target - hi_target
        lm_tot = TARGET_MIX["low"] + TARGET_MIX["medium"]
        lo_target = int(round(remain * TARGET_MIX["low"] / lm_tot))
        md_target = remain - lo_target
    else:
        lo_target = int(round(n_target * TARGET_MIX["low"]))
        md_target = int(round(n_target * TARGET_MIX["medium"]))
        hi_target = n_target - lo_target - md_target

    complexity_target = {"low": lo_target, "medium": md_target, "high": hi_target}

    # Bucket: (category, complexity) -> candidates
    buckets = defaultdict(list)
    for p in pairs:
        buckets[(p["category"], p["complexity"])].append(p)
    for lst in buckets.values():
        rng.shuffle(lst)

    selected = []
    used_chunks = set()

    # Target matrix cells: cat_quota[c] * complexity_mix fraction
    total_complex_sum = sum(complexity_target.values()) or 1
    complexity_frac = {k: v / total_complex_sum for k, v in complexity_target.items()}

    # First pass: hit per-(cat,complexity) cell target
    cell_target = {}
    for cat, q in cat_quota.items():
        for comp, frac in complexity_frac.items():
            cell_target[(cat, comp)] = int(round(q * frac))

    # Pass 1: fill cells
    for (cat, comp), tgt in cell_target.items():
        cands = buckets.get((cat, comp), [])
        take = 0
        for p in cands:
            if take >= tgt:
                break
            if p["positive_chunk_id"] in used_chunks:
                continue
            selected.append(p)
            used_chunks.add(p["positive_chunk_id"])
            take += 1

    # Pass 2: if under target, top up from any unused candidates across categories,
    # preferring complexity buckets that are still under-filled.
    if len(selected) < n_target:
        all_remaining = [
            p for lst in buckets.values() for p in lst
            if p["positive_chunk_id"] not in used_chunks
        ]
        rng.shuffle(all_remaining)
        # Track per-complexity how close we are
        got_complex = Counter(p["complexity"] for p in selected)
        # Sort remaining: first those whose complexity still needs more
        def score(p):
            c = p["complexity"]
            deficit = complexity_target.get(c, 0) - got_complex.get(c, 0)
            return -deficit  # more deficit -> earlier
        all_remaining.sort(key=score)
        for p in all_remaining:
            if len(selected) >= n_target:
                break
            selected.append(p)
            used_chunks.add(p["positive_chunk_id"])
            got_complex[p["complexity"]] += 1

    return selected, cat_quota, cat_raw, complexity_target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-unqa", action="store_true",
                    help="Include pairs without QA grading (treated as borderline).")
    ap.add_argument("--strict-realistic", action="store_true",
                    help="Also drop pairs with realistic=0.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    triples = load_pairs()
    print(f"[load] raw triples (pre-filter): {len(triples)}")

    qa = load_qa()
    print(f"[load] QA grading rows: {len(qa)}")

    # Join + filter
    kept = []
    unqa_count = 0
    dropped_filter = 0
    for t in triples:
        key = (t["positive_chunk_id"], t["question"])
        g = qa.get(key, {})
        if not g:
            unqa_count += 1
            if not args.include_unqa:
                continue
            # Use complexity hint if available, else 'medium' default
            comp = t["complexity_hint"] or "medium"
            t2 = dict(t)
            t2["complexity"] = comp
            t2["qa_scores"] = {"answerable": None, "realistic": None, "specific": None}
            kept.append(t2)
            continue
        if not passes_hard_filter(g, args.strict_realistic):
            dropped_filter += 1
            continue
        comp = (g.get("complexity") or t["complexity_hint"] or "medium").lower()
        if comp not in ("low", "medium", "high"):
            comp = "medium"
        t2 = dict(t)
        t2["complexity"] = comp
        t2["qa_scores"] = {
            "answerable": int(g.get("answerable", 0)),
            "realistic": int(g.get("realistic", 0)),
            "specific": int(g.get("specific", 0)),
        }
        kept.append(t2)

    print(f"[filter] triples w/o QA: {unqa_count} "
          f"({'kept' if args.include_unqa else 'dropped'})")
    print(f"[filter] dropped by hard filter: {dropped_filter}")
    print(f"[filter] candidates post-filter: {len(kept)}")

    if not kept:
        print("[fatal] no candidates after filtering — aborting.", file=sys.stderr)
        sys.exit(2)

    selected, cat_quota, cat_raw, complex_target = select(kept, args.n, args.seed)

    # Report distributions
    print("\n=== Final selection: {} pairs ===".format(len(selected)))
    by_cat = Counter(p["category"] for p in selected)
    by_comp = Counter(p["complexity"] for p in selected)
    by_src = Counter(p["source"] for p in selected)

    print("\nBy category (count | share | quota | corpus%):")
    for cat in sorted(set(list(by_cat) + list(cat_quota))):
        c = by_cat.get(cat, 0)
        share = c / max(1, len(selected))
        print(f"  {cat:20s} {c:5d}  {share*100:5.1f}%  "
              f"quota={cat_quota.get(cat,0):5d}  corpus={cat_raw.get(cat,0)*100:5.1f}%")

    print("\nBy complexity (count | share | target):")
    for comp in ("low", "medium", "high"):
        c = by_comp.get(comp, 0)
        share = c / max(1, len(selected))
        print(f"  {comp:8s} {c:5d}  {share*100:5.1f}%  target={complex_target.get(comp,0)}")

    print("\nBy source:")
    for s, c in by_src.most_common():
        print(f"  {s:14s} {c}")

    # Write
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for p in selected:
            out = {
                "question": p["question"],
                "positive_chunk_id": p["positive_chunk_id"],
                "chunk_text": p["chunk_text"],
                "category": p["category"],
                "section_ref": p["section_ref"],
                "complexity": p["complexity"],
                "qa_scores": p["qa_scores"],
                "source": p["source"],
            }
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"\n[write] {args.out}  ({len(selected)} lines)")


if __name__ == "__main__":
    main()
