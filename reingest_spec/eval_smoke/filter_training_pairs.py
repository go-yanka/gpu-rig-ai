#!/usr/bin/env python3
"""filter_training_pairs.py — wire existing 5,781 QA pairs to G2 for scoped smoke runs.

Problem this solves (codified from 2026-04-23):
------------------------------------------------
We have ~5,781 pre-generated QA pairs in `D:/_gpu_rig_ai/eval/training_pairs/`
across 7 jsonl files. They were produced against v1 chunk_ids (INT hash). v2
uses SHA256 chunk_ids that will NEVER match, AND the pairs have no `doc_id`.

For a scoped smoke (e.g. 5 docs via ingest_v2.py --doc-ids), we need to filter
those 5,781 pairs down to "questions whose answer lives in the smoke docs."
We do this by: for each pair, check whether `why_this_chunk` (the rationale
the pair-generator wrote describing what chunk it came from) has strong
substring overlap with any chunk text in the smoke manifest. If yes, we mint
`expected_doc_id = <the matching doc>` and emit it to the smoke gold set.

This retains the 12h of investment in those 5,781 pairs rather than drafting
from scratch (the CLAUDE.md inventory rule). Output is consumable by
gate_g2_dual_judge.py (which accepts `expected_doc_ids` or `expected_text`).

Usage:
  python3 filter_training_pairs.py \\
    --manifest /opt/indian-legal-ai/data/ingest_manifest_v2.sqlite \\
    --pairs-dir /opt/indian-legal-ai/eval/training_pairs/ \\
    --doc-ids cbic-notification-msts:1008308,cbic-circular-msts:1001000,... \\
    --min-overlap 40 \\
    --out eval_smoke/training_pairs_5doc.jsonl

Output schema (one per line):
  {"id": "<src-file>:<line>",
   "query": "<question>",
   "category": "<category>",
   "expected_doc_ids": ["<doc_id>"],
   "expected_text": "<why_this_chunk substring matched>",
   "source_pair": {<full original pair>}}
"""
from __future__ import annotations
import argparse
import json
import sqlite3
import sys
from pathlib import Path


def _load_smoke_chunks(manifest_path: str, doc_ids: list[str]) -> dict[str, list[dict]]:
    """Return {doc_id: [{chunk_id, text}, ...]} for the smoke docs."""
    con = sqlite3.connect(manifest_path)
    con.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(doc_ids))
    rows = con.execute(
        f"SELECT chunk_id, doc_id, payload_json FROM chunks "
        f"WHERE is_canonical=1 AND doc_id IN ({placeholders})",
        doc_ids,
    ).fetchall()
    out: dict[str, list[dict]] = {}
    for r in rows:
        p = json.loads(r["payload_json"])
        text = (p.get("text") or "").strip()
        if text:
            out.setdefault(r["doc_id"], []).append({"chunk_id": r["chunk_id"], "text": text})
    con.close()
    return out


def _ngram_overlap(a: str, b: str, n: int = 5) -> int:
    """Rough character-ngram intersection count (cheap approximation of
    substring similarity; good enough for filtering, not for G3)."""
    a = " ".join(a.lower().split())
    b = " ".join(b.lower().split())
    if len(a) < n or len(b) < n:
        return 0
    A = {a[i:i+n] for i in range(len(a) - n + 1)}
    B = {b[i:i+n] for i in range(len(b) - n + 1)}
    return len(A & B)


def _find_best_doc(why: str, chunks_by_doc: dict[str, list[dict]], min_overlap: int):
    """For a `why_this_chunk` rationale, find the smoke doc with strongest overlap.
    Returns (doc_id, best_chunk_text) or (None, None) if below threshold."""
    best_doc, best_score, best_text = None, 0, ""
    for doc_id, chunks in chunks_by_doc.items():
        for ch in chunks:
            s = _ngram_overlap(why, ch["text"])
            if s > best_score:
                best_score, best_doc, best_text = s, doc_id, ch["text"]
    if best_score >= min_overlap:
        return best_doc, best_text
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="ingest_manifest_v2.sqlite path (must have chunks table populated)")
    ap.add_argument("--pairs-dir", required=True,
                    help="Directory containing qa_*.jsonl / pairs_*.jsonl training pairs")
    ap.add_argument("--doc-ids", required=True,
                    help="Comma-separated doc_ids to scope")
    ap.add_argument("--min-overlap", type=int, default=40,
                    help="Minimum 5-gram overlap to accept a pair (default 40)")
    ap.add_argument("--out", required=True)
    # 2026-04-24 A-to-Z failure reporting. Prior version silently dropped pairs
    # on JSONDecodeError / missing-q / missing-why / below-overlap. Now every
    # drop is categorized, counted, sampled to `{out}.dropped.json`, and a
    # summary printed. --strict-max-drop-pct fails the run if drop rate exceeds
    # threshold — protects against a corrupt pair file silently yielding a
    # tiny matched set that looks "normal".
    ap.add_argument("--strict-max-drop-pct", type=float, default=100.0,
                    help="Fail run if drop rate > this percent (default 100 = warn-only)")
    args = ap.parse_args()

    doc_ids = [d.strip() for d in args.doc_ids.split(",") if d.strip()]
    print(f"[filter] loading smoke chunks for {len(doc_ids)} docs", file=sys.stderr)
    chunks_by_doc = _load_smoke_chunks(args.manifest, doc_ids)
    n_chunks = sum(len(v) for v in chunks_by_doc.values())
    print(f"[filter] loaded {n_chunks} canonical chunks across {len(chunks_by_doc)} docs",
          file=sys.stderr)
    if n_chunks == 0:
        print("[filter] ERROR: no chunks loaded — was ingest run against this manifest?",
              file=sys.stderr)
        sys.exit(2)

    pairs_dir = Path(args.pairs_dir)
    jsonl_files = sorted(pairs_dir.glob("qa_*.jsonl")) + sorted(pairs_dir.glob("pairs_*.jsonl"))
    print(f"[filter] scanning {len(jsonl_files)} pair files", file=sys.stderr)

    n_in = n_kept = 0
    drop_reasons: dict = {"json_decode": 0, "missing_q_or_why": 0, "below_overlap": 0}
    drop_samples: dict = {"json_decode": [], "missing_q_or_why": [], "below_overlap": []}
    SAMPLE_CAP = 20

    def _record_drop(reason: str, rec: dict):
        drop_reasons[reason] += 1
        if len(drop_samples[reason]) < SAMPLE_CAP:
            drop_samples[reason].append(rec)

    with open(args.out, "w") as fout:
        for jf in jsonl_files:
            with open(jf) as fin:
                for lineno, line in enumerate(fin, 1):
                    line = line.strip()
                    if not line:
                        continue
                    n_in += 1
                    rec_id = f"{jf.name}:{lineno}"
                    try:
                        p = json.loads(line)
                    except json.JSONDecodeError as e:
                        _record_drop("json_decode",
                                     {"id": rec_id, "error": str(e),
                                      "line_prefix": line[:120]})
                        continue
                    q = p.get("question") or p.get("q") or ""
                    why = p.get("why_this_chunk") or p.get("why") or ""
                    if not q or not why:
                        _record_drop("missing_q_or_why",
                                     {"id": rec_id, "has_q": bool(q), "has_why": bool(why)})
                        continue
                    doc_id, matched_text = _find_best_doc(why, chunks_by_doc, args.min_overlap)
                    if doc_id:
                        out_rec = {
                            "id": rec_id,
                            "query": q,
                            "category": p.get("category", ""),
                            "expected_doc_ids": [doc_id],
                            "expected_text": matched_text[:500],
                            "source_pair": p,
                        }
                        fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                        n_kept += 1
                    else:
                        _record_drop("below_overlap",
                                     {"id": rec_id, "query": q[:160],
                                      "why_prefix": why[:160]})

    total_drops = sum(drop_reasons.values())
    drop_pct = (100.0 * total_drops / n_in) if n_in else 0.0
    print(f"[filter] scanned={n_in} kept={n_kept} dropped={total_drops} "
          f"({drop_pct:.1f}%) → {args.out}", file=sys.stderr)
    print(f"[filter] drop reasons: {drop_reasons}", file=sys.stderr)

    dropped_path = args.out + ".dropped.json"
    with open(dropped_path, "w", encoding="utf-8") as f:
        json.dump({
            "scanned": n_in, "kept": n_kept,
            "dropped_total": total_drops, "dropped_pct": round(drop_pct, 2),
            "reasons": drop_reasons,
            "samples": drop_samples,
            "min_overlap": args.min_overlap,
            "strict_max_drop_pct": args.strict_max_drop_pct,
        }, f, indent=2, ensure_ascii=False)
    print(f"[filter] drop report → {dropped_path}", file=sys.stderr)

    if n_kept < 20:
        print(f"[filter] WARNING: only {n_kept} pairs matched (min-overlap={args.min_overlap}). "
              f"Consider lowering --min-overlap.", file=sys.stderr)

    if drop_pct > args.strict_max_drop_pct:
        print(f"[filter FAIL] drop rate {drop_pct:.1f}% > --strict-max-drop-pct "
              f"{args.strict_max_drop_pct}% — refusing to proceed on degraded sample.",
              file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
