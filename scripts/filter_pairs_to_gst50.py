#!/usr/bin/env python3
"""Filter unified cbic_pairs_v2.jsonl to the 50 GST doc_ids for Stage M gate runs.

Two filter modes because the legacy migrated rows have heterogeneous provenance:
  (1) by doc_id — exact match on pair.doc_id ∈ gst50_ids  (Format B origin)
  (2) by chunk_id — match pair.chunk_id ∈ {chunks of gst50 docs in v2 manifest}
      (Format A origin — those rows don't carry doc_id but the chunk_id comes
      from the v1 corpus; only useful if we re-lookup against v2 chunks table).

Because v2 chunks are NEW (different chunk_ids from v1), most Format A rows
will NOT match the v2 gst50 chunk_ids. So this filter primarily relies on
the doc_id match for Format B rows. Output shows coverage per query_type and
per domain so we can decide whether hand-authoring top-up is needed.

CLI:
  python filter_pairs_to_gst50.py \\
    --in D:/_gpu_rig_ai/data/training_corpus/cbic_pairs_v2.jsonl \\
    --doc-ids D:/_gpu_rig_ai/reingest_spec/gst50_doc_ids.csv \\
    --out D:/_gpu_rig_ai/data/training_corpus/cbic_pairs_v2_gst50_candidate.jsonl
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--doc-ids", type=Path, required=True,
                    help="CSV or newline file of gst50 doc_ids")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    raw = args.doc_ids.read_text(encoding="utf-8").strip()
    ids = set()
    for tok in raw.replace("\n", ",").split(","):
        t = tok.strip()
        if t: ids.add(t)
    print(f"[filter] {len(ids)} target doc_ids")

    kept = []
    cnt_total = 0
    cnt_kept = 0
    by_doc = Counter()
    by_type = Counter()
    by_gen = Counter()
    by_diff = Counter()

    with open(args.inp, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            cnt_total += 1
            try:
                r = json.loads(line)
            except Exception: continue
            if r.get("doc_id") in ids:
                kept.append(r)
                cnt_kept += 1
                by_doc[r["doc_id"]] += 1
                by_type[r.get("query_type") or "(unset)"] += 1
                by_gen[r.get("generator") or "(unset)"] += 1
                by_diff[r.get("difficulty") or "(unset)"] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # missing doc ids (no pairs found)
    missing = sorted(ids - set(by_doc.keys()))

    report = {
        "input": str(args.inp),
        "output": str(args.out),
        "total_scanned": cnt_total,
        "total_kept": cnt_kept,
        "target_docs": len(ids),
        "docs_with_pairs": len(by_doc),
        "docs_without_pairs": len(missing),
        "missing_doc_ids": missing,
        "by_query_type": dict(by_type),
        "by_generator": dict(by_gen),
        "by_difficulty": dict(by_diff),
        "pairs_per_doc_histogram": Counter(by_doc.values()),
        "top_docs_by_pair_count": by_doc.most_common(10),
    }
    # counter hist in json-safe form
    report["pairs_per_doc_histogram"] = {
        f"{k} pairs": v for k, v in sorted(report["pairs_per_doc_histogram"].items())
    }
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
