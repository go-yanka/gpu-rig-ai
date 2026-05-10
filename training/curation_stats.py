#!/usr/bin/env python3
"""Summarize curated_pairs.jsonl into a markdown report.

Usage: python curation_stats.py [--in PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_IN = Path("D:/_gpu_rig_ai/training/curated_pairs.jsonl")
DEFAULT_OUT = Path("D:/_gpu_rig_ai/consults/curation_stats_20260422.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    pairs = []
    with args.inp.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))

    n = len(pairs)
    by_cat = Counter(p["category"] for p in pairs)
    by_comp = Counter(p["complexity"] for p in pairs)
    by_src = Counter(p["source"] for p in pairs)
    by_cat_comp = defaultdict(Counter)
    for p in pairs:
        by_cat_comp[p["category"]][p["complexity"]] += 1

    warnings = []
    for cat, c in by_cat.items():
        share = c / max(1, n)
        if share > 0.40:
            warnings.append(f"Category `{cat}` exceeds 40% cap ({share*100:.1f}%).")
        if share < 0.05 and c > 0:
            warnings.append(f"Category `{cat}` below 5% floor ({share*100:.1f}%).")
    for comp, target_frac in (("low", 0.35), ("medium", 0.45), ("high", 0.20)):
        got = by_comp.get(comp, 0) / max(1, n)
        if abs(got - target_frac) > 0.10:
            warnings.append(
                f"Complexity `{comp}` off target ({got*100:.1f}% vs {target_frac*100:.0f}%)."
            )
    if by_comp.get("high", 0) == 0:
        warnings.append("No HIGH-complexity pairs selected.")

    lines = []
    lines.append(f"# Curation stats — {args.inp.name}")
    lines.append("")
    lines.append(f"- **Total pairs**: {n}")
    lines.append(f"- **Unique chunk_ids**: {len({p['positive_chunk_id'] for p in pairs})}")
    lines.append("")
    lines.append("## By category")
    lines.append("")
    lines.append("| category | count | share |")
    lines.append("|---|---:|---:|")
    for cat, c in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {cat} | {c} | {c/n*100:.1f}% |")
    lines.append("")
    lines.append("## By complexity")
    lines.append("")
    lines.append("| complexity | count | share |")
    lines.append("|---|---:|---:|")
    for comp in ("low", "medium", "high"):
        c = by_comp.get(comp, 0)
        lines.append(f"| {comp} | {c} | {c/max(1,n)*100:.1f}% |")
    lines.append("")
    lines.append("## By source")
    lines.append("")
    lines.append("| source | count | share |")
    lines.append("|---|---:|---:|")
    for s, c in by_src.most_common():
        lines.append(f"| {s} | {c} | {c/n*100:.1f}% |")
    lines.append("")
    lines.append("## Category x Complexity")
    lines.append("")
    comps = ("low", "medium", "high")
    lines.append("| category | " + " | ".join(comps) + " | total |")
    lines.append("|---|" + "---:|" * (len(comps) + 1))
    for cat in sorted(by_cat_comp):
        row = by_cat_comp[cat]
        tot = sum(row.values())
        lines.append(f"| {cat} | " + " | ".join(str(row.get(c, 0)) for c in comps) + f" | {tot} |")
    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("- None.")
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
