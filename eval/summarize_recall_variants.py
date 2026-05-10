#!/usr/bin/env python3
"""Read the combined JSONL of variant results and produce a markdown summary."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path):
    by_variant = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        by_variant[r["variant"]].append(r)
    return by_variant


def stats(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0, "hit1": 0, "hit5": 0, "errs": 0, "elapsed_s": 0.0}
    h1 = sum(1 for r in rows if r.get("hit_at_1"))
    h5 = sum(1 for r in rows if r.get("hit_at_5"))
    errs = sum(1 for r in rows if r.get("error"))
    elapsed = sum(float(r.get("elapsed_s") or 0) for r in rows)
    return {"n": n, "hit1": h1, "hit5": h5, "errs": errs, "elapsed_s": elapsed}


def by_cat(rows):
    d = defaultdict(list)
    for r in rows:
        d[r.get("category") or "unknown"].append(r)
    return {c: stats(rs) for c, rs in d.items()}


def find_fixed(baseline, variant_rows):
    """Items where baseline missed but this variant hit."""
    b = {r["id"]: r for r in baseline}
    fixed = []
    for r in variant_rows:
        br = b.get(r["id"])
        if br and not br.get("hit_at_5") and r.get("hit_at_5"):
            fixed.append((br, r))
    return fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    by_v = load(args.jsonl)
    order = ["baseline", "hyde", "meta_filter", "wider_rerank"]
    base = by_v.get("baseline", [])
    base_stats = stats(base)

    lines = []
    lines.append("# Recall@5 variants — 2026-04-22")
    lines.append("")
    lines.append(f"Gold set: {base_stats['n']} items. Live CBIC RAG API (`/query`, top-5 citations).")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Variant | N | hit@1 | hit@5 | Δ vs baseline (hit@5) | errors | total elapsed (s) | avg s/item |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for v in order:
        rows = by_v.get(v, [])
        s = stats(rows)
        if s["n"] == 0:
            continue
        h1 = f"{s['hit1']}/{s['n']} ({100*s['hit1']/s['n']:.2f}%)"
        h5 = f"{s['hit5']}/{s['n']} ({100*s['hit5']/s['n']:.2f}%)"
        if v == "baseline":
            delta = "—"
        else:
            base_rate = 100 * base_stats["hit5"] / max(base_stats["n"], 1)
            this_rate = 100 * s["hit5"] / max(s["n"], 1)
            delta = f"{this_rate - base_rate:+.2f} pp"
        avg = s["elapsed_s"] / max(s["n"], 1)
        lines.append(f"| {v} | {s['n']} | {h1} | {h5} | {delta} | {s['errs']} | {s['elapsed_s']:.1f} | {avg:.2f} |")

    lines.append("")
    lines.append("## Per-category hit@5")
    lines.append("")
    variants_present = [v for v in order if v in by_v]
    # Gather categories across baseline
    cats = sorted({r.get("category") or "unknown" for r in base})
    header = "| category | " + " | ".join(variants_present) + " |"
    sep = "|---|" + "|".join(["---"] * len(variants_present)) + "|"
    lines.append(header)
    lines.append(sep)
    for c in cats:
        row = [c]
        for v in variants_present:
            rs = [r for r in by_v.get(v, []) if (r.get("category") or "unknown") == c]
            s = stats(rs)
            if s["n"]:
                row.append(f"{s['hit5']}/{s['n']} ({100*s['hit5']/s['n']:.1f}%)")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Representative items fixed by variants (baseline miss → variant hit)")
    lines.append("")
    for v in ["hyde", "meta_filter", "wider_rerank"]:
        rows = by_v.get(v, [])
        if not rows:
            continue
        fixed = find_fixed(base, rows)
        lines.append(f"### {v} — fixed {len(fixed)} items baseline missed")
        lines.append("")
        for br, vr in fixed[:4]:
            lines.append(f"- **{vr['id']}** ({vr.get('category')})")
            lines.append(f"  - Q: {vr['question'][:180]}")
            if v == "hyde" and vr.get("hyde_text"):
                lines.append(f"  - HyDE: {vr['hyde_text'][:200]}")
            if v == "meta_filter":
                lines.append(f"  - Steered Q: {vr.get('used_question','')[:180]}")
            exp_hits = {k: v_ for k, v_ in (vr.get("per_entity") or {}).items() if v_}
            lines.append(f"  - Matched entities: {exp_hits}")
        lines.append("")

    # Also: items that regressed
    lines.append("## Regressions (baseline hit → variant miss)")
    lines.append("")
    for v in ["hyde", "meta_filter", "wider_rerank"]:
        rows = by_v.get(v, [])
        if not rows:
            continue
        b_map = {r["id"]: r for r in base}
        regs = [r for r in rows
                if b_map.get(r["id"]) and b_map[r["id"]].get("hit_at_5")
                and not r.get("hit_at_5")]
        lines.append(f"- {v}: {len(regs)} regressions")

    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    # Pick best by hit@5
    best = None
    best_rate = -1.0
    for v in order:
        s = stats(by_v.get(v, []))
        if s["n"] == 0:
            continue
        rate = 100 * s["hit5"] / s["n"]
        if rate > best_rate:
            best_rate = rate
            best = v
    base_rate = 100 * base_stats["hit5"] / max(base_stats["n"], 1)
    lines.append(f"- Best variant by hit@5: **{best}** at {best_rate:.2f}% "
                 f"(baseline {base_rate:.2f}%, delta {best_rate-base_rate:+.2f} pp).")
    lines.append("- See per-category table for category-specific behaviour.")
    lines.append("- Any variant with < baseline recall should be rejected outright.")
    lines.append("- Consider stacking: HyDE + meta_filter + wider_rerank if each is individually positive.")
    lines.append("")

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
