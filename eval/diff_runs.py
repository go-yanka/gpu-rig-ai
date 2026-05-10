#!/usr/bin/env python3
"""Diff two eval runs produced by run_eval.py.

Usage:
    python diff_runs.py --base runs/2026_04_20_base --new runs/2026_04_21_b22b
    python diff_runs.py --base ... --new ... --out diff.md

Prints a markdown table: per-query delta + aggregate delta + regression list.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_run(path: Path) -> dict:
    p = path / "run.json"
    if not p.exists():
        print(f"ERROR: {p} not found", file=sys.stderr)
        sys.exit(2)
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", default=None, help="optional markdown output path")
    ap.add_argument("--regress-threshold", type=float, default=-0.5,
                    help="per-item delta below which an item is 'regressed' (default -0.5)")
    args = ap.parse_args()

    base = load_run(Path(args.base))
    new = load_run(Path(args.new))

    b_by_id = {r["id"]: r for r in base["results"]}
    n_by_id = {r["id"]: r for r in new["results"]}
    all_ids = sorted(set(b_by_id) | set(n_by_id))

    rows = []
    regressions = []
    improvements = []
    for iid in all_ids:
        b = b_by_id.get(iid)
        n = n_by_id.get(iid)
        b_pts = b["points"] if b else None
        n_pts = n["points"] if n else None
        delta = (n_pts - b_pts) if (b_pts is not None and n_pts is not None) else None
        b_lat = b.get("latency_ms") if b else None
        n_lat = n.get("latency_ms") if n else None
        lat_delta = (n_lat - b_lat) if (b_lat is not None and n_lat is not None) else None
        rows.append({
            "id": iid,
            "cat": (n or b)["category"],
            "base_pts": b_pts, "new_pts": n_pts, "delta": delta,
            "base_lat": b_lat, "new_lat": n_lat, "lat_delta": lat_delta,
        })
        if delta is not None:
            if delta <= args.regress_threshold:
                regressions.append((iid, delta))
            elif delta > 0:
                improvements.append((iid, delta))

    b_agg = base["aggregate"]
    n_agg = new["aggregate"]
    total_delta = n_agg["total_points"] - b_agg["total_points"]
    pct_delta = (n_agg.get("pct_score") or 0) - (b_agg.get("pct_score") or 0)

    lines = [
        "# Eval Run Diff",
        "",
        f"- **Base**: `{args.base}`  ({b_agg['total_points']} / {b_agg['max_points']}  =  {b_agg['pct_score']}%)",
        f"- **New**:  `{args.new}`   ({n_agg['total_points']} / {n_agg['max_points']}  =  {n_agg['pct_score']}%)",
        f"- **Total points delta**: {total_delta:+.3f}  ({pct_delta:+.2f} pct-pt)",
        f"- **Regressions** (delta <= {args.regress_threshold}): {len(regressions)}",
        f"- **Improvements** (delta > 0): {len(improvements)}",
        f"- **Latency median**: base {b_agg.get('latency_ms_median')}ms -> new {n_agg.get('latency_ms_median')}ms",
        f"- **Latency p95**: base {b_agg.get('latency_ms_p95')}ms -> new {n_agg.get('latency_ms_p95')}ms",
        "",
    ]

    # Regression gate flag
    base_pct = b_agg.get("pct_score") or 0
    if base_pct and (pct_delta < -5.0):
        lines.append(f"**GATE: REGRESSION** — aggregate dropped by {pct_delta:.2f} pct-pts (>5% threshold). Consider revert.")
    else:
        lines.append(f"GATE: OK — aggregate change {pct_delta:+.2f} pct-pts.")
    lines.append("")

    if regressions:
        lines += ["## Regressions", "", "| id | delta |", "|---|---:|"]
        for iid, d in sorted(regressions, key=lambda x: x[1]):
            lines.append(f"| {iid} | {d:+.2f} |")
        lines.append("")

    lines += ["## Per-item",
              "",
              "| id | cat | base | new | delta | base_lat | new_lat | lat_delta |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['cat']} | "
            f"{r['base_pts']} | {r['new_pts']} | "
            f"{r['delta'] if r['delta'] is not None else '-'} | "
            f"{r['base_lat']} | {r['new_lat']} | "
            f"{r['lat_delta'] if r['lat_delta'] is not None else '-'} |"
        )

    out = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"[diff] written to {args.out}")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
