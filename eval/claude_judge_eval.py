#!/usr/bin/env python3
"""CBIC RAG eval with Claude CLI as LLM-as-judge.

Mirrors run_eval.py but uses the local `claude` CLI (via `claude -p`) as the
judge instead of qwen3-14B. Primary purpose: get an independent baseline to
test the hypothesis that our qwen3 judge is biased / deflated.

Usage:
    python claude_judge_eval.py --gold ../eval/gold_set.yaml \
           --api http://192.168.1.107:9500 \
           --out runs/p1_2_claude_judge_<ts> \
           [--throttle 15]   # seconds between claude calls (rate-limit safety)
           [--limit N]       # smoke test

Reuses same 0-3 rubric as run_eval.py for apples-to-apples comparison.
Writes per_item.jsonl with BOTH a claude_score (if available) and whatever
metadata was in the RAG response, plus a summary.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import yaml
    import requests
except ImportError as e:
    print(f"ERROR: missing dep: {e}", file=sys.stderr)
    sys.exit(2)


# ---------- rubric (mirrors run_eval.py) --------------------------------------

JUDGE_SYSTEM = (
    "You are a strict Indian tax-law evaluator. Given a question and a "
    "candidate answer, rate the answer on a 0-3 scale for whether it correctly "
    "resolves the question:\n"
    "  0 = wrong or misleading\n"
    "  1 = partially correct, misses key statutory reference or conclusion\n"
    "  2 = mostly correct, minor gaps\n"
    "  3 = fully correct with correct statutory citation and clear conclusion\n"
    "Reply with a single integer 0, 1, 2, or 3 — nothing else."
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def match_any(needle: str, haystack: str) -> bool:
    return _norm(needle) in _norm(haystack)


def deterministic_points(item: dict, answer: str, verified_quotes: list) -> dict:
    """Compute the non-judge points (matches run_eval.score_item)."""
    secs_exp = item.get("expected_sections", []) or []
    secs_match = [s for s in secs_exp if match_any(s, answer)]

    rules_exp = item.get("expected_rules", []) or []
    rules_match = [r for r in rules_exp if match_any(r, answer)]

    notifs_exp = item.get("expected_notifications", []) or []
    notifs_match = [n for n in notifs_exp if match_any(n, answer)]

    kws_exp = item.get("expected_conclusion_keywords", []) or []
    kws_match = [k for k in kws_exp if match_any(k, answer)]

    forbidden = item.get("must_not_say", []) or []
    forbidden_hits = [f for f in forbidden if match_any(f, answer)]

    verbatim_required = bool(item.get("must_cite_verbatim"))
    verbatim_pass = (not verbatim_required) or len(verified_quotes) >= 1

    pts = (
        float(len(secs_match))
        + float(len(rules_match))
        + float(len(notifs_match))
        + float(len(kws_match))
        - float(len(forbidden_hits))
        + (1.0 if (verbatim_required and verbatim_pass) else 0.0)
    )
    max_pts = (
        float(len(secs_exp))
        + float(len(rules_exp))
        + float(len(notifs_exp))
        + float(len(kws_exp))
        + (1.0 if verbatim_required else 0.0)
    )
    return {
        "det_points": pts,
        "det_max": max_pts,
        "sections_matched": secs_match,
        "rules_matched": rules_match,
        "notifications_matched": notifs_match,
        "keywords_matched": kws_match,
        "forbidden_hits": forbidden_hits,
        "verbatim_required": verbatim_required,
        "verbatim_pass": verbatim_pass,
    }


# ---------- API query ---------------------------------------------------------

def call_rag(api_base: str, question: str, timeout: float) -> dict:
    try:
        r = requests.post(
            f"{api_base.rstrip('/')}/query",
            json={"question": question},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}


# ---------- Claude CLI judge --------------------------------------------------

def _resolve_claude_cmd() -> str:
    """Resolve the claude CLI path. On Windows we need claude.cmd, not claude."""
    explicit = os.environ.get("CLAUDE_CLI_PATH")
    if explicit and Path(explicit).exists():
        return explicit
    if sys.platform == "win32":
        candidates = [
            Path(os.environ.get("APPDATA", "")) / "npm" / "claude.cmd",
            Path.home() / "AppData/Roaming/npm/claude.cmd",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
    return "claude"


CLAUDE_CMD = _resolve_claude_cmd()


def call_claude_judge(question: str, answer: str, retries: int = 2) -> int | None:
    """Invoke `claude -p <prompt>` and parse a 0-3 integer from the response."""
    if not answer.strip():
        return 0
    prompt = (
        f"{JUDGE_SYSTEM}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CANDIDATE ANSWER:\n{answer}\n\n"
        f"Score (0-3):"
    )
    last_err = None
    for attempt in range(retries + 1):
        try:
            env = os.environ.copy()
            # Claude CLI refuses to run nested inside another Claude session;
            # unset to allow a fresh headless invocation.
            env.pop("CLAUDECODE", None)
            env.pop("CLAUDE_CODE_ENTRYPOINT", None)
            proc = subprocess.run(
                [CLAUDE_CMD, "-p", prompt, "--output-format", "text"],
                capture_output=True,
                text=True,
                timeout=120,
                encoding="utf-8",
                env=env,
                shell=False,
            )
            if proc.returncode != 0:
                last_err = f"rc={proc.returncode}: {proc.stderr[:200]}"
                time.sleep(5 * (attempt + 1))
                continue
            out = proc.stdout.strip()
            m = re.search(r"[0-3]", out)
            if m:
                return int(m.group(0))
            last_err = f"no-int-in-output: {out[:80]}"
        except subprocess.TimeoutExpired:
            last_err = "timeout"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(5 * (attempt + 1))
    print(f"    [claude-judge] FAIL after retries: {last_err}", file=sys.stderr)
    return None


# ---------- runner ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(Path(__file__).parent / "gold_set.yaml"))
    ap.add_argument("--api", default="http://192.168.1.107:9500")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--throttle", type=float, default=0.0,
                    help="seconds to wait between Claude judge calls")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default=None, help="filter by category")
    args = ap.parse_args()

    gold_path = Path(args.gold)
    gold = yaml.safe_load(gold_path.read_text(encoding="utf-8"))
    items = gold.get("items", gold) if isinstance(gold, dict) else gold

    if args.only:
        items = [it for it in items if it.get("category") == args.only]
    if args.limit and args.limit > 0:
        items = items[:args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_item_path = out_dir / "per_item.jsonl"
    summary_path = out_dir / "summary.md"

    print(f"[cfg] items={len(items)} api={args.api} out={out_dir}", file=sys.stderr)
    print(f"[cfg] throttle={args.throttle}s judge=claude-cli", file=sys.stderr)

    t_start = time.time()
    results = []
    with per_item_path.open("w", encoding="utf-8") as fh:
        for i, item in enumerate(items, 1):
            q = item["question"]
            t0 = time.time()
            resp = call_rag(args.api, q, args.timeout)
            rag_lat = (time.time() - t0) * 1000.0
            answer = (
                resp.get("answer_markdown")
                or resp.get("answer")
                or resp.get("text")
                or ""
            )
            verified_quotes = resp.get("verified_quotes", []) or []

            det = deterministic_points(item, answer, verified_quotes)

            claude_score: int | None = None
            if not resp.get("_error"):
                t1 = time.time()
                claude_score = call_claude_judge(q, answer)
                judge_lat = (time.time() - t1) * 1000.0
            else:
                judge_lat = 0.0

            claude_pts = float(claude_score) if claude_score is not None else 0.0
            total_pts = det["det_points"] + claude_pts
            total_max = det["det_max"] + 3.0

            rec = {
                "id": item.get("id"),
                "category": item.get("category"),
                "subcategory": item.get("subcategory"),
                "difficulty": item.get("difficulty"),
                "question": q,
                "answer": answer,
                "verified_quotes_count": len(verified_quotes),
                "rag_latency_ms": round(rag_lat),
                "judge_latency_ms": round(judge_lat),
                "claude_score": claude_score,
                "claude_pts": claude_pts,
                **det,
                "total_pts": total_pts,
                "total_max": total_max,
                "error": resp.get("_error"),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            results.append(rec)

            elapsed_min = (time.time() - t_start) / 60.0
            print(
                f"[{i}/{len(items)}] {rec['id']:32s} "
                f"det={det['det_points']:.1f}/{det['det_max']:.1f} "
                f"claude={claude_score} "
                f"rag={rag_lat/1000:.1f}s j={judge_lat/1000:.1f}s "
                f"elapsed={elapsed_min:.1f}m",
                file=sys.stderr,
            )

            if args.throttle > 0 and i < len(items):
                time.sleep(args.throttle)

    # Aggregate
    total_pts = sum(r["total_pts"] for r in results)
    total_max = sum(r["total_max"] for r in results)
    pct = 100.0 * total_pts / total_max if total_max else 0.0

    det_only_pts = sum(r["det_points"] for r in results)
    det_only_max = sum(r["det_max"] for r in results)
    det_only_pct = 100.0 * det_only_pts / det_only_max if det_only_max else 0.0

    claude_scores = [r["claude_score"] for r in results if r["claude_score"] is not None]
    mean_claude = sum(claude_scores) / len(claude_scores) if claude_scores else 0.0

    # per-category breakdown
    by_cat = {}
    for r in results:
        c = r["category"] or "?"
        d = by_cat.setdefault(c, {"n": 0, "pts": 0.0, "max": 0.0, "claude_sum": 0, "claude_n": 0})
        d["n"] += 1
        d["pts"] += r["total_pts"]
        d["max"] += r["total_max"]
        if r["claude_score"] is not None:
            d["claude_sum"] += r["claude_score"]
            d["claude_n"] += 1

    lines = [
        f"# CBIC RAG eval — Claude judge",
        f"- date: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- gold: {gold_path}",
        f"- items: {len(results)}",
        f"- api: {args.api}",
        f"- judge: claude-cli (Max subscription)",
        "",
        f"## Headline",
        f"**Total: {total_pts:.1f} / {total_max:.1f} = {pct:.2f}%**",
        f"Deterministic-only (no judge): {det_only_pts:.1f} / {det_only_max:.1f} = {det_only_pct:.2f}%",
        f"Mean Claude judge score: {mean_claude:.2f} / 3 (n={len(claude_scores)})",
        "",
        f"## Per category",
        f"| category | n | pts | max | pct | mean_claude |",
        f"|---|---:|---:|---:|---:|---:|",
    ]
    for c in sorted(by_cat):
        d = by_cat[c]
        p = 100.0 * d["pts"] / d["max"] if d["max"] else 0.0
        mc = d["claude_sum"] / d["claude_n"] if d["claude_n"] else 0.0
        lines.append(f"| {c} | {d['n']} | {d['pts']:.1f} | {d['max']:.1f} | {p:.2f}% | {mc:.2f} |")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[done] summary: {summary_path}", file=sys.stderr)
    print(f"[done] headline: {total_pts:.1f}/{total_max:.1f} = {pct:.2f}%", file=sys.stderr)


if __name__ == "__main__":
    main()
