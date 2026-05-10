#!/usr/bin/env python3
"""Stage F — G2 Reasoning gate (PARALLEL variant) — dual judge ensemble.

Differences from gate_g2_dual_judge.py:
 - Claude judge invoked via local CLI (`claude -p "<prompt>"`), NOT Anthropic API key.
   Reason: user standing preference in MEMORY.md — "for claude we use claude cli".
 - Gemini judge unchanged (GEMINI_API_KEY from D:/_gpu_rig_ai/.env on host).
 - Per-item work (query + 2 judges) submitted to a ThreadPoolExecutor.
   Default concurrency: --workers 8.  Speedup ~5-8x versus serial.
 - Preserves gate semantics: BOTH mean >= threshold AND agreement_ok AND
   errors <= allow_errors.  Errors counted, never silently averaged away.

Writes gate_g2_parallel_result.json.  Exits 0 pass / 2 fail / 3 errors>allow.

Origin: 2026-04-24 GST50 scale test — serial G2 projected ~60 min for 500 pairs;
parallel version targets ~10 min.
"""
from __future__ import annotations
import argparse, json, os, sys, time, urllib.request, re, subprocess, shlex
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

HERE = Path(__file__).parent
API = "http://127.0.0.1:9500/query"
DEFAULT_GOLD = HERE.parent / "eval" / "v2_gold.json"
OUT = HERE / "gate_g2_parallel_result.json"

GEMINI_MODEL = os.environ.get("GEMINI_JUDGE_MODEL", "gemini-2.5-pro")
GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              f"{GEMINI_MODEL}:generateContent?key={{key}}")
CLAUDE_BIN = os.environ.get("CLAUDE_CLI_BIN", "/usr/bin/claude")

JUDGE_PROMPT = """You are evaluating the factual correctness of a RAG system's answer
against the user's question and the retrieved evidence.

QUESTION:
{question}

EXPECTED (from gold set — may be partial hints, not exhaustive):
{expected}

SYSTEM ANSWER:
{answer}

RETRIEVED EVIDENCE (top snippets used to answer):
{evidence}

Score factual correctness from 0.0 (completely wrong / hallucinated) to
1.0 (fully correct, well-grounded in evidence). Penalize:
- factual contradictions with the evidence
- fabricated citations / section numbers
- missing critical qualifications (provisos, exceptions)
Reward grounded, precise answers that cite the retrieved material.

Respond ONLY with a single JSON object: {{"score": <float 0..1>, "reason": "<1 line>"}}"""


def _post(url, body, headers, timeout=60):
    req = urllib.request.Request(url, method="POST",
        data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _extract_json(s):
    m = re.search(r"\{[^{}]*\"score\"[^{}]*\}", s, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in judge output: {s[:200]}")
    return json.loads(m.group(0))


def judge_gemini(prompt, key):
    url = GEMINI_URL.format(key=key)
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 200}}
    r = _post(url, body, {"Content-Type": "application/json"}, timeout=60)
    txt = r["candidates"][0]["content"]["parts"][0]["text"]
    return _extract_json(txt)


def judge_claude_cli(prompt, timeout=120):
    """Call local `claude -p` CLI. Returns dict {score, reason}."""
    # Use stdin for the prompt to avoid shell arg-size / quoting issues.
    proc = subprocess.run(
        [CLAUDE_BIN, "-p", prompt],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude cli rc={proc.returncode} stderr={proc.stderr[:300]}")
    return _extract_json(proc.stdout)


def query_v2(q, collection, k=10):
    body = {"question": q, "k": k, "collection": collection}
    req = urllib.request.Request(API, method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def _evidence_str(resp, n=5):
    hits = resp.get("hits") or resp.get("results") or []
    out = []
    for h in hits[:n]:
        p = h.get("payload") or h
        out.append(f"- [{p.get('section_ref', '')}] {str(p.get('text', ''))[:400]}")
    return "\n".join(out)


def _judge_one(g, collection, gkey):
    """Run query + both judges for one gold item. Returns (row, errors)."""
    errors = []
    try:
        resp = query_v2(g["query"], collection)
        answer = resp.get("answer") or resp.get("response") or ""
        evidence = _evidence_str(resp)
    except Exception as e:
        err = f"query: {type(e).__name__}: {e}"
        return ({"id": g.get("id"), "error": err},
                [{"id": g.get("id"), "stage": "query", "error": err}])

    prompt = JUDGE_PROMPT.format(
        question=g["query"],
        expected=json.dumps({k: g.get(k) for k in
            ("expected_section_refs", "expected_terms", "expected_doc_ids")
            if g.get(k)}),
        answer=answer, evidence=evidence)

    try: gj = judge_gemini(prompt, gkey)
    except Exception as e:
        gj = {"score": None, "reason": f"err:{e}"}
        errors.append({"id": g.get("id"), "stage": "judge_gemini",
                       "error": f"{type(e).__name__}: {e}"})
    try: cj = judge_claude_cli(prompt)
    except Exception as e:
        cj = {"score": None, "reason": f"err:{e}"}
        errors.append({"id": g.get("id"), "stage": "judge_claude_cli",
                       "error": f"{type(e).__name__}: {e}"})

    return ({"id": g.get("id"), "gemini": gj, "claude": cj}, errors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="cbic_v2_gst50")
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--disagree-thresh", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel worker threads (default 8)")
    ap.add_argument("--allow-errors", type=int, default=0)
    args = ap.parse_args()

    gkey = os.environ.get("GEMINI_API_KEY")
    if not gkey:
        print("[G2p] GEMINI_API_KEY required (source D:/_gpu_rig_ai/.env)", file=sys.stderr)
        sys.exit(2)
    if not Path(CLAUDE_BIN).exists():
        print(f"[G2p] claude CLI not found at {CLAUDE_BIN}", file=sys.stderr)
        sys.exit(2)

    gold = json.loads(args.gold.read_text()).get("queries", [])
    if args.limit: gold = gold[:args.limit]

    rows, errors = [], []
    g_scores, c_scores = [], []
    done_n = 0
    lock = Lock()
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_judge_one, g, args.collection, gkey): g for g in gold}
        for fut in as_completed(futs):
            row, errs = fut.result()
            with lock:
                rows.append(row)
                errors.extend(errs)
                if "gemini" in row and row["gemini"].get("score") is not None:
                    g_scores.append(row["gemini"]["score"])
                if "claude" in row and row["claude"].get("score") is not None:
                    c_scores.append(row["claude"]["score"])
                done_n += 1
                if done_n % 10 == 0 or done_n == len(gold):
                    el = time.time() - t0
                    print(f"[G2p] {done_n}/{len(gold)}  errors={len(errors)}  "
                          f"elapsed={el:.1f}s", flush=True)

    g_mean = sum(g_scores) / len(g_scores) if g_scores else 0.0
    c_mean = sum(c_scores) / len(c_scores) if c_scores else 0.0
    both_pass = g_mean >= args.threshold and c_mean >= args.threshold
    agreement = abs(g_mean - c_mean) <= 0.1
    passed = both_pass and agreement and len(errors) <= args.allow_errors

    # per-item disagreement flags
    review_count = 0
    for r in rows:
        gs = r.get("gemini", {}).get("score") if "gemini" in r else None
        cs = r.get("claude", {}).get("score") if "claude" in r else None
        if gs is not None and cs is not None and abs(gs - cs) > args.disagree_thresh:
            r["needs_review"] = True
            review_count += 1

    out = {"gate": "G2_parallel", "collection": args.collection,
           "n": len(gold), "n_scored_gemini": len(g_scores),
           "n_scored_claude": len(c_scores),
           "gemini_mean": round(g_mean, 4), "claude_mean": round(c_mean, 4),
           "threshold": args.threshold,
           "agreement_ok": agreement, "both_pass_threshold": both_pass,
           "errors": len(errors), "allow_errors": args.allow_errors,
           "needs_review_count": review_count,
           "workers": args.workers,
           "wall_seconds": round(time.time() - t0, 1),
           "pass_gate": passed, "ts": time.time(), "per_item": rows}
    args.out.write_text(json.dumps(out, indent=2))

    if errors:
        fail_path = str(args.out) + ".errors.json"
        with open(fail_path, "w", encoding="utf-8") as f:
            json.dump({"ts": time.time(), "count": len(errors), "records": errors}, f, indent=2)
        print(f"[G2p ERRORS] {len(errors)} items errored — see {fail_path}")
        for e in errors[:10]:
            print(f"  - {e['id']} [{e['stage']}]: {e['error']}")

    print(json.dumps({k: v for k, v in out.items() if k != "per_item"}, indent=2))
    if len(errors) > args.allow_errors:
        print(f"[G2p FAIL] {len(errors)} errors > allow_errors={args.allow_errors}")
        sys.exit(3)
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
