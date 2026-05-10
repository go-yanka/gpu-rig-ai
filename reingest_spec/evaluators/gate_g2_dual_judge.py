#!/usr/bin/env python3
"""Stage F — G2 Reasoning gate: dual-judge ensemble (Gemini + Claude).

For each gold query, call /query against cbic_v2 to get the v2 answer, then ask
both Gemini (2.0 Flash via google generativeai API) and Claude (Anthropic API)
to score factual correctness on a 0-1 scale.

Pass: BOTH judges' mean score >= 0.95 AND they agree (|gemini - claude| <= 0.1
aggregate, or per-item disagreement >0.25 flagged as "needs_review").

API keys from env: GEMINI_API_KEY, ANTHROPIC_API_KEY.
Writes gate_g2_result.json. Exits 0/2.

Pattern reused from probe_v17_judge.py (judge call shape).
"""
from __future__ import annotations
import argparse, json, os, sys, time, urllib.request, urllib.error, re, subprocess
from pathlib import Path

HERE = Path(__file__).parent
API = "http://127.0.0.1:9500/query"
DEFAULT_GOLD = HERE.parent / "eval" / "v2_gold.json"  # reingest_spec/eval/
OUT = HERE / "gate_g2_result.json"

# M1 resolved: use current-gen judges that actually exist.
# Override via env if the rig has different model access.
GEMINI_MODEL = os.environ.get("GEMINI_JUDGE_MODEL", "gemini-2.5-pro")
CLAUDE_MODEL = os.environ.get("CLAUDE_JUDGE_MODEL", "claude-sonnet-4-5")
GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              f"{GEMINI_MODEL}:generateContent?key={{key}}")
CLAUDE_URL = "https://api.anthropic.com/v1/messages"

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


def judge_claude(prompt, key):
    """HTTP path (legacy) — used only if CLAUDE_USE_CLI=0."""
    body = {"model": CLAUDE_MODEL,
            "max_tokens": 200, "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}]}
    headers = {"Content-Type": "application/json",
               "x-api-key": key,
               "anthropic-version": "2023-06-01"}
    r = _post(CLAUDE_URL, body, headers, timeout=90)
    txt = r["content"][0]["text"]
    return _extract_json(txt)


def judge_claude_cli(prompt):
    """2026-05-08: Claude CLI shellout (matches SPEC §1: 'Gemini + Claude CLI').
    Uses local `claude` binary which authenticates via Pro/Max subscription.
    No ANTHROPIC_API_KEY required.
    """
    res = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True, text=True, timeout=180,
    )
    if res.returncode != 0:
        raise RuntimeError(f"claude CLI exit {res.returncode}: {(res.stderr or '')[:300]}")
    out = res.stdout or ""
    return _extract_json(out)


def query_v2(q, collection, k=10):
    body = {"question": q, "k": k, "collection": collection}
    req = urllib.request.Request(API, method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def _evidence_str(resp, n=5):
    hits = resp.get("hits") or resp.get("results") or []
    out = []
    for h in hits[:n]:
        p = h.get("payload") or h
        out.append(f"- [{p.get('section_ref', '')}] {str(p.get('text', ''))[:400]}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="cbic_v2")
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--disagree-thresh", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--allow-errors", type=int, default=0,
                    help="Max permitted per-item errors before the gate fails (default 0 = strict)")
    args = ap.parse_args()

    gkey = os.environ.get("GEMINI_API_KEY")
    akey = os.environ.get("ANTHROPIC_API_KEY")  # only needed if CLAUDE_USE_CLI=0
    use_cli = os.environ.get("CLAUDE_USE_CLI", "1") == "1"
    if not gkey:
        print("[G2] GEMINI_API_KEY required", file=sys.stderr)
        sys.exit(2)
    if not use_cli and not akey:
        print("[G2] ANTHROPIC_API_KEY required when CLAUDE_USE_CLI=0", file=sys.stderr)
        sys.exit(2)

    gold = json.loads(args.gold.read_text()).get("queries", [])
    if args.limit: gold = gold[:args.limit]

    # 2026-04-24 A-to-Z failure reporting. The original code silently recorded
    # score=None on ANY judge error ("err:...") and aggregated mean over
    # whatever partial set of scored queries existed — so 50 judge timeouts
    # out of 200 would still yield a "mean" on the remaining 150, and the
    # gate could falsely pass on an incomplete denominator. Now: judge errors
    # are counted separately; the gate FAILS loudly unless --allow-errors N
    # is supplied AND the count is below N.
    rows = []
    g_scores, c_scores = [], []
    errors: list = []
    for i, g in enumerate(gold):
        try:
            resp = query_v2(g["query"], args.collection)
            answer = resp.get("answer") or resp.get("response") or ""
            evidence = _evidence_str(resp)
        except Exception as e:
            err = f"query: {type(e).__name__}: {e}"
            rows.append({"id": g.get("id"), "error": err})
            errors.append({"id": g.get("id"), "stage": "query", "error": err})
            continue

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
        try:
            cj = judge_claude_cli(prompt) if use_cli else judge_claude(prompt, akey)
        except Exception as e:
            cj = {"score": None, "reason": f"err:{e}"}
            errors.append({"id": g.get("id"), "stage": "judge_claude",
                           "error": f"{type(e).__name__}: {e}"})

        gs = gj.get("score"); cs = cj.get("score")
        needs_review = (gs is not None and cs is not None
                        and abs(gs - cs) > args.disagree_thresh)
        if gs is not None: g_scores.append(gs)
        if cs is not None: c_scores.append(cs)
        rows.append({"id": g.get("id"), "gemini": gj, "claude": cj,
                     "needs_review": needs_review})
        if i % 10 == 0:
            print(f"[G2] {i}/{len(gold)}  errors={len(errors)}", flush=True)

    g_mean = sum(g_scores) / len(g_scores) if g_scores else 0.0
    c_mean = sum(c_scores) / len(c_scores) if c_scores else 0.0
    both_pass = g_mean >= args.threshold and c_mean >= args.threshold
    agreement = abs(g_mean - c_mean) <= 0.1
    allow = getattr(args, "allow_errors", 0)
    passed = both_pass and agreement and len(errors) <= allow

    out = {"gate": "G2", "collection": args.collection,
           "n": len(gold), "n_scored_gemini": len(g_scores),
           "n_scored_claude": len(c_scores),
           "gemini_mean": round(g_mean, 4), "claude_mean": round(c_mean, 4),
           "threshold": args.threshold,
           "agreement_ok": agreement, "both_pass_threshold": both_pass,
           "errors": len(errors), "allow_errors": allow,
           "needs_review_count": sum(1 for r in rows if r.get("needs_review")),
           "pass_gate": passed, "ts": time.time(), "per_item": rows}
    args.out.write_text(json.dumps(out, indent=2))
    if errors:
        fail_path = str(args.out) + ".errors.json"
        with open(fail_path, "w", encoding="utf-8") as f:
            json.dump({"ts": time.time(), "count": len(errors), "records": errors}, f, indent=2)
        print(f"[G2 ERRORS] {len(errors)} items errored — see {fail_path}")
        for e in errors[:10]:
            print(f"  - {e['id']} [{e['stage']}]: {e['error']}")
    print(json.dumps({k: v for k, v in out.items() if k != "per_item"}, indent=2))
    if len(errors) > allow:
        print(f"[G2 FAIL] {len(errors)} errors > allow_errors={allow} "
              f"— refusing to report pass on incomplete judge coverage.")
        sys.exit(3)
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
