#!/usr/bin/env python3
"""Stage H — G4 Refusal gate (GROUNDED variant) — corpus-intrinsic refusal.

Replaces the θ-threshold-on-retrieval-score approach. Asks /retrieve with
`grounded:true` which calls qwen3-14b on the top-K reranked chunks and emits
a {"grounded": "yes"|"partial"|"no"} verdict.

Refusal definition (configurable via --refuse-on):
  - "no"          : refuse only when grounded=="no"  (lenient)
  - "no_or_partial": refuse on "no" OR "partial"     (strict, default)

Why this exists (2026-04-25): the CBIC corpus is intentionally cross-statute
(TDS-under-GST, transfer pricing JWG, allied_acts, etc.). A topic classifier
("is this CBIC?") would false-positive-refuse legitimate cross-statute
queries. The right primitive is GROUNDEDNESS — does the evidence actually
contain enough to answer the question? See JOURNAL.md 2026-04-25 (later 9).

Pass: refusal_rate >= --threshold (default 0.95).

Writes gate_g4_grounded_result.json. Exits 0/2.
"""
from __future__ import annotations
import argparse, json, sys, time, urllib.request, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
API = "http://127.0.0.1:9500/retrieve"
DEFAULT_ADV = HERE.parent / "eval" / "v2_adversarial.json"
OUT = HERE / "gate_g4_grounded_result.json"


def _query(q, collection, k=5):
    body = {"question": q, "k": k, "collection": collection, "grounded": True}
    req = urllib.request.Request(API, method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="cbic_v2")
    ap.add_argument("--adv", type=Path, default=DEFAULT_ADV)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.95,
                    help="Required refusal rate (default 0.95).")
    ap.add_argument("--refuse-on", choices=["no", "no_or_partial"],
                    default="no_or_partial",
                    help="What grounded verdicts count as a refusal.")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel /retrieve workers. qwen3 is --parallel 1 so "
                         "true concurrency caps at 1 — keep workers low.")
    ap.add_argument("--allow-errors", type=int, default=0)
    args = ap.parse_args()

    adv = json.loads(args.adv.read_text()).get("queries", [])
    per_map = {}
    refused = 0
    errors = []
    lock = threading.Lock()
    done = 0
    refuse_set = ({"no"} if args.refuse_on == "no"
                  else {"no", "partial"})

    def _run(idx, q):
        qtxt = q["query"] if isinstance(q, dict) else q
        qid = q.get("id") if isinstance(q, dict) else f"adv_{idx}"
        try:
            resp = _query(qtxt, args.collection, k=args.k)
            return idx, qid, resp, None
        except Exception as e:
            return idx, qid, None, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(_run, i, q) for i, q in enumerate(adv)]
        for fut in as_completed(futs):
            idx, qid, resp, err = fut.result()
            with lock:
                done += 1
                if err is not None:
                    per_map[idx] = {"id": qid, "error": err, "refused": False}
                    errors.append({"id": qid, "error": err})
                else:
                    g = resp.get("grounded")
                    is_refusal = g in refuse_set
                    if is_refusal:
                        refused += 1
                    per_map[idx] = {"id": qid, "grounded": g,
                                    "grounded_reason": resp.get("grounded_reason"),
                                    "grounded_ms": resp.get("grounded_ms"),
                                    "refused": is_refusal}
                if done % 10 == 0:
                    print(f"[G4-grounded] {done}/{len(adv)} refused={refused} errors={len(errors)}",
                          flush=True)

    per = [per_map[i] for i in range(len(adv))]
    n = len(adv)
    rate = refused / n if n else 0.0
    passed = rate >= args.threshold and len(errors) <= args.allow_errors
    out = {"gate": "G4-grounded", "collection": args.collection,
           "n": n, "refused": refused, "refusal_rate": round(rate, 4),
           "threshold": args.threshold, "refuse_on": args.refuse_on,
           "errors": len(errors), "allow_errors": args.allow_errors,
           "pass_gate": passed, "ts": time.time(),
           "leaks": [p for p in per if not p.get("refused") and "error" not in p][:200],
           "per_item": per}
    args.out.write_text(json.dumps(out, indent=2))
    if errors:
        fp = str(args.out) + ".errors.json"
        with open(fp, "w") as f:
            json.dump({"ts": time.time(), "count": len(errors), "records": errors}, f, indent=2)
        print(f"[G4-grounded INFRA-ERRORS] see {fp}")
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_item", "leaks")}, indent=2))
    if len(errors) > args.allow_errors:
        sys.exit(3)
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
