#!/usr/bin/env python3
"""Stage H — G4 Refusal gate: adversarial OOC refusal via theta threshold.

Reads v2_adversarial.json and theta_v2.json (produced by theta_tune.py).
For each adversarial query, calls /query against cbic_v2 and verifies that
the top hit's score is below θ (i.e. the system would refuse).

Pass: refusal_rate >= 1.00 per SPEC §1 G4 (100% refusal on OOC adversarials).
Override via --threshold for diagnostic/tune runs.

Writes gate_g4_result.json. Exits 0/2.
"""
from __future__ import annotations
import argparse, json, sys, time, urllib.request, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
API = "http://127.0.0.1:9500/retrieve"
DEFAULT_ADV = HERE.parent / "eval" / "v2_adversarial.json"  # reingest_spec/eval/
DEFAULT_THETA = HERE.parent / "theta_v2.json"
OUT = HERE / "gate_g4_result.json"


def _query(q, collection, k=10):
    body = {"question": q, "k": k, "collection": collection}
    req = urllib.request.Request(API, method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read())


def _top_score(resp):
    hits = resp.get("hits") or resp.get("results") or []
    return max((h.get("rerank_score") if h.get("rerank_score") is not None else h.get("score", 0) for h in hits), default=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="cbic_v2")
    ap.add_argument("--adv", type=Path, default=DEFAULT_ADV)
    ap.add_argument("--theta-file", type=Path, default=DEFAULT_THETA)
    ap.add_argument("--theta", type=float, default=None,
                    help="override theta; else read from --theta-file")
    ap.add_argument("--threshold", type=float, default=1.00,
                    help="refusal-rate pass threshold (SPEC §1 G4 = 1.00)")
    ap.add_argument("--out", type=Path, default=OUT)
    # 2026-04-24 A-to-Z failure reporting: an errored adversarial query is NOT
    # a refusal and NOT a leak — it is a BROKEN MEASUREMENT. Strict default:
    # any per-query error → non-zero exit. --allow-errors N for flaky infra.
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel /retrieve workers (default 8). 2026-04-25.")
    ap.add_argument("--allow-errors", type=int, default=0,
                    help="Max permitted per-query errors before the gate fails (default 0 = strict)")
    args = ap.parse_args()

    if args.theta is not None:
        theta = args.theta
    else:
        if not args.theta_file.exists():
            print(f"[G4] missing theta file {args.theta_file} — run theta_tune.py first",
                  file=sys.stderr)
            sys.exit(2)
        t = json.loads(args.theta_file.read_text())
        theta = t.get("theta")
        if theta is None:
            print("[G4] theta_v2.json has no 'theta' (infeasible tune) — G4 cannot run",
                  file=sys.stderr)
            sys.exit(2)

    adv = json.loads(args.adv.read_text()).get("queries", [])
    per_map: dict = {}
    refused = 0
    errors: list = []
    lock = threading.Lock()
    done_n = 0

    def _run_one(idx, q):
        qtxt = q["query"] if isinstance(q, dict) else q
        qid = q.get("id") if isinstance(q, dict) else f"adv_{idx}"
        try:
            sc = _top_score(_query(qtxt, args.collection))
            return idx, qid, sc, None
        except Exception as e:
            return idx, qid, None, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(_run_one, i, q) for i, q in enumerate(adv)]
        for fut in as_completed(futs):
            idx, qid, sc, err = fut.result()
            with lock:
                done_n += 1
                if err is not None:
                    per_map[idx] = {"id": qid, "error": err, "refused": False}
                    errors.append({"id": qid, "error": err})
                else:
                    is_refusal = sc < theta
                    if is_refusal:
                        refused += 1
                    per_map[idx] = {"id": qid, "top_score": round(sc, 4), "refused": is_refusal}
                if done_n % 10 == 0:
                    print(f"[G4] {done_n}/{len(adv)}  refused={refused} errors={len(errors)}", flush=True)
    per = [per_map[i] for i in range(len(adv))]

    n = len(adv)
    rate = refused / n if n else 0.0
    passed = rate >= args.threshold and len(errors) <= args.allow_errors
    out = {"gate": "G4", "collection": args.collection,
           "theta": theta, "n": n, "refused": refused,
           "refusal_rate": round(rate, 4),
           "threshold": args.threshold,
           "errors": len(errors), "allow_errors": args.allow_errors,
           "pass_gate": passed, "ts": time.time(),
           "leaks": [p for p in per if not p.get("refused") and "error" not in p][:200],
           "per_item": per}
    args.out.write_text(json.dumps(out, indent=2))
    if errors:
        fail_path = str(args.out) + ".errors.json"
        with open(fail_path, "w", encoding="utf-8") as f:
            json.dump({"ts": time.time(), "count": len(errors), "records": errors}, f, indent=2)
        print(f"[G4 INFRA-ERRORS] {len(errors)} queries errored — see {fail_path}")
        for e in errors[:10]:
            print(f"  - {e['id']}: {e['error']}")
    print(json.dumps({k: v for k, v in out.items() if k != "per_item"}, indent=2))
    if len(errors) > args.allow_errors:
        print(f"[G4 FAIL] {len(errors)} errors > allow_errors={args.allow_errors} "
              f"— refusing to report aggregate on incomplete sample.")
        sys.exit(3)
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
