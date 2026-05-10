#!/usr/bin/env python3
"""theta_tune.py (B6) — calibrate score threshold θ for cbic_v2.

Extends probe_v16_theta.py. Reads score distributions of gold + adversarial
queries against the v2 collection, picks θ such that:
  (a) gold recall@θ >= TARGET_GOLD_RECALL (default 0.95)
  (b) adversarial refusal@θ >= TARGET_ADV_REFUSE (default 0.90)
If no θ satisfies both, fail hard and write diagnostics.

Writes: reingest_spec/theta_v2.json  (consumed by api_v2 at serve-time)

Run (on rig):
  python3 theta_tune.py --collection cbic_v2
  python3 theta_tune.py --collection cbic_v2 --gold-recall 0.97 --adv-refuse 0.92
"""
from __future__ import annotations
import argparse, json, sys, urllib.request
from pathlib import Path

# 2026-04-25 O8: tune steps configurable via --steps. Default 200 (was hardcoded 400; 200 empirically same theta within 0.01).
STEPS_OVERRIDE = None

HERE = Path(__file__).parent
DEFAULT_GOLD = Path("/opt/indian-legal-ai/rag/cbic_rag/eval_set.json")
DEFAULT_ADV = Path("/opt/indian-legal-ai/rag/cbic_rag/eval_set_adversarial.json")
OUT = HERE / "theta_v2.json"
API = "http://127.0.0.1:9500/retrieve"  # dense-only fast path (2026-04-24)


def _score(resp):
    hits = resp.get("hits") or resp.get("results") or []
    return max((h.get("rerank_score") if h.get("rerank_score") is not None else h.get("score", 0) for h in hits), default=0)


def _query(q, collection, k=10):
    body = {"question": q, "k": k, "collection": collection}
    req = urllib.request.Request(API, method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def collect_scores(queries, collection, tag, workers=8):
    """Parallelized score collection. workers=8 saturates the 5-GPU embed pool + reranker --parallel 4."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    scores = [None] * len(queries)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut2idx = {ex.submit(_query, q, collection): i for i, q in enumerate(queries)}
        for fut in as_completed(fut2idx):
            i = fut2idx[fut]
            try:
                scores[i] = _score(fut.result())
            except Exception as e:
                print(f"  {tag}[{i}] err {e}")
                scores[i] = None
            done += 1
            if done % 50 == 0:
                print(f"  {tag} {done}/{len(queries)}", flush=True)
    return [s for s in scores if s is not None]


def pick_theta(gold_scores, adv_scores, target_gold_recall, target_adv_refuse):
    """Sweep θ over a dense grid; return the lowest θ that satisfies both
    constraints (or None if infeasible)."""
    if not gold_scores or not adv_scores:
        return None, {"reason": "empty_scores"}
    lo = min(min(gold_scores), min(adv_scores))
    hi = max(max(gold_scores), max(adv_scores))
    steps = STEPS_OVERRIDE if STEPS_OVERRIDE is not None else 200
    best = None
    for i in range(steps + 1):
        theta = lo + (hi - lo) * i / steps
        gold_recall = sum(1 for s in gold_scores if s >= theta) / len(gold_scores)
        adv_refuse = sum(1 for s in adv_scores if s < theta) / len(adv_scores)
        if gold_recall >= target_gold_recall and adv_refuse >= target_adv_refuse:
            if best is None or theta < best["theta"]:
                best = {"theta": round(theta, 4),
                        "gold_recall": round(gold_recall, 4),
                        "adv_refuse": round(adv_refuse, 4)}
    return best, {"lo": lo, "hi": hi, "steps": steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200, help="Binary search steps (default 200, was 400). 2026-04-25 O8.")
    ap.add_argument("--collection", default="cbic_v2")
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--adv", type=Path, default=DEFAULT_ADV)
    ap.add_argument("--gold-recall", type=float, default=0.95)
    ap.add_argument("--adv-refuse", type=float, default=0.90)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--scores-in", type=Path, default=None,
                    help="skip query phase, load prior scores JSON")
    args = ap.parse_args()
    global STEPS_OVERRIDE
    STEPS_OVERRIDE = args.steps

    if args.scores_in and args.scores_in.exists():
        data = json.loads(args.scores_in.read_text())
        gold_scores, adv_scores = data["gold_scores"], data["adv_scores"]
    else:
        gold = json.loads(args.gold.read_text())["queries"]
        adv = json.loads(args.adv.read_text()).get("queries", [])
        gold_qs = [q["query"] for q in gold]
        adv_qs = [q["query"] for q in adv]
        print(f"[theta_tune] gold={len(gold_qs)} adv={len(adv_qs)} coll={args.collection}")
        gold_scores = collect_scores(gold_qs, args.collection, "gold")
        adv_scores = collect_scores(adv_qs, args.collection, "adv")

    best, diag = pick_theta(gold_scores, adv_scores,
                            args.gold_recall, args.adv_refuse)

    result = {
        "collection": args.collection,
        "target_gold_recall": args.gold_recall,
        "target_adv_refuse": args.adv_refuse,
        "n_gold": len(gold_scores), "n_adv": len(adv_scores),
        "gold_min": min(gold_scores) if gold_scores else None,
        "gold_p50": sorted(gold_scores)[len(gold_scores)//2] if gold_scores else None,
        "adv_max": max(adv_scores) if adv_scores else None,
        "adv_p90": sorted(adv_scores)[int(len(adv_scores)*0.9)] if adv_scores else None,
        "gold_scores": gold_scores,
        "adv_scores": adv_scores,
        "theta": best["theta"] if best else None,
        "achieved_gold_recall": best["gold_recall"] if best else None,
        "achieved_adv_refuse": best["adv_refuse"] if best else None,
        "feasible": best is not None,
        "diag": diag,
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("gold_scores", "adv_scores")}, indent=2))
    if not best:
        print("[theta_tune] INFEASIBLE — gold/adv distributions overlap; fix retrieval first")
        sys.exit(2)
    print(f"[theta_tune] OK θ={best['theta']}  gold_recall={best['gold_recall']}  adv_refuse={best['adv_refuse']}")


if __name__ == "__main__":
    main()
