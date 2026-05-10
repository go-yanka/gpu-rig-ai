#!/usr/bin/env python3
"""V16: theta_retrieve threshold stability.
Score distribution of top-1 for (170 gold queries) vs (20 OOC queries).
Pass: OOC max score < gold min score * 0.9 -> clean separation -> pick theta in middle.
Needs gold + adversarial sets locally; runs from laptop via Qdrant search.
Assumes embedding server reachable at 127.0.0.1:9088 on rig — if not, use rig shell.
"""
import json, urllib.request, sys
from pathlib import Path

QDRANT = "http://192.168.1.107:6343"
COLL = "cbic_v1"
GOLD_PATH = Path("/opt/indian-legal-ai/rag/cbic_rag/eval_set.json")
ADV_PATH = Path("/opt/indian-legal-ai/rag/cbic_rag/eval_set_adversarial.json")
OUT = Path("D:/_gpu_rig_ai/reingest_spec/v16_result.json")

# Note: this probe needs embeddings for queries. Easiest path is to call the rig's
# /query endpoint (which does routing+embedding internally) rather than re-implement.
# That endpoint is localhost-only — so this probe MUST be run on the rig.

API = "http://127.0.0.1:9500/query"

def query(q, k=10):
    body = {"query": q, "k": k}
    req = urllib.request.Request(API, method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())

def score_of(resp):
    hits = resp.get("hits") or resp.get("results") or []
    return max((h.get("score", 0) for h in hits), default=0)

def main():
    gold = json.loads(GOLD_PATH.read_text())
    adv = json.loads(ADV_PATH.read_text()) if ADV_PATH.exists() else {"queries": []}

    gold_qs = [q["query"] for q in gold.get("queries", [])]
    adv_qs = [q["query"] for q in adv.get("queries", [])]
    print(f"[V16] gold={len(gold_qs)} adv={len(adv_qs)}")

    gold_scores = []
    for i, q in enumerate(gold_qs):
        try:
            gold_scores.append(score_of(query(q)))
        except Exception as e:
            print(f"  gold[{i}] err {e}")
        if i % 20 == 0: print(f"  gold {i}/{len(gold_qs)}")

    adv_scores = []
    for i, q in enumerate(adv_qs):
        try:
            adv_scores.append(score_of(query(q)))
        except Exception as e:
            print(f"  adv[{i}] err {e}")

    gs = sorted(gold_scores); as_ = sorted(adv_scores)
    def pct(arr, p): return arr[int(len(arr)*p)] if arr else 0
    summary = {
        "probe": "V16",
        "gold_n": len(gs), "adv_n": len(as_),
        "gold_min": gs[0] if gs else None,
        "gold_p10": pct(gs, 0.1), "gold_p50": pct(gs, 0.5),
        "adv_max": as_[-1] if as_ else None,
        "adv_p90": pct(as_, 0.9), "adv_p50": pct(as_, 0.5),
        "separation_ok": bool(gs and as_ and as_[-1] < gs[0] * 0.9),
        "proposed_theta": round((gs[0] + as_[-1]) / 2, 3) if gs and as_ else None,
        "pass_gate": bool(gs and as_ and as_[-1] < gs[0] * 0.9),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
