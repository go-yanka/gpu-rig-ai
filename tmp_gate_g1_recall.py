#!/usr/bin/env python3
"""Stage E — G1 Accuracy gate: recall@10.

Reads v2_gold.json (produced by Stage C). For each gold query, calls /query against
cbic_v2 and checks whether the expected section_ref or doc_id appears in top-10.

Gold-item schema assumed (tolerant — accepts any subset):
  { "id": "q001", "query": "...", "category": "gst",
    "expected_doc_ids": ["..."],              # list of doc_ids that satisfy the query
    "expected_section_refs": ["Section 16(2)"], # optional
    "expected_chunk_ids": ["..."]              # optional, strictest
  }
A hit is any of: chunk_id in expected_chunk_ids, doc_id in expected_doc_ids,
or section_ref (case-insensitive substring) matches any expected_section_refs.

Pass: recall@10 >= 0.95.
Writes gate_g1_result.json. Exits 0/2.

Pattern reused from probe_v16_theta.py (API call shape) and theta_tune.py.
"""
from __future__ import annotations
import argparse, json, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).parent
API = "http://127.0.0.1:9500/query"
RETRIEVE_API = "http://127.0.0.1:9500/retrieve"
DEFAULT_GOLD = HERE.parent / "eval" / "v2_gold.json"  # reingest_spec/eval/
OUT = HERE / "gate_g1_result.json"


def _query(q, collection, k=10, endpoint=API):
    # H2 fix: use the real `collection` field on QueryReq (added to api.py),
    # not a fabricated `_collection` filter that Qdrant silently ignored.
    body = {"question": q, "k": k, "collection": collection}
    req = urllib.request.Request(endpoint, method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read())


def _hits(resp):
    return resp.get("hits") or resp.get("results") or []


def _hit_summary(h, max_text=180):
    """Compact diagnostic record for one retrieved hit."""
    p = h.get("payload") or h
    txt = str(p.get("text") or "")[:max_text].replace("\n", " ")
    return {
        "doc_id": p.get("doc_id"),
        "section_ref": p.get("section_ref"),
        "is_table": p.get("is_table"),
        "doc_type": p.get("doc_type"),
        "category": p.get("category"),
        "score": h.get("score"),
        "chunk_id": (h.get("id") or p.get("chunk_id"))[:16] if (h.get("id") or p.get("chunk_id")) else None,
        "text_preview": txt,
    }


def _norm_gold(g):
    """H3 fix: normalize singular/plural schema differences between curator
    output (expected_doc_id, expected_section, expected_chunk_id — all scalar)
    and evaluator expectations (plural lists). Also drops expected_chunk_id
    entirely because v1 chunk_ids are INT hashes, v2 chunk_ids are SHA256
    strings — they will never match; chunk-id fallback is meaningless."""
    docs = g.get("expected_doc_ids") or ([g["expected_doc_id"]] if g.get("expected_doc_id") else [])
    secs = g.get("expected_section_refs") or ([g["expected_section"]] if g.get("expected_section") else [])
    text = g.get("expected_text") or ""
    return {"docs": {str(d) for d in docs if d},
            "secs": [str(s).lower() for s in secs if s],
            "text": text.strip().lower()}


def _is_hit(gold_item, hits):
    g = _norm_gold(gold_item)
    for h in hits:
        payload = h.get("payload") or h
        did = str(payload.get("doc_id") or "")
        sref = str(payload.get("section_ref") or "").lower()
        ctext = str(payload.get("text") or "").lower()
        # Primary: doc_id exact match (stable v1↔v2 per ingest_v2.phase1)
        if did and did in g["docs"]:
            return True
        # Secondary: section ref bidirectional substring
        if sref and any((es and (es in sref or sref in es)) for es in g["secs"]):
            return True
        # Tertiary: expected_text substring appears in retrieved chunk
        if g["text"] and len(g["text"]) > 20 and g["text"] in ctext:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="cbic_v2")
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--retrieve-only", action="store_true", help="Hit /retrieve (no LLM) instead of /query")
    # 2026-04-24 A-to-Z failure reporting. A query that errors (API timeout,
    # 5xx, JSON decode) is NOT a miss — it is a BROKEN MEASUREMENT. Strict
    # default: any per-query error → exit non-zero. --allow-errors N permits
    # at most N errored queries (for flaky-infra debugging only).
    ap.add_argument("--allow-errors", type=int, default=0,
                    help="Max permitted per-query errors before the gate fails (default 0 = strict)")
    args = ap.parse_args()

    gold = json.loads(args.gold.read_text()).get("queries", [])
    per_item = []
    miss_records: list = []  # 2026-04-24: full diagnostic per miss
    hits_n = 0
    errors: list = []  # {id, error}
    for i, g in enumerate(gold):
        q = g["query"]
        try:
            resp = _query(q, args.collection, k=args.k, endpoint=(RETRIEVE_API if args.retrieve_only else API))
            retrieved = _hits(resp)
            ok = _is_hit(g, retrieved)
            err = None
        except Exception as e:
            ok = False
            err = f"{type(e).__name__}: {e}"
            retrieved = []
            errors.append({"id": g.get("id"), "error": err})
        per_item.append({"id": g.get("id"), "hit": ok, "error": err})
        if ok:
            hits_n += 1
        else:
            # 2026-04-24 verbose miss logging — captures everything needed
            # to diagnose without re-running. query, gold, top-10 with
            # section_ref/doc_id/is_table/doc_type/score/preview.
            miss_records.append({
                "idx": i,
                "query": q,
                "category": g.get("category"),
                "expected_doc_id": g.get("expected_doc_id") or g.get("expected_doc_ids"),
                "expected_section": g.get("expected_section") or g.get("expected_section_refs"),
                "expected_chunk_id": g.get("expected_chunk_id"),
                "_source": g.get("_source"),
                "error": err,
                "top_k": [_hit_summary(h) for h in retrieved[:args.k]],
            })
        if i % 25 == 0:
            print(f"[G1] {i}/{len(gold)}  hits={hits_n} errors={len(errors)}", flush=True)

    n = len(gold)
    recall = hits_n / n if n else 0.0
    passed = recall >= args.threshold and len(errors) <= args.allow_errors
    out = {"gate": "G1", "collection": args.collection,
           "n": n, "hits": hits_n, "recall_at_k": round(recall, 4),
           "k": args.k, "threshold": args.threshold,
           "errors": len(errors), "allow_errors": args.allow_errors,
           "pass_gate": passed, "ts": time.time(),
           "misses": [p["id"] for p in per_item if not p["hit"]][:200],
           "per_item": per_item}
    args.out.write_text(json.dumps(out, indent=2))
    # 2026-04-24: write detailed miss diagnostics alongside main result.
    if miss_records:
        miss_path = str(args.out).replace(".json", ".misses.json")
        with open(miss_path, "w", encoding="utf-8") as f:
            json.dump({"ts": time.time(), "collection": args.collection,
                       "n": n, "miss_count": len(miss_records),
                       "records": miss_records}, f, indent=2)
        print(f"[G1 MISSES] {len(miss_records)} miss diagnostics written to {miss_path}")
    # Write structured failure artifact when there are infra errors.
    if errors:
        fail_path = str(args.out) + ".errors.json"
        with open(fail_path, "w", encoding="utf-8") as f:
            json.dump({"ts": time.time(), "count": len(errors), "records": errors}, f, indent=2)
        print(f"[G1 INFRA-ERRORS] {len(errors)} queries errored — see {fail_path}")
        for e in errors[:10]:
            print(f"  - {e['id']}: {e['error']}")
    print(json.dumps({k: v for k, v in out.items() if k != "per_item"}, indent=2))
    if len(errors) > args.allow_errors:
        print(f"[G1 FAIL] {len(errors)} errors > allow_errors={args.allow_errors} "
              f"— refusing to report aggregate on incomplete sample.")
        sys.exit(3)
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
