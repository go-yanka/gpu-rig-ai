#!/usr/bin/env python3
"""M6 — Hard-negative miner for the Stage-M orchestrator.

For each gold QA pair, retrieves top-K chunks from the given Qdrant collection
via the rag_api_v2 /retrieve endpoint, filters out the POSITIVE (chunks whose
doc_id ∈ expected_doc_ids or chunk_id == expected_chunk_id), and emits the
remaining hits as HARD NEGATIVES.

These hard negatives feed:
  - reranker training (learn to demote high-similarity wrong chunks)
  - G4 adversarial gate construction (confusable queries)
  - retrieval robustness eyeballing during scale-up

CLI (matches run_scale_gate.py M6 stage signature):
  python3 mine_hard_negatives.py \\
    --collection cbic_v2_gst50 \\
    --gold /path/to/pairs_unified.jsonl \\
    --out  /path/to/hardneg_gst50.jsonl \\
    --k 5

Input gold schema (unified — produced by scripts/migrate_pairs_to_v2.py):
  {"query_id": "...", "question": "...", "doc_id": "...",
   "chunk_id": "...", "answer": "...", "query_type": "...",
   "generator": "...", "difficulty": "...", ...}
  Older schemas with expected_doc_ids / expected_chunk_ids / expected_section_refs
  are ALSO tolerated for parity with gate_g1_recall.py's _norm_gold().

Output JSONL (one row per (query, negative) pair):
  {"query_id": "...", "question": "...",
   "positive_doc_id": "...", "positive_chunk_id": "...",
   "neg_doc_id": "...", "neg_chunk_id": "...",
   "neg_section_ref": "...", "neg_score": 0.78, "rank": 2,
   "neg_text_preview": "first 240 chars..."}

Exit codes:
  0 — ran to completion (hardneg file written)
  1 — usage / setup error
  2 — too many retrieval failures (>10%) — caller should halt

Origin: 2026-04-24 GST50 scale test.  Orchestrator SKIP-ed this stage until
miner was installed.  Pattern reused from gate_g1_recall.py retrieval shape.
"""
from __future__ import annotations
import argparse, json, sys, time, urllib.request, urllib.error
from pathlib import Path


DEFAULT_API = "http://127.0.0.1:9500/retrieve"
DEFAULT_FALLBACK_API = "http://127.0.0.1:9500/query"
FAIL_FRAC_HALT = 0.10  # caller halts if >10% queries fail to retrieve


def _norm_gold(g: dict) -> dict:
    """Mirror of gate_g1_recall._norm_gold — tolerate both the unified schema
    (singular doc_id/chunk_id) and older plural expected_* variants."""
    docs = (g.get("expected_doc_ids")
            or ([g["expected_doc_id"]] if g.get("expected_doc_id") else None)
            or ([g["doc_id"]] if g.get("doc_id") else []))
    chunks = (g.get("expected_chunk_ids")
              or ([g["expected_chunk_id"]] if g.get("expected_chunk_id") else None)
              or ([g["chunk_id"]] if g.get("chunk_id") else []))
    secs = (g.get("expected_section_refs")
            or ([g["expected_section"]] if g.get("expected_section") else [])
            or ([g["section_ref"]] if g.get("section_ref") else []))
    return {
        "docs":   {str(d) for d in docs if d},
        "chunks": {str(c) for c in chunks if c},
        "secs":   [str(s).lower() for s in secs if s],
    }


def _get_query_text(g: dict) -> str:
    return (g.get("question") or g.get("query") or g.get("q") or "").strip()


def _get_query_id(g: dict) -> str:
    return str(g.get("query_id") or g.get("id") or g.get("qid") or "")


def _retrieve(api_url: str, question: str, collection: str, k: int, timeout: int = 45):
    body = {"question": question, "k": k, "collection": collection}
    req = urllib.request.Request(
        api_url, method="POST",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _hits(resp) -> list:
    return resp.get("hits") or resp.get("results") or []


def _is_positive(hit: dict, g_norm: dict) -> bool:
    """True if this hit IS the expected positive — we exclude it from negatives."""
    payload = hit.get("payload") or hit
    did  = str(payload.get("doc_id") or "")
    cid  = str(payload.get("chunk_id") or payload.get("id") or hit.get("id") or "")
    sref = str(payload.get("section_ref") or "").lower()
    if cid and cid in g_norm["chunks"]:
        return True
    if did and did in g_norm["docs"]:
        return True
    if sref and any(es and (es in sref or sref in es) for es in g_norm["secs"]):
        return True
    return False


def _load_gold(path: Path) -> list[dict]:
    """Accept either JSONL (one obj per line) or a top-level JSON list."""
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw[0] == "[":
        return json.loads(raw)
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"[mine_hardneg] skip malformed line: {e}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True,
                    help="Qdrant collection to retrieve against, e.g. cbic_v2_gst50")
    ap.add_argument("--gold", type=Path, required=True,
                    help="Unified pairs file (JSONL or JSON list)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSONL for hard negatives")
    ap.add_argument("--k", type=int, default=5,
                    help="Top-K to retrieve; negatives are ranks 2..K after "
                         "the positive is excluded (default 5)")
    ap.add_argument("--api", default=DEFAULT_API,
                    help=f"Retrieval endpoint (default {DEFAULT_API}); will "
                         f"fall back to {DEFAULT_FALLBACK_API} on 404")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="Drop negatives below this score (default 0.0 = keep all)")
    ap.add_argument("--max-neg-per-query", type=int, default=None,
                    help="Cap negatives emitted per query (default: k-1)")
    ap.add_argument("--timeout", type=int, default=45)
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N gold items (smoke)")
    args = ap.parse_args()

    if not args.gold.exists():
        print(f"[mine_hardneg] gold file not found: {args.gold}", file=sys.stderr)
        sys.exit(1)

    gold = _load_gold(args.gold)
    if args.limit:
        gold = gold[: args.limit]
    print(f"[mine_hardneg] {len(gold)} gold items; collection={args.collection} k={args.k}")

    api = args.api
    max_neg = args.max_neg_per_query or max(args.k - 1, 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_fail = 0
    n_neg_total = 0
    n_no_pos = 0   # positive NOT in top-k (retrieval miss — still emit the k as weak negatives)
    t0 = time.time()

    with open(args.out, "w", encoding="utf-8") as fout:
        for i, g in enumerate(gold):
            q = _get_query_text(g)
            qid = _get_query_id(g) or f"idx_{i}"
            if not q:
                n_fail += 1
                continue
            gnorm = _norm_gold(g)
            try:
                resp = _retrieve(api, q, args.collection, args.k, args.timeout)
            except urllib.error.HTTPError as e:
                if e.code == 404 and api != DEFAULT_FALLBACK_API:
                    api = DEFAULT_FALLBACK_API
                    print(f"[mine_hardneg] /retrieve returned 404, falling back to /query")
                    try:
                        resp = _retrieve(api, q, args.collection, args.k, args.timeout)
                    except Exception as e2:
                        print(f"[mine_hardneg] qid={qid} FAIL: {e2}", file=sys.stderr)
                        n_fail += 1
                        continue
                else:
                    print(f"[mine_hardneg] qid={qid} FAIL http {e.code}: {e}", file=sys.stderr)
                    n_fail += 1
                    continue
            except Exception as e:
                print(f"[mine_hardneg] qid={qid} FAIL {type(e).__name__}: {e}", file=sys.stderr)
                n_fail += 1
                continue

            hits = _hits(resp)
            found_pos = any(_is_positive(h, gnorm) for h in hits)
            if not found_pos:
                n_no_pos += 1

            emitted = 0
            for rank, h in enumerate(hits, start=1):
                if _is_positive(h, gnorm):
                    continue
                payload = h.get("payload") or h
                score = float(h.get("score") or h.get("_score") or 0.0)
                if score < args.min_score:
                    continue
                row = {
                    "query_id": qid,
                    "question": q,
                    "positive_doc_id": next(iter(gnorm["docs"]), None),
                    "positive_chunk_id": next(iter(gnorm["chunks"]), None),
                    "neg_doc_id": payload.get("doc_id"),
                    "neg_chunk_id": payload.get("chunk_id") or payload.get("id") or h.get("id"),
                    "neg_section_ref": payload.get("section_ref"),
                    "neg_score": score,
                    "rank": rank,
                    "neg_text_preview": (payload.get("text") or "")[:240],
                    "positive_found_in_topk": found_pos,
                    "generator": g.get("generator"),
                    "query_type": g.get("query_type"),
                    "difficulty": g.get("difficulty"),
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                emitted += 1
                n_neg_total += 1
                if emitted >= max_neg:
                    break

            n_ok += 1
            if (i + 1) % 100 == 0:
                dt = time.time() - t0
                print(f"[mine_hardneg] {i+1}/{len(gold)} ok={n_ok} fail={n_fail} "
                      f"no_pos_in_topk={n_no_pos} negs={n_neg_total} ({dt:.1f}s)")

    dt = time.time() - t0
    summary = {
        "gold_total": len(gold),
        "queries_ok": n_ok,
        "queries_failed": n_fail,
        "queries_pos_missing_in_topk": n_no_pos,
        "negatives_emitted": n_neg_total,
        "elapsed_s": round(dt, 1),
        "fail_frac": round(n_fail / max(len(gold), 1), 4),
        "out": str(args.out),
        "collection": args.collection,
        "k": args.k,
    }
    print(json.dumps(summary, indent=2))

    fail_frac = n_fail / max(len(gold), 1)
    if fail_frac > FAIL_FRAC_HALT:
        print(f"[mine_hardneg] fail_frac={fail_frac:.3f} > {FAIL_FRAC_HALT} — HALT",
              file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
