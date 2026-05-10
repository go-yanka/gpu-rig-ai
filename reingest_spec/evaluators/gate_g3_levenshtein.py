#!/usr/bin/env python3
"""Stage G — G3 Evidence gate: near-miss Levenshtein recovery.

When a gold expected_section (or expected_text) is NOT in top-10 by exact/ID
match (per G1 logic), check normalized Levenshtein similarity between the
retrieved top-1 text and the expected section text. If similarity >= 0.95 we
count it as a near-miss hit (catches OCR noise / paraphrase cases that
semantic + exact matching missed).

Normalization: NFKC + lowercase + collapse whitespace (per SPEC.md §1 G3 row
and R12 risk mitigation).

THRESHOLD DRIFT FIX 2026-04-23: default --sim-threshold raised from 0.85 → 0.95
to match SPEC.md §1 G3 row verbatim ("normalized Levenshtein ≥0.95 fallback").
The 0.85 was a lingering pre-amendment value; flagged by codification audit.
Do NOT lower without a SPEC amendment + JOURNAL entry.

Pass: (G1_hits + G3_near_misses) / N >= 0.95.
Writes gate_g3_result.json. Exits 0/2.

Gold-item schema additions (tolerant):
  expected_text: "<the authoritative section text>"  # optional; if absent we
    try expected_section_refs joined as a proxy pattern.
"""
from __future__ import annotations
import argparse, json, sys, time, unicodedata, urllib.request, re, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
API = "http://127.0.0.1:9500/query"
DEFAULT_GOLD = HERE.parent / "eval" / "v2_gold.json"  # reingest_spec/eval/
OUT = HERE / "gate_g3_result.json"


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _lev(a: str, b: str) -> int:
    """Iterative Levenshtein. O(len(a)*len(b)). Fine for chunk-sized strings."""
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j-1] + 1, prev[j] + 1, prev[j-1] + cost)
        prev = cur
    return prev[-1]


def _sim(a: str, b: str) -> float:
    a, b = _normalize(a), _normalize(b)
    if not a and not b: return 1.0
    m = max(len(a), len(b))
    if m == 0: return 1.0
    return 1.0 - _lev(a, b) / m


def _query(q, collection, k=10, endpoint=None):
    url = endpoint or API
    body = {"question": q, "k": k, "collection": collection}
    req = urllib.request.Request(url, method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _hits(resp):
    return resp.get("hits") or resp.get("results") or []


def _g1_hit(g, hits):
    exp_chunks = set(g.get("expected_chunk_ids") or ([str(g["expected_chunk_id"])] if g.get("expected_chunk_id") else []))
    exp_docs = set(g.get("expected_doc_ids") or ([g["expected_doc_id"]] if g.get("expected_doc_id") else []))
    exp_secs = [s.lower() for s in (g.get("expected_section_refs") or ([g["expected_section"]] if g.get("expected_section") else []))]
    for h in hits:
        p = h.get("payload") or h
        cid = str(p.get("chunk_id") or h.get("id") or "")
        did = str(p.get("doc_id") or "")
        sref = str(p.get("section_ref") or "").lower()
        if cid and cid in exp_chunks: return True
        if did and did in exp_docs: return True
        # Defect D fix (2026-04-26): shared-PDF chunks store all sharing
        # doc_ids in payload.linked_doc_ids. Mirrors gate_g1_recall._is_hit.
        ldids = p.get("linked_doc_ids") or []
        if ldids and any(d in exp_docs for d in ldids):
            return True
        if sref and any(es and (es in sref or sref in es) for es in exp_secs):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="cbic_v2")
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--sim-threshold", type=float, default=0.95)
    ap.add_argument("--pass-threshold", type=float, default=0.95)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--retrieve-only", action="store_true", help="Hit /retrieve (dense, ~40ms) instead of /query (LLM, ~5s). Needed for fast gates.")
    ap.add_argument("--out", type=Path, default=OUT)
    # 2026-04-24 A-to-Z failure reporting: strict default.
    ap.add_argument("--allow-errors", type=int, default=0,
                    help="Max per-query infra errors before gate fails (default 0 = strict)")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel /retrieve workers (default 8). Mirrors G1 parallelization 2026-04-25.")
    args = ap.parse_args()

    gold = json.loads(args.gold.read_text()).get("queries", [])
    per_map: dict = {}
    g1_hits = g3_saves = 0
    errors: list = []
    endpoint = "http://127.0.0.1:9500/retrieve" if args.retrieve_only else API
    lock = threading.Lock()
    done_n = 0

    def _run_one(idx, g):
        try:
            hits = _hits(_query(g["query"], args.collection, k=args.k, endpoint=endpoint))
            return idx, g, hits, None
        except Exception as e:
            return idx, g, None, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(_run_one, i, g) for i, g in enumerate(gold)]
        for fut in as_completed(futs):
            idx, g, hits, err = fut.result()
            with lock:
                done_n += 1
                if err is not None:
                    per_map[idx] = {"id": g.get("id"), "error": err, "classification": "error"}
                    errors.append({"id": g.get("id"), "error": err})
                elif _g1_hit(g, hits):
                    g1_hits += 1
                    per_map[idx] = {"id": g.get("id"), "classification": "g1_hit"}
                else:
                    expected_text = g.get("expected_text") or " ".join(
                        g.get("expected_section_refs") or []) or " ".join(
                        g.get("expected_terms") or [])
                    top1_text = ""
                    if hits:
                        p = hits[0].get("payload") or hits[0]
                        top1_text = str(p.get("text") or "")
                    sim = _sim(expected_text, top1_text) if expected_text else 0.0
                    if sim >= args.sim_threshold:
                        g3_saves += 1
                        per_map[idx] = {"id": g.get("id"), "classification": "g3_near_miss",
                                        "sim": round(sim, 3)}
                    else:
                        per_map[idx] = {"id": g.get("id"), "classification": "miss",
                                        "sim": round(sim, 3)}
                if done_n % 25 == 0:
                    print(f"[G3] {done_n}/{len(gold)}  g1={g1_hits} g3_saves={g3_saves}", flush=True)
    per = [per_map[i] for i in range(len(gold))]

    n = len(gold)
    combined = (g1_hits + g3_saves) / n if n else 0.0
    passed = combined >= args.pass_threshold and len(errors) <= args.allow_errors
    out = {"gate": "G3", "collection": args.collection,
           "n": n, "g1_hits": g1_hits, "g3_near_misses": g3_saves,
           "combined_recall": round(combined, 4),
           "sim_threshold": args.sim_threshold,
           "pass_threshold": args.pass_threshold,
           "errors": len(errors), "allow_errors": args.allow_errors,
           "pass_gate": passed, "ts": time.time(),
           "per_item": per}
    args.out.write_text(json.dumps(out, indent=2))
    if errors:
        fail_path = str(args.out) + ".errors.json"
        with open(fail_path, "w", encoding="utf-8") as f:
            json.dump({"ts": time.time(), "count": len(errors), "records": errors}, f, indent=2)
        print(f"[G3 INFRA-ERRORS] {len(errors)} queries errored — see {fail_path}")
        for e in errors[:10]:
            print(f"  - {e['id']}: {e['error']}")
    print(json.dumps({k: v for k, v in out.items() if k != "per_item"}, indent=2))
    if len(errors) > args.allow_errors:
        print(f"[G3 FAIL] {len(errors)} errors > allow_errors={args.allow_errors} "
              f"— refusing to report aggregate on incomplete sample.")
        sys.exit(3)
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
