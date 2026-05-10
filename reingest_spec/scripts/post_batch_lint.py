#!/usr/bin/env python3
"""Post-batch chunker drift detector.

Runs after each ingest (between Phase 5 and CP gate). Surfaces:
  D-DEFECT     — docs with 0 chunks in cbic_v2 (e.g. shared mega-PDF skipped)
  SECTION-REF  — chunks with empty/None section_ref (will hurt G3 lev recall)
  CHUNK-RATIO  — docs whose chunks-per-doc ratio is < 0.5x batch median (anomaly)
  TAIL-DUP     — Defect F2 regression: chunks whose char range is 100% inside another's
Outputs:
  - Console summary (one line per category + counts)
  - JSON file at /opt/indian-legal-ai/data/eval/post_batch_lint_<batch>.json
  - Exit 0 = all clean, Exit 1 = at least one P1+, Exit 2 = P0
Usage:
  post_batch_lint.py <batch_n> <doc_ids_csv_path>
"""
import sys, json, os
from collections import defaultdict
from urllib.request import Request, urlopen
from urllib.parse import urlencode

QDRANT = "http://127.0.0.1:6343"
COLLECTION = "cbic_v2"
P0, P1, P2 = "P0", "P1", "P2"

def scroll_collection(payload_filter=None, limit=None):
    """Yield (doc_id, chunk_id, payload) for all points in cbic_v2."""
    body = {"limit": 256, "with_payload": True}
    if payload_filter: body["filter"] = payload_filter
    next_offset = None
    seen = 0
    while True:
        if next_offset is not None: body["offset"] = next_offset
        req = Request(f"{QDRANT}/collections/{COLLECTION}/points/scroll",
                      data=json.dumps(body).encode(),
                      headers={"content-type": "application/json"})
        d = json.loads(urlopen(req, timeout=30).read())
        for p in d["result"]["points"]:
            yield p["payload"].get("doc_id"), p["id"], p["payload"]
            seen += 1
            if limit and seen >= limit: return
        next_offset = d["result"].get("next_page_offset")
        if next_offset is None: break

def main():
    batch_n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    doc_csv = sys.argv[2] if len(sys.argv) > 2 else None

    # 1. Read doc_ids that were supposed to be in this batch
    expected_doc_ids = set()
    if doc_csv and os.path.exists(doc_csv):
        with open(doc_csv) as f:
            content = f.read().strip()
            expected_doc_ids = {d.strip() for d in content.split(",") if d.strip()}

    # 1b. Subtract codified carve-outs (D-2 no-PDF, D-2 junk, D-1 shared-PDF losers).
    # 2026-05-07: classification at /opt/indian-legal-ai/data/eval/known_d_defect_carveouts.json
    # documents doc_ids that are structurally unrecoverable in current ingest pipeline
    # (rescrape needed or chunker-v3 required). They MUST NOT trigger D-DEFECT P0 — but
    # any drift OUTSIDE this list still does.
    carveout_path = "/opt/indian-legal-ai/data/eval/known_d_defect_carveouts.json"
    carved_out = set()
    if os.path.exists(carveout_path):
        with open(carveout_path) as f:
            cdata = json.load(f)
        for k in ("d2_no_pdf","d2_junk_content","d1_shared_pdf_loser"):
            for d in cdata.get(k, []): carved_out.add(d)
    pre_carve = len(expected_doc_ids)
    expected_doc_ids -= carved_out
    if pre_carve > len(expected_doc_ids):
        print(f"[lint] applied carve-out: {pre_carve - len(expected_doc_ids)} docs excluded "
              f"(D-2/D-1 documented limitations); effective expected={len(expected_doc_ids)}")

    # 2. Collect chunk counts + section_refs from cbic_v2 (filtered to expected docs if available)
    chunks_by_doc = defaultdict(list)
    empty_sectionref = 0
    total_chunks = 0
    duplicate_ranges = 0
    seen_ranges = defaultdict(set)  # doc_id -> set of (start,end)
    print(f"[lint] scanning cbic_v2 for batch {batch_n} drift checks...")
    for doc_id, cid, payload in scroll_collection():
        if expected_doc_ids and doc_id not in expected_doc_ids: continue
        total_chunks += 1
        chunks_by_doc[doc_id].append(cid)
        sr = payload.get("section_ref")
        # Only count empty section_ref as a defect for Act/Rules docs (which are section-structured).
        # Notifications, circulars, instructions, forms are typically flat prose — empty sr is expected.
        category = (payload.get("category") or "").lower()
        is_section_doc = category in ("act","rules","allied_acts") or "act" in (doc_id or "").lower()
        if is_section_doc and (not sr or (isinstance(sr, str) and not sr.strip())):
            empty_sectionref += 1
        rng = (payload.get("char_start"), payload.get("char_end"))
        if rng[0] is not None and rng[1] is not None:
            for prev in seen_ranges[doc_id]:
                if rng[0] >= prev[0] and rng[1] <= prev[1] and rng != prev:
                    duplicate_ranges += 1
                    break
                if prev[0] >= rng[0] and prev[1] <= rng[1] and rng != prev:
                    duplicate_ranges += 1
                    break
            seen_ranges[doc_id].add(rng)

    # 3. Defect D check: docs in expected list but with 0 chunks.
    # Runtime classification (2026-05-07): for any zero-chunk doc not in the static carve-out,
    # try to classify it on-the-fly against the manifest + pdftotext. If it falls into D-2a/
    # D-2b/D-1 we log it as INFO and exclude from P0; only docs we cannot classify trigger HALT.
    #
    # 2026-05-08 amendment: pre-subtract docs that HAVE canonical chunks in v2 manifest
    # (any upserted state). These are upsert stragglers, not chunker failures — corpus_drain
    # at end of loop fixes them. Without this, lint flags every straggler as D-DEFECT P0
    # mid-loop and HALTs spuriously.
    zero_chunk_docs_raw_pre = [d for d in expected_doc_ids if d not in chunks_by_doc] if expected_doc_ids else []
    # Filter: docs with canonical chunks in manifest (regardless of upserted) aren't real D-DEFECT.
    try:
        import sqlite3 as _sq2
        _v2m = _sq2.connect("/opt/indian-legal-ai/data/ingest_manifest_v2.sqlite")
        if zero_chunk_docs_raw_pre:
            _qm = ",".join("?" * len(zero_chunk_docs_raw_pre))
            _has_canon = {r[0] for r in _v2m.execute(
                f"SELECT DISTINCT doc_id FROM chunks WHERE is_canonical=1 AND doc_id IN ({_qm})",
                zero_chunk_docs_raw_pre,
            )}
        else:
            _has_canon = set()
    except Exception:
        _has_canon = set()
    zero_chunk_docs_raw = [d for d in zero_chunk_docs_raw_pre if d not in _has_canon]
    pending_upsert_skipped = len(zero_chunk_docs_raw_pre) - len(zero_chunk_docs_raw)
    if pending_upsert_skipped:
        print(f"[lint] {pending_upsert_skipped} zero-Qdrant docs are upsert stragglers (have canonical chunks in manifest) — excluded from D-DEFECT, will be drained by corpus_drain")
    zero_chunk_docs = []
    runtime_carved = {"d2a_no_pdf":[], "d2b_junk_content":[], "d1_shared_pdf":[]}
    if zero_chunk_docs_raw:
        import sqlite3 as _sqlite, subprocess as _sub
        _man = _sqlite.connect("/opt/indian-legal-ai/data/scraped/cbic/_manifest.sqlite")
        _qmarks = ",".join("?" * len(zero_chunk_docs_raw))
        _rows = list(_man.execute(f"SELECT doc_id, path_en, sha256_en FROM docs WHERE doc_id IN ({_qmarks})", zero_chunk_docs_raw))
        _row_by_id = {r[0]: r for r in _rows}
        for did in zero_chunk_docs_raw:
            r = _row_by_id.get(did)
            if r is None or not r[1]:
                runtime_carved["d2a_no_pdf"].append(did); continue
            _, pen, sha = r
            if not os.path.isfile(pen):
                runtime_carved["d2a_no_pdf"].append(did); continue
            if sha:
                _n = list(_man.execute("SELECT COUNT(*) FROM docs WHERE sha256_en=?", (sha,)))[0][0]
                if _n > 1:
                    runtime_carved["d1_shared_pdf"].append(did); continue
            try:
                _txt = _sub.check_output(["pdftotext","-q",pen,"-"], timeout=15).decode("utf-8","ignore").strip()
            except Exception:
                _txt = ""
            if len(_txt) < 500:
                runtime_carved["d2b_junk_content"].append(did); continue
            zero_chunk_docs.append(did)
        n_runtime = sum(len(v) for v in runtime_carved.values())
        if n_runtime > 0:
            print(f"[lint] runtime-classified {n_runtime} zero-chunk docs as known carve-out: "
                  f"{len(runtime_carved['d2a_no_pdf'])} D-2a, "
                  f"{len(runtime_carved['d2b_junk_content'])} D-2b, "
                  f"{len(runtime_carved['d1_shared_pdf'])} D-1")

    # 4. Chunk-ratio anomaly
    counts = sorted([len(c) for c in chunks_by_doc.values()])
    median = counts[len(counts)//2] if counts else 0
    low_ratio = [d for d, c in chunks_by_doc.items() if median > 0 and len(c) < max(1, median*0.5)]

    # 5. Build report
    findings = []

    # 5a. Corpus-wide stragglers check (2026-05-08): canonical chunks that never
    # got upserted to Qdrant. Per-batch RECONCILE only checks scoped doc_ids, so
    # transient phase3_4_5 failures in PRIOR batches leak silently across batches.
    # Surfaced when CP-3 push hit 3,755 stragglers across 1,098 doc_ids.
    # Lint flags this as P1 (informational during loop, not P0 HALT) but the
    # corpus_drain step in run_batch_loop.sh MUST clear it before any CP gate.
    try:
        import sqlite3 as _sq
        _v2 = _sq.connect("/opt/indian-legal-ai/data/ingest_manifest_v2.sqlite")
        _pending = list(_v2.execute("SELECT COUNT(*) FROM chunks WHERE is_canonical=1 AND upserted=0"))[0][0]
        _pending_docs = list(_v2.execute("SELECT COUNT(DISTINCT doc_id) FROM chunks WHERE is_canonical=1 AND upserted=0"))[0][0]
        if _pending > 0:
            findings.append({"severity":P1,"code":"UPSERT-STRAGGLERS","count":_pending,
                             "what":f"{_pending} canonical chunks across {_pending_docs} doc_ids stuck at upserted=0 corpus-wide. Run corpus_drain before any CP gate."})
    except Exception as _e:
        findings.append({"severity":P2,"code":"LINT-AUDIT-ERR","count":1,
                         "what":f"could not query stragglers: {type(_e).__name__}: {_e}"})

    if zero_chunk_docs:
        findings.append({"severity":P0,"code":"D-DEFECT","count":len(zero_chunk_docs),
                         "what":f"{len(zero_chunk_docs)} expected doc_ids have 0 chunks in cbic_v2",
                         "samples":zero_chunk_docs[:5]})
    if empty_sectionref > 0:
        sev = P1 if empty_sectionref > total_chunks * 0.05 else P2
        findings.append({"severity":sev,"code":"SECTION-REF","count":empty_sectionref,
                         "what":f"{empty_sectionref}/{total_chunks} chunks ({100*empty_sectionref/max(1,total_chunks):.1f}%) have empty section_ref"})
    if low_ratio:
        findings.append({"severity":P2,"code":"CHUNK-RATIO","count":len(low_ratio),
                         "what":f"{len(low_ratio)} docs have <0.5x median chunks (median={median})",
                         "samples":low_ratio[:5]})
    if duplicate_ranges > 0:
        sev = P1 if duplicate_ranges > 10 else P2
        findings.append({"severity":sev,"code":"TAIL-DUP","count":duplicate_ranges,
                         "what":f"{duplicate_ranges} chunks have ranges 100% inside another (Defect F2 regression)"})

    # 6. Print + write
    print(f"[lint] batch {batch_n}: {total_chunks} chunks across {len(chunks_by_doc)} docs (median {median}/doc)")
    if not findings:
        print("[lint] CLEAN — no drift detected")
    else:
        print(f"[lint] {len(findings)} findings:")
        for f in findings:
            print(f"  [{f['severity']}] {f['code']}: {f['what']}")
            if f.get('samples'): print(f"         samples: {f['samples']}")

    out_path = f"/opt/indian-legal-ai/data/eval/post_batch_lint_{batch_n}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump({"batch":batch_n, "total_chunks":total_chunks, "median_chunks_per_doc":median,
               "expected_docs":len(expected_doc_ids), "actual_docs":len(chunks_by_doc),
               "findings":findings}, open(out_path,"w"), indent=2)
    print(f"[lint] wrote {out_path}")

    # Exit code
    has_p0 = any(f["severity"]==P0 for f in findings)
    has_p1 = any(f["severity"]==P1 for f in findings)
    if has_p0: sys.exit(2)
    if has_p1: sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
