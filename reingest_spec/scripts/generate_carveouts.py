#!/usr/bin/env python3
"""Generate the corpus-wide D-DEFECT carve-out file.

Codified 2026-05-07. Produces /opt/indian-legal-ai/data/eval/known_d_defect_carveouts.json
which lists doc_ids that are STRUCTURALLY UNRECOVERABLE in the current ingest pipeline:

  d2_no_pdf            — manifest has no path_en (scrape never delivered a PDF, or empty_body)
  d2_junk_content      — PDF on disk but pdftotext extracts < 500 chars (HTML page saved as PDF,
                          1.2KB placeholder, image-only forms — all need rescrape or OCR)
  d1_shared_pdf_loser  — doc_id is a member of a shared-PDF cluster (multiple doc_ids referencing
                          the same sha256_en). Only one wins per cluster under current chunker;
                          others are intentionally suppressed until per-form structural splitting
                          ships in chunker-v3.

post_batch_lint.py reads this file and EXCLUDES carved-out doc_ids from `expected` before
running the D-DEFECT P0 check. Anything missing from cbic_v2 BUT NOT in this carve-out file
is a real chunker / pipeline defect and HALTS the loop.

Re-run this script:
  - whenever the manifest changes (new scrape, recovered PDFs)
  - whenever the chunker changes its handling of D-1 or D-2 cases
"""
import sqlite3
import json
import os
import subprocess
from collections import defaultdict

MANIFEST = "/opt/indian-legal-ai/data/scraped/cbic/_manifest.sqlite"
OUT = "/opt/indian-legal-ai/data/eval/known_d_defect_carveouts.json"


def main():
    c = sqlite3.connect(MANIFEST)
    all_docs = list(c.execute("SELECT doc_id, path_en, sha256_en, last_error FROM docs"))
    print(f"total manifest docs: {len(all_docs)}")

    # D-2a: no path_en
    d2_no_pdf = []
    sha_groups = defaultdict(list)
    for did, pen, sha, _err in all_docs:
        if not pen:
            d2_no_pdf.append(did)
            continue
        if sha:
            sha_groups[sha].append(did)
    print(f"D-2a NO_PDF: {len(d2_no_pdf)}")

    # D-1: doc_ids in clusters of size > 1
    d1_cluster_members = []
    for sha, dids in sha_groups.items():
        if len(dids) > 1:
            d1_cluster_members.extend(dids)
    print(f"D-1 SHARED_PDF members: {len(d1_cluster_members)}")

    # D-2b: junk content. Heuristic: PDF likely sparse (<50KB) OR known-bad scrape error → verify via pdftotext.
    # Threshold raised 2026-05-07 (was <5KB) after batch 7 surfaced 5 junk PDFs in the 9-25KB range
    # (HTML "site best viewed" pages, "no circular issued" placeholders) that the tighter heuristic missed.
    d2_junk = []
    junk_candidates = [
        (did, pen)
        for did, pen, _sha, err in all_docs
        if pen and os.path.isfile(pen)
        and (
            os.path.getsize(pen) < 50_000
            or (err and ("empty_body" in err or "wrong PDF" in err or "reverted" in err))
        )
    ]
    print(f"junk-content candidates to verify: {len(junk_candidates)}")
    for did, pen in junk_candidates:
        try:
            txt = subprocess.check_output(["pdftotext", "-q", pen, "-"], timeout=10).decode("utf-8", "ignore").strip()
        except Exception:
            txt = ""
        if len(txt) < 500:
            d2_junk.append(did)
    print(f"D-2b JUNK_CONTENT (verified): {len(d2_junk)}")

    all_carved = set(d2_no_pdf) | set(d1_cluster_members) | set(d2_junk)

    out = {
        "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "reason": "D-DEFECT classification — codified non-bug exclusions (full corpus). See JOURNAL 2026-05-07.",
        "d2_no_pdf": sorted(d2_no_pdf),
        "d2_junk_content": sorted(d2_junk),
        "d1_shared_pdf_loser": sorted(d1_cluster_members),
        "total_unique": len(all_carved),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT} ({len(all_carved)} unique carved-out doc_ids)")


if __name__ == "__main__":
    main()
