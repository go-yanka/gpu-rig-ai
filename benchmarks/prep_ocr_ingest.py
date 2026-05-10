#!/usr/bin/env python3
"""After OCR finishes: create doc_id→sha256 symlinks in ocr_cache/
and write targets TSV for ingest_ocr_worker.py.

The existing ingester expects `{doc_id_safe}.txt` files with `---PAGE N---` format.
Our ocr_full_gemini.py wrote `{sha256}.txt`. Both formats match on content."""
import sqlite3, os
from pathlib import Path

QA = "/opt/indian-legal-ai/data/scraped/cbic/_qa.sqlite"
MANIFEST = "/opt/indian-legal-ai/data/scraped/cbic/_manifest.sqlite"
OCR_CACHE = Path("/opt/indian-legal-ai/data/ocr_cache")
TARGETS = Path("/opt/indian-legal-ai/data/ocr_targets_851.tsv")

def safe(did): return did.replace("/", "_").replace(":", "_")

def main():
    # Map doc_id → full row from qa table (image_only=1)
    cq = sqlite3.connect(QA)
    qa_rows = list(cq.execute(
        "SELECT source, category, subcategory, doc_id, lang, path, sha256 "
        "FROM qa WHERE image_only=1 AND path IS NOT NULL"
    ))
    print(f"qa image_only rows: {len(qa_rows)}")

    # Look up titles from manifest by (source, category, subcategory, doc_id)
    cm = sqlite3.connect(MANIFEST)
    man = {}
    for src, cat, sub, did, title in cm.execute(
        "SELECT source, category, subcategory, doc_id, title FROM docs"
    ):
        man[(src, cat, sub, did)] = title or ""
    print(f"manifest docs: {len(man)}")

    linked = skipped_missing = skipped_nodata = 0
    rows_out = []
    for src, cat, sub, did, lang, path, sha256 in qa_rows:
        src_file = OCR_CACHE / f"{sha256}.txt"
        if not src_file.exists():
            skipped_missing += 1
            continue
        # Check it has content
        if src_file.stat().st_size < 20:
            skipped_nodata += 1
            continue
        link = OCR_CACHE / f"{safe(did)}.txt"
        if link.exists() or link.is_symlink():
            try: link.unlink()
            except: pass
        try:
            link.symlink_to(src_file.name)  # relative
            linked += 1
        except Exception as e:
            print(f"  symlink err {did}: {e}")
            continue
        title = man.get((src, cat, sub, did), "")
        rows_out.append((did, cat, sub, path, title))

    # Write targets TSV
    with open(TARGETS, "w", encoding="utf-8") as f:
        for did, cat, sub, pdf, title in rows_out:
            f.write(f"{did}\t{cat}\t{sub}\t{pdf}\t{title}\n")

    print(f"linked: {linked}, missing_ocr: {skipped_missing}, empty: {skipped_nodata}")
    print(f"targets written: {TARGETS} ({len(rows_out)} rows)")

if __name__ == "__main__":
    main()
