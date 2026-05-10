"""Quality audit for downloaded PDFs.

Reads the CBIC manifest + filesystem, runs structural/content checks on every
PDF, stores results back into the manifest, and prints a summary report.

Fails a PDF if ANY of:
  - not %PDF magic
  - size < 400 bytes (almost certainly an error page)
  - pypdf can't parse structure
  - 0 pages
  - <50 chars of extractable text in first 3 pages (image-only scan --
    flagged for OCR later, not deleted)

Cross-cut checks:
  - duplicate SHA-256 across different docs (server served same file twice)
  - Hindi file byte-identical to English (wrong language served)
"""
from __future__ import annotations
import os, sys, json, sqlite3, hashlib, argparse, time
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional

try:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError
except ImportError:
    from PyPDF2 import PdfReader  # type: ignore
    from PyPDF2.errors import PdfReadError  # type: ignore


SCHEMA_EXTRA = """
CREATE TABLE IF NOT EXISTS qa (
  source TEXT NOT NULL,
  category TEXT NOT NULL,
  subcategory TEXT NOT NULL,
  doc_id TEXT NOT NULL,
  lang TEXT NOT NULL,           -- 'en' | 'hi'
  path TEXT,
  bytes INTEGER,
  sha256 TEXT,
  magic_ok INTEGER,
  size_ok INTEGER,
  pypdf_ok INTEGER,
  pages INTEGER,
  text_chars INTEGER,
  image_only INTEGER,            -- 1 if <50 chars -- needs OCR
  status TEXT,                   -- 'ok' | 'image_only' | 'corrupt' | 'error_page' | 'missing'
  reason TEXT,
  checked_at TEXT,
  PRIMARY KEY (source, category, subcategory, doc_id, lang)
);
"""


def check_pdf(path: Path) -> dict:
    out = {
        "magic_ok": 0, "size_ok": 0, "pypdf_ok": 0,
        "pages": 0, "text_chars": 0, "image_only": 0,
        "status": "missing", "reason": "", "bytes": 0, "sha256": None,
    }
    if not path.exists():
        out["reason"] = "file does not exist"
        return out

    raw = path.read_bytes()
    out["bytes"] = len(raw)
    out["sha256"] = hashlib.sha256(raw).hexdigest()

    if raw[:4] != b"%PDF":
        out["status"] = "corrupt"
        out["reason"] = f"bad magic bytes: {raw[:8]!r}"
        return out
    out["magic_ok"] = 1

    if out["bytes"] < 400:
        out["status"] = "error_page"
        out["reason"] = f"suspiciously small ({out['bytes']} bytes)"
        return out
    out["size_ok"] = 1

    try:
        reader = PdfReader(str(path))
        n_pages = len(reader.pages)
        out["pages"] = n_pages
        out["pypdf_ok"] = 1
        if n_pages == 0:
            out["status"] = "corrupt"
            out["reason"] = "zero pages"
            return out
        # extract text from first 3 pages
        text = ""
        for pg in reader.pages[:3]:
            try:
                text += pg.extract_text() or ""
            except Exception:
                pass
        chars = len(text.strip())
        out["text_chars"] = chars
        if chars < 50:
            out["image_only"] = 1
            out["status"] = "image_only"
            out["reason"] = f"only {chars} extractable chars in first 3 pages"
            return out
        out["status"] = "ok"
        return out
    except PdfReadError as e:
        out["status"] = "corrupt"
        out["reason"] = f"pypdf: {e}"
        return out
    except Exception as e:
        out["status"] = "corrupt"
        out["reason"] = f"exception: {e}"
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default="/opt/indian-legal-ai/data/scraped/cbic/_manifest.sqlite")
    ap.add_argument("--qa-db",
                    default="/opt/indian-legal-ai/data/scraped/cbic/_qa.sqlite",
                    help="separate DB for QA results so the downloader's "
                         "write-lock on _manifest.sqlite is never contested")
    ap.add_argument("--source", default="cbic")
    ap.add_argument("--limit", type=int, default=0,
                    help="max rows to check (0 = all)")
    ap.add_argument("--category", default=None,
                    help="restrict to one category, e.g. 'gst'")
    ap.add_argument("--only-new", action="store_true",
                    help="skip files already in qa table")
    ap.add_argument("--summary-only", action="store_true",
                    help="don't re-check; just print summary from qa table")
    args = ap.parse_args()

    # open manifest READ-ONLY (mode=ro) so we never race with the downloader
    mcon = sqlite3.connect(f"file:{args.manifest}?mode=ro&immutable=0",
                            uri=True, timeout=30.0)
    mcon.row_factory = sqlite3.Row
    # QA results go to a DIFFERENT file so QA writes never touch the manifest
    con = sqlite3.connect(args.qa_db, timeout=30.0)
    con.executescript(SCHEMA_EXTRA)
    con.execute("PRAGMA journal_mode=WAL")
    con.row_factory = sqlite3.Row

    if not args.summary_only:
        where = "source=? AND downloaded_at IS NOT NULL"
        params = [args.source]
        if args.category:
            where += " AND category=?"
            params.append(args.category)
        rows = mcon.execute(
            f"SELECT source, category, subcategory, doc_id, path_en, path_hi, "
            f"sha256_en, sha256_hi FROM docs WHERE {where}",
            params,
        ).fetchall()
        total = len(rows)
        print(f"[qa] {total} downloaded docs to audit", flush=True)

        checked = 0
        t0 = time.time()
        for i, r in enumerate(rows, 1):
            if args.limit and checked >= args.limit:
                break
            for lang in ("en", "hi"):
                pkey = f"path_{lang}"
                p = r[pkey]
                if not p:
                    continue
                # already audited?
                if args.only_new:
                    hit = con.execute(
                        "SELECT 1 FROM qa WHERE source=? AND category=? AND "
                        "subcategory=? AND doc_id=? AND lang=?",
                        (r["source"], r["category"], r["subcategory"],
                         r["doc_id"], lang),
                    ).fetchone()
                    if hit:
                        continue
                result = check_pdf(Path(p))
                con.execute(
                    "INSERT OR REPLACE INTO qa "
                    "(source, category, subcategory, doc_id, lang, path, "
                    " bytes, sha256, magic_ok, size_ok, pypdf_ok, pages, "
                    " text_chars, image_only, status, reason, checked_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (r["source"], r["category"], r["subcategory"],
                     r["doc_id"], lang, p,
                     result["bytes"], result["sha256"],
                     result["magic_ok"], result["size_ok"],
                     result["pypdf_ok"], result["pages"],
                     result["text_chars"], result["image_only"],
                     result["status"], result["reason"],
                     time.strftime("%Y-%m-%d %H:%M:%S")))
                checked += 1
            if i % 100 == 0:
                con.commit()
                dt = time.time() - t0
                print(f"[qa] {i}/{total} rows; {checked} files; "
                      f"{checked/max(dt,0.01):.1f} files/s", flush=True)
        con.commit()
        print(f"[qa] completed: {checked} files audited in "
              f"{time.time()-t0:.1f}s", flush=True)

    # -------- summary report ----------
    print()
    print("=" * 70)
    print("QUALITY REPORT")
    print("=" * 70)

    by_status = Counter()
    total = 0
    for r in con.execute("SELECT status FROM qa WHERE source=?",
                         (args.source,)):
        by_status[r["status"]] += 1
        total += 1
    if total == 0:
        print("No rows in qa table yet.")
        return
    print(f"Total PDFs audited: {total}")
    print(f"{'status':20s}  {'count':>7s}  {'pct':>6s}")
    print("-" * 40)
    for st in ("ok", "image_only", "corrupt", "error_page", "missing"):
        n = by_status.get(st, 0)
        if n:
            print(f"{st:20s}  {n:>7d}  {100*n/total:>5.1f}%")

    # per-category
    print()
    print("By category / language:")
    for r in con.execute("""
        SELECT category, lang, status, COUNT(*) n
        FROM qa WHERE source=?
        GROUP BY category, lang, status
        ORDER BY category, lang, status
    """, (args.source,)):
        print(f"  {r['category']:18s} {r['lang']:4s} {r['status']:14s} "
              f"{r['n']:>6d}")

    # duplicate SHA-256 (server served same file for multiple docs)
    print()
    dupes = con.execute("""
        SELECT sha256, COUNT(*) n
        FROM qa WHERE source=? AND sha256 IS NOT NULL AND status IN ('ok','image_only')
        GROUP BY sha256 HAVING n > 1
        ORDER BY n DESC LIMIT 15
    """, (args.source,)).fetchall()
    if dupes:
        print(f"Duplicate SHA-256 groups (top 15): "
              f"{sum(r['n'] for r in dupes)} files share content")
        for r in dupes:
            print(f"  {r['sha256'][:16]}...  appears in {r['n']} docs")
    else:
        print("No duplicate SHA-256 groups.")

    # EN/HI same file
    same_lang = con.execute("""
        SELECT e.category, COUNT(*) n
        FROM qa e JOIN qa h ON e.source=h.source AND e.category=h.category
             AND e.subcategory=h.subcategory AND e.doc_id=h.doc_id
        WHERE e.lang='en' AND h.lang='hi' AND e.sha256 = h.sha256
          AND e.source=?
        GROUP BY e.category
    """, (args.source,)).fetchall()
    if same_lang:
        print()
        print("Hindi file byte-identical to English (wrong lang served):")
        for r in same_lang:
            print(f"  {r['category']}: {r['n']} docs")

    # image-only (OCR candidates)
    n_ocr = by_status.get("image_only", 0)
    if n_ocr:
        print()
        print(f"[!] {n_ocr} PDFs are image-only (need OCR before ingestion)")


if __name__ == "__main__":
    main()
