"""
tariff_ingest.py — extracts rate rows from CBIC rate notification PDFs into SQLite.

Sentinel: tariff_v1
Target DB: /opt/indian-legal-ai/tariff.db

Usage:
    python tariff_ingest.py --init-only
    python tariff_ingest.py --pdf /path/to/notification.pdf \
        --notification-id 1/2017-CT(Rate) --effective-from 2017-07-01
    python tariff_ingest.py --pdf-dir /path/to/notifications/ --index index.json

NOTE: The Docling extraction logic is stubbed (TODO). init_db() + CLI wiring + insert_rows()
      are functional. Finish extract_tables_from_pdf() once real CBIC rate PDFs are in hand.
"""
import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

# TODO: pip install docling
# from docling.document_converter import DocumentConverter

DB_PATH = '/opt/indian-legal-ai/tariff.db'
SCHEMA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tariff_schema.sql')


def init_db(db_path: str = DB_PATH, schema_file: str = SCHEMA_FILE) -> None:
    """Create tariff.db from schema if missing. Idempotent (schema uses IF NOT EXISTS)."""
    Path(os.path.dirname(db_path)).mkdir(parents=True, exist_ok=True)
    with open(schema_file, 'r', encoding='utf-8') as f:
        ddl = f.read()
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(ddl)
        conn.commit()
        # Verify sentinel
        row = conn.execute("SELECT value FROM tariff_meta WHERE key='sentinel'").fetchone()
        print(f"[init_db] db={db_path} sentinel={row[0] if row else 'MISSING'}")
    finally:
        conn.close()


def extract_tables_from_pdf(pdf_path: str) -> list[dict]:
    """
    TODO: Use Docling to extract tables from a CBIC rate notification PDF.

    Return a list of dicts with keys matching tariff_rate columns:
        hsn, sac, description, rate_igst, rate_cgst, rate_sgst, rate_cess,
        doc_page, schedule, chapter

    Notes on CBIC rate notification structure:
      - Notification 1/2017-CT(Rate) has 6 schedules (I=5%, II=12%, III=18%, IV=28%, V=3%, VI=0.25%).
      - Each schedule is a table: [S.No, Chapter/Heading/Sub-heading/Tariff item, Description].
      - "Chapter" or "heading" rows have no specific HSN — forward-fill the HS chapter for children.
      - Merged cells: Docling returns None for spanned cells; forward-fill from the last
        non-None row in the same column before emitting a row.
      - Detect schedule via section header text ("Schedule I - 2.5%") — the IGST rate is
        2 * CGST rate, and CGST == SGST per intra-state rule.

    For now this raises NotImplementedError; wire up after installing docling and
    inspecting 2-3 real rate notification PDFs.
    """
    raise NotImplementedError(
        "TODO: implement Docling-based table extraction. See docstring."
    )


def insert_rows(
    conn: sqlite3.Connection,
    rows: list[dict],
    notification_id: str,
    effective_from: str,
    effective_to: str | None = None,
    pdf_path: str | None = None,
    notification_date: str | None = None,
) -> int:
    """
    Bulk insert. Deduplicate on (COALESCE(hsn,sac), effective_from, notification_id).
    Returns number of rows inserted (skipped duplicates do not count).
    """
    cur = conn.cursor()
    inserted = 0
    for r in rows:
        code = r.get('hsn') or r.get('sac')
        if not code or not r.get('description'):
            continue
        # Dedup check
        existing = cur.execute(
            """SELECT id FROM tariff_rate
               WHERE COALESCE(hsn, sac) = ?
                 AND effective_from = ?
                 AND notification_id = ?""",
            (code, effective_from, notification_id),
        ).fetchone()
        if existing:
            continue
        cur.execute(
            """INSERT INTO tariff_rate
               (hsn, sac, description, rate_igst, rate_cgst, rate_sgst, rate_cess,
                effective_from, effective_to, notification_id, notification_date,
                doc_page, pdf_path, schedule, chapter)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                r.get('hsn'),
                r.get('sac'),
                r.get('description'),
                r.get('rate_igst'),
                r.get('rate_cgst'),
                r.get('rate_sgst'),
                r.get('rate_cess'),
                effective_from,
                effective_to,
                notification_id,
                notification_date,
                r.get('doc_page'),
                pdf_path,
                r.get('schedule'),
                r.get('chapter'),
            ),
        )
        inserted += 1
    conn.commit()
    return inserted


def ingest_pdf(
    pdf_path: str,
    notification_id: str,
    effective_from: str,
    effective_to: str | None,
    db_path: str,
    notification_date: str | None = None,
) -> int:
    """Extract + insert for a single PDF. Returns rows inserted."""
    rows = extract_tables_from_pdf(pdf_path)
    conn = sqlite3.connect(db_path)
    try:
        n = insert_rows(
            conn, rows,
            notification_id=notification_id,
            effective_from=effective_from,
            effective_to=effective_to,
            pdf_path=pdf_path,
            notification_date=notification_date,
        )
    finally:
        conn.close()
    print(f"[ingest_pdf] {pdf_path}: inserted {n} rows")
    return n


def ingest_dir(pdf_dir: str, index_path: str, db_path: str) -> int:
    """
    Ingest a directory of PDFs using an index.json file:
        [{"pdf": "file.pdf", "notification_id": "...", "effective_from": "...",
          "effective_to": null, "notification_date": "..."}, ...]
    """
    with open(index_path, 'r', encoding='utf-8') as f:
        index = json.load(f)
    total = 0
    for entry in index:
        pdf = os.path.join(pdf_dir, entry['pdf'])
        total += ingest_pdf(
            pdf_path=pdf,
            notification_id=entry['notification_id'],
            effective_from=entry['effective_from'],
            effective_to=entry.get('effective_to'),
            db_path=db_path,
            notification_date=entry.get('notification_date'),
        )
    print(f"[ingest_dir] total rows inserted: {total}")
    return total


def main():
    ap = argparse.ArgumentParser(description='CBIC tariff rate PDF → SQLite ingester')
    ap.add_argument('--pdf', help='single PDF to ingest')
    ap.add_argument('--pdf-dir', help='directory of PDFs (use with --index)')
    ap.add_argument('--index', help='index.json describing PDFs in --pdf-dir')
    ap.add_argument('--notification-id', help="e.g., '1/2017-CT(Rate)'")
    ap.add_argument('--notification-date')
    ap.add_argument('--effective-from', help='ISO date, e.g., 2017-07-01')
    ap.add_argument('--effective-to', default=None)
    ap.add_argument('--db', default=DB_PATH)
    ap.add_argument('--init-only', action='store_true',
                    help='Create DB schema only; do not ingest')
    args = ap.parse_args()

    init_db(args.db)
    if args.init_only:
        return 0

    if args.pdf:
        if not (args.notification_id and args.effective_from):
            ap.error('--pdf requires --notification-id and --effective-from')
        ingest_pdf(
            pdf_path=args.pdf,
            notification_id=args.notification_id,
            effective_from=args.effective_from,
            effective_to=args.effective_to,
            db_path=args.db,
            notification_date=args.notification_date,
        )
    elif args.pdf_dir:
        if not args.index:
            ap.error('--pdf-dir requires --index')
        ingest_dir(args.pdf_dir, args.index, args.db)
    else:
        print('[main] --init-only completed (no --pdf/--pdf-dir given)')
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
