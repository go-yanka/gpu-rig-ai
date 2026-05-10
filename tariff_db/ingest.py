"""
tariff.db ingestion pipeline.

Usage:
    python ingest.py build        # init + load all seed data + verify
    python ingest.py init         # init empty DB only
    python ingest.py verify       # run sample queries

Functions:
    init_db(path)                       - create empty DB with schema + indices
    load_hsn_codes(csv_path, conn)      - load codes table from CSV
    load_notification(notif_id, src, conn)
                                        - parse and load one notification (CSV stub OR PDF/JSON)
    load_exemption_table(notif_id, csv_path, conn)
                                        - load 50/2017-Cus S.No. tables
    verify(conn)                        - run 5 sample queries from schema doc
"""
from __future__ import annotations

import csv
import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
SCHEMA_SQL = HERE / "schema.sql"
SEED = HERE / "seed"
DEFAULT_DB = HERE / "tariff.db"


# ---------- helpers ----------

def _row(d: dict, *keys) -> tuple:
    return tuple(d.get(k) or None for k in keys)


def _none_if_empty(v):
    if v is None:
        return None
    v = v.strip() if isinstance(v, str) else v
    return v if v not in ("", None) else None


# ---------- init ----------

def init_db(path: str | os.PathLike = DEFAULT_DB) -> sqlite3.Connection:
    path = Path(path)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    with open(SCHEMA_SQL, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()
    print(f"[init] created {path}")
    return conn


# ---------- codes ----------

def load_hsn_codes(csv_path: str | os.PathLike, conn: sqlite3.Connection) -> int:
    """Load codes table from CSV. Columns: code,code_type,level,parent_code,description,chapter."""
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((
                r["code"].strip(),
                r["code_type"].strip(),
                int(r["level"]),
                _none_if_empty(r.get("parent_code")),
                r["description"].strip(),
                int(r["chapter"]) if _none_if_empty(r.get("chapter")) else None,
            ))
    # Two-pass: insert parents first (by level), then children. We disable FK to simplify.
    conn.execute("PRAGMA foreign_keys = OFF")
    rows.sort(key=lambda x: x[2])  # sort by level ascending
    conn.executemany(
        "INSERT OR REPLACE INTO codes(code,code_type,level,parent_code,description,chapter) VALUES (?,?,?,?,?,?)",
        rows,
    )
    conn.execute("PRAGMA foreign_keys = ON")
    conn.commit()
    print(f"[codes] loaded {len(rows)} rows from {Path(csv_path).name}")
    return len(rows)


# ---------- notifications ----------

def load_notifications_metadata(csv_path: str | os.PathLike, conn: sqlite3.Connection) -> int:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            rows.append((
                r["notif_id"].strip(),
                r["series"].strip(),
                int(r["number"]),
                int(r["year"]),
                r["issued_on"].strip(),
                r["effective_from"].strip(),
                _none_if_empty(r.get("superseded_by")),
                _none_if_empty(r.get("title")),
                _none_if_empty(r.get("source_doc_id")),
            ))
    conn.executemany(
        "INSERT OR REPLACE INTO notifications VALUES (?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    print(f"[notifications] loaded {len(rows)} rows")
    return len(rows)


def load_notification(notif_id: str, source: str | os.PathLike, conn: sqlite3.Connection) -> int:
    """
    Load one notification's payload.

    `source` is either:
      - a rates CSV (01/2017-CT(R) style): columns code,levy_type,rate_pct,rate_specific,
        condition_no,schedule,sno,effective_from,effective_to
      - a list-membership CSV (02/2017-CT(R), 05/2017-CT(R), 13/2017-CT(R) style): columns
        code,list_type,sno,description,effective_from,effective_to
      - a PDF path / JSON — NOT IMPLEMENTED in this stub. The pipeline is PDF-ready:
        drop in a parser in _parse_notification_pdf() and wire it here.

    Detection: first line of CSV header determines which loader is called.
    """
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix.lower() == ".pdf":
        raise NotImplementedError(
            "PDF parsing for notifications is not implemented in this seed pipeline. "
            "Use pre-parsed CSVs in seed/ for now. Wire a pdfplumber/tabula parser into "
            "_parse_notification_pdf() to extend."
        )

    if p.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported source: {p}")

    with open(p, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = set(reader.fieldnames or [])
        rows = list(reader)

    if "levy_type" in header:
        return _load_rates_rows(notif_id, rows, conn)
    if "list_type" in header:
        return _load_list_rows(notif_id, rows, conn)
    raise ValueError(f"Cannot determine CSV kind for {p} - header was {header}")


def _load_rates_rows(notif_id: str, rows: list[dict], conn: sqlite3.Connection) -> int:
    tuples = []
    for r in rows:
        tuples.append((
            r["code"].strip(),
            r["levy_type"].strip(),
            float(r["rate_pct"]) if _none_if_empty(r.get("rate_pct")) else None,
            _none_if_empty(r.get("rate_specific")),
            _none_if_empty(r.get("condition_no")),
            _none_if_empty(r.get("schedule")),
            int(r["sno"]) if _none_if_empty(r.get("sno")) else None,
            r["effective_from"].strip(),
            _none_if_empty(r.get("effective_to")),
            notif_id,
            None,
        ))
    conn.executemany(
        "INSERT INTO rates(code,levy_type,rate_pct,rate_specific,condition_no,schedule,sno,"
        "effective_from,effective_to,notif_id,amended_by) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        tuples,
    )
    conn.commit()
    print(f"[rates] loaded {len(tuples)} rows for {notif_id}")
    return len(tuples)


def _load_list_rows(notif_id: str, rows: list[dict], conn: sqlite3.Connection) -> int:
    tuples = []
    for r in rows:
        tuples.append((
            _none_if_empty(r.get("code")),
            r["list_type"].strip(),
            int(r["sno"]) if _none_if_empty(r.get("sno")) else None,
            _none_if_empty(r.get("description")),
            notif_id,
            r["effective_from"].strip(),
            _none_if_empty(r.get("effective_to")),
        ))
    conn.executemany(
        "INSERT INTO list_membership(code,list_type,sno,description,notif_id,"
        "effective_from,effective_to) VALUES (?,?,?,?,?,?,?)",
        tuples,
    )
    conn.commit()
    print(f"[list_membership] loaded {len(tuples)} rows for {notif_id}")
    return len(tuples)


# ---------- exemptions ----------

def load_exemption_table(notif_id: str, source: str | os.PathLike, conn: sqlite3.Connection) -> int:
    """
    Load 50/2017-Cus-style S.No. exemption table from CSV.
    Columns: sno,code,description,std_rate,igst_rate,condition_no,effective_from,effective_to
    """
    p = Path(source)
    with open(p, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        tuples = []
        for r in reader:
            tuples.append((
                notif_id,
                int(r["sno"]),
                _none_if_empty(r.get("code")),
                r["description"].strip(),
                _none_if_empty(r.get("std_rate")),
                _none_if_empty(r.get("igst_rate")),
                _none_if_empty(r.get("condition_no")),
                r["effective_from"].strip(),
                _none_if_empty(r.get("effective_to")),
            ))
    conn.executemany(
        "INSERT OR REPLACE INTO exemptions(notif_id,sno,code,description,std_rate,igst_rate,"
        "condition_no,effective_from,effective_to) VALUES (?,?,?,?,?,?,?,?,?)",
        tuples,
    )
    conn.commit()
    print(f"[exemptions] loaded {len(tuples)} rows for {notif_id}")
    return len(tuples)


# ---------- verify ----------

SAMPLE_QUERIES = [
    (
        "Q1: GST rate on HSN 1006 as of 2023-01-01",
        """
        SELECT levy_type, rate_pct, notif_id FROM rates
        WHERE code='1006' AND levy_type IN ('CGST','SGST','IGST')
          AND effective_from<='2023-01-01'
          AND (effective_to IS NULL OR effective_to>'2023-01-01')
        """,
    ),
    (
        "Q2: Is HSN 8703 on RCM currently?",
        """
        SELECT notif_id, sno FROM list_membership
        WHERE code='8703' AND list_type='RCM' AND effective_to IS NULL
        """,
    ),
    (
        "Q3: Latest rate change for HSN 2202",
        """
        SELECT notif_id, effective_from, rate_pct FROM rates
        WHERE code='2202' AND levy_type='IGST'
        ORDER BY effective_from DESC LIMIT 1
        """,
    ),
    (
        "Q4: 50/2017-Cus S.No. 404 detail",
        """
        SELECT code, description, std_rate, condition_no FROM exemptions
        WHERE notif_id='50/2017-Cus' AND sno=404
        """,
    ),
    (
        "Q5: All inverted-duty goods in chapter 54",
        """
        SELECT lm.code, c.description FROM list_membership lm JOIN codes c USING(code)
        WHERE lm.list_type='INVERTED_DUTY' AND c.chapter=54 AND lm.effective_to IS NULL
        """,
    ),
]


def verify(conn: sqlite3.Connection) -> None:
    print("\n" + "=" * 60)
    print("VERIFY: sample queries")
    print("=" * 60)
    for label, sql in SAMPLE_QUERIES:
        print(f"\n-- {label}")
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        if not rows:
            print("  (no rows)")
        else:
            print("  " + " | ".join(cols))
            for r in rows:
                print("  " + " | ".join("" if v is None else str(v) for v in r))

    # Row counts summary
    print("\n-- Table row counts --")
    for tbl in ("codes", "notifications", "rates", "list_membership", "exemptions"):
        n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl}: {n}")


# ---------- driver ----------

def build_all(db_path: str | os.PathLike = DEFAULT_DB) -> sqlite3.Connection:
    conn = init_db(db_path)

    load_hsn_codes(SEED / "codes.csv", conn)
    load_notifications_metadata(SEED / "notifications.csv", conn)

    load_notification("01/2017-CT(R)", SEED / "rates_01_2017_CTR.csv", conn)
    load_notification("02/2017-CT(R)", SEED / "list_02_2017_CTR_exempt.csv", conn)
    load_notification("05/2017-CT(R)", SEED / "list_05_2017_CTR_inverted.csv", conn)
    load_notification("13/2017-CT(R)", SEED / "list_13_2017_CTR_rcm.csv", conn)
    load_exemption_table("50/2017-Cus", SEED / "exemptions_50_2017_Cus.csv", conn)

    verify(conn)
    return conn


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    db = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DB
    if cmd == "init":
        init_db(db).close()
    elif cmd == "verify":
        conn = sqlite3.connect(str(db))
        verify(conn)
        conn.close()
    elif cmd == "build":
        conn = build_all(db)
        conn.close()
    else:
        print(__doc__)
        sys.exit(1)
