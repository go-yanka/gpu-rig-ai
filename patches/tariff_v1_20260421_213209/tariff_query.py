"""
tariff_query.py — exact-match HSN/SAC/description lookups against SQLite tariff.db.

Sentinel: tariff_v1
Target DB: /opt/indian-legal-ai/tariff.db

Public API:
    is_rate_query(q)           -> bool
    lookup(q, asof=None)       -> list[dict]  (most-recent effective_from first)
"""
import re
import sqlite3
import datetime
from typing import Optional, List, Dict

DB_PATH = '/opt/indian-legal-ai/tariff.db'

HSN_RE = re.compile(r'\bHSN\s*[:\-]?\s*(\d{2,8})\b', re.IGNORECASE)
SAC_RE = re.compile(r'\bSAC\s*[:\-]?\s*(\d{2,8})\b', re.IGNORECASE)
RATE_RE = re.compile(
    r'\b(?:gst|igst|cgst|sgst|rate|tariff|duty)\s+(?:on|for|of|applicable\s+to)\s+',
    re.IGNORECASE,
)


def is_rate_query(q: str) -> bool:
    """True if the question looks like a rate/HSN/SAC lookup."""
    if not q:
        return False
    return bool(HSN_RE.search(q) or SAC_RE.search(q) or RATE_RE.search(q))


def lookup(q: str, asof: Optional[str] = None, db_path: str = DB_PATH) -> List[Dict]:
    """
    Lookup rate rows matching the question.

    Resolution order:
      1. HSN code exact match (if 'HSN <digits>' in q)
      2. SAC code exact match
      3. FTS5 fallback on description

    Filters by effective_from <= asof <= COALESCE(effective_to, infinity).
    Returns rows ordered by effective_from DESC (most recent first).
    """
    asof = asof or datetime.date.today().isoformat()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        h = HSN_RE.search(q)
        s = SAC_RE.search(q)
        if h:
            code = h.group(1)
            rows = cur.execute(
                """SELECT * FROM tariff_rate
                   WHERE hsn = ?
                     AND effective_from <= ?
                     AND (effective_to IS NULL OR effective_to >= ?)
                   ORDER BY effective_from DESC""",
                (code, asof, asof),
            ).fetchall()
        elif s:
            code = s.group(1)
            rows = cur.execute(
                """SELECT * FROM tariff_rate
                   WHERE sac = ?
                     AND effective_from <= ?
                     AND (effective_to IS NULL OR effective_to >= ?)
                   ORDER BY effective_from DESC""",
                (code, asof, asof),
            ).fetchall()
        else:
            # FTS fallback on description
            tokens = re.findall(r'\w+', q.lower())
            fts_q = ' '.join(tokens)[:200]
            if not fts_q:
                return []
            rows = cur.execute(
                """SELECT tariff_rate.*
                   FROM tariff_rate
                   JOIN tariff_fts ON tariff_rate.id = tariff_fts.rowid
                   WHERE tariff_fts MATCH ?
                     AND tariff_rate.effective_from <= ?
                     AND (tariff_rate.effective_to IS NULL
                          OR tariff_rate.effective_to >= ?)
                   ORDER BY bm25(tariff_fts) LIMIT 5""",
                (fts_q, asof, asof),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
