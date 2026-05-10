"""
tariff.db query router.

    maybe_route(query, conn=None) -> (result, reason)

`result` is:
    - list[dict] of SQL rows (possibly empty) when SQL route matched AND executed
    - None when the query doesn't match any structured-lookup rule

`reason` is a human-readable tag describing which rule fired (or 'no-match').

Design:
    - Regex-match the query against the rule set from p2_1_tariff_db_schema.md.
    - Pick the most specific rule that fires and run a targeted SQL.
    - NEVER fail closed: if SQL match returns zero rows, caller should fall through to RAG.
      We signal that via (result=[], reason='...-empty') so the caller can decide.
    - Wall-clock goal: sub-ms on a 60MB SQLite. All relevant indexes are in schema.sql.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

HERE = Path(__file__).resolve().parent
DEFAULT_DB = HERE / "tariff.db"


# ---------- regex rules (from schema.md "Router Rule") ----------

RE_HSN_SAC = re.compile(
    r"\b(?:HSN|SAC|chapter\s+heading)\s*(?P<code>\d{2,8})\b",
    re.IGNORECASE,
)
# bare HSN code (no keyword): e.g. "rate on 8703" — more permissive
RE_BARE_CODE = re.compile(
    r"\b(?P<code>\d{4,8})\b",
)
RE_RATE_PHRASE = re.compile(
    r"\b(rate|duty|BCD|IGST|CGST|SGST|GST|compensation\s+cess|comp\s*cess)\s+(on|of|for)\b",
    re.IGNORECASE,
)
RE_NOTIF = re.compile(
    r"\b(?:notification|notif|nt\.?|notif\.?)\s*(?P<num>\d{1,3})\s*/\s*(?P<yr>\d{4})\b",
    re.IGNORECASE,
)
RE_NOTIF_BARE = re.compile(
    r"\b(?P<num>\d{1,3})\s*/\s*(?P<yr>\d{4})\s*[- ]?\s*(?P<series>CT\(R\)|IT\(R\)|UT\(R\)|Cus|ADD)\b",
    re.IGNORECASE,
)
RE_SNO = re.compile(
    r"\bS\.?\s*No\.?\s*(?P<sno>\d+)\b",
    re.IGNORECASE,
)
RE_NOTIF_SNO_TRIPLE = re.compile(
    r"\b(?:S\.?\s*No\.?\s*)?(?P<sno>\d+)\b[^\n]{0,80}?\b(?P<num>50/2017|01/2017|02/2017|12/2017|13/2017|05/2017)\b",
    re.IGNORECASE,
)
RE_LIST_KEYWORD = re.compile(
    r"\b(RCM|reverse\s+charge|inverted\s+duty|negative\s+list|nil\s*rated|nil-rated|exempt(?:ed|ion)?)\b",
    re.IGNORECASE,
)
RE_DATE = re.compile(
    r"\b(?P<y>20\d{2})[-/](?P<m>\d{1,2})[-/](?P<d>\d{1,2})\b|"
    r"\b(?P<d2>\d{1,2})[-/](?P<m2>\d{1,2})[-/](?P<y2>20\d{2})\b|"
    r"\b(?:as\s+of|on)\s+(?P<y3>20\d{2})\b"
)


# ---------- helpers ----------

def _rows(cur) -> List[Dict[str, Any]]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _extract_date(q: str) -> Optional[str]:
    m = RE_DATE.search(q)
    if not m:
        return None
    if m.group("y"):
        return f"{m.group('y')}-{int(m.group('m')):02d}-{int(m.group('d')):02d}"
    if m.group("y2"):
        return f"{m.group('y2')}-{int(m.group('m2')):02d}-{int(m.group('d2')):02d}"
    if m.group("y3"):
        return f"{m.group('y3')}-12-31"  # treat "as of 2023" as end-of-year
    return None


def _extract_notif_id(q: str) -> Optional[str]:
    # Prefer explicit "50/2017-Cus" style
    m = RE_NOTIF_BARE.search(q)
    if m:
        series = m.group("series").upper().replace(" ", "")
        return f"{int(m.group('num'))}/{m.group('yr')}-{series}"
    m = RE_NOTIF.search(q)
    if m:
        # Guess series from surrounding context
        rest = q.lower()
        if "cus" in rest or "customs" in rest or "bcd" in rest:
            series = "Cus"
        elif "igst" in rest or "it(r)" in rest or "integrated" in rest:
            series = "IT(R)"
        elif "ut(r)" in rest:
            series = "UT(R)"
        else:
            series = "CT(R)"
        return f"{int(m.group('num'))}/{m.group('yr')}-{series}"
    return None


def _pick_code(q: str) -> Optional[str]:
    m = RE_HSN_SAC.search(q)
    if m:
        return m.group("code")
    # bare code only if accompanied by a rate/duty phrase
    if RE_RATE_PHRASE.search(q):
        m = RE_BARE_CODE.search(q)
        if m:
            c = m.group("code")
            # avoid catching years etc. by requiring 4+ digits and not 19xx/20xx
            if not (c.startswith("19") or c.startswith("20")) and len(c) in (4, 6, 8):
                return c
    return None


# ---------- main entry ----------

def maybe_route(
    query: str,
    conn: Optional[sqlite3.Connection] = None,
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """
    Try to answer `query` from tariff.db. Returns (result, reason).

    result = None  -> no SQL rule matched; caller MUST fall through to RAG.
    result = []    -> a rule matched but DB returned zero rows; caller SHOULD fall through to RAG
                      but may surface the route_reason as diagnostic.
    result = [...] -> authoritative SQL rows.
    """
    close_after = False
    if conn is None:
        conn = sqlite3.connect(str(DEFAULT_DB))
        close_after = True
    try:
        q = query.strip()

        # --- Rule A: S.No. X of 50/2017 / 01/2017 / 12/2017 ---
        m = RE_NOTIF_SNO_TRIPLE.search(q)
        if m:
            sno = int(m.group("sno"))
            num = m.group("num")  # e.g. '50/2017'
            series = "Cus" if num.startswith("50/") else "CT(R)"
            notif_id = f"{num}-{series}"
            cur = conn.execute(
                "SELECT notif_id, sno, code, description, std_rate, igst_rate, condition_no "
                "FROM exemptions WHERE notif_id=? AND sno=?",
                (notif_id, sno),
            )
            rows = _rows(cur)
            if rows:
                return rows, f"exemption-lookup:{notif_id}:sno={sno}"
            # fallback to list_membership
            cur = conn.execute(
                "SELECT notif_id, sno, code, description, list_type FROM list_membership "
                "WHERE notif_id=? AND sno=?",
                (notif_id, sno),
            )
            rows = _rows(cur)
            return (rows if rows else []), (
                f"notif-sno-lookup:{notif_id}:sno={sno}" + ("" if rows else "-empty")
            )

        # --- Rule B: list-keyword (RCM / inverted duty / exempt / nil) ---
        lm = RE_LIST_KEYWORD.search(q)
        if lm:
            kw = lm.group(1).lower()
            list_type = (
                "RCM" if "rcm" in kw or "reverse" in kw
                else "INVERTED_DUTY" if "inverted" in kw
                else "EXEMPT" if "exempt" in kw
                else "NIL"
            )
            code = _pick_code(q)
            if code:
                # NIL and EXEMPT are colloquially interchangeable — try both.
                types = [list_type] + (["EXEMPT"] if list_type == "NIL" else [])
                placeholders = ",".join("?" * len(types))
                cur = conn.execute(
                    f"SELECT code, list_type, sno, description, notif_id, effective_from, effective_to "
                    f"FROM list_membership WHERE code=? AND list_type IN ({placeholders}) "
                    f"AND (effective_to IS NULL OR effective_to > date('now'))",
                    (code, *types),
                )
                rows = _rows(cur)
                return (rows if rows else []), (
                    f"list-membership:{list_type}:code={code}" + ("" if rows else "-empty")
                )
            # No code — list all currently-in-force entries of that type (capped)
            cur = conn.execute(
                "SELECT code, list_type, sno, description, notif_id FROM list_membership "
                "WHERE list_type=? AND (effective_to IS NULL OR effective_to > date('now')) "
                "ORDER BY sno LIMIT 50",
                (list_type,),
            )
            rows = _rows(cur)
            return (rows if rows else []), (
                f"list-membership:{list_type}:all" + ("" if rows else "-empty")
            )

        # --- Rule C: rate/duty lookup by HSN/SAC code ---
        code = _pick_code(q)
        if code and (RE_RATE_PHRASE.search(q) or RE_HSN_SAC.search(q)):
            date = _extract_date(q) or "9999-12-31"
            # Customs BCD? look in exemptions too
            is_customs = bool(re.search(r"\b(BCD|customs|import\s+duty)\b", q, re.IGNORECASE))
            cur = conn.execute(
                "SELECT levy_type, rate_pct, rate_specific, schedule, sno, notif_id, "
                "effective_from, effective_to FROM rates "
                "WHERE code=? AND effective_from <= ? "
                "AND (effective_to IS NULL OR effective_to > ?) "
                "ORDER BY levy_type",
                (code, date, date),
            )
            rows = _rows(cur)
            if is_customs or not rows:
                cur2 = conn.execute(
                    "SELECT notif_id, sno, description, std_rate, igst_rate, condition_no "
                    "FROM exemptions WHERE code=?",
                    (code,),
                )
                xrows = _rows(cur2)
                if xrows:
                    rows = rows + [{"_customs": True, **r} for r in xrows]
            return (rows if rows else []), (
                f"rate-lookup:code={code}:date={date}" + ("" if rows else "-empty")
            )

        # --- Rule D: notification-only lookup (no sno) ---
        notif_id = _extract_notif_id(q)
        if notif_id:
            cur = conn.execute(
                "SELECT notif_id, series, number, year, issued_on, effective_from, title "
                "FROM notifications WHERE notif_id=?",
                (notif_id,),
            )
            rows = _rows(cur)
            return (rows if rows else []), (
                f"notif-meta:{notif_id}" + ("" if rows else "-empty")
            )

        return None, "no-match"
    finally:
        if close_after:
            conn.close()


if __name__ == "__main__":
    import sys, json
    q = " ".join(sys.argv[1:]) or "What is the GST rate on HSN 1006?"
    res, reason = maybe_route(q)
    print(f"QUERY:  {q}")
    print(f"REASON: {reason}")
    print(f"RESULT: {json.dumps(res, indent=2, default=str)}")
