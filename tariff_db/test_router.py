"""
Test the tariff.db router against 10 bucket-1 gold items.

Each gold item specifies:
    - query: the natural-language prompt
    - must_route: True if router MUST match (non-None result) — miss = fail
    - expects: optional dict of substring or value assertions on first returned row
    - not_route: True if router MUST NOT match (None result) — route = fail (negative test)

Run:
    python test_router.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from router import maybe_route

HERE = Path(__file__).resolve().parent
DB = HERE / "tariff.db"


GOLD = [
    {
        "id": 1,
        "query": "What is the GST rate on HSN 1006?",
        "must_route": True,
        "expects": {"levy_type_any": ["CGST", "SGST", "IGST"], "rate_pct_in": [2.5, 5.0]},
    },
    {
        "id": 2,
        "query": "IGST rate on HSN 8703 motor cars",
        "must_route": True,
        "expects": {"rate_pct_in": [28.0, 14.0]},
    },
    {
        "id": 3,
        "query": "Is HSN 0801 (cashew) under RCM?",
        "must_route": True,
        "expects": {"list_type_eq": "RCM"},
    },
    {
        "id": 4,
        "query": "Show S.No. 404 of 50/2017-Cus",
        "must_route": True,
        "expects": {"sno_eq": 404, "notif_id_eq": "50/2017-Cus"},
    },
    {
        "id": 5,
        "query": "Is notification 13/2017-CT(R) still in force?",
        "must_route": True,
        "expects": {"notif_id_eq": "13/2017-CT(R)"},
    },
    {
        "id": 6,
        "query": "GST rate on chapter heading 2202 as of 2023-01-01",
        "must_route": True,
        "expects": {"rate_pct_in": [14.0, 28.0]},
    },
    {
        "id": 7,
        "query": "Goods under inverted duty structure in chapter 54",
        "must_route": True,
        "expects": {"list_type_eq": "INVERTED_DUTY"},
    },
    {
        "id": 8,
        "query": "Nil-rated supply of fresh milk HSN 0401",
        "must_route": True,
        "expects": {},
    },
    {
        "id": 9,
        "query": "BCD on imported rice HSN 1006",
        "must_route": True,
        "expects": {},
    },
    {
        "id": 10,
        "query": "How do I file a GST refund under inverted duty structure?",
        "must_route": True,  # this fires on 'inverted duty' keyword; SQL returns list
        "expects": {},
    },
    # negative control
    {
        "id": 11,
        "query": "Explain the intent of section 16 of the CGST Act regarding input tax credit eligibility",
        "must_route": False,
        "not_route": True,
    },
]


def _check_expects(expects: dict, rows: list) -> tuple[bool, str]:
    if not rows:
        return False, "no rows returned"
    r = rows[0]
    for k, v in expects.items():
        if k.endswith("_eq"):
            field = k[:-3]
            if r.get(field) != v:
                return False, f"{field}={r.get(field)!r} != {v!r}"
        elif k.endswith("_in"):
            field = k[:-3]
            vals = [row.get(field) for row in rows]
            if not any(val in v for val in vals):
                return False, f"none of {field} values {vals} in {v}"
        elif k.endswith("_any"):
            field = k[:-4]
            vals = [row.get(field) for row in rows]
            if not any(val in v for val in vals):
                return False, f"none of {field} values {vals} in {v}"
    return True, "ok"


def main() -> int:
    if not DB.exists():
        print(f"[ERROR] {DB} does not exist. Run: python ingest.py build")
        return 2
    conn = sqlite3.connect(str(DB))
    passed = 0
    failed = 0
    for g in GOLD:
        q = g["query"]
        res, reason = maybe_route(q, conn)
        routed = res is not None and len(res) > 0

        if g.get("not_route"):
            ok = res is None
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] #{g['id']} (neg) route={routed} reason={reason!r}")
            print(f"        Q: {q}")
            if ok:
                passed += 1
            else:
                failed += 1
            continue

        if g["must_route"]:
            if not routed:
                status = "MISS" if res is None else "EMPTY"
                print(f"[{status}] #{g['id']} reason={reason!r}")
                print(f"        Q: {q}")
                if res is not None:
                    # empty result set but rule fired — partial credit, still fail
                    pass
                failed += 1
                continue
            ok, why = _check_expects(g.get("expects", {}), res)
            status = "PASS" if ok else "BADDATA"
            print(f"[{status}] #{g['id']} reason={reason!r}  rows={len(res)}  {why}")
            print(f"        Q: {q}")
            if len(res) <= 3:
                for row in res[:3]:
                    print(f"        -> {row}")
            else:
                print(f"        -> (first) {res[0]}")
            if ok:
                passed += 1
            else:
                failed += 1
    conn.close()
    total = passed + failed
    print("\n" + "=" * 60)
    print(f"RESULT: {passed}/{total} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
