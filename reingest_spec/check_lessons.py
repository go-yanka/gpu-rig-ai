#!/usr/bin/env python3
"""check_lessons.py — verifies LESSONS_APPLIED.md rows against real code.

For every row whose STATUS claims APPLIED, parse the "v2 fix location" cell
(format: `path:line identifier` or `path identifier`) and confirm the
identifier is still present in the file. If not, print FAIL and exit 1.

Run this as Stage-B/C/D/E/F exit-gate check. Must exit 0 before proceeding.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
LEDGER = ROOT / "reingest_spec" / "LESSONS_APPLIED.md"

ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|([^|]+)\|([^|]+)\|\s*(APPLIED|PARTIAL|MISSING|BROKEN|INCOMPLETE|PENDING|UNVERIFIED)[^|]*\|", re.I)
LOC_RE = re.compile(r"`([^`]+)`")


def main() -> int:
    if not LEDGER.exists():
        print(f"[FAIL] {LEDGER} missing")
        return 2
    text = LEDGER.read_text(encoding="utf-8")
    fails = []
    checked = 0
    for line in text.splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        row_n, pain, loc, status = m.groups()
        status = status.strip().upper()
        if status != "APPLIED":
            continue
        code = LOC_RE.search(loc)
        if not code:
            fails.append(f"row {row_n}: no `code-ref` in location cell")
            continue
        ref = code.group(1)
        # strip leading line number "path.py:NNN ident" → "path.py"
        parts = ref.split()
        path_part = parts[0]
        ident = " ".join(parts[1:]).strip() or None
        path_only = path_part.split(":")[0]
        # Resolve: try project root, then reingest_spec/, then cbic_rag/
        candidates = [ROOT / path_only,
                      ROOT / "reingest_spec" / path_only,
                      ROOT / "cbic_rag" / path_only,
                      ROOT / "rag" / "cbic_rag" / path_only]  # rig layout
        full = next((p for p in candidates if p.exists()), None)
        if full is None:
            fails.append(f"row {row_n}: {path_only} not found under {ROOT}")
            continue
        content = full.read_text(encoding="utf-8", errors="replace")
        if ident and ident not in content:
            # try last token only
            tok = ident.split()[-1]
            if tok not in content:
                fails.append(f"row {row_n}: identifier '{ident}' not in {path_only}")
                continue
        checked += 1
    if fails:
        print(f"[FAIL] {len(fails)} ledger rows failed verification:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(f"[OK] {checked} APPLIED rows verified in ledger")
    return 0


if __name__ == "__main__":
    sys.exit(main())
