#!/usr/bin/env python3
"""V12: Inspect 7 OCR'd PDFs that produced <20-byte output.
Classify: genuinely blank | OCR failure | re-OCR with higher DPI fixes.
Run on rig.
"""
import sqlite3, json, os
from pathlib import Path

QA = "/opt/indian-legal-ai/data/scraped/cbic/_qa.sqlite"
OCR = Path("/opt/indian-legal-ai/data/ocr_cache")
OUT = Path("/opt/indian-legal-ai/data/probes/v12_result.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    c = sqlite3.connect(QA)
    rows = c.execute("SELECT sha256, path, pages FROM qa WHERE image_only=1").fetchall()
    empties = []
    for sha, path, pages in rows:
        t = OCR / f"{sha}.txt"
        if not t.exists(): continue
        sz = t.stat().st_size
        if sz < 20:
            empties.append({"sha": sha[:12], "path": path, "pages": pages,
                            "ocr_bytes": sz, "pdf_exists": os.path.exists(path),
                            "pdf_bytes": os.path.getsize(path) if os.path.exists(path) else 0})
    summary = {"probe": "V12", "empty_count": len(empties), "empties": empties,
               "pass_gate": "manual", "action": "re-OCR each at dpi=300 with table prompt, then classify"}
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
