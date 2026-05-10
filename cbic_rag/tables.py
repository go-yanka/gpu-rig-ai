"""Table-aware extraction for CBIC tax PDFs.

Tax documents are full of rate tables, HSN code tables, notification schedules.
`pdftotext -layout` concatenates columns weirdly and loses row structure. We use
pdfplumber to detect tables per page and emit them as markdown-style pipe tables,
which preserve semantic rows for dense/sparse embedding.

Integration contract: `extract_tables_markdown(path, page_no)` returns a string
with markdown tables for that page, or '' if none. The chunker prepends this to
the page text so the first chunk of each tabled page contains the structured data.
"""
from __future__ import annotations
from typing import List

def _rows_to_md(rows: List[List[str]]) -> str:
    if not rows or not rows[0]:
        return ''
    # pad rows to same width
    w = max(len(r) for r in rows)
    rows = [(r + [''] * (w - len(r))) for r in rows]
    def clean(c):
        if c is None: return ''
        return str(c).replace('\n', ' ').replace('|', '/').strip()
    head = rows[0]; body = rows[1:]
    out = ['| ' + ' | '.join(clean(c) for c in head) + ' |']
    out.append('|' + '|'.join([' --- '] * w) + '|')
    for r in body:
        out.append('| ' + ' | '.join(clean(c) for c in r) + ' |')
    return '\n'.join(out)


def extract_tables_markdown(pdf_path: str, page_no: int) -> str:
    """Return markdown block for tables on 1-indexed page_no, or '' if none/error."""
    try:
        import pdfplumber
    except Exception:
        return ''
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_no < 1 or page_no > len(pdf.pages):
                return ''
            page = pdf.pages[page_no - 1]
            tables = page.extract_tables() or []
            if not tables:
                return ''
            blocks = []
            for i, t in enumerate(tables):
                md = _rows_to_md(t)
                if md:
                    blocks.append(f'[TABLE {i+1}]\n{md}\n[/TABLE {i+1}]')
            return '\n\n'.join(blocks)
    except Exception:
        return ''
