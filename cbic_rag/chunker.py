"""Page-aware chunking with character offsets.

Produces chunks of ~512 tokens (rough 2000 char target) with 15% overlap.
Each chunk carries (page, char_start, char_end) so the UI can deep-link.

Output chunk dict schema:
  doc_id, file_path, title, category, subcategory, download_source, source_url,
  page, char_start, char_end, text
"""
from __future__ import annotations
import subprocess, os, re, json
from dataclasses import dataclass, asdict
from typing import List, Iterable

CHAR_TARGET = 1800   # ~450 tokens
CHAR_MAX    = 2400
OVERLAP     = 270    # ~15%

USE_OCR    = os.environ.get('CHUNK_OCR', '0') == '1'
USE_TABLES = os.environ.get('CHUNK_TABLES', '0') == '1'
OCR_MIN_CHARS = int(os.environ.get('OCR_MIN_CHARS', '80'))

@dataclass
class Chunk:
    doc_id: str
    file_path: str
    title: str
    category: str
    subcategory: str
    download_source: str
    source_url: str
    page: int            # 1-based page number
    char_start: int      # char offset within that page
    char_end: int
    text: str


def extract_pages(pdf_path: str) -> List[str]:
    """Return a list of page texts using pdftotext; layout-preserving."""
    try:
        out = subprocess.check_output(
            ['pdftotext', '-q', '-layout', '-enc', 'UTF-8', pdf_path, '-'],
            stderr=subprocess.DEVNULL, timeout=90)
        # pdftotext uses form feed (\f) as page separator
        pages = out.decode('utf-8', 'ignore').split('\f')
        # last one is usually trailing empty
        if pages and pages[-1].strip() == '':
            pages = pages[:-1]
        return pages
    except Exception:
        return []


def _split_text(text: str, char_target: int, char_max: int, overlap: int) -> List[tuple]:
    """Return list of (start, end) spans within `text`.

    Prefers to break at paragraph boundaries, then sentence, then char_max.
    Overlap is applied between consecutive chunks.
    """
    text = text.replace('\r\n', '\n')
    n = len(text)
    if n == 0:
        return []
    spans = []
    i = 0
    while i < n:
        end = min(i + char_target, n)
        # Don't break mid-paragraph if we can help it
        if end < n:
            # prefer two-newline paragraph break
            nxt = text.find('\n\n', end, min(i + char_max, n))
            if nxt != -1:
                end = nxt
            else:
                # fallback to sentence break
                m = None
                for mm in re.finditer(r'[.?!]\s+', text[end:min(i + char_max, n)]):
                    m = mm
                if m:
                    end = end + m.end()
        spans.append((i, end))
        if end == n:
            break
        i = max(end - overlap, i + 1)
    return spans


def chunk_doc(meta: dict, pdf_path: str) -> Iterable[Chunk]:
    """Chunk a single PDF. `meta` carries doc_id, title, category, subcategory,
    download_source, source_url.

    Optional: CHUNK_OCR=1 runs tesseract on pages that pdftotext emptied.
              CHUNK_TABLES=1 prepends pdfplumber-extracted tables to the first
              chunk of each page that has them.
    """
    pages = extract_pages(pdf_path)
    if not pages:
        return
    for pnum, ptext in enumerate(pages, start=1):
        # OCR fallback if pdftotext page is near-empty
        if USE_OCR and len(ptext.strip()) < OCR_MIN_CHARS:
            try:
                from ocr import ocr_page
                ocr_text = ocr_page(pdf_path, pnum)
                if ocr_text and len(ocr_text) > len(ptext):
                    ptext = ocr_text
            except Exception:
                pass
        # Optional table enrichment: markdown block prepended to first chunk
        table_md = ''
        if USE_TABLES:
            try:
                from tables import extract_tables_markdown
                table_md = extract_tables_markdown(pdf_path, pnum)
            except Exception:
                table_md = ''

        spans = _split_text(ptext, CHAR_TARGET, CHAR_MAX, OVERLAP)
        for idx, (s, e) in enumerate(spans):
            ch = ptext[s:e].strip()
            if idx == 0 and table_md:
                ch = table_md + '\n\n' + ch
            if len(ch) < 60:     # skip tiny
                continue
            yield Chunk(
                doc_id=meta['doc_id'],
                file_path=pdf_path,
                title=meta.get('title') or os.path.basename(pdf_path),
                category=meta.get('category') or '',
                subcategory=meta.get('subcategory') or '',
                download_source=meta.get('download_source') or 'cbic_primary',
                source_url=meta.get('source_url') or '',
                page=pnum,
                char_start=s,
                char_end=e,
                text=ch,
            )
