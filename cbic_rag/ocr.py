"""OCR fallback for image-only CBIC PDFs.

Many older CBIC scans are rasterized — `pdftotext` returns empty or near-empty.
When the chunker sees a page with <100 chars of text, we fall back to
tesseract OCR via pytesseract + pdf2image (or pdftoppm).

Hindi + English: tesseract langs are 'eng+hin'. Install with:
    apt install tesseract-ocr tesseract-ocr-hin poppler-utils

Integration: `ocr_page(pdf_path, page_no)` returns extracted text (or '').
"""
from __future__ import annotations
import os, subprocess, tempfile

OCR_LANGS = os.environ.get('OCR_LANGS', 'eng+hin')
OCR_DPI   = int(os.environ.get('OCR_DPI', '300'))


def ocr_page(pdf_path: str, page_no: int) -> str:
    """Rasterize one page via pdftoppm and run tesseract on it."""
    with tempfile.TemporaryDirectory() as td:
        img_prefix = os.path.join(td, 'pg')
        try:
            subprocess.run(
                ['pdftoppm', '-r', str(OCR_DPI), '-f', str(page_no),
                 '-l', str(page_no), '-png', pdf_path, img_prefix],
                check=True, capture_output=True, timeout=60,
            )
        except Exception:
            return ''
        # pdftoppm names like pg-<page>.png (with zero-padding)
        pngs = sorted([f for f in os.listdir(td) if f.endswith('.png')])
        if not pngs:
            return ''
        img = os.path.join(td, pngs[0])
        try:
            out = subprocess.run(
                ['tesseract', img, '-', '-l', OCR_LANGS, '--psm', '6'],
                check=True, capture_output=True, timeout=120,
            )
            return out.stdout.decode('utf-8', errors='replace').strip()
        except Exception:
            return ''


def looks_scanned(text: str, min_chars: int = 80) -> bool:
    """Heuristic: page text shorter than threshold = probably scanned."""
    return len(text.strip()) < min_chars
