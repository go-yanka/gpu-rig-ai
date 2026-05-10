#!/usr/bin/env python3
"""B-OCR-API: Gemini 2.0 Flash vision OCR on 10 CBIC PDFs.
Same 10 PDFs logic as b_ocr_vl.py (seed=42). Measure s/page, chars, cost.
Target: <4 s/page, <$0.01/page."""
import os, json, time, random, base64, io, urllib.request
from pathlib import Path
from pdf2image import convert_from_path

random.seed(42)
CORPUS = Path("/opt/indian-legal-ai/data/scraped/cbic")
OUT = Path("/mnt/d/_gpu_rig_ai/benchmarks/b_ocr_api_results.json")

# Load key from .env
ENV = Path("/mnt/d/_gpu_rig_ai/.env")
for line in ENV.read_text().splitlines():
    if line.startswith("GEMINI_API_KEY="):
        API_KEY = line.split("=", 1)[1].strip()
        break

MODEL = "gemini-2.0-flash"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

PROMPT = "Extract all text from this scanned document page verbatim. Preserve structure, tables, lists. Output only the extracted text, no commentary, no markdown fences."

# Gemini 2.0 Flash pricing (as of 2026-04)
PRICE_IN = 0.10 / 1_000_000   # $/input token
PRICE_OUT = 0.40 / 1_000_000  # $/output token


def ocr_page(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    body = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "image/png", "data": b64}},
                {"text": PROMPT},
            ]
        }],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 4096},
    }
    req = urllib.request.Request(URL, method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(body).encode())
    resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
    text = resp["candidates"][0]["content"]["parts"][0]["text"]
    usage = resp.get("usageMetadata", {})
    in_tok = usage.get("promptTokenCount", 0)
    out_tok = usage.get("candidatesTokenCount", 0)
    return text, in_tok, out_tok


pdfs = list(CORPUS.rglob("*.pdf"))
random.shuffle(pdfs)
pdfs = pdfs[:10]

per_pdf = []
total_pages = 0
total_in = 0
total_out = 0
total_chars = 0
t0 = time.perf_counter()

for pdf in pdfs:
    t = time.perf_counter()
    try:
        images = convert_from_path(str(pdf), dpi=150, first_page=1, last_page=3)
    except Exception as e:
        print(f"  {pdf.name}: convert err {e}", flush=True)
        continue
    chars = 0
    pdf_in = 0
    pdf_out = 0
    for img in images:
        try:
            text, in_t, out_t = ocr_page(img)
            chars += len(text)
            pdf_in += in_t
            pdf_out += out_t
        except Exception as e:
            print(f"  {pdf.name}: ocr err {e}", flush=True)
    dt = time.perf_counter() - t
    npg = len(images)
    total_pages += npg
    total_in += pdf_in
    total_out += pdf_out
    total_chars += chars
    per_pdf.append({
        "pdf": pdf.name, "pages": npg, "seconds": dt,
        "s_per_page": dt / max(npg, 1), "chars": chars,
        "in_tok": pdf_in, "out_tok": pdf_out,
    })
    print(f"  {pdf.name}: {npg}pg {dt:.1f}s {dt/max(npg,1):.2f}s/pg chars={chars} in={pdf_in} out={pdf_out}", flush=True)

wall = time.perf_counter() - t0
s_per_pg = wall / max(total_pages, 1)
cost = total_in * PRICE_IN + total_out * PRICE_OUT
cost_per_pg = cost / max(total_pages, 1)

result = {
    "model": MODEL,
    "n_pdfs": len(per_pdf),
    "total_pages": total_pages,
    "wall_seconds": wall,
    "seconds_per_page": s_per_pg,
    "total_in_tok": total_in,
    "total_out_tok": total_out,
    "total_chars": total_chars,
    "cost_usd": cost,
    "cost_per_page_usd": cost_per_pg,
    "target_s_per_page": 4.0,
    "passed": s_per_pg < 4.0,
    "projected_471pdfs_x15pg_hours": (471 * 15 * s_per_pg) / 3600,
    "projected_471pdfs_x15pg_cost_usd": 471 * 15 * cost_per_pg,
    "per_pdf": per_pdf,
}
OUT.write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != "per_pdf"}, indent=2))
print("VERDICT:", "PASS" if result["passed"] else "FAIL")
