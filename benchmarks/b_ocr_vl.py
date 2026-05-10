#!/usr/bin/env python3
"""B-OCR-VL: Qwen2.5-VL-7B Vulkan GPU OCR on 10 CBIC PDFs.
Measure wall time and s/page. Target <4 s/page."""
import json, time, random, base64, io
from pathlib import Path
from pdf2image import convert_from_path
import urllib.request

random.seed(42)
CORPUS = Path("/opt/indian-legal-ai/data/scraped/cbic")
OUT = Path("/mnt/d/_gpu_rig_ai/benchmarks/b_ocr_vl_results.json")
URL = "http://127.0.0.1:8780/v1/chat/completions"

PROMPT = "Extract all text from this scanned document page verbatim. Preserve structure. Output only the text, no commentary."

def ocr_page(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    body = {
        "model": "qwen2.5-vl",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": PROMPT},
        ]}],
        "max_tokens": 1500,
        "temperature": 0.1,
    }
    req = urllib.request.Request(URL, method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(body).encode())
    resp = json.loads(urllib.request.urlopen(req, timeout=300).read())
    return resp["choices"][0]["message"]["content"]

pdfs = list(CORPUS.rglob("*.pdf"))
random.shuffle(pdfs)
pdfs = pdfs[:10]

per_pdf = []
total_pages = 0
t0 = time.perf_counter()

for pdf in pdfs:
    t = time.perf_counter()
    try:
        images = convert_from_path(str(pdf), dpi=150, first_page=1, last_page=3)
    except Exception as e:
        print(f"  {pdf.name}: convert err {e}", flush=True)
        continue
    chars = 0
    for img in images:
        try:
            text = ocr_page(img)
            chars += len(text)
        except Exception as e:
            print(f"  {pdf.name}: ocr err {e}", flush=True)
    dt = time.perf_counter() - t
    npg = len(images)
    total_pages += npg
    per_pdf.append({"pdf": pdf.name, "pages": npg, "seconds": dt, "s_per_page": dt/max(npg,1), "chars": chars})
    print(f"  {pdf.name}: {npg}pg {dt:.1f}s {dt/max(npg,1):.2f}s/pg chars={chars}", flush=True)

wall = time.perf_counter() - t0
s_per_pg = wall / max(total_pages, 1)
result = {
    "n_pdfs": len(per_pdf),
    "total_pages": total_pages,
    "wall_seconds": wall,
    "seconds_per_page": s_per_pg,
    "target_s_per_page": 4.0,
    "passed": s_per_pg < 4.0,
    "projected_471pdfs_x15pg_hours": (471 * 15 * s_per_pg) / 3600,
    "per_pdf": per_pdf,
}
OUT.write_text(json.dumps(result, indent=2))
print(json.dumps({k:v for k,v in result.items() if k != 'per_pdf'}, indent=2))
print("VERDICT:", "PASS" if result["passed"] else "FAIL")
