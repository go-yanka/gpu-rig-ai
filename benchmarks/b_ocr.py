#!/usr/bin/env python3
"""B-OCR: RapidOCR CPU on 10 CBIC PDFs. Target <4s/page."""
import json, time, random, os
from pathlib import Path
from pdf2image import convert_from_path
from rapidocr_onnxruntime import RapidOCR

random.seed(42)
CORPUS = Path("/opt/indian-legal-ai/data/scraped/cbic")
OUT = Path("/mnt/d/_gpu_rig_ai/benchmarks/b_ocr_results.json")

pdfs = list(CORPUS.rglob("*.pdf"))
random.shuffle(pdfs)
pdfs = pdfs[:10]
print(f"Sampled {len(pdfs)} PDFs", flush=True)

ocr = RapidOCR()
per_pdf = []
total_pages = 0
t0 = time.perf_counter()

for pdf in pdfs:
    t = time.perf_counter()
    try:
        images = convert_from_path(str(pdf), dpi=200, first_page=1, last_page=5)
    except Exception as e:
        print(f"  {pdf.name}: convert err {e}", flush=True)
        continue
    pages_text = []
    for img in images:
        result, _ = ocr(img)
        if result:
            pages_text.append("\n".join(r[1] for r in result))
    dt = time.perf_counter() - t
    npg = len(images)
    total_pages += npg
    per_pdf.append({"pdf": pdf.name, "pages": npg, "seconds": dt, "s_per_page": dt/max(npg,1), "chars": sum(len(t) for t in pages_text)})
    print(f"  {pdf.name}: {npg}pg {dt:.1f}s {dt/max(npg,1):.2f}s/pg chars={per_pdf[-1]['chars']}", flush=True)

wall = time.perf_counter() - t0
s_per_pg = wall / max(total_pages, 1)
result = {
    "n_pdfs": len(per_pdf),
    "total_pages": total_pages,
    "wall_seconds": wall,
    "seconds_per_page": s_per_pg,
    "target_seconds_per_page": 4.0,
    "passed": s_per_pg < 4.0,
    "per_pdf": per_pdf,
    "projected_471_pdfs_hours": (471 * (total_pages/max(len(per_pdf),1)) * s_per_pg) / 3600,
}
OUT.write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != 'per_pdf'}, indent=2))
print("VERDICT:", "PASS" if result["passed"] else "FAIL")
