#!/usr/bin/env python3
"""B-OCR parallel: RapidOCR on 4 CPU workers, 10 PDFs."""
import json, time, random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from pdf2image import convert_from_path

random.seed(42)
CORPUS = Path("/opt/indian-legal-ai/data/scraped/cbic")
OUT = Path("/mnt/d/_gpu_rig_ai/benchmarks/b_ocr_parallel_results.json")

def ocr_one(pdf_path):
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    t = time.perf_counter()
    try:
        images = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=5)
    except Exception as e:
        return {"pdf": Path(pdf_path).name, "error": str(e)}
    pages = 0
    chars = 0
    for img in images:
        result, _ = ocr(img)
        pages += 1
        if result:
            chars += sum(len(r[1]) for r in result)
    dt = time.perf_counter() - t
    return {"pdf": Path(pdf_path).name, "pages": pages, "seconds": dt, "s_per_page": dt/max(pages,1), "chars": chars}

if __name__ == "__main__":
    pdfs = list(CORPUS.rglob("*.pdf"))
    random.shuffle(pdfs)
    pdfs = [str(p) for p in pdfs[:10]]
    print(f"Running OCR on {len(pdfs)} PDFs with 4 workers", flush=True)

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(ocr_one, pdfs))
    wall = time.perf_counter() - t0

    total_pages = sum(r.get("pages", 0) for r in results)
    total_sec = sum(r.get("seconds", 0) for r in results)
    for r in results:
        print(f"  {r}", flush=True)

    effective_s_per_page = wall / max(total_pages, 1)
    serial_s_per_page = total_sec / max(total_pages, 1)
    speedup = serial_s_per_page / max(effective_s_per_page, 0.001)
    verdict = {
        "n_pdfs": len(results),
        "total_pages": total_pages,
        "wall_seconds": wall,
        "effective_s_per_page_parallel": effective_s_per_page,
        "avg_s_per_page_serial_sum": serial_s_per_page,
        "speedup": speedup,
        "target_s_per_page": 4.0,
        "passed": effective_s_per_page < 4.0,
        "projected_471pdfs_x15pg_hours": (471 * 15 * effective_s_per_page) / 3600,
    }
    OUT.write_text(json.dumps({**verdict, "per_pdf": results}, indent=2))
    print(json.dumps(verdict, indent=2))
    print("VERDICT:", "PASS" if verdict["passed"] else "FAIL")
