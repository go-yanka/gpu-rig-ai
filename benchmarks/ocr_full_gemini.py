#!/usr/bin/env python3
"""Full OCR run: 851 image-only CBIC PDFs via Gemini 2.0 Flash.
Resumable (skip if output exists). Parallel (ThreadPoolExecutor).
Output: /opt/indian-legal-ai/data/ocr_cache/<sha256>.txt
Progress log: /opt/indian-legal-ai/data/ocr_run.log
"""
import os, sys, json, time, io, base64, sqlite3, urllib.request, urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from pdf2image import convert_from_path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # disable decompression-bomb guard (CBIC scans can be huge)

QA_DB = "/opt/indian-legal-ai/data/scraped/cbic/_qa.sqlite"
OUT_DIR = Path("/opt/indian-legal-ai/data/ocr_cache")
LOG = Path("/opt/indian-legal-ai/data/ocr_run.log")
RESULTS = Path("/mnt/d/_gpu_rig_ai/benchmarks/ocr_full_results.json")
WORKERS = 3  # lower for tail (big PDFs) — each does 3 pages parallel = 9 concurrent
DPI = 150
MAX_PAGES_PER_PDF = 200  # safety cap
PRICE_IN = 0.10 / 1_000_000
PRICE_OUT = 0.40 / 1_000_000

ENV = Path("/mnt/d/_gpu_rig_ai/.env")
for line in ENV.read_text().splitlines():
    if line.startswith("GEMINI_API_KEY="):
        API_KEY = line.split("=", 1)[1].strip()
        break

MODEL = "gemini-2.0-flash"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"
PROMPT = "Extract all text from this scanned document page verbatim. Preserve structure, tables, lists. Output only the extracted text, no commentary, no markdown fences."

OUT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def ocr_page(img, retries=2):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    body = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/png", "data": b64}},
            {"text": PROMPT},
        ]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 4096},
    }
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(URL, method="POST",
                headers={"Content-Type": "application/json"},
                data=json.dumps(body).encode())
            resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
            cand = resp.get("candidates", [{}])[0]
            parts = cand.get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            usage = resp.get("usageMetadata", {})
            return text, usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0), None
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"
            if e.code == 429:  # rate limit
                time.sleep(5 + attempt * 10)
            elif e.code >= 500:
                time.sleep(2 + attempt * 3)
            else:
                break  # 400 etc — no retry
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(2 + attempt * 3)
    return "", 0, 0, last_err


def process_pdf(row):
    source, category, subcategory, doc_id, lang, path, bytes_, sha256, pages = row
    out_file = OUT_DIR / f"{sha256}.txt"
    meta_file = OUT_DIR / f"{sha256}.meta.json"
    if out_file.exists() and meta_file.exists():
        return {"sha256": sha256, "status": "skip", "pages": pages, "chars": 0, "in_tok": 0, "out_tok": 0, "seconds": 0}

    t0 = time.perf_counter()
    try:
        last = min(pages or MAX_PAGES_PER_PDF, MAX_PAGES_PER_PDF)
        images = convert_from_path(path, dpi=DPI, first_page=1, last_page=last)
    except Exception as e:
        log(f"  CONV ERR {sha256[:8]} {Path(path).name}: {e}")
        return {"sha256": sha256, "status": "convert_err", "error": str(e), "pages": 0, "chars": 0, "in_tok": 0, "out_tok": 0, "seconds": 0}

    # Page-level parallelism: 3 threads per PDF (on top of 10 PDF-workers)
    results_by_idx = [None] * len(images)
    PAGE_WORKERS = 3
    def _do(idx, img):
        return idx, ocr_page(img)
    with ThreadPoolExecutor(max_workers=PAGE_WORKERS) as ppe:
        futs = [ppe.submit(_do, i, img) for i, img in enumerate(images)]
        for f in as_completed(futs):
            idx, (text, it, ot, err) = f.result()
            results_by_idx[idx] = (text, it, ot, err)
    parts = []
    in_tok = out_tok = 0
    errs = 0
    for idx, (text, it, ot, err) in enumerate(results_by_idx, 1):
        if err:
            errs += 1
            parts.append(f"\n[PAGE {idx} OCR ERROR: {err}]\n")
        else:
            parts.append(f"\n--- PAGE {idx} ---\n{text}")
        in_tok += it; out_tok += ot

    full = "".join(parts)
    out_file.write_text(full)
    dt = time.perf_counter() - t0
    meta = {
        "sha256": sha256, "path": path, "pages": len(images),
        "chars": len(full), "in_tok": in_tok, "out_tok": out_tok,
        "errors": errs, "seconds": dt,
        "cost_usd": in_tok * PRICE_IN + out_tok * PRICE_OUT,
        "model": MODEL, "dpi": DPI,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_file.write_text(json.dumps(meta, indent=2))
    return {"sha256": sha256, "status": "ok" if errs == 0 else "partial",
            "pages": len(images), "chars": len(full), "in_tok": in_tok, "out_tok": out_tok,
            "seconds": dt, "errors": errs}


def main():
    c = sqlite3.connect(QA_DB)
    cur = c.cursor()
    rows = list(cur.execute(
        "SELECT source, category, subcategory, doc_id, lang, path, bytes, sha256, pages "
        "FROM qa WHERE image_only=1 AND path IS NOT NULL ORDER BY pages ASC"
    ))
    log(f"START: {len(rows)} PDFs, workers={WORKERS}")
    t0 = time.perf_counter()

    done = 0
    tot_pages = tot_chars = tot_in = tot_out = tot_skip = tot_err = 0
    results = []

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process_pdf, r): r for r in rows}
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            done += 1
            if res["status"] == "skip":
                tot_skip += 1
            elif res["status"] == "convert_err":
                tot_err += 1
            else:
                tot_pages += res["pages"]
                tot_chars += res["chars"]
                tot_in += res["in_tok"]
                tot_out += res["out_tok"]
            if done % 20 == 0 or done == len(rows):
                el = time.perf_counter() - t0
                cost = tot_in * PRICE_IN + tot_out * PRICE_OUT
                rate = tot_pages / max(el, 1)
                log(f"  [{done}/{len(rows)}] pages={tot_pages} chars={tot_chars} cost=${cost:.3f} rate={rate:.2f} pg/s skip={tot_skip} err={tot_err}")

    wall = time.perf_counter() - t0
    cost = tot_in * PRICE_IN + tot_out * PRICE_OUT
    summary = {
        "total_pdfs": len(rows),
        "completed": done,
        "skipped_existing": tot_skip,
        "convert_errors": tot_err,
        "total_pages_ocrd": tot_pages,
        "total_chars": tot_chars,
        "total_in_tok": tot_in,
        "total_out_tok": tot_out,
        "cost_usd": cost,
        "wall_seconds": wall,
        "wall_hours": wall / 3600,
        "avg_s_per_page": wall / max(tot_pages, 1),
    }
    RESULTS.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    log(f"DONE: {json.dumps(summary)}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
