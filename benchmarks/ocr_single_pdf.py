#!/usr/bin/env python3
"""OCR a single PDF by sha256. Runs pages with high parallelism.
Usage: python3 ocr_single_pdf.py <sha256> [PAGE_WORKERS=8]
"""
import os, sys, json, time, io, base64, sqlite3, urllib.request, urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from pdf2image import convert_from_path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

QA_DB = "/opt/indian-legal-ai/data/scraped/cbic/_qa.sqlite"
OUT_DIR = Path("/opt/indian-legal-ai/data/ocr_cache")
DPI = 150
PRICE_IN = 0.10 / 1_000_000
PRICE_OUT = 0.40 / 1_000_000

sha = sys.argv[1]
PAGE_WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 8

ENV = Path("/mnt/d/_gpu_rig_ai/.env")
for line in ENV.read_text().splitlines():
    if line.startswith("GEMINI_API_KEY="):
        API_KEY = line.split("=", 1)[1].strip()
        break
MODEL = "gemini-2.0-flash"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"
PROMPT = "Extract all text from this scanned document page verbatim. Preserve structure, tables, lists. Output only the extracted text, no commentary, no markdown fences."

def ocr_page(img, retries=3):
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
            resp = json.loads(urllib.request.urlopen(req, timeout=180).read())
            cand = resp.get("candidates", [{}])[0]
            parts = cand.get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            usage = resp.get("usageMetadata", {})
            return text, usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0), None
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"
            if e.code == 429:
                time.sleep(5 + attempt * 10)
            elif e.code >= 500:
                time.sleep(2 + attempt * 3)
            else:
                break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(2 + attempt * 3)
    return "", 0, 0, last_err


def main():
    c = sqlite3.connect(QA_DB)
    row = c.execute("SELECT path, pages FROM qa WHERE sha256=? AND image_only=1", (sha,)).fetchone()
    if not row:
        print(f"NOT FOUND: {sha}"); sys.exit(1)
    path, pages = row
    out_file = OUT_DIR / f"{sha}.txt"
    meta_file = OUT_DIR / f"{sha}.meta.json"
    if meta_file.exists():
        print(f"SKIP exists: {sha[:12]}"); return

    t0 = time.perf_counter()
    print(f"[{sha[:12]}] rendering {pages}pg @ {DPI}dpi...", flush=True)
    images = convert_from_path(path, dpi=DPI, first_page=1, last_page=pages)
    print(f"[{sha[:12]}] rendered {len(images)}pg in {time.perf_counter()-t0:.1f}s, OCR with {PAGE_WORKERS} workers", flush=True)

    results_by_idx = [None] * len(images)
    def _do(idx, img):
        r = ocr_page(img)
        if idx % 20 == 0:
            print(f"[{sha[:12]}] done page {idx}", flush=True)
        return idx, r
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
        "sha256": sha, "path": path, "pages": len(images),
        "chars": len(full), "in_tok": in_tok, "out_tok": out_tok,
        "errors": errs, "seconds": dt,
        "cost_usd": in_tok * PRICE_IN + out_tok * PRICE_OUT,
        "model": MODEL, "dpi": DPI,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_file.write_text(json.dumps(meta, indent=2))
    print(f"[{sha[:12]}] DONE {len(images)}pg {len(full)}ch errs={errs} {dt:.1f}s ${meta['cost_usd']:.4f}", flush=True)

if __name__ == "__main__":
    main()
