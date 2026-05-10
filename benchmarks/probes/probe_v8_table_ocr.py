#!/usr/bin/env python3
"""V8: Gemini 2.0 Flash table-aware OCR prompt quality.
Re-OCR 10 pages from docs known to have tables with table-aware prompt.
Manual review needed after: >=8/10 tables reconstructable as markdown.
Run on rig (needs poppler for pdf2image + GEMINI_API_KEY in .env).
"""
import os, sys, json, time, io, base64, urllib.request
from pathlib import Path

ENV = Path("/mnt/d/_gpu_rig_ai/.env")
for line in ENV.read_text().splitlines():
    if line.startswith("GEMINI_API_KEY="):
        API_KEY = line.split("=",1)[1].strip(); break

MODEL = "gemini-2.0-flash"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"
PROMPT_TABLE = (
    "Extract all text from this scanned page verbatim. CRITICAL: if the page contains tables, "
    "reproduce them as GitHub-flavored markdown tables with aligned columns. Preserve every "
    "numeric value, every column header, every row label. Do NOT merge cells. Do NOT summarize. "
    "Output plain text + markdown tables only, no fences, no commentary."
)

# Caller supplies table-page paths via arg or env TABLE_SAMPLES (comma-sep)
OUT = Path("/opt/indian-legal-ai/data/probes/v8_result.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

def ocr(img_bytes):
    b64 = base64.b64encode(img_bytes).decode()
    body = {"contents": [{"parts": [
        {"inline_data": {"mime_type": "image/png", "data": b64}},
        {"text": PROMPT_TABLE}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 4096}}
    req = urllib.request.Request(URL, method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=120).read())
    return "".join(p.get("text","") for p in r["candidates"][0]["content"]["parts"])

def main():
    samples_arg = os.environ.get("V8_SAMPLES", "")
    if not samples_arg:
        print("set V8_SAMPLES=pdf1:page,pdf2:page,... (full paths)")
        print("Recommended pick: 10 PDFs from qa table with 'schedule' or 'notification' containing tables")
        sys.exit(1)
    from pdf2image import convert_from_path
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    results = []
    for spec in samples_arg.split(","):
        path, page = spec.rsplit(":", 1); page = int(page)
        imgs = convert_from_path(path, dpi=200, first_page=page, last_page=page)
        buf = io.BytesIO(); imgs[0].save(buf, format="PNG")
        t0 = time.time()
        try:
            txt = ocr(buf.getvalue())
            dt = time.time() - t0
            md_tables = txt.count("| ")  # rough marker for md tables
            results.append({"pdf": path, "page": page, "chars": len(txt),
                            "md_table_lines": md_tables, "seconds": round(dt,1),
                            "preview": txt[:500]})
            print(f"  {Path(path).name}:p{page} {len(txt)}ch md_tbl_lines={md_tables} {dt:.1f}s")
        except Exception as e:
            results.append({"pdf": path, "page": page, "err": str(e)})

    summary = {"probe": "V8", "samples": len(results), "results": results,
               "manual_review_required": True,
               "pass_gate": "manual"}
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"[V8] wrote {OUT}")

if __name__ == "__main__":
    main()
