#!/usr/bin/env python3
import base64, json, subprocess, time, pathlib, urllib.request, sys

DOCS = [
    ("cbic-form-msts_1000237", "/opt/indian-legal-ai/data/scraped/cbic/customs/forms/unknown/SCMTR_Form_I_hi.pdf"),
    ("cbic-form-msts_1000238", "/opt/indian-legal-ai/data/scraped/cbic/customs/forms/unknown/SCMTR_Forms_II_hi.pdf"),
    ("cbic-order-msts_1000344", "/opt/indian-legal-ai/data/scraped/cbic/customs/orders/2023/off-odr-carr-01h_hi.pdf"),
    ("cbic-circular-msts_1002210", "/opt/indian-legal-ai/data/scraped/cbic/central_excise/circulars/2003/686-03-cx.pdf"),
    ("cbic-notification-msts_1004843", "/opt/indian-legal-ai/data/scraped/cbic/central_excise/notifications/2007/cent41-2k7.pdf"),
]

OUT = pathlib.Path("/tmp/ocr_poc_out")
OUT.mkdir(exist_ok=True)

PROMPT = ("You are a strict mechanical OCR system. Extract all printed text from this image verbatim. "
          "Do not correct typos. Do not format as a table unless a table is drawn. Do not add conversational filler. "
          "Output the exact English and Hindi text exactly as it appears. Preserve layout and left-to-right column order "
          "(emit one column fully before the next). Ignore stamps, seals, and handwritten signatures. "
          "Mark any illegible region as [UNREADABLE].")

def ocr_page(png_path):
    b64 = base64.b64encode(png_path.read_bytes()).decode()
    payload = {
        "model": "qwen2.5-vl",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
        }],
        "temperature": 0.0, "top_p": 1.0, "max_tokens": 2048
    }
    t0 = time.time()
    req = urllib.request.Request(
        "http://localhost:9600/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"], time.time() - t0

results = []
for doc_id, pdf_path in DOCS:
    if not pathlib.Path(pdf_path).exists():
        print(f"SKIP {doc_id} missing {pdf_path}", flush=True)
        continue
    png_prefix = f"/tmp/ocr_poc_{doc_id}"
    subprocess.run(["pdftoppm", "-r", "150", "-png", "-f", "1", "-l", "2", pdf_path, png_prefix], check=True)
    pngs = sorted(pathlib.Path("/tmp").glob(f"ocr_poc_{doc_id}-*.png"))
    for png in pngs[:2]:
        try:
            text, dt = ocr_page(png)
            (OUT / f"{png.stem}.txt").write_text(text, encoding="utf-8")
            deva = sum(1 for c in text if '\u0900' <= c <= '\u097f')
            results.append({"doc": doc_id, "png": png.name, "chars": len(text), "devanagari": deva, "latency_s": round(dt,1)})
            print(f"OK {png.name} {dt:.1f}s chars={len(text)} devanagari={deva}", flush=True)
        except Exception as e:
            results.append({"doc": doc_id, "png": png.name, "error": str(e)})
            print(f"FAIL {png.name} {e}", flush=True)

pathlib.Path("/tmp/ocr_poc_results.json").write_text(json.dumps(results, indent=2))
print("WROTE /tmp/ocr_poc_results.json", flush=True)
