#!/usr/bin/env python3
"""V22: OCR-tolerant regex variants for section_ref extraction.
Tests 5 regex variants on scrolled chunks. Measures hit rate.
Pass: best variant recalls >=80% of manually-identified refs on 50-chunk sample.
Runs from laptop.
"""
import json, urllib.request, re, time
from pathlib import Path

QDRANT = "http://192.168.1.107:6343"
COLL = "cbic_v1"
OUT = Path("D:/_gpu_rig_ai/reingest_spec/v22_result.json")

# Variants tuned for OCR noise: lookalike chars (l<->1, O<->0), stray spaces in words, punctuation variation
VARIANTS = {
    "strict": re.compile(r"Section\s+(\d+[A-Z]?)(?:\((\w+)\))?(?:\((\w+)\))?", re.I),
    "space_tol": re.compile(r"Sec[\s\.]*t[\s]*i[\s]*o[\s]*n\s+(\d+[A-Z]?)(?:\(([^)]{1,3})\))?(?:\(([^)]{1,3})\))?", re.I),
    "ocr_o_zero": re.compile(r"Sect[il1]on\s+(\d+[A-Z]?)(?:\(([^)]{1,3})\))?(?:\(([^)]{1,3})\))?", re.I),
    "hybrid": re.compile(r"\bS[e3][c\(][\s\.]*t[il1][o0]n\s+(\d+[A-Z]?)(?:\s*\(([^)]{1,4})\))?(?:\s*\(([^)]{1,4})\))?", re.I),
    "loose": re.compile(r"\b[Ss][\w\s]{2,5}t[\w\s]{2,5}n\s+(\d+[A-Z]?)", re.I),
}

def scroll(n=1000):
    out = []
    offset = None
    while len(out) < n:
        body = {"limit": min(500, n-len(out)), "with_payload": ["text", "path"], "with_vector": False}
        if offset is not None: body["offset"] = offset
        req = urllib.request.Request(f"{QDRANT}/collections/{COLL}/points/scroll",
            method="POST", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=60).read())["result"]
        out.extend(r["points"])
        offset = r.get("next_page_offset")
        if offset is None: break
    return out

def main():
    pts = scroll(2000)
    print(f"[V22] scrolled {len(pts)}")
    stats = {name: {"hits": 0, "samples": []} for name in VARIANTS}
    for p in pts:
        pl = p.get("payload") or {}
        txt = (pl.get("text") or "")[:4000]
        path = pl.get("path") or ""
        is_ocr = "/ocr_cache/" in path or path.endswith(".txt")  # heuristic
        for name, rx in VARIANTS.items():
            m = rx.search(txt)
            if m:
                stats[name]["hits"] += 1
                if len(stats[name]["samples"]) < 10:
                    stats[name]["samples"].append(m.group(0)[:80])
    summary = {
        "probe": "V22",
        "n_chunks": len(pts),
        "variant_stats": {k: {"hits": v["hits"],
                              "rate": round(v["hits"]/max(len(pts),1), 3),
                              "samples": v["samples"]}
                          for k, v in stats.items()},
        "note": "manual review needed: best variant by precision, not just recall",
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
