#!/usr/bin/env python3
"""V3: langdetect reliability on CBIC chunks.
Sample 200 chunks across collection, run langdetect, flag 20 for manual review.
Pass: <=5% unknown; manual spot-check needed for final accuracy.
Runs on rig (needs langdetect). If not installed: pip install langdetect
"""
import json, urllib.request, random
from pathlib import Path
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
except ImportError:
    print("pip install langdetect"); raise

QDRANT = "http://127.0.0.1:6343"  # rig-local
COLL = "cbic_v1"
OUT = Path("/opt/indian-legal-ai/data/probes/v3_result.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

def scroll(n=200):
    out = []
    offset = None
    while len(out) < n:
        body = {"limit": min(500, n*2), "with_payload": ["text"], "with_vector": False}
        if offset is not None: body["offset"] = offset
        req = urllib.request.Request(f"{QDRANT}/collections/{COLL}/points/scroll",
            method="POST", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=60).read())["result"]
        out.extend(r["points"])
        offset = r.get("next_page_offset")
        if offset is None: break
    random.Random(42).shuffle(out)
    return out[:n]

def main():
    pts = scroll(200)
    print(f"[V3] sampled {len(pts)}")
    counts = {}
    review = []
    errors = 0
    for p in pts:
        txt = (p.get("payload") or {}).get("text") or ""
        snippet = txt[:500]
        try:
            lang = detect(snippet) if snippet.strip() else "empty"
        except Exception:
            lang = "err"; errors += 1
        counts[lang] = counts.get(lang, 0) + 1
        if len(review) < 20:
            review.append({"lang": lang, "snippet": snippet[:200]})
    summary = {
        "probe": "V3",
        "n": len(pts),
        "lang_counts": counts,
        "errors": errors,
        "manual_review_samples": review,
        "pass_gate": errors / max(len(pts), 1) <= 0.05,
        "note": "manual spot-check of 20 samples required to confirm accuracy",
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!="manual_review_samples"}, indent=2))

if __name__ == "__main__":
    main()
