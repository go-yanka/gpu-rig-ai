#!/usr/bin/env python3
"""B2 gate: audit cbic_v1 Qdrant collection — chunk health without full re-ingest.
Measures: total count, payload field fill-rate, text-length distribution,
category balance, duplicate detection (sample), language spread.
Output: /tmp/b2_chunk_audit.json"""
import json, urllib.request
from collections import Counter
from pathlib import Path

QURL = "http://127.0.0.1:6343"
COLL = "cbic_v1"
OUT = Path("/mnt/d/_gpu_rig_ai/benchmarks/b2_chunk_audit.json")

def q(url, method="GET", body=None):
    req = urllib.request.Request(url, method=method)
    if body is not None:
        req.data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    return json.loads(urllib.request.urlopen(req, timeout=60).read())

# Collection info
info = q(f"{QURL}/collections/{COLL}")["result"]
total = info["points_count"]
print(f"total points: {total}")

# Scroll through N points, sample payload
SAMPLE = 5000
SCROLL_BATCH = 500
sampled = []
offset = None
while len(sampled) < SAMPLE:
    body = {"limit": SCROLL_BATCH, "with_payload": True, "with_vector": False}
    if offset is not None:
        body["offset"] = offset
    r = q(f"{QURL}/collections/{COLL}/points/scroll", "POST", body)["result"]
    pts = r["points"]
    sampled.extend(pts)
    offset = r.get("next_page_offset")
    if offset is None: break

print(f"sampled {len(sampled)} points")

# Analyze
fields = Counter()
field_fill = Counter()
cats = Counter()
langs = Counter()
doc_types = Counter()
text_lens = []
doc_ids = Counter()
sources = Counter()
no_text = 0
short_text = 0  # <50 chars

for p in sampled:
    pl = p.get("payload", {}) or {}
    for k in pl:
        fields[k] += 1
        if pl.get(k) not in (None, "", []):
            field_fill[k] += 1
    cat = pl.get("category") or pl.get("subcategory") or "?"
    cats[cat] += 1
    langs[pl.get("lang") or pl.get("language") or "?"] += 1
    doc_types[pl.get("doc_type") or pl.get("type") or "?"] += 1
    sources[pl.get("source") or "?"] += 1
    t = pl.get("text") or pl.get("chunk_text") or ""
    text_lens.append(len(t))
    if not t: no_text += 1
    elif len(t) < 50: short_text += 1
    d = pl.get("doc_id") or pl.get("path")
    if d: doc_ids[d] += 1

text_lens.sort()
n = len(text_lens)
def pct(p): return text_lens[int(n * p)] if n else 0

dup_doc_ids = sum(1 for d,c in doc_ids.items() if c > 20)  # suspicious: >20 chunks from 1 doc

report = {
    "collection": COLL,
    "total_points": total,
    "sampled": len(sampled),
    "fields_present_in_any": dict(fields.most_common()),
    "field_fill_rate": {k: round(field_fill[k]/len(sampled), 3) for k in fields},
    "category_dist": dict(cats.most_common(15)),
    "lang_dist": dict(langs.most_common()),
    "doc_type_dist": dict(doc_types.most_common(15)),
    "source_dist": dict(sources.most_common()),
    "text_len_stats": {
        "min": text_lens[0] if text_lens else 0,
        "p10": pct(0.1), "p50": pct(0.5), "p90": pct(0.9), "p99": pct(0.99),
        "max": text_lens[-1] if text_lens else 0,
        "mean": round(sum(text_lens)/n, 1) if n else 0,
    },
    "anomalies": {
        "empty_text_chunks": no_text,
        "very_short_<50_chars": short_text,
        "unique_docs_sampled": len(doc_ids),
        "docs_with_>20_chunks": dup_doc_ids,
        "max_chunks_per_doc": max(doc_ids.values()) if doc_ids else 0,
    },
}
OUT.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

# Gate verdicts
gates = []
gates.append(("G1 text fill >99%", field_fill.get("text",0) + field_fill.get("chunk_text",0) >= len(sampled) * 0.99))
gates.append(("G2 category fill >95%", sum(1 for c in cats if c != "?") / max(len(cats),1) > 0 and cats.get("?", 0) / len(sampled) < 0.05))
gates.append(("G3 median chunk 200-2000 chars", 200 <= pct(0.5) <= 2000))
gates.append(("G4 no >20% empty", no_text / len(sampled) < 0.2))
gates.append(("G5 >3 categories", len(cats) >= 3))
print("\n=== GATES ===")
for name, ok in gates:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
