#!/usr/bin/env python3
"""V24: Payload validator dry-run rejection rate.
Chunk 100 docs with new chunker v2, run schema validator (no upsert), count rejects.
Pass: <=2% reject rate.
Run on rig (chunker v2 must exist).
"""
import sys, json, sqlite3
from pathlib import Path

sys.path.insert(0, "/opt/indian-legal-ai/rag/cbic_rag")

QA = "/opt/indian-legal-ai/data/scraped/cbic/_qa.sqlite"
OUT = Path("/opt/indian-legal-ai/data/probes/v24_result.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

REQUIRED_FIELDS = [
    "chunk_id", "doc_id", "sha256", "source", "category",
    "lang", "text", "embed_text", "chunk_type", "text_source",
]

def validate(chunk):
    missing = [f for f in REQUIRED_FIELDS if f not in chunk or chunk[f] in (None, "")]
    if missing: return False, f"missing:{','.join(missing)}"
    if not isinstance(chunk.get("text"), str) or len(chunk["text"]) < 50:
        return False, "text<50"
    if len(chunk["text"]) > 6000:
        return False, "text>6000"
    return True, None

def main():
    try:
        from chunker import chunk_document
    except ImportError:
        summary = {"probe": "V24", "status": "need_chunker_v2"}
        OUT.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2)); return

    c = sqlite3.connect(QA)
    paths = [r[0] for r in c.execute(
        "SELECT path FROM qa WHERE path IS NOT NULL ORDER BY RANDOM() LIMIT 100")]
    total = 0; rejected = 0; reasons = {}
    for p in paths:
        try: chunks = chunk_document(p)
        except Exception as e: continue
        for ch in chunks:
            total += 1
            ok, why = validate(ch)
            if not ok:
                rejected += 1
                reasons[why] = reasons.get(why, 0) + 1

    summary = {
        "probe": "V24", "docs": len(paths), "total_chunks": total,
        "rejected": rejected,
        "reject_rate": round(rejected/max(total,1), 4),
        "reasons": reasons,
        "pass_gate": (rejected/max(total,1)) <= 0.02,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
