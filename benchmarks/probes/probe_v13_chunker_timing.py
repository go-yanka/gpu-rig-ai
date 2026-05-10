#!/usr/bin/env python3
"""V13: Chunker run-time on 100 typical docs.
Pass: <=15 minutes for 100 -> <=3hr for 1150 docs projected.
Run on rig.
"""
import sys, time, json, sqlite3, random
from pathlib import Path

sys.path.insert(0, "/opt/indian-legal-ai/rag/cbic_rag")

QA = "/opt/indian-legal-ai/data/scraped/cbic/_qa.sqlite"
OUT = Path("/opt/indian-legal-ai/data/probes/v13_result.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    try:
        from chunker import chunk_document
    except ImportError:
        print("need chunker.chunk_document(path); adapt to actual chunker API"); raise

    c = sqlite3.connect(QA)
    rows = c.execute("SELECT path FROM qa WHERE path IS NOT NULL ORDER BY RANDOM() LIMIT 100").fetchall()
    paths = [r[0] for r in rows]
    print(f"[V13] chunking {len(paths)} random docs")

    t0 = time.time()
    total_chunks = 0; errors = 0
    for i, p in enumerate(paths):
        try:
            ch = chunk_document(p)
            total_chunks += len(ch)
        except Exception as e:
            errors += 1
        if i % 20 == 0:
            print(f"  [{i}/100] chunks_sofar={total_chunks} errs={errors}")
    dt = time.time() - t0
    summary = {
        "probe": "V13", "docs": len(paths), "chunks": total_chunks, "errors": errors,
        "seconds": round(dt, 1),
        "seconds_per_doc": round(dt/len(paths), 2),
        "projected_1150_docs_minutes": round(dt/len(paths)*1150/60, 1),
        "pass_gate": dt <= 900,  # 15 min for 100
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
