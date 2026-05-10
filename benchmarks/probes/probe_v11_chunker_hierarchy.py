#!/usr/bin/env python3
"""V11: Chunker v2 parent_hierarchy_text emission.
Chunk 5 known-structure docs, dump first 20 chunks, manual verify.
Pass: 5/5 docs have correct hierarchy breadcrumb on first chunk of each section.
Run on rig; requires chunker v2 implementation (DRAFT: test against current chunker, flag gaps).
"""
import sys, json
from pathlib import Path

sys.path.insert(0, "/opt/indian-legal-ai/rag/cbic_rag")

OUT = Path("/opt/indian-legal-ai/data/probes/v11_result.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Pick 5 well-structured docs (replace with real paths from manifest)
KNOWN_DOCS = [
    # (path, expected_first_section_breadcrumb)
    # TODO: fill in 5 actual rig paths before running
]

def main():
    if not KNOWN_DOCS:
        print("TODO: populate KNOWN_DOCS with 5 rig paths + expected breadcrumbs")
        summary = {"probe": "V11", "status": "not_runnable_yet",
                   "action": "populate KNOWN_DOCS list after reviewing manifest"}
        OUT.write_text(json.dumps(summary, indent=2))
        return

    try:
        from chunker import chunk_document
    except ImportError as e:
        print(f"chunker v1 may lack chunk_document; need chunker v2: {e}")
        summary = {"probe": "V11", "status": "needs_chunker_v2",
                   "current_chunker_emits_hierarchy": False}
        OUT.write_text(json.dumps(summary, indent=2))
        return

    results = []
    for path, expected in KNOWN_DOCS:
        chunks = chunk_document(path)
        first = chunks[0] if chunks else {}
        actual = first.get("parent_hierarchy_text", "")
        results.append({"path": path, "expected": expected, "actual": actual,
                        "match": expected in actual if expected else False})

    passes = sum(1 for r in results if r["match"])
    summary = {"probe": "V11", "n": len(results), "passes": passes,
               "pass_gate": passes == len(results), "results": results}
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
