#!/usr/bin/env python3
"""V14: must_cite_verbatim audit on 170-query gold set.
Pass: 100% of citation-critical queries have non-empty must_cite_verbatim.
Run on rig (reads eval_set.json).
"""
import json
from pathlib import Path

GOLD = Path("/opt/indian-legal-ai/rag/cbic_rag/eval_set.json")
OUT = Path("/opt/indian-legal-ai/data/probes/v14_result.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

REFUSAL_SUBS = {"refuse", "refuse_direct_tax"}

def is_citation_critical(q):
    # Refusal/OOC queries must refuse, not cite — exclude them
    if q.get("subcategory") in REFUSAL_SUBS:
        return False
    # heuristic: difficulty complex/advanced OR category has explicit section in expected_sections
    return (q.get("difficulty") in ("complex", "advanced")
            or bool(q.get("expected_sections"))
            or "section" in (q.get("query") or "").lower()
            or "notification" in (q.get("query") or "").lower())

def main():
    data = json.loads(GOLD.read_text())
    qs = data.get("queries", [])
    critical = [q for q in qs if is_citation_critical(q)]
    missing = [q for q in critical if not q.get("must_cite_verbatim")]
    summary = {
        "probe": "V14",
        "total_queries": len(qs),
        "citation_critical": len(critical),
        "missing_must_cite_verbatim": len(missing),
        "missing_ids": [q.get("id") for q in missing[:20]],
        "pass_gate": len(missing) == 0,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
