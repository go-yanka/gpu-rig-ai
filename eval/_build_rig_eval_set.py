"""Convert D:/_gpu_rig_ai/eval/gold_set.yaml (v2, 170 items) to rig eval_set.json schema.

Rig schema (from /opt/indian-legal-ai/rag/cbic_rag/eval.py):
  {"version": N, "notes": str, "queries": [{"id", "query", "category", "expected_terms": [str,...]}, ...]}

expected_terms is derived by merging the gold set's term-like fields so hit_score()
(substring match in retrieved chunk text) still works. All richer fields (subcategory,
difficulty, expected_sections, expected_rules, expected_notifications,
expected_conclusion_keywords, must_not_say, must_cite_verbatim, notes) are preserved
verbatim so a future richer harness can consume them.

IDs: we preserve the human-readable gold IDs (e.g. gst_pos_001). eval.py treats id as
an opaque string, so no renumbering needed.
"""
from __future__ import annotations
import json
from pathlib import Path
import yaml

SRC = Path(r"D:\_gpu_rig_ai\eval\gold_set.yaml")
DST = Path(r"D:\_gpu_rig_ai\eval\eval_set_rig_v2.json")


def derive_expected_terms(item: dict) -> list[str]:
    terms: list[str] = []
    for field in ("expected_sections", "expected_rules",
                  "expected_notifications", "expected_conclusion_keywords"):
        v = item.get(field) or []
        for s in v:
            if s and isinstance(s, str):
                terms.append(s.strip())
    # dedupe preserving order
    seen = set()
    out = []
    for t in terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def main() -> None:
    with SRC.open("r", encoding="utf-8") as f:
        gold = yaml.safe_load(f)

    items = gold["items"]
    queries = []
    for it in items:
        q = {
            "id": it["id"],
            "query": it["question"],
            "category": it.get("category", ""),
            "expected_terms": derive_expected_terms(it),
            # preserved metadata (ignored by current eval.py, used by future harness)
            "subcategory": it.get("subcategory"),
            "difficulty": it.get("difficulty"),
            "expected_sections": it.get("expected_sections", []),
            "expected_rules": it.get("expected_rules", []),
            "expected_notifications": it.get("expected_notifications", []),
            "expected_conclusion_keywords": it.get("expected_conclusion_keywords", []),
            "must_not_say": it.get("must_not_say", []),
            "must_cite_verbatim": it.get("must_cite_verbatim", False),
            "notes": it.get("notes"),
        }
        queries.append(q)

    out = {
        "version": gold.get("version", 2),
        "created": gold.get("created"),
        "notes": gold.get("notes", ""),
        "queries": queries,
    }
    DST.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] wrote {DST}  items={len(queries)}  size={DST.stat().st_size}")


if __name__ == "__main__":
    main()
