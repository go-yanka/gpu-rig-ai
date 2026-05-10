#!/usr/bin/env python3
"""Retrieval recall@k audit — runs on rig, uses rig's retriever module.

For each gold item, retrieve top-k chunks and check whether any of the gold
expected_sections / expected_rules / expected_notifications / doc_ids appear
in those chunks (via section_ref, doc_number, hierarchy, or text substring).

Output: JSONL per item + summary line.

Usage (on rig):
    cd /opt/indian-legal-ai/rag/cbic_rag
    python3 /tmp/recall_audit_rig.py --gold /tmp/gold_set.yaml \
            --out /tmp/recall_audit.jsonl --k 5
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/opt/indian-legal-ai/rag/cbic_rag")

import yaml
import retriever


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def chunk_contains(chunk: dict, needle: str) -> bool:
    """True if the chunk plausibly covers the cited entity."""
    n = norm(needle)
    if not n:
        return False
    fields = [
        chunk.get("section_ref", ""),
        chunk.get("doc_number", ""),
        chunk.get("hierarchy", ""),
        chunk.get("title", ""),
        chunk.get("text", "")[:2000],
    ]
    hay = " ".join(str(f) for f in fields)
    return n in norm(hay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    gold = yaml.safe_load(Path(args.gold).read_text(encoding="utf-8"))
    items = gold.get("items") if isinstance(gold, dict) else gold

    total_checks = 0
    total_hits = 0
    recall1 = recall5 = 0
    per_item = []
    for i, item in enumerate(items, 1):
        q = item["question"]
        try:
            chunks = retriever.retrieve(q, k=max(args.k, 10))
        except Exception as e:
            print(f"[{i}/{len(items)}] {item['id']} ERR: {e}", file=sys.stderr)
            per_item.append({"id": item["id"], "error": str(e)})
            continue

        top = chunks[: args.k]
        expected = {
            "sections": item.get("expected_sections", []) or [],
            "rules": item.get("expected_rules", []) or [],
            "notifications": item.get("expected_notifications", []) or [],
        }
        all_entities = [e for lst in expected.values() for e in lst]
        if not all_entities:
            # if no structured expectations, use conclusion keywords as weak proxy
            all_entities = item.get("expected_conclusion_keywords", []) or []

        hit_k1 = False
        hit_k5 = False
        per_entity = {}
        for ent in all_entities:
            hit_ranks = []
            for rank, c in enumerate(chunks, 1):
                if chunk_contains(c, ent):
                    hit_ranks.append(rank)
            per_entity[ent] = hit_ranks[:5]
            if hit_ranks:
                if hit_ranks[0] == 1:
                    hit_k1 = True
                if hit_ranks[0] <= args.k:
                    hit_k5 = True
            total_checks += 1
            if hit_ranks:
                total_hits += 1
        if hit_k1:
            recall1 += 1
        if hit_k5:
            recall5 += 1

        rec = {
            "id": item["id"],
            "category": item.get("category"),
            "question": q,
            "n_expected": len(all_entities),
            "hit_at_1": hit_k1,
            "hit_at_k": hit_k5,
            "per_entity_hits": per_entity,
            "top_k_meta": [
                {
                    "rank": j + 1,
                    "doc_id": c.get("doc_id"),
                    "doc_number": c.get("doc_number"),
                    "section_ref": c.get("section_ref"),
                    "title": (c.get("title") or "")[:80],
                    "score": c.get("score"),
                }
                for j, c in enumerate(top)
            ],
        }
        per_item.append(rec)
        print(f"[{i}/{len(items)}] {item['id']:32s} "
              f"ents={len(all_entities)} k1={int(hit_k1)} k5={int(hit_k5)}", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in per_item:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(items)
    print(f"\n=== Recall audit @ k={args.k} ===", file=sys.stderr)
    print(f"items:           {n}", file=sys.stderr)
    print(f"any-entity@1:    {recall1}/{n} = {100*recall1/n:.2f}%", file=sys.stderr)
    print(f"any-entity@{args.k}: {recall5}/{n} = {100*recall5/n:.2f}%", file=sys.stderr)
    if total_checks:
        print(f"per-entity hit:  {total_hits}/{total_checks} = {100*total_hits/total_checks:.2f}%",
              file=sys.stderr)


if __name__ == "__main__":
    main()
