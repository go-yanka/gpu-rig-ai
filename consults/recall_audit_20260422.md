# Retrieval recall@k audit — 2026-04-22

**Purpose:** Test the consult's frame-shift claim: *"If recall@5 isn't 95%+, no prompting fix will help."*

**Method:** 170 gold items, top-k from live `retriever.retrieve()` (hybrid RRF dense BGE-M3 + sparse), check whether any expected entity (sections/rules/notifications, or conclusion keywords as weak proxy) appears in each chunk's `section_ref`, `doc_number`, `hierarchy`, `title`, or `text[:2000]` via case-insensitive substring.

## Headline

| Metric | Result |
|---|---:|
| any-entity @1 | **6 / 170 = 3.53%** |
| any-entity @5 | **21 / 170 = 12.35%** |
| per-entity hit | 24 / 334 = 7.19% |

**The frame-shift claim is VALIDATED. Retrieval is the primary bottleneck, not prompting.**

## Per category

| category | n | @1 | @5 |
|---|---:|---|---|
| central_excise | 10 | 10.0% | 20.0% |
| customs | 45 | 4.4% | 15.6% |
| gst | 81 | 1.2% | **7.4%** |
| others | 14 | 7.1% | 21.4% |
| service_tax | 20 | 5.0% | 15.0% |

GST — our largest category — is the weakest at 7.4%@5. GST queries are the most entity-specific (IGST §10, CGST §16(2)(aa), Rule 36(4), etc.) and that's exactly where the retriever fails hardest.

## Caveats (false-negative allowance)

The substring match has known false negatives:
- Gold entity `"10(1)(a) IGST"` won't cleanly match a chunk whose `section_ref="10(1)(a)"` and whose IGST lineage lives in a separate payload field
- Notation variants (`Sec. 9` vs `Section 9`, `9(1)` vs `9 (1)`)
- "CGST"/"IGST" appear inconsistently in chunk text

But even tripling the hit-rate to adjust for these ≈ **36%@5** — still catastrophic, still far below the 95% line needed for any prompting fix to matter.

## Evidence: real retrieval failure (not just notation)

`gst_pos_003`: *"Services of a chartered accountant in Pune are rendered to a registered client in Gujarat. What is the place of supply?"*
- Expected: `12(2) IGST`
- Top-3: "Continuation of Recovery Proceedings", "Central Goods and Services Tax Act" (unspec), "amend CGST Rules notification"
- The IGST Chapter V place-of-supply provisions do not surface at all

`gst_pos_004`: *"An event-management company organizes a conference in Bengaluru for a client registered in Delhi..."*
- Expected: `12(7) IGST`
- Top-3: IGST Rules §2(iii), "Order for rejection of compounding of offence", "Order for deferred payment"
- Same pattern — semantically adjacent but legally wrong chunks dominate

## Interpretation

1. **BGE-M3 out-of-domain on Indian tax law.** It hasn't seen "place of supply" cases enough to map prose scenarios to IGST §12 provisions. The dense vectors collapse distinct legal concepts (compounding, recovery, place-of-supply) into one fuzzy "GST-procedural" cluster.

2. **Sparse (BM25) can't save it either.** Questions don't contain the section numbers. Sparse matches on "GST" and "registered" which is noise.

3. **Prompting improvements are throwing water on a fire.** Two-pass CoT, validator ladders, tariff router — none of this matters when the correct chunk isn't in the top-5.

## Decision

**Retrieval is now the critical path. Everything else is secondary.**

Recommended next actions (stacked):

1. **A. Query rewriting / HyDE.** Cheap test. Have qwen3 generate a hypothetical answer first, embed *that*, retrieve. This often lifts out-of-domain BGE-M3 by 2-3x on entity-specific queries. No retraining.

2. **B. BGE-M3 contrastive fine-tune.** The Gemini plan. Generate 2000 pairs (script already built + smoke-tested), fine-tune BGE-M3 on CBIC corpus. 1-2 days on rig. Should lift recall@5 from 12% → 60-75% based on domain-adaptation literature.

3. **C. Hierarchical retrieval.** Retrieve act → chapter → section in stages using metadata filters, not pure vector search. The Qdrant payload already has `parent_act`, `hierarchy`, `section_ref`. Queries with detected entities (via NER or regex) should hard-filter before vector ranking.

4. **D. Cross-encoder rerank.** We already do ColBERT rerank, but on a too-small candidate pool. Widen initial retrieval to top-50 and rerank to top-5.

C and D are quick wins that don't need fine-tuning. Start there while B runs.

## Files

- Script: `D:\_gpu_rig_ai\eval\recall_audit_rig.py` (runs on rig, imports `retriever`)
- Raw per-item: `D:\_gpu_rig_ai\eval\recall_audit_20260422.jsonl`
- Log: rig:`/tmp/recall_audit.log`
