
---
## 2026-04-25 (final) — Multi-set scale validation complete

| Set | n_gold | raw recall@10 | adjusted (excl shared-PDF forms) | Phase2 result |
|-----|-------|--------------|---------------------------------|---------------|
| Set 1 (GST50) | 28  | 0.9643 | 0.9643 | 42/42 ok |
| Set 2 (mixed) | 309 | 0.8997 | 0.9754 (excl 24) | 50/50 ok (post-Defect C) |
| Set 3 (mixed) | 375 | 0.9147 | 0.9635 (excl 19) | 50/50 ok (post-Defect C ext) |
| Set 4 (mixed) | 409 | 0.8704 | 0.9674 (excl 41) | 50/50 ok (Defect C ext) |

**Conclusion:** chunker recipe is PROVEN at >=95% adjusted recall@10 across
4 doc-type-mixed scopes (192 docs, 1121 gold queries). Raw fails 95% only
because of Defect D (shared-source-PDF doc_ids).

### Defects found and resolved
- **Defect A (qwen3 timeout/max_tokens)**: bumped to 4096/180s + L4 brace
  recovery. Resolved most truncation cases (Set 1 win).
- **Defect B (table_regions=null)**: nullguard `for tr in (plan.table_regions or [])`.
  Resolved.
- **Defect C (qwen3 regex-repetition trap on form/circular/notification/
  instruction)**: prefix-bypass for forms + repetition detector with default-
  plan fallback covering all 9 known prefixes + generic catch-all. Resolved
  in Sets 2-4 (zero phase2 failures by Set 4).

### Defects deferred (NOT chunker scope)
- **Defect D (shared-PDF doc_ids)**: e.g. `CGST-Rules-2017-Part-B-Forms.pdf`
  is referenced by ~50 form doc_ids. Chunker reads entire PDF for each
  doc_id; first one absorbs all 243 chunks, rest dedupe to 0. Needs
  per-form page-range metadata in manifest. ~84 forms in full corpus
  affected. Out of scope for current chunker work.
- **Defect E (form retrieval semantic gap)**: even when forms have chunks,
  scenario-style queries don't match form-field labels. 5/16 queries miss
  on form 1000192 with 243 chunks. Needs query-side rewriting or hybrid
  sparse boost. Out of scope.

### Levers in reserve (for future tightening)
- L1: per-doc-type reranker bias (table penalty for non-table queries)
- L2: BM25 sparse + dense RRF fusion
- L3: parent-doc stitching at retrieval time (top-k expand to siblings)
- L4: Add `doc_type` to payload metadata (currently null for many)

### Path to full re-ingest (14,925 docs)
The chunker recipe is locked. Pre-flight needed before full corpus run:
1. Inventory shared-PDF doc_ids: how many of 14,925 share a path with another?
2. If Defect D affects >5% of docs, build manifest enrichment pass to set
   page_offset per doc_id before phase2.
3. Otherwise, proceed; expected raw recall ~88-92%, adjusted ~96-97%.

