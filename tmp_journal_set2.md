
---
## 2026-04-25 — M5 GST50 → SET2 SCALE TEST

**Set 1 (GST50, cbic_v2_gst50_sem):** G1 recall@10 = 0.9643 (27/28) — PASS
**Set 2 (50 mixed, cbic_v2_set2):** G1 raw recall = 0.835 (258/309) — FAIL on raw,
PASS on adjusted basis 0.9736 (258/265) when 44 queries against 4 phase2-failed
docs are excluded.

### Phase2 failure mode (NEW)
qwen3-14b classifier enters a regex-repetition trap on certain doc types
(forms, sometimes instructions). Output emits "hard_boundaries" containing the
same `\s*\d+\.\s*\w+\s*\d*\.` fragment repeated until max_tokens (4096) hit.
JSON never closes, no comma to truncate to → L4 brace recovery cannot fix.

**Set 2 phase2 failures (4/50 = 8%):**
- cbic-form-msts:1000130, 1000184, 1000193 (all 3 form docs)
- cbic-instruction-msts:1000455

### Diagnosis of remaining 7 truly-retrieval misses (Set 2)
- 5x notifications: scenario-style queries like "Our client, a food processing
  company..." that don't match section keywords directly
- 1x circular, 1x rule: similar pattern
This 7/265 = 2.6% miss is consistent with Set 1's 1/28 (3.6%) — chunker recipe
is robust across doc types.

### Fix needed before full re-ingest
**Defect C (NEW): qwen3 classifier repetition trap on form/instruction docs.**
Options:
1. Detect repetition in last_raw (>=3x same 50-char substring) -> fallback to
   default plan based on doc_id prefix
2. Pre-classify by doc_id prefix (cbic-form-msts:* -> fixed form template,
   skip qwen3 entirely)
3. Add stop sequences to qwen3 sampling

Option 2 cheapest + most reliable. Forms have a known fixed structure;
classifier was overkill anyway.

### Status
- Chunker recipe: PROVEN at >=95% adjusted across 2 doc-type-mixed scopes
- Classifier robustness: BLOCKING for full re-ingest (8% failure rate would
  drop ~1190 of 14925 docs)
- Path: implement Defect C fix -> rerun Set 2 -> Sets 3/4 -> full re-ingest
