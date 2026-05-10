
---

## 2026-04-25 ADDENDUM — Multi-set scale validation complete

### Proven recipe (chunker locked)

The Phase 2 → Phase 3-5 → G1 pipeline has been validated on 4 doc-type-mixed sets
(192 docs / 1121 gold queries) at ≥95% **adjusted** recall@10. See
`memory/project_cbic_reingest_v2.md` 2026-04-25 (later) block for full numbers.

Mandatory environment for any phase invocation:
```
DENSE_ONLY=1
EMBED_GPUS=4,5,6
RADV_DEBUG=nodcc
GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1
PYTHONPATH=/home/user/.local/lib/python3.10/site-packages:/opt/indian-legal-ai/rag/cbic_rag
```

Phase 2 hardening that is now permanent:
- qwen3 max_tokens=4096, timeout=180s
- `_tolerant_json_loads` L4 (best-effort brace completion)
- `table_regions or []` nullguard
- **Defect C: `_DEFAULT_PLANS_BY_PREFIX` (10 prefix templates + `_GENERIC_`),
  `_BYPASS_QWEN_PREFIXES=("cbic-form-msts",)`, `_detect_repetition()` early-break.**

### Failure modes that still drop raw recall (NOT chunker scope)

- **Defect D (shared-source-PDF doc_ids):** when N doc_ids point at the same PDF,
  the chunker reads the whole PDF for each, dedup absorbs all but the first.
  ~50 GST form doc_ids share `CGST-Rules-2017-Part-B-Forms.pdf`.
  **Pre-flight check before full re-ingest:**
  ```sql
  SELECT path_en, COUNT(*) AS n FROM docs
  WHERE phase1_done=1 GROUP BY path_en HAVING n>1 ORDER BY n DESC;
  ```
  If shared-PDF docs exceed 5% of corpus, build a manifest enrichment
  pass to populate per-doc_id `page_offset` before phase2.

- **Defect E (form retrieval semantic gap):** form chunks (field labels) don't
  match scenario-style queries. Retrieval-side lever (BM25+RRF or query
  rewriting). Don't try to fix in chunker.

### Per-set runner (use for any new 50-doc cohort)

```bash
ssh -i ~/.ssh/id_ed25519_rig root@192.168.1.107
cd /opt/indian-legal-ai/reingest_spec

# 1. phase2 (chunk)
IDS=$(paste -sd, eval/scale_sets/setN/doc_ids.csv)
DENSE_ONLY=1 EMBED_GPUS=4,5,6 RADV_DEBUG=nodcc GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1 \
  PYTHONPATH=/home/user/.local/lib/python3.10/site-packages:/opt/indian-legal-ai/rag/cbic_rag \
  python3 ingest_v2.py --phase phase2 --doc-ids "$IDS" --allow-phase2-failures 10

# 2. phase3_4_5 (embed + upsert into a fresh per-set collection)
DENSE_ONLY=1 EMBED_GPUS=4,5,6 RADV_DEBUG=nodcc GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1 \
  QDRANT_COLL_V2=cbic_v2_setN \
  PYTHONPATH=/home/user/.local/lib/python3.10/site-packages:/opt/indian-legal-ai/rag/cbic_rag \
  python3 ingest_v2.py --phase phase3_4_5

# 3. G1 evaluator with verbose miss diagnostics
python3 evaluators/gate_g1_recall.py \
  --collection cbic_v2_setN \
  --gold eval/scale_sets/setN/v2_gold_setN.json \
  --retrieve-only --out evaluators/gate_g1_setN.json
# Misses written to evaluators/gate_g1_setN.misses.json
```

### Acceptance criteria (per cohort)

- Adjusted recall@10 ≥ 0.95 (excluding queries against shared-PDF docs and
  form scenario-queries until Defects D/E ship)
- Phase2 should be 0 raises on any cohort going forward (Defect C generic
  fallback covers all observed prefixes)
- Per-cohort miss diagnostics MUST be reviewed before declaring pass —
  random recall noise can mask real regressions

