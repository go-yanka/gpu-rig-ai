# CBIC RAG — Dual Plan v3 (Locked Spec)

**Date:** 2026-04-21
**Status:** Ready to execute. All open design questions resolved across 3 external consultation rounds + a final 9-question tie-breaker pass.
**Supersedes:** `cbic_rag_dual_plan.md` (v1/v2 with change markers).

This document is the single source of truth for the fix-current and greenfield-rebuild plans. No more consultations before A4 ships.

---

## 0. Scope

- **Part A** — end-to-end fix to the live CBIC RAG system (`b11b12a8_v1` backend on `http://192.168.1.107:9500`). Order is gated by eval harness.
- **Part B** — blueprint for rebuilding from scratch for the next corpus (Income Tax Act + Rules + Circulars, later MCA, Labour, RBI).
- **Appendix** — the 9 follow-up questions and their resolved answers, preserved for audit.

---

## 1. Locked design decisions (from 3 consult rounds + tie-breaker)

| # | Decision | Rationale | Source |
|---|---|---|---|
| D1 | **Retrieval before generation** — fix chunks surfacing before touching the model | Root cause is retrieval, not prompting | All 3 rounds |
| D2 | **Two-pass structured extraction** (JSON → validate → prose) is the permanent verbatim-quote fix | Makes faithfulness mechanical, not prompt-dependent | All 3 rounds |
| D3 | **A4 before P2** — two-pass before parent-doc retrieval | A4 is the structural win; P2 is a refinement | Round #3 + #4 |
| D4 | **Drop P2 entirely** — AST chunking in greenfield supersedes it | Don't build throwaway concatenation logic | Round #3 |
| D5 | **No Qdrant hard-filter in P1** — BM25 multiplier ×3-4 on `doc_type='act'` instead | Preserves visibility of clarifying circulars | Round #3 |
| D6 | **Pass 2 sees JSON only, chunks NOT re-fed** | Prevents re-awakening the paraphrase instinct | FQ1 tie-breaker (re-paraphrase observed more than connective-invention) |
| D7 | **`verbatim_span` length 80-450 chars + must contain one complete clause** (regex boundary) | Eliminates gaming via short high-freq phrases | FQ2 tie-breaker |
| D8 | **ONE Qdrant collection** with strong payload indexes (greenfield) | Filter penalty is sub-ms; two collections doubles ops surface | FQ3 tie-breaker |
| D9 | **Markdown-header split + 15% sentence-boundary overlap** for unstructured circulars | Preserves paragraph numbers lawyers cite | FQ4 tie-breaker |
| D10 | **Tariff tables bypass RAG entirely** — go to a separate SQLite tariff DB for HSN/rate queries | Semantic search can't do exact rate lookups | FQ5/FQ6 (new critical gap) |
| D11 | **json_schema via llama.cpp native API**, not raw GBNF | Optimized for structured JSON; raw GBNF risks infinite-loop | Round #3 |
| D12 | **Cap 6 sub-questions per extraction pass** | Schema violates / truncates at 8+ on qwen3-14B Vulkan | FQ7 |
| D13 | **Decomposition verifier pass** — tiny LLM call to check sub-Q coverage before extraction | Decomposition is the weak link; retry-once is cheap insurance | FQ8 |
| D14 | **Per-difficulty eval gates**, not single aggregate | Two-pass trades fluency for faithfulness on hard queries | FQ9 |
| D15 | **Skip generator fine-tuning** — mode-collapse risk + 2-5k-pair data cost > expected lift | All 4 rounds converged |
| D16 | **Embedding contrastive fine-tune deferred** — evaluate after A4 ships | Higher ROI than generator LoRA, but only if retrieval still gaps | Round #1 + #2 |
| D17 | **Case law out of v1** for greenfield | Different retrieval architecture (IRAC); derails timeline | Round #3 + #4 |
| D18 | **Table extraction is a first-class pipeline**, not an afterthought | Rate/classification queries are dead without it | Round #3 |

---

## PART A — Fix the current system

### A0. Baseline eval (measurement, no code change)

Run `D:\_gpu_rig_ai\eval\run_eval.py --workers 2` against live `b11b12a8_v1`. Save to `D:\_gpu_rig_ai\eval\runs\baseline_<ts>\`.

**Metrics (per category, per difficulty, and aggregate):**
- Section-coverage %
- Expected-keyword %
- Forbidden-word violations
- Verbatim-quote gate pass rate (per-answer, aggregate)
- LLM-as-judge 0-3
- p50 / p95 latency

**Deploy gate (applied to every subsequent step):**
- Basic tier: no aggregate regression >10%
- Intermediate tier: no aggregate regression >15%
- Complex tier: latency may rise up to +20% if verbatim-gate pass rate improves >25%
- No single-query regression >25% on any metric

**Time:** 20-30 min.
**Artifact:** `runs/baseline_<ts>/summary.md`, referenced by all subsequent diff runs.

---

### A1. P1 — Retrieval boost (BM25 score multiplier, no hard filter)

**Goal:** Surface Act/Rules chunks when the query explicitly names a Section/Rule, without blinding the LLM to clarifying circulars.

**Design:**

1. **Classify query (regex only, zero-LLM latency):**
   ```python
   STATUTE_REGEX = r'\b(?:Section|Sec\.?|s\.|Rule|Article)\s+(\d+[A-Z]?)(?:\s*\(\s*(\d+[A-Z]?)\s*\))?(?:\s*\(\s*([a-z])\s*\))?'
   LEGAL_PHRASE_SET = {
       'composite supply', 'mixed supply', 'place of supply', 'time of supply',
       'bill-to-ship-to', 'input tax credit', 'reverse charge', 'zero-rated',
       'export of services', 'DRC-01B', 'GSTR-1', 'GSTR-3B', 'GSTR-2B',
       'section 75(12)', 'rule 86b', 'rule 88c', 'rule 88d', 'rule 36(4)',
   }

   def classify_query(q: str) -> dict:
       hard = STATUTE_REGEX.search(q)
       soft = any(p in q.lower() for p in LEGAL_PHRASE_SET)
       return {'tier': 'hard' if hard else ('soft' if soft else 'none'),
               'refs': hard.groups() if hard else None}
   ```

2. **Apply multipliers at RRF fusion time:**
   - `hard`: BM25 score × 3.5 on chunks where `doc_type IN ('act','rules')`. Dense score unchanged.
   - `soft`: BM25 score × 1.8 on same set.
   - `none`: unchanged.

3. **Preserve existing union with unfiltered retrieval** (already a property of RRF on unfiltered collection). No Qdrant payload filter added.

4. **Post-rerank rule:** in the final top-12 passed to the LLM, enforce **minimum 2 non-Act chunks** to retain clarifying circulars. If fewer, backfill from the next-best non-Act hits.

**Files:**
- `retriever.py`: add `classify_query()`, integrate multiplier in fusion step, enforce min-non-Act rule post-MMR.
- `api.py`: log `retrieval_tier`, `statute_refs_matched`, `non_act_count` to `timings`.

**Sentinel:** `p1_v1`
**Patch paths:** `/opt/indian-legal-ai/patches/p1_<ts>/`, `D:\_gpu_rig_ai\patches\p1_<ts>\`, both pre/post mirrors.
**Time:** 1-1.5 hr code + 20 min eval.

---

### A2. Corpus refresh

**Gap:** CGST Rules PDF is Dec 2022 (missing Rule 88D inserted Aug 2023); IGST Act PDF is Mar 2020 (missing Finance Act 2023 amendments).

**Actions:**
1. Download latest consolidated CGST Rules + IGST Act from CBIC portal.
2. Ingest via existing recipe (single-GPU Vulkan BGE-M3 on GPU 5). See `ingest_playbook_cbic.md`.
3. Mark old chunk IDs as `superseded: true` rather than deleting (preserves trace for historical answers).
4. Verify Rule 88D and Finance Act 2023 amendments retrieve cleanly with a targeted query.

**Time:** 30-45 min.
**Gate:** no eval regression (corpus add only).

---

### A3. Two-pass structured extraction (Option A) — **THE fix**

This is the structural change. Replaces the current single-shot reason-and-cite generation path.

**Flow:**

```
query
  ↓
classify_query() → tier + refs
  ↓
HyDE (existing, cached) + hybrid retrieve + ColBERT rerank + MMR
  ↓
decompose(query) → [sub_q1, sub_q2, …]   (qwen3-14B temp=0.1, LRU-cached)
  ↓
VERIFY decomposition (D13):
  tiny LLM pass: "does this list cover the original query without dropping/adding facets?"
  if FAIL → retry decompose once, then proceed best-effort
  ↓
EXTRACT (Pass 1): qwen3-14B + llama.cpp json_schema, temp=0, /no_think
  INPUT: chunks (with IDs) + sub-questions
  OUTPUT: JSON {sub_answers: [{sub_question, cited_chunk_id, verbatim_span, conclusion}]}
  CAP: 6 sub-questions per pass; split into multiple passes if >6
  ↓
VALIDATE (Python, no LLM):
  for each sub_answer:
    assert cited_chunk_id in retrieved_chunks
    assert len(verbatim_span) >= 80 and len(verbatim_span) <= 450
    assert one_complete_clause(verbatim_span)   # regex: contains . ? ! ; terminator
    pass1 = substring_check(verbatim_span, chunks[cited_chunk_id].text, canon=NFKC+ws)
    if not pass1:
      pass2 = jaccard_6gram(verbatim_span, chunks[cited_chunk_id].text) >= 0.80
      if not pass2:
        pass3 = bge_m3_cosine(verbatim_span, chunks[cited_chunk_id].text) >= 0.92
        if not pass3: DROP (move to suspicious_spans[])
  ↓
SYNTHESIZE (Pass 2): qwen3-14B, temp=0, /no_think
  INPUT: VALIDATED JSON ONLY + original question  (chunks NOT re-fed — D6)
  SYS_PROMPT (<300 tokens):
    "Render these verified legal facts into a practitioner advisory.
     Rules:
     - Copy each verbatim_span EXACTLY. Do not alter wording.
     - Cite [S#] immediately after each quote, matching cited_chunk_id.
     - Do NOT introduce any legal claim not present in VERIFIED FACTS.
     - Format: one paragraph per sub_question. Final paragraph = overall
       conclusion stitched from per-sub conclusions. No new conclusions."
  OUTPUT: final prose answer with inline [S#] citations
  ↓
Final validator (existing): placeholder-leak check, repeat-sentence check, cited-section-in-chunk check.
  ↓
Return { answer, verified_spans[], suspicious_spans[], timings, retrieval_tier, … }
```

**Key implementation notes:**
- **D6 — Pass 2 is JSON-only.** Chunks are tokenized only in Pass 1. Pass 2 context is ~500-1500 tokens total. Cache-friendly prefix.
- **D7 — Span length 80-450 chars + clause requirement** — reject `"Provided that"` style gaming.
- **D11 — llama.cpp `json_schema` parameter** (native). If schema violation observed >2%, fall back to regex-parse + retry-once.
- **D12 — 6-sub-question cap** per extraction pass. For 7+, split into multiple Pass-1 calls, merge JSON.
- **D13 — Decomposition verifier** adds ~1-2s but prevents cascade failures.

**Files:**
- New `validator.py`: centralized substring + Jaccard + BGE-M3 cosine + clause regex.
- `storyformat.py`: new `EXTRACTION_SYS` (<400 tokens), new `SYNTHESIS_SYS` (<300 tokens), new `DECOMP_VERIFY_SYS` (<200 tokens). Delete stale hard-reasoning-rules block; reasoning now lives in the structure, not the prompt.
- `api.py`: new `two_pass_generate()` replacing current generation path. Feature-flag `TWO_PASS_ENABLED` env var so A/B rollback is zero-code.
- `retriever.py`: unchanged from P1.

**Sentinel:** `two_pass_v1`
**Time:** 3-4 hr code + 30 min eval.
**Risk:** medium. Named risks + mitigations:
- Pass 2 connective-tissue invention → SYS_PROMPT strictness + final validator + eval harness.
- json_schema violations at high sub-Q counts → 6-cap + retry + regex fallback.
- Decomposition drops a sub-Q → D13 verifier + retry-once.

**Feature flag:** `TWO_PASS_ENABLED=1`. Setting to `0` reverts to live path instantly without redeploy. Ship WITH the flag ON. Keep it togglable for 48 hrs post-deploy.

---

### A4. Table Extraction pipeline (new — gap flagged round #3)

**Goal:** Rate / HSN / tariff queries return exact cell values, not paraphrased summaries.

**Scope:** GST rate schedules (CGST Rate Notifications), Customs Tariff First Schedule, rate-amending notifications.

**Design:**

1. **Extract tables at ingestion time:**
   - Run Docling on all rate-notification PDFs. Docling handles merged cells on Indian tariff format.
   - For each extracted table, emit rows as structured records: `{hsn, sac, description, rate_igst, rate_cgst, rate_sgst, rate_cess, effective_from, effective_to, notification_id, doc_page, pdf_path}`.
   - Store in a new SQLite DB: `/opt/indian-legal-ai/tariff.db` with FTS5 index on `description` and B-tree index on `hsn`, `sac`, `effective_from`.

2. **Route rate queries in the API:**
   - If query matches `(HSN|SAC)\s*\d{2,8}` or `what.*rate.*(on|for)` or explicit rate keywords ("IGST rate", "GST rate"), send to a new endpoint path `/v1/rate-query` (internal).
   - `/v1/rate-query` does SQL lookup first, then a short RAG pass for the interpretive explanation (cited to the notification chunk via the existing pipeline).
   - Response format: `{rate_table_hit: {…}, interpretive_answer: "…", citations: […]}`.

3. **UI:**
   - New rate card in Answer tab for `rate_table_hit` (HSN, description, rate breakdown, effective date, notification link).
   - Below: interpretive paragraph with inline citations.

**Gate:** 10 new rate-lookup gold queries added to `gold_set.yaml`. Pass rate must be ≥95% on rate card (SQL is deterministic, so 100% expected for covered HSNs; 95% leaves room for gaps).

**[v3 tweak — round #5]** For chunks that contain inline tables (not tariff rate schedules — those go to SQLite; this is *other* embedded tables in circulars like penalty schedules, procedure steps, etc.), set payload `is_table=true` at ingest. When the validator encounters a verbatim_span derived from an `is_table=true` chunk, bypass 6-gram Jaccard and fall back to **exact substring after aggressive whitespace normalization only**. Tabular formatting (pipes, newlines, spaces) breaks n-gram overlap in ways that don't indicate hallucination.

**Time:** 4-5 hr (Docling ingest + SQLite schema + endpoint + UI).
**Sentinel:** `tariff_v1`
**Risk:** low — new code path, doesn't touch existing RAG.

---

### A5. P3 — Query-class routing

Four classes detected by regex + keyword classifier:

| Class | Detection | Pipeline |
|---|---|---|
| **Definitional** | "what does Sec X say", "define Y", single Section/Rule ref, no facts | Skip decomposition. P1-hard retrieve. Single-pass extract (Pass 1 only). Render span directly. |
| **Scenario** | 2+ facts stated, "given that…", multi-fact questions | Full pipeline: P1 + decompose + verify + two-pass. |
| **Reconciliation** | Keyword set: GSTR-1/3B/2B, Rule 88C/D, DRC-01B, Section 75(12), Rule 86B | Two-pass + inject reconciliation calibration rules (from `rag_reasoning_calibrations.md`) into EXTRACTION_SYS *for this class only*. |
| **Rate/Classification** | HSN/SAC regex, "what rate", notification numbers | Route to `/v1/rate-query` (A4 pipeline). |

**Files:** `api.py` — new `route_query()` dispatcher. Per-class SYS_PROMPT variants (each <400 tokens, well under the 800-token regression threshold).

**Sentinel:** `p3_v1`
**Time:** 2 hr code + 20 min eval.

---

### A6. Training fallback ladder (DEFERRED — only if A3+A1+A4+A5 underperforms)

Defer until after A0 → A1 → A2 → A3 → A4 → A5 all shipped and measured.

Trigger condition: if verbatim-gate pass rate remains <90% (or <85% on retrieval-sensitive queries) after A5.

**Revised fallback order** (consult rounds 4 + 5 converge, 2026-04-21 — supersedes earlier single-path "embedding fine-tune first" framing):

1. **Base model upgrade first** — swap qwen3-14B for **Qwen2.5-32B-Instruct** or **DeepSeek-R1-Distill-Qwen-32B**. Both are open-weights, both strong on structured-output / citation tasks, both run on the existing Vulkan rig. Per consult #2, instruction-following and citation fidelity scale ~linearly with parameters; a 32B base often closes the gap with zero training.
2. **BGE-M3 embedding contrastive fine-tune** — harvest (query, relevant-chunk) positive pairs from E10 upvotes + gold set, fine-tune BGE-M3 via MultipleNegativesRankingLoss on a rented Nvidia A100 (2-4 hrs), deploy adapter.
3. **Generator distillation from open-weights** — use Llama-3.3-70B-Instruct / DeepSeek-V3 / Qwen3-72B to generate ~5k survivor pairs (see `training_data_generation_plan.md`), distill into 14B or 32B base. Permissive licenses only — Claude 3.5 / GPT-4o outputs are ToS-barred for training competing models.
4. **LoRA on the generator — last resort.** Known mode-collapse risk at 14B (fixes quoting, can break multi-hop reasoning). Only if steps 1-3 failed.

Stop at the first step that closes the verbatim-gate threshold.

---

### A7. Feedback loop (continuous)

- E10 downvotes → weekly review → validated ones become new eval gold cases.
- Run full eval weekly. Regression → revert most-recent deploy, diagnose.
- Amendment-graph sidecar (see Part B, D18 idea applied here): SQLite `doc_supersession (doc_id, supersedes, superseded_by, effective_from, effective_to)` — populate at ingest time, query at answer time to flag outdated citations.

---

### A8. Deploy cadence (one dimension per deploy)

| # | Step | Dimension | Risk | Ship / Revert gate |
|---|---|---|---|---|
| 0 | Baseline eval | measurement | none | — |
| 1 | P1 retrieval boost | retrieval | low | per-tier eval gate (A0) |
| 2 | Corpus refresh | corpus | none | no regression |
| 3 | Two-pass extraction (A3) | generation | medium | per-tier gate + suspicious-spans% should *decrease* |
| 4 | Table extraction (A4) | new pipeline | low | 95% rate-query pass rate |
| 5 | Query-class routing (A5) | prompt+retrieval | low-med | per-class eval gate |
| 6 | Embedding fine-tune (A6) | retrieval | medium | only if A3-A5 didn't close the gap |

**Never mix dimensions in a single deploy.** Each step has its own sentinel and own patch artifact set.

---

### A9. Infrastructure (already stable — restating)

- Patch paths: `/opt/indian-legal-ai/patches/<feat>_<ts>/` (rig) + `D:\_gpu_rig_ai\patches\<feat>_<ts>\` (laptop). Never `/tmp`, never `AppData\Local\Temp`.
- Every deploy writes 3 artifacts: `*.bak.<feat>.<ts>`, `*.patched.<feat>.<ts>`, `apply.py`.
- systemd `cbic-rag-api.service`: `Restart=on-failure`, `StandardOutput=journal`, `MemoryMax=6G`.
- SSH flags everywhere: `-o ControlMaster=no -o ControlPath=none`.
- Feature flags for reversible rollout (A3 uses `TWO_PASS_ENABLED=1|0`).

---

## PART B — Greenfield rebuild (next corpus, e.g. Income Tax)

Hardware assumption unchanged: 4-core, 3× AMD Vulkan GPUs, no ROCm, Qdrant, llama.cpp, no cloud.

### B0. Day-1 principles

1. Verbatim citation trust (every legal claim → substring-verified quote).
2. Eval harness EXISTS before first answer. 100-Q stratified gold set: 40% definitional, 30% scenario, 20% reconciliation, 10% adversarial.
3. Handle definitional + scenario + reconciliation + rate queries equally well, on day 1.
4. Temporal correctness baked in (Section X as of date D).
5. Two-pass extraction is the default generation path.
6. Table extraction is a first-class pipeline.
7. No single patch is ever on `/tmp`.

### B1. Corpus + temporal model

- Current + Historical architecture (D8): single Qdrant collection for "current" chunks. Separate `<corpus>_historical` collection for superseded chunks. Queries default to current; temporal queries (with explicit as-of date) hit historical.
- Every chunk carries: `effective_from`, `effective_to`, `supersedes: [doc_id]`, `superseded_by: [doc_id]`.
- New: **amendment-graph SQLite sidecar** (D: `/opt/indian-legal-ai/amendments.db`) — O(1) chain traversal without vector search.

### B2. Parser

Docling primary. RapidOCR Vulkan (GPU 4/6) for scanned-only PDFs. Never CPU tesseract / CPU RapidOCR.

### B3. Chunking

- **Acts + Rules:** AST-style — chunk at Section/Sub-section/Clause boundaries, never mid-clause. Target 400-800 tokens, cap 1500.
- **Circulars + Notifications + unstructured docs (D9):** markdown-header split (LlamaIndex `MarkdownHeaderTextSplitter` or Docling-native) + 15% sentence-boundary overlap.
- **Tables:** extracted as structured rows to SQLite (see B8), NOT chunked for vector search.

Every chunk carries:
```json
{
  "hierarchy_path": ["Income-tax Act, 1961", "Chapter IV", "Section 14"],
  "section_ref": "14(1)(a)",
  "section_canonical": "S.14(1)(a)",
  "doc_type": "act" | "rules" | "notification" | "circular" | "case",
  "doc_id": "itact:1961:14_1_a",
  "effective_from": "1962-04-01",
  "effective_to": null,
  "parent_section_id": "itact:1961:14",
  "subsection_order": 3,
  "cross_refs_outbound": ["S.10(13A)", "R.2A"],
  "page": 27,
  "pdf_path": "…"
}
```

### B4. Vector index — ONE collection (D8)

Single Qdrant collection `<corpus>_v1`. Payload indexes on `doc_type`, `category`, `effective_from/to`, `section_ref`. BM25 + BGE-M3 dense, RRF fusion. Query-time `doc_type` filters + BM25 multipliers replace the two-collection model.

### B5. Query-class router (day 1)

Same 4 classes as A5, built in from the start.

### B6. Generation — two-pass from day 1

Same as A3. No single-pass fallback.

### B7. Verifier

Substring (NFKC canon) + 6-gram Jaccard ≥0.80 + BGE-M3 cosine ≥0.92. All three.

### B8. Table pipeline (parallel to RAG, day 1)

SQLite tariff DB with FTS5 + B-tree indexes. Rate queries bypass RAG entirely; RAG only for interpretive paragraph (A4 design).

### B9. Eval harness

- Built before first answer. 100-Q stratified.
- Runner + diff + regression gate same as `D:\_gpu_rig_ai\eval\`.
- Per-difficulty gates (D14).
- **No deploy without green eval.**

### B10. Observability

Per-query logs: `retrieval_tier`, `class`, `sub_question_count`, `chunks_retrieved`, `verified_spans`, `suspicious_spans`, `gate_pass_rate`, `latencies_ms`, `tokens_in/out`. `/v1/meta` exposes live sentinel + collection sizes + effective-as-of.

### B11. What we WOULD NOT change from CBIC

BGE-M3 on Vulkan GPU 5. qwen3-14B-hermes. Qdrant. ColBERT + MMR. These are working.

### B12. What we WOULD do differently vs CBIC

- Eval harness FIRST.
- AST chunking from start.
- Two-pass from start.
- Single collection with strong payload filters (NOT two collections — corrected via FQ3 benchmark).
- Markdown-header split for unstructured (not semantic).
- Versioned docs with amendment graph from start.
- Query-class router from start.
- Table pipeline as parallel first-class pipeline.
- llama.cpp `json_schema` native API, not raw GBNF.
- Persistent patch paths from day 1.

---

## 2. Immediate execution order

1. **NOW:** A0 baseline eval (20-30 min).
2. **Next:** A3 two-pass extraction (3-4 hr). **This is the priority per all 4 consult rounds.**
3. **Then:** A1 P1 retrieval boost (1-1.5 hr).
4. **Then:** A2 corpus refresh (30-45 min).
5. **Then:** A4 table extraction (4-5 hr).
6. **Then:** A5 query-class routing (2 hr).
7. **Defer:** A6 embedding fine-tune; reassess after A5.

Total critical path to ship-grade: **~12-14 hrs active coding** split across however many deploy windows we want.

---

## Appendix — FQ resolutions (round 4)

For future reviewers: these are the resolved answers that locked design decisions D6-D14.

**FQ1 — Pass 2 re-feeding**
Answer: **Do NOT re-feed chunks in Pass 2.** Observed failure mode: re-paraphrasing with chunks in context is more common than connective-tissue invention with JSON-only. Strict synthesis prompt + final validator catch drift. → D6.

**FQ2 — Verbatim span length**
Answer: **80-450 chars + at least one complete clause** (regex `[^.?!;]+[.?!;]`). Eliminates gaming via "Provided that" style high-freq fragments. → D7.

**FQ3 — Single vs two collections**
Answer: Qdrant payload filter latency on 108k chunks with proper indexes is sub-ms. **One collection wins on ops simplicity** with identical performance. → D8.

**FQ4 — Unstructured circular chunking**
Answer: **Markdown-header split + 15% sentence-boundary overlap.** Semantic chunking loses explicit paragraph numbers that lawyers cite. → D9.

**FQ5 / FQ6 — Tariff tables + HSN lookup**
Answer: **Docling extract → separate SQLite tariff DB → bypass RAG for pure rate queries.** RAG only for interpretive explanation. Quote-verification at cell level with `table_row_id`. → D10 + new pipeline A4.

**FQ7 — JSON schema reliability**
Answer: Stable up to 6-7 sub-questions at 8-10k context on qwen3-14B Vulkan. **Cap at 6 per pass**; split for 7+. → D12.

**FQ8 — Decomposition quality**
Answer: Decomposition occasionally drops/merges sub-questions. **Add a cheap verifier pass** with retry-once. → D13.

**FQ9 — A4 evaluation threshold**
Answer: **Per-difficulty-tier gate** (basic ≤10%, intermediate ≤15%, complex +20% latency ok if gate +25%). Aggregate alone is too blunt. → D14.

---

## Sign-off

This plan is locked. Any future design change requires a new eval-regression test case and a documented rationale. We ship A0 → A3 → A1 → A2 → A4 → A5 in that order, each gated by per-difficulty eval.

No more consultations before A3 is live.
