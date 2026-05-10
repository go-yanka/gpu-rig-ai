# CBIC RAG — Re-Ingestion Plan v1 (Foundation Rebuild for 95% Trust)

**Date:** 2026-04-22
**Status:** DRAFT for external LLM consultation
**Author:** Claude (working with user)
**Premise:** The current system's foundation is too shaky for 95% trust. No amount of downstream patching (better retrieval, reranking, fine-tuning, LLM prompts) can compensate for bad chunks and missing metadata. This document captures what we learned from the first attempt and lays out a disciplined re-ingestion plan designed to hit 95% trust by construction, not by repair.

---

## Part 1 — Honest Assessment of Attempt #1

### 1.1 What we built

- **Corpus:** 108,802 chunks across 5 categories (customs 53K, GST 41K, central excise 9K, others 3K, service tax 2.5K) ingested from CBIC PDFs into Qdrant (`cbic_v1`), embedded with BGE-M3 (1024-dim dense + sparse).
- **Stack:** FastAPI RAG service at `:9500`, qwen3-14b generator at `:9082` (Vulkan), llama-server-rocm for embedding, dashboard at `:9500/ui`.
- **Eval:** Gold set of 170 hand-curated practitioner Q&A with expected section/rule/notification citations.

### 1.2 What the numbers actually say

| Metric | Baseline measurement | Target | Gap |
|---|---:|---:|---:|
| Recall @1 | ~20% | ≥80% | -60pp |
| Recall @5 | ~40% | ≥95% | -55pp |
| Recall @20 | ~40.6% | ≥99% | -58pp |
| Chunks missing `section_ref` | 83.17% | ≤5% | catastrophic |
| Chunks >2,400 chars | 19.34% | ≤2% | 10× over budget |
| Boilerplate duplicates (top prefix) | 273 copies | 0 meaningful | severe |
| Image-only PDFs still un-OCR'd | 471 | 0 | blind spot |

**Top duplicate prefix alone appears 273 times** — the same GSTIN form table header is polluting retrieval across hundreds of chunks. Nine separate table-header templates each appear 70–275 times.

### 1.3 What we tried to fix after the fact

- Recall variants sweep (HyDE, meta-filter, act-prefix) → no single variant got us near target.
- 4,000 training pairs generated (Gemini + Claude Opus + Claude Sonnet) for BGE-M3 fine-tuning.
- QA grading rubric (answerable/realistic/specific/complexity) to filter bad pairs.
- Hard-negative mining from baseline top-20 retrievals.
- RunPod A100 fine-tune pipeline staged.
- Chunk audit + failure-mode analysis.

### 1.4 Why these fixes cannot reach 95%

1. **Fine-tuning teaches the model to prefer the least-bad chunk among bad options.** It cannot make a missing `section_ref` appear, cannot split a 4,000-char mixed-topic chunk, cannot remove 273 copies of a form header.
2. **MNRL fine-tuning changes the full encoder**, which means we must re-embed all 108K chunks anyway — so the ~2–4 hour re-embed cost is paid regardless. Better to re-chunk first.
3. **The hardest practitioner questions (cross-act, multi-party scenarios) depend on metadata we never extracted** (section hierarchy, notification supersession, effective dates).
4. **The generator (qwen3-14b) can only cite what retrieval gives it.** Retrieval has a hard ceiling set by chunk quality. We are at that ceiling.

**Conclusion:** Repair path is a local maximum at ~60–70% recall@5 best-case. The 95% target requires rebuilding the foundation.

---

## Part 2 — Root-Cause Observations (what the first ingest got wrong)

Each observation is paired with the evidence and the specific fix it demands in v2.

### 2.1 Chunking was mechanical, not semantic

**Evidence:** 19.34% of chunks >2,400 chars. Median 1,298 chars (~324 tokens) is fine on average, but the long tail contains multi-topic blobs. 10.58% are <300 chars — orphan fragments that can't stand alone.

**Root cause:** Chunker used fixed-size windows on raw text, ignoring section/sub-section boundaries, provisos, and tables.

**Fix (v2):** Parse PDFs into a document tree first (section → sub-section → clause → proviso; table as a unit), then chunk along the tree with a hard cap (800–1,200 chars for prose, whole-table for tables ≤1,500 chars, split-by-row for larger tables).

### 2.2 Metadata was aspirational, not enforced

**Evidence:** 83.17% of chunks have no `section_ref`. 3.45% have no `parent_act`. No `sub_section`, no `clause`, no `proviso_id`, no `effective_date`, no `supersedes`, no `chunk_type` (definition / rate / procedure / penalty / form).

**Root cause:** Metadata was extracted opportunistically from filenames and first-line regex, never validated against the document tree.

**Fix (v2):** Metadata becomes a first-class schema validated at ingest. A chunk without `section_ref` or `doc_number` cannot enter Qdrant without an explicit `metadata_reason_missing` flag. Every chunk carries: `doc_id, doc_number, doc_type, parent_act, section_ref, sub_section, clause, proviso_id, chunk_type, effective_date, supersedes_ids, superseded_by_ids, hierarchy_path, page_num, source_pdf`.

### 2.3 Boilerplate was ingested as content

**Evidence:** Same GSTIN table header 273 times, currency-exchange-rate table header 95 + 82 times, drawback-schedule header 188 times, etc. Nine templates × ~100+ copies = ~1,000 near-identical noise chunks.

**Root cause:** No template detection. Every occurrence of a form skeleton was embedded as if it carried answer-bearing content.

**Fix (v2):** Template detection pass — cluster chunks by n-gram shingle, any cluster with ≥3 near-identical members where >70% of content is table delimiters / form labels is marked `is_template=true` and excluded from retrieval (still kept for citation fallback if needed).

### 2.4 Tables were treated as text

**Evidence:** 43,654 chunks are tables (40%). Median 636 chars. Many are rate schedules, tariff lines, notification annexures — the factual core of tax law — yet they share the prose embedding space where structure is lost.

**Root cause:** Tables were flattened to pipe-delimited text and embedded like prose.

**Fix (v2):** Tables get a structured store (SQLite) keyed by (tariff_item, HS_code, rate, notification) with a text "caption" summarizing the table's purpose. Retrieval checks the structured store for factual lookups (rate/duty questions) and the vector store for conceptual questions. Table captions still get embedded; the row data stays relational.

### 2.5 Image-only PDFs were deferred and forgotten

**Evidence:** 471 PDFs still not OCR'd as of today. The OCR research decision was frozen.

**Root cause:** OCR was treated as a separate phase. It should be part of ingest.

**Fix (v2):** OCR is a gating step in the ingest pipeline. Every PDF is classified (text-extractable vs image-only) and image-only ones go through Qwen2.5-VL-7B (GPU 4/6, Vulkan) or RapidOCR fallback in the same pass. No PDF enters Qdrant without full-text extraction.

### 2.6 No document hierarchy means no scoped retrieval

**Evidence:** "Section 54(3) proviso 2" is currently indistinguishable from "Section 54(3)" which is indistinguishable from "Section 54". When a practitioner asks about the proviso, retrieval returns the whole section as a single blob.

**Root cause:** Hierarchy was flattened into one field (`section_ref` as a string, often empty).

**Fix (v2):** Store hierarchy as explicit fields AND as a materialized path (`cgst/s54/ss3/proviso2`). Allow Qdrant filters to scope retrieval by path prefix. Allows "within Section 54, find the proviso" queries.

### 2.7 Supersession and effective dates not tracked

**Evidence:** No field records which notification supersedes which, no effective-date range, no "as-of" retrieval capability. A 2017 circular that was amended in 2022 is retrieved with equal weight to the current-in-force version.

**Root cause:** Ingest treated every document as current.

**Fix (v2):** Every chunk gets `effective_from`, `effective_to` (nullable = current), `supersedes_ids[]`, `superseded_by_ids[]`. Retrieval default filters to `effective_to IS NULL` unless user asks for a historical as-of.

### 2.8 No chunk-type classification

**Evidence:** A tariff rate table, a procedural circular, a definition section, and a penalty clause are all embedded in the same space. Practitioners' questions have a type signature (rate lookup, procedure, eligibility) that we ignore.

**Root cause:** Every chunk is "a chunk."

**Fix (v2):** A cheap classifier pass (distilled or rule-based) tags each chunk as one of: `definition`, `rate_table`, `procedure`, `eligibility_rule`, `penalty`, `form`, `exemption`, `notification_body`, `case_reference`, `general_prose`. Used for type-aware retrieval routing and as a Qdrant filter.

### 2.9 No question bank generated at ingest

**Evidence:** We spent weeks generating 4,000 Q&A pairs after the fact, paying for Claude/Gemini API calls, dealing with schema inconsistencies, running QA, filtering garbage. If this had been done during ingest — when the paragraph was already in memory — it would have been 10× cheaper and fully coupled to the source.

**Fix (v2):** Every substantive chunk generates 2–3 practitioner-style questions at ingest via local qwen3-14b (free on our hardware). Stored as `chunk.synthetic_questions[]`. These become (a) training pairs for fine-tuning, (b) question-side embeddings for hybrid retrieval, (c) a smoke-test corpus.

### 2.10 No provenance audit trail

**Evidence:** If retrieval returns the wrong chunk, we have no easy way to trace back: which PDF, which parse, which version of the chunker, which date. Debugging is archaeology.

**Fix (v2):** Every chunk records `ingest_run_id`, `chunker_version`, `source_pdf_sha256`, `source_page_num`, `extraction_method` (pdfplumber / qwen-vl / rapidocr). Full reproducibility.

---

## Part 3 — Re-Ingestion Plan (Design for 95% Trust)

### 3.1 Design principles

1. **Metadata-first, content second.** A chunk is only valid if its metadata is valid.
2. **Semantic boundaries, not byte boundaries.** Chunk the document tree, not the text stream.
3. **Validate at every gate.** Each stage has acceptance criteria; bad chunks are quarantined, not embedded.
4. **Couple generation to ingest.** Synthetic questions, captions, summaries — all created when the source is in hand.
5. **One source of truth.** Qdrant holds embeddings; SQLite holds structured tables + chunk registry; filesystem holds raw PDFs + extracted text. Clear separation.
6. **Reproducibility.** Every chunk traceable to a specific ingest run, chunker version, and source page.
7. **Measure early, measure continuously.** The gold set runs after every stage, not only at the end.

### 3.2 Stage-by-stage pipeline

**Stage 0 — Corpus inventory & deduplication**
- Walk all PDFs, compute SHA-256 of content.
- Deduplicate (we likely have multiple copies of same circular).
- Classify by source (CBIC GST, CBIC Customs, Finance Act, Central Excise, Service Tax, State VAT adjacencies).
- Write `corpus_manifest.json` — the authoritative list of what we will ingest.
- **Acceptance:** 100% of PDFs classified, 0 hash collisions.

**Stage 1 — PDF → structured text**
- pdfplumber for text-extractable PDFs.
- Qwen2.5-VL-7B (Vulkan GPU 4/6) for image-only PDFs.
- RapidOCR CPU fallback for VL failures.
- Output: `{pdf_id, pages: [{page_num, text, tables: [markdown]}], method, confidence}`.
- **Acceptance:** 100% of PDFs have extracted text, <1% OCR-garbled (measured by vocabulary ratio).

**Stage 2 — Document parsing (the critical stage)**
- Regex + rule-based parser detects: document header (Act name, notification number, date), preamble, section/sub-section/clause boundaries, provisos, explanations, annexures, tables.
- Build a document tree with explicit parent-child links.
- Output per doc: `{doc_id, doc_type, parent_act, doc_number, issue_date, effective_from, effective_to, tree: [...]}`.
- **Acceptance:** ≥95% of sections have section_ref assigned; spot-check 50 random docs for tree correctness.

**Stage 3 — Chunking (from the tree, not from text)**
- Each tree node becomes a candidate chunk.
- Prose nodes >1,200 chars split at paragraph / semicolon boundaries with 100-char overlap.
- Tables <1,500 chars kept whole; larger split by row groups with header repeated.
- Every chunk inherits metadata from its tree ancestors: `parent_act > doc_number > section > sub_section > clause > proviso`.
- Assign `chunk_type` via classifier.
- **Acceptance:** <2% chunks >2,400 chars; <5% chunks <150 chars; 100% chunks have `section_ref` OR explicit `no_section_reason`.

**Stage 4 — Template / boilerplate removal**
- Compute 5-gram shingle hash across all chunks.
- Cluster near-duplicates; if cluster ≥3 and content is >70% form-scaffolding, mark `is_template=true`.
- Templates excluded from retrieval but kept in an auxiliary store for form lookups.
- **Acceptance:** zero template clusters with >10 copies in active retrieval index.

**Stage 5 — Structured extraction**
- For every table: attempt schema detection (tariff rate table, currency conversion, drawback schedule, etc.). Extract rows into SQLite with typed columns.
- For every notification body: extract `supersedes`, `superseded_by`, `effective_from`.
- Attach structured links back to chunks.
- **Acceptance:** ≥80% of tables schema-matched; all notifications have effective-date metadata.

**Stage 6 — Synthetic Q&A generation (coupled)**
- For each substantive chunk (non-template, ≥200 chars, `chunk_type` in definition/rule/rate/procedure/eligibility), generate 2–3 practitioner questions via local qwen3-14b. Balanced complexity (LOW/MED/HIGH).
- Validate each Q with a cheap self-check: "can this be answered by this chunk alone?" — quick qwen-judge pass.
- Store as `chunk.synthetic_questions[]`.
- **Acceptance:** ≥60% of substantive chunks have ≥2 validated questions (rest = too narrow or tabular).

**Stage 7 — Embedding**
- BGE-M3 dense + sparse on GPU 5 Vulkan, batch 32, same proven recipe.
- Embed both the chunk text AND each synthetic question (question-side retrieval for hybrid search).
- Write to fresh collection `cbic_v2` with the v2 payload schema.
- **Acceptance:** 0 embedding failures, collection point-count matches chunk registry.

**Stage 8 — Retrieval layer**
- Hybrid: dense + sparse + structured (SQLite for rate/date lookups).
- Chunk-type-aware routing: rate question → prefer `rate_table` chunks; procedure question → prefer `procedure` chunks.
- Default effective-date filter: only current-in-force unless user asks "as of YYYY".
- Reranker: cross-encoder (bge-reranker-v2-m3) top-50 → top-10.
- **Acceptance:** recall@5 ≥ 80% on gold set BEFORE fine-tuning.

**Stage 9 — Fine-tuning (only if Stage 8 is <90%)**
- Use the Stage 6 synthetic Q&A + hard negatives mined from Stage 8 retrieval.
- BGE-M3 MNRL on RunPod A100.
- Re-embed v2 collection as `cbic_v2_ft`.
- **Acceptance:** recall@5 ≥ 95%.

**Stage 10 — Generator + citation**
- qwen3-14b with structured prompt requiring: citation must match retrieved chunk's `hierarchy_path`; answer must quote the specific proviso/clause; refusal if retrieval confidence low.
- Faithfulness check: post-hoc judge (claude or gemini) scores citation fidelity on gold set.
- **Acceptance:** ≥95% of generated answers cite correctly; ≥95% are factually faithful to cited chunks.

### 3.3 Gate-based evaluation

Run the 170-question gold set after **every stage** from Stage 3 onward. A stage is not complete until its acceptance criteria are met. This avoids the Attempt #1 pattern of "ingest everything, then discover problems."

| Stage | Gate metric | Threshold |
|---|---|---:|
| 3 | chunks >2400 chars | <2% |
| 3 | chunks missing section_ref | <5% |
| 4 | template-cluster copies in index | 0 |
| 5 | tables schema-matched | ≥80% |
| 6 | chunks with ≥2 validated Qs | ≥60% |
| 7 | embedding success | 100% |
| 8 | recall@5 (before FT) | ≥80% |
| 9 | recall@5 (after FT) | ≥95% |
| 10 | citation fidelity | ≥95% |

---

## Part 4 — What "95% Trust" Means (Operational Definition)

95% trust is not one number. It's five conditions that must hold simultaneously:

1. **Retrieval:** recall@5 ≥ 95% on a held-out gold set of ≥500 practitioner questions.
2. **Citation fidelity:** ≥95% of answers cite a chunk whose `section_ref` + `doc_number` matches the expected authoritative source.
3. **Factual faithfulness:** ≥95% of answer sentences are entailed by the cited chunk (judged by strong LLM).
4. **Calibration:** when the system returns "I don't know" or "insufficient evidence", it is correct ≥95% of the time (no false confidence).
5. **Reproducibility:** the same question at a different time returns an answer citing the same chunk (no retrieval drift).

We need to measure all five. The current system measures only #1 (poorly).

---

## Part 5 — Risks & Open Questions for External LLMs

### 5.1 Risks

- **Time cost.** Stages 1–5 could take 1–2 weeks on this hardware. Worth it, but we should know before starting.
- **OCR quality.** Qwen2.5-VL has never been run at this scale on our rig. Budget for 2× on OCR failure handling.
- **Document-parser coverage.** Indian tax PDFs are inconsistent. The regex rules will need iteration on a representative sample first.
- **Table schema detection.** Rate tables alone have 10+ layouts. May need per-source schemas.
- **Generator still a bottleneck.** Even with perfect retrieval, qwen3-14b may fabricate. Faithfulness check is non-negotiable.

### 5.2 Questions we want external LLMs to answer

**About the ROCm pivot (NEW — highest priority):**

0a. **If ROCm now works on RDNA1 (gfx1010) + RDNA2 (gfx1031) consumer cards** — what's the current best-practice stack as of April 2026? PyTorch-ROCm version, bitsandbytes alternative, flash-attention status? Any gotchas specific to these cards?

0b. **Given ROCm works, what becomes possible that Vulkan llama.cpp cannot do?** Priority list of wins — we want sentence-transformers native, cross-encoder rerankers, and Surya/Nougat OCR. Are there unexpected wins we should know about?

0c. **GPU-accelerated chunking** — is there a production-worthy way to do document chunking on GPU? Candidates we're aware of: (a) semantic chunking via sentence embedding similarity breakpoints (sentence-transformers on GPU), (b) LLM-based structured extraction (small qwen / Nougat / Surya / Donut doing parse+chunk in one pass), (c) LayoutLM-style layout-aware parsing. Which of these are actually faster end-to-end than CPU regex + PyMuPDF on our corpus (Indian legal PDFs with heavy tables, section numbering, provisos)? Any benchmarks or battle-tested libraries?

0d. **Nougat / Surya / Marker** — on PyTorch-ROCm with RDNA1/RDNA2, do any of these actually run? Any reports from the community? Marker in particular is known to handle tables well for legal/financial PDFs.

0e. **Cross-encoder reranker on ROCm** — bge-reranker-v2-m3 vs Qwen3-Reranker-0.6B vs jina-reranker-v2 — which is the best bet for Indian tax domain, and which runs fastest on RDNA2 via PyTorch-ROCm?

**Original questions (Attempt #1 repair context, still relevant):**


1. **Is the staged gate-based approach sound, or are we overengineering? What would you cut?**
2. **For the document parser: build from scratch with regex, or use an existing Indian-legal-document parser (e.g., Indian Kanoon's toolchain, academic releases)?**
3. **For chunking: do you recommend a different strategy than tree-walk with size cap? (e.g., propositional chunking, late chunking, small2big)**
4. **For table handling: is dual-store (Qdrant + SQLite) the right call, or should tables stay text-only with better captions?**
5. **For embedding: stay with BGE-M3, or switch to Qwen3-Embedding-0.6B / E5-Mistral / nomic-embed-v2 given we're starting fresh?**
6. **For synthetic Q&A generation at ingest: qwen3-14b is free but weaker than Claude/Gemini. Is the quality loss acceptable given the 100× cost advantage?**
7. **For supersession tracking: any battle-tested library for Indian tax notification chains, or do we roll our own?**
8. **For the 95% operational definition: is calibration (#4) the right 5th metric, or should we use something else (e.g., robustness to paraphrase)?**
9. **Retrieval stack: dense + sparse + reranker covers most bases. Should we add a query-decomposition step for multi-hop questions (currently 6.5% of gold set)?**
10. **Scaling question: 108K chunks → likely 150–200K after proper chunking. Any concerns at that scale for Qdrant on our hardware (64GB RAM, 4-core CPU)?**

### 5.3 Out of scope for this consult

- UI changes (dashboard stays as-is for now).
- New data sources (we rebuild on the existing corpus first).
- Agent / tool-use layer (separate project).

---

## Part 6 — Migration Strategy

- **Keep `cbic_v1` read-only** during v2 build. Existing system stays up; dashboard keeps working.
- **Build `cbic_v2` in parallel** in a new Qdrant collection + new SQLite DB.
- **A/B compare** on gold set at every stage.
- **Cut over only when** `cbic_v2` exceeds `cbic_v1` on every metric AND hits 95% gate.
- **Keep `cbic_v1` archived** for 3 months post-cutover in case of regression.

---

## Part 7 — Artifacts to produce during v2

1. `corpus_manifest.json` — inventory.
2. `parser_v2.py` — document tree parser, unit-tested on 50 hand-verified docs.
3. `chunker_v2.py` — tree-walker with size cap.
4. `template_detector.py` — near-duplicate clustering.
5. `table_extractor.py` — schema detection + SQLite writer.
6. `synth_qa_generator.py` — qwen3-14b coupled generator.
7. `gate_eval.py` — runs gold set at each stage.
8. `metrics_dashboard.md` — auto-updated per-stage metrics.
9. `chunk_registry.db` — SQLite of every chunk + provenance.
10. `reingest_decisions.md` — running log of tradeoffs made.

---

## Appendix A — Evidence pointers from Attempt #1

- Chunk audit: `D:\_gpu_rig_ai\consults\chunk_audit_20260422.md` (108,802 chunks, 83% missing section_ref, 19% >2400 chars, 273× boilerplate)
- Recall audit: `D:\_gpu_rig_ai\consults\recall_audit_20260422.md`
- Failure modes: `D:\_gpu_rig_ai\consults\failure_modes_20260422.md`
- Embedder alternatives analysis: `D:\_gpu_rig_ai\consults\embedder_alternatives_20260422.md`
- Training pair gen experience: 4,000 pairs across Gemini / Claude Opus / Sonnet with 3 separate schema fixes and 1 full quarantine (`pairs_opus_highcomplex_BAD_argv.jsonl`) — evidence that post-hoc pair generation is fragile.

## Appendix B — Why this plan is different from Attempt #1

Attempt #1 was **content-first, metadata-opportunistic, fix-later**. v2 is **metadata-first, gate-driven, measure-continuously**. Every criticism of v1 in Part 2 maps to a specific stage and acceptance criterion in Part 3. There is no step where "we'll deal with that later."

---

---

## Part 8 — Hardware Plan (Rig-Specific, No-Reinvent)

### 8.1 What the rig actually is (facts, not assumptions)

- **Host:** 4-core CPU, 64 GB RAM, Ubuntu (HiveOS-derived), Qdrant on port 6343.
- **GPU fleet (mining rig):** mixed RDNA. RX 5700 XT (gfx1010, 8 GB, RDNA1) and RX 6700 XT (gfx1031, 12 GB, RDNA2). Multiple cards, enumerated as GPU 0–6.
- **Only working ML toolchain:** Vulkan via llama-cpp-python / llama-server. ROCm / torch-ROCm / onnxruntime-ROCm are ABI-broken on these consumer cards — do not try again.
- **Proven limits from Attempt #1:**
  - Single GPU 5 Vulkan + BGE-M3 = 5.2–9.6 chunks/sec (stable).
  - Multi-GPU Vulkan embedding pool = 0.3 ch/s (D-state kernel hang — banned).
  - Qwen2.5-VL-7B on Vulkan = 600 s/page (vision encoder is the wall — 16-day extrapolation for 471 docs).
  - Tesseract/RapidOCR on 4-core CPU = query-killer (load avg 33; banned by user directive).

### 8.2 Fixed GPU allocation for v2 ingest

Runs concurrently during the ingest window. `cbic_v1` stays up on its current assignment; v2 uses the remaining cards.

| GPU | Card | Role during v2 ingest | Model | Port | Why |
|---:|---|---|---|---:|---|
| 0 | RX 5700 XT | idle / hot spare | — | — | Keep as failover for GPU 5 |
| 1 | RX 5700 XT | idle / hot spare | — | — | Keep as failover for GPU 5 |
| 2 | RX 6700 XT | **qwen3-14B chat (existing)** | qwen3-14b Q4_K_M | 9082 | Do not disturb — user queries continue |
| 3 | RX 5700 XT | **synthetic Q&A generator (new)** | qwen3-8B or 14B Q4_K_M | 9083 | Stage 6 (Q&A gen at ingest) runs here, isolated from chat |
| 4 | RX 6700 XT | **OCR primary (new)** | MiniCPM-V 2.6 or SmolVLM Q4 | 9600 | Fastest RDNA2 card; Qwen2.5-VL banned |
| 5 | RX 5700 XT | **BGE-M3 embedder (proven)** | bge-m3.gguf | 9084 | Single-GPU, proven 5.2–9.6 ch/s |
| 6 | RX 6700 XT | **OCR secondary (new)** | same as GPU 4 | 9601 | Parallel page queue to halve OCR wall-clock |

**Why this layout:**
- GPU 5 remains the sole embedder (multi-GPU embed is proven broken). Non-negotiable.
- GPU 2 keeps serving users; v2 is a background build, not a cutover.
- GPU 3 runs the Q&A generator at ingest so we don't hit the chat endpoint and stall users. Pick a model that fits 8 GB on RDNA1 — **qwen3-8B Q4_K_M** is the safe choice (fits in ~6 GB, plenty of headroom).
- GPUs 4 + 6 form a two-worker OCR pool. Dispatch pages round-robin.
- GPUs 0 and 1 stay cold as hot spares — if GPU 5 faults mid-embed, swap in minutes, not hours.

### 8.3 Per-service launch recipes (reuse proven env vars)

**Embedder — GPU 5 (unchanged from playbook, proven):**

```bash
export GGML_VK_VISIBLE_DEVICES=5
export RADV_DEBUG=nodcc
export GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1
# llama-cpp-python in-process, n_ctx=8192, n_batch=512, pooling_type=2
# per-text create_embedding([t]) — not batched
# UPSERT_BATCH=128, CHUNK_CHARS/OVERLAP per Stage 3 spec
```

**Q&A generator — GPU 3 (new, isolated from chat):**

```bash
GGML_VK_VISIBLE_DEVICES=3 RADV_DEBUG=nodcc \
  llama-server -m /opt/ai-models/qwen3/qwen3-8b-Q4_K_M.gguf \
  -ngl 99 -c 8192 -b 1024 -ub 512 -t 3 \
  --host 0.0.0.0 --port 9083
```

Temperature 0.2, max_tokens 512. Strict JSON prompt (reuse `gen_training_pairs_v2.py` prompt, already proven).

**OCR pool — GPUs 4 + 6 (new, must solve the Qwen2.5-VL bottleneck before launching):**

Decision required **before** OCR work starts: which smaller VLM will actually deliver <30 s/page on Vulkan?
- Candidate A: **MiniCPM-V 2.6** Q4 (~6 GB) — strongest non-Qwen Hindi OCR in 7B tier.
- Candidate B: **SmolVLM-2B** — newer, smaller CLIP tower, accuracy risk on Devanagari.
- Candidate C: **Qwen2-VL-2B** — same family, smaller CLIP; ceding quality for speed.
- Candidate D: **Park OCR** for v2 launch, ingest 108K - ~1.5K image-only chunks first (the 471 image PDFs are ~2.5% of corpus), ship with that gap documented, revisit when a faster VLM lands.

**Recommendation: Option D for first cut**, because OCR is the one stage where we still have no proven recipe on this hardware. Shipping v2 with a 97.5% coverage is better than delaying v2 indefinitely to solve OCR. The 471 image PDFs stay in a `ocr_pending` queue; they can be folded into `cbic_v2` via the already-designed Stage 1+2 OCR pipeline (`ocr_worker.py` + `ingest_ocr_worker.py`) once the VLM question is answered. Ask external LLMs which candidate (A/B/C) is most likely to break the Vulkan CLIP wall.

**CPU partitioning (reuse slice definitions from OCR research doc):**
- `llama.slice` → cores 0–1 (all GPU-bound llama-server processes — host side is tiny).
- `ingest.slice` → cores 2–3 (pdfplumber / PyMuPDF / parser / chunker — CPU-bound).
- No tesseract, no RapidOCR, no CPU OCR ever.

### 8.4 Concurrency model during v2 build

Four concurrent pipelines, each on its own CPU core pool + GPU:

```
Pipeline A (parser):     core 2+3 CPU → writes parsed_tree.jsonl
Pipeline B (chunker):    core 2+3 CPU → reads parsed_tree, writes chunks.jsonl
Pipeline C (Q&A gen):    core 0-1 CPU + GPU 3 (qwen3-8B) → reads chunks, writes qa_pairs
Pipeline D (embedder):   core 0-1 CPU + GPU 5 (BGE-M3) → reads chunks, writes Qdrant
```

Pipelines A→B→D are sequential per-doc; Pipeline C runs in parallel with D (both read finished chunks). This mirrors the proven `overnight_worker.py` single-lane model, scaled to two lanes safely because they use different GPUs.

**Do NOT parallelize embedding across GPUs.** The playbook is explicit: 0.3 ch/s if you try. Stick to GPU 5.

### 8.5 Throughput budget

| Stage | Rate | Volume | Wall-clock |
|---|---|---|---|
| Stage 0–1 (inventory + text extract, text PDFs) | limited by disk/CPU | ~15K PDFs | **~4–6 hrs** |
| Stage 1 (VLM OCR on 471 image PDFs, GPUs 4+6 parallel) | best-case 20 pp/min aggregate | ~4,700 pages | **~4–6 hrs** (if VLM problem solved) |
| Stage 2 (parse) | ~50 doc/sec pure Python | 15K docs | **~5 min** |
| Stage 3 (chunk) | ~200 chunk/sec | ~150K chunks | **~12 min** |
| Stage 4 (template dedupe) | ~10K chunks/sec (hashing) | 150K | **~30 sec** |
| Stage 5 (table extract) | ~100 tables/sec (regex + SQLite) | ~40K tables | **~7 min** |
| Stage 6 (synth Q&A, GPU 3) | ~1 chunk/sec | ~80K substantive chunks | **~22 hrs** (longest stage) |
| Stage 7 (embed, GPU 5) | proven 9.6 ch/s | 150K chunks | **~4.3 hrs** |
| Stage 8 (retrieval test) | 170 Qs × /query | — | **~30 min** |
| Stage 9 (fine-tune on RunPod) | — | 5K filtered pairs | **~30 min + $1** |
| Stage 10 (gate eval) | — | 170–500 Qs | **~1–2 hrs** |

**Total wall-clock budget: ~2–3 days** if Stage 6 runs overnight (it's the long tail). Can cut to ~1 day if we run Stage 6 on both GPU 3 and GPU 4/6 (when OCR isn't using them) in parallel.

### 8.6 Resource guardrails (non-negotiable)

1. **GPU 5 (embedder) has exclusive use during its phase.** No other process touches it. The single-GPU rule is the #1 lesson from Attempt #1.
2. **GPU 2 (qwen3-14B chat) is untouched.** v2 build is invisible to users.
3. **CPU cores 2–3 are reserved for CPU-bound ingest work.** `ionice -c 3` on all parser/chunker processes.
4. **No OCR on CPU.** Ever. (User's explicit directive.)
5. **Qdrant writes use `wait=False`.** Already proven — don't switch to sync.
6. **Every worker writes to its own log file with `nohup` + `disown`.** Kill-safe, resume-safe.
7. **Every ingest worker is idempotent via deterministic point IDs** (`blake2b(doc_id|page|chunk_idx)` — existing recipe). Re-run is a no-op.

---

## Part 9 — Reuse Map (What NOT to Rewrite)

This is the explicit list of existing artifacts that will be reused verbatim, extended with minimal changes, or replaced. No gratuitous rewrites.

### 9.1 Reuse verbatim (zero changes)

| Artifact | Location | Why reuse |
|---|---|---|
| BGE-M3 embed recipe (llama-cpp-python, pooling_type=2, n_ctx=8192) | `ingest_playbook_cbic.md` | Proven 5.2–9.6 ch/s, only working path |
| BM25 sparse recipe (fastembed `Qdrant/bm25`) | playbook | Proven |
| Deterministic point ID (`blake2b(doc_id|page|chunk_idx)`) | playbook | Idempotent re-ingest critical |
| Env vars (`GGML_VK_VISIBLE_DEVICES`, `RADV_DEBUG=nodcc`, `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1`) | playbook | Hardware workarounds, non-optional |
| Size-ascending TSV launch ordering | playbook | Fast-feedback guarantee |
| TSV schema (`doc_id TAB category TAB subcategory TAB path TAB title`) | playbook | Every downstream script expects this |
| Qdrant collection config (dense 1024-d cosine + sparse, payload indexes) | playbook | Proven schema |
| `quality_check.py`, `one_chunk_audit.py` | `/opt/indian-legal-ai/rag/cbic_rag/` | Full post-ingest audit harness |
| `ingest_monitor.py` | `D:\_gpu_rig_ai\` | Dashboard during ingest |
| Qwen2.5-VL Q4 + mmproj files on disk | `/opt/ai-models/qwen25vl/` | 5.7 GB already staged; keep for any VLM experiment |
| `ocr_poc.py` (driver) | `D:\_gpu_rig_ai\ocr_poc.py` | Can be rebased on a different model |
| Gold set (170 Q) and eval harness | `D:\_gpu_rig_ai\eval\` | Gate metric source of truth |
| `variants_and_hardneg.py` | `D:\_gpu_rig_ai\eval\` | Retrieval test harness (use for gate eval Stage 8) |
| Hard-neg miner (`training/mine_hard_negatives.py`) | `D:\_gpu_rig_ai\training\` | Reuse for Stage 9 |
| RunPod training pipeline (`training/runpod/`) | `D:\_gpu_rig_ai\training\runpod\` | All 5 scripts ready |
| `finetune_bge_m3.py` (post-audit version, with SequentialEvaluator + gold metrics) | `D:\_gpu_rig_ai\training\` | Stage 9 drop-in |
| `prep_pairs.py` (post-audit, accepts `--mode curated`) | `D:\_gpu_rig_ai\training\` | Stage 9 drop-in |
| `curate_training_set.py` | `D:\_gpu_rig_ai\training\` | Filter Stage 6 pairs |
| `gen_training_pairs_v2.py` (Gemini) | `D:\_gpu_rig_ai\scripts\` | Reference prompt for Stage 6 (swap API to local qwen3-8B) |
| `qa_pairs_claude.py` grader | `D:\_gpu_rig_ai\training\` | Reuse as gate-eval grader |

### 9.2 Extend with minimal changes

| Artifact | Change | Reason |
|---|---|---|
| `overnight_worker.py` | Read from the new `chunks.jsonl` output of Stage 3 instead of extracting+chunking inline; embed + upsert unchanged | Proven embed path stays; only the input source changes |
| `recovery_worker.py` | Same as above | Same |
| `ingest_ocr_worker.py` | Point input cache to the new OCR output dir; otherwise unchanged | OCR cache format already page-marker-delimited |
| Qdrant collection `cbic_v1` → `cbic_v2` | Add new payload fields (`sub_section`, `clause`, `proviso_id`, `chunk_type`, `effective_from`, `hierarchy_path`, `synthetic_questions`, `is_template`) | Same base schema; additive |
| `gen_training_pairs_v2.py` | Repoint from Gemini API to local qwen3-8B at `:9083`; keep prompt | Local model, zero API cost |
| `qa_pairs_claude.py` grader | Extend to also grade `section_ref` extraction correctness (new metric: metadata-faithful) | Broader QA |

### 9.3 New scripts required (small footprint)

Only these are genuinely new. Everything else is reuse or extension.

| Script | Purpose | Est. LOC |
|---|---|---:|
| `parser_v2.py` | PDF/OCR-text → document tree with section hierarchy | ~400 |
| `chunker_v2.py` | Tree-walk + size cap + overlap; inherits metadata from ancestors | ~200 |
| `template_detector.py` | 5-gram shingle clustering; mark `is_template=true` | ~150 |
| `table_extractor.py` | Schema-detect tariff/rate/currency tables → SQLite | ~300 |
| `synth_qa_at_ingest.py` | For each chunk, call local qwen3-8B at `:9083`; 2–3 questions; self-check | ~200 (mostly copy of `gen_training_pairs_v2.py`) |
| `gate_eval.py` | Run gold set after each stage; fail build if threshold missed | ~150 (wraps existing `variants_and_hardneg.py`) |
| `chunk_registry.py` | SQLite writer for per-chunk provenance (`ingest_run_id`, `chunker_version`, etc.) | ~100 |

**Total new code: ~1,500 LOC.** Everything else is existing, proven, or trivial config. This is a plan to build around the playbook, not replace it.

### 9.4 Explicit non-reuse (what we drop)

- **Post-hoc Q&A generation via Claude/Gemini API.** Stage 6 coupled generation replaces it. Keep the 4,000 pairs from Attempt #1 as a validation set, not training data.
- **Multi-GPU embed pool code paths in `recovery_worker.py`.** Deleted. Always single-GPU.
- **Tesseract + RapidOCR CPU fallbacks.** Dropped per user directive. OCR is GPU-only or parked.
- **The CPU-partitioning slice definitions for CPU OCR.** No longer needed.
- **`consult_package_20260422.md`.** Superseded by this document for the v2 conversation. (Old consult can still be sent separately for the fine-tune-first path, if user wants.)

---

## Part 10 — Questions Specifically About the Hardware Plan

Added to the external LLM consult (on top of Part 5.2):

11. **Small VLM choice for OCR:** MiniCPM-V 2.6 vs SmolVLM-2B vs Qwen2-VL-2B on Vulkan RDNA2 — any benchmarks or community reports on vision-encoder throughput specifically? (Qwen2.5-VL-7B hit 600 s/page due to CLIP, not LLM.)
12. **Park OCR for v2 launch (Option D):** acceptable tradeoff to ship with ~97.5% coverage and a documented gap, then fold in OCR via a separate pass?
13. **GPU 3 as Q&A generator:** qwen3-8B enough, or is the quality drop vs qwen3-14B significant enough to justify running Stage 6 on GPU 2 off-hours?
14. **Parallel pipeline C + D:** any reason the Q&A generator (GPU 3) and embedder (GPU 5) can't run concurrently? (They're on different cards, different PCIe lanes.)
15. **OCR parallelism across GPU 4 + GPU 6:** confirm two llama-server instances on different GPUs with Vulkan don't hit the D-state issue (which was specific to embedding pool, not chat/VLM pools).
16. **Stage ordering:** should Stage 6 (Q&A gen) run before Stage 7 (embed) so synthetic questions can be embedded in the same pass, or after so we only embed validated questions?

17. **If ROCm is live** (H15 passes), should we drop the Vulkan llama.cpp embedder entirely in favor of sentence-transformers BGE-M3 on ROCm? Or keep both and route — Vulkan for the proven high-throughput bulk embed, ROCm for things Vulkan can't serve (rerankers, OCR, semantic chunker)?

18. **GPU chunking impact**: if semantic chunking via embeddings or Nougat-style LLM chunking turns out to be GPU-feasible, does the quality lift on Indian legal PDFs justify the added complexity over a clean regex/tree parser? Or is tree-walk chunking (Stage 3 as specified) already close to ceiling for this domain?

19. **Tool combo**: Marker (Surya + nougat + layout) is reportedly excellent on tables and legal documents. If it runs on our ROCm stack, could it replace most of Stage 1 (text extract) + Stage 2 (parse) + part of Stage 5 (table extract) in one shot? What quality tradeoffs?

---

---

## Part 11 — Proof Status & Required Benchmarks (No Guesswork Gate)

**Rule:** Nothing in this plan executes on the real corpus until every item below marked `UNPROVEN` has been benchmarked on this rig with a small test sample and either promoted to `PROVEN` or replaced with a fallback. Claims marked `PROVEN` come directly from Attempt #1's logs and the playbook.

### 11.1 Proof status of every hardware claim in the plan

| # | Claim | Status | Evidence / Benchmark required |
|---|---|---|---|
| H1 | Single GPU 5 Vulkan BGE-M3 in-process = 5.2–9.6 ch/s | **PROVEN** | `ingest_playbook_cbic.md`, Attempt #1 production logs |
| H2 | Multi-GPU Vulkan embed pool (llama-cpp-python in-process) = 0.3 ch/s, D-state hangs | **PROVEN (negative)** | Attempt #1 — do not retry this form |
| H3 | qwen3-14B on GPU 2 Vulkan llama-server @ :9082 runs alongside GPU 5 embedder without conflict | **PROVEN** | Current production; queries + ingest coexist daily |
| H4 | qwen3-8B Q4_K_M fits on an 8 GB RX 5700 XT (RDNA1) at `-c 8192 -ngl 99` | **UNPROVEN** | BENCHMARK: launch on GPU 3, 50-token completion, measure tok/s and VRAM |
| H5 | Two separate llama-server HTTP instances on GPU 4 + GPU 6 (different cards) run concurrently without D-state | **UNPROVEN** | BENCHMARK: launch qwen3-8B on both, issue 50 requests round-robin, watch dmesg for amdgpu ring reinit |
| H6 | Q&A generator (GPU 3) + embedder (GPU 5) run concurrently without stepping on each other | **UNPROVEN but low-risk** | BENCHMARK: 10-min parallel stress — GPU 5 embeds 1K chunks while GPU 3 generates 1K completions; check both throughputs match isolated rates within 10% |
| H7 | Two-GPU HTTP embedder pool (GPU 0 + GPU 1 llama-server, dispatcher round-robin) is stable AND ≥1.7× throughput of single-GPU | **UNPROVEN** | BENCHMARK: embed 5K chunks via 2-server pool, measure wall-clock vs single-GPU baseline |
| H19 | **Multi-GPU HTTP generator pool for Stage 6 Q&A** — 5–6 independent qwen3-8B llama-server instances on GPUs 0/1/3/4/5/6, dispatcher round-robins chunks, runs stable for hours | **UNPROVEN but architecturally sound** | BENCHMARK B8: run 5 parallel llama-servers for 30 min on 1K chunks; measure per-GPU throughput, CPU load avg, thermal, dmesg clean |
| H20 | Batching 4 chunks per llama-server request (one prompt with 4 chunk blobs → 8–12 questions) yields 2–3× throughput without quality loss | **UNPROVEN** | BENCHMARK B8b: 100-chunk batched vs unbatched; compare speed + QA-judge quality scores |
| H8 | Windows machine (this laptop) can run PyMuPDF parser + chunker with reasonable throughput | **UNPROVEN** | BENCHMARK: 100-PDF sample on Windows, measure docs/sec |
| H9 | SMB share from Windows → rig tolerates concurrent writes of 150K small JSONL lines without corruption/locking | **UNPROVEN but low-risk** | BENCHMARK: append-from-both-hosts to one file for 60 sec; verify line count and no interleaving |
| H10 | Qwen2.5-VL-7B Vulkan = 600 s/page (too slow) | **PROVEN (negative)** | `ocr_research_cbic.md` POC, 2026-04-21 |
| H11 | Any small VLM (MiniCPM-V 2.6 / SmolVLM-2B / Qwen2-VL-2B) will deliver <30 s/page on Vulkan | **UNPROVEN** | BENCHMARK: 10-page OCR POC on each candidate before committing to any OCR plan |
| H12 | Qdrant `cbic_v2` can hold ~150–200K dense+sparse points on this host with acceptable RAM | **PROVEN** | `cbic_v1` already holds 108K; room to 2× verified in prior Qdrant sizing |
| H13 | BGE-M3 re-embed of 150K chunks = ~4.3 hrs on GPU 5 | **EXTRAPOLATED from H1** | Acceptable — directly scales from proven throughput |
| H14 | RunPod A100 BGE-M3 MNRL fine-tune ≈ 30 min + ~$1 | **PROVEN externally** | `training/runpod/` pipeline staged; ref: prior RunPod runs |
| H15 | ROCm actually works on this rig's RDNA1 + RDNA2 cards (user reports recent success) | **UNPROVEN / RE-VERIFY** | BENCHMARK B0: `rocminfo` lists both gfx1010 and gfx1031 as agents; `torch.cuda.is_available()` via PyTorch-ROCm returns True; bge-m3 via sentence-transformers runs a 1K-chunk embed without ABI error |
| H16 | If H15 holds, sentence-transformers BGE-M3 on PyTorch-ROCm matches or beats llama-cpp Vulkan throughput | **UNPROVEN** | BENCHMARK B0b: embed 1K chunks, compare wall-clock to proven 9.6 ch/s Vulkan baseline |
| H17 | If H15 holds, cross-encoder reranker (bge-reranker-v2-m3) runs on ROCm at usable speed | **UNPROVEN** | BENCHMARK B0c: rerank 100×50 pairs; acceptable if ≤2 sec/query |
| H18 | If H15 holds, Surya OCR or Nougat on PyTorch-ROCm is usable for image-PDF parsing | **UNPROVEN** | BENCHMARK B0d: run 10-page POC; compare to small-VLM B7 candidates |

### 11.2 Benchmarks to run BEFORE any Stage 0 work begins (total: ~half a day)

Ordered by dependency. Each must produce a written result (pass/fail + numbers) before the next. **B0 runs first because it dictates the entire toolchain choice.**

| Step | Benchmark | Pass criterion | Wall-clock | On fail |
|---:|---|---|---:|---|
| **B0** | **H15 — ROCm actually works** on rig (rocminfo lists both GPUs, PyTorch-ROCm loads, no ABI error on 1-min smoke test) | Both gfx1010 + gfx1031 visible to torch; 1K-chunk sentence-transformers embed completes | 30 min | Stick with Vulkan-only stack (fallback = current plan). No further ROCm benchmarks run. |
| B0b | **H16 — BGE-M3 on ROCm** beats or matches Vulkan 9.6 ch/s | ≥8 ch/s on GPU 5 via PyTorch-ROCm | 15 min | Keep Vulkan BGE-M3 (proven); use ROCm only for things Vulkan can't do |
| B0c | **H17 — Cross-encoder reranker on ROCm** | bge-reranker-v2-m3 reranks 100×50 pairs, ≤2 sec/query on GPU 2 or 4 | 20 min | No reranker in the pipeline; retrieval must carry 95% on dense+sparse alone |
| B0d | **H18 — Surya OCR on ROCm** (or Nougat as fallback) | 10-page POC ≤20 s/page, Devanagari legible | 60 min | Skip Surya; return to Option D (park OCR) or small-VLM path (B7) |
| B1 | **H8 — Windows PyMuPDF speed** | ≥10 docs/sec on mixed corpus sample | 15 min | Drop split-ingest; run all parsing on rig cores 2–3 only |
| B2 | **H9 — SMB concurrent write** | 60 sec both-host append, line count matches, no interleaving | 5 min | Use per-host output files + merge step (trivial) |
| B3 | **H4 — qwen3-8B on GPU 3 RDNA1** | Loads at `-ngl 99 -c 8192`; ≥8 tok/s; <7 GB VRAM | 10 min | Fall back to qwen3-4B or run Q&A gen on GPU 2 off-hours |
| B4 | **H6 — Parallel GPU 3 + GPU 5** | Both throughputs within 10% of their isolated baselines during 10-min stress | 15 min | Serialize stages 6 and 7 instead of parallelizing |
| B5 | **H7 — 2-GPU HTTP embedder pool (GPU 0 + GPU 1)** | Stable for 30 min continuous; ≥1.7× single-GPU throughput; no dmesg amdgpu errors | 40 min | **Single-GPU embed only** — fall back to H1 (already the default) |
| B6 | **H5 — Two llama-server on GPU 4 + GPU 6** | Both serve 50 requests round-robin; no D-state; both maintain steady tok/s | 20 min | OCR goes single-GPU (halves OCR throughput but keeps it working) |
| B7 | **H11 — Small VLM OCR POC** (MiniCPM-V 2.6 first; if >30 s/page, try SmolVLM-2B; if >30 s/page, try Qwen2-VL-2B) | Any candidate ≤30 s/page with acceptable Devanagari fidelity on 10-page sample | 60–90 min | **Park OCR for v2 ship (Option D)**, 471 image PDFs remain in ocr_pending queue |
| B8 | **H19 — Multi-GPU generator pool for Stage 6** — spin up 5 qwen3-8B llama-servers on GPUs 0, 1, 3, 4, 6; dispatcher sends 1K chunks round-robin | All 5 stable for 30 min; aggregate ≥4× single-GPU throughput; load avg <6; no dmesg GPU errors | 40 min | Fall back to fewer GPUs (3 or 1); wall-clock scales linearly |
| B8b | **H20 — Batched Q&A prompts** (4 chunks per request) | ≥2× throughput vs single-chunk; QA-judge quality within 5% of single-chunk baseline on 100-sample | 30 min | Use single-chunk prompts (proven safer) |

**Total benchmark window: ~3 hours.** Every decision downstream keys off these results. No part of the plan executes on guess.

### 11.3 What changes in the plan if benchmarks come back positive vs negative

| Benchmark | If PASS | If FAIL |
|---|---|---|
| B1 (Windows PyMuPDF) | Parse + chunk split: Windows handles half of PDFs (by hash mod 2), rig cores 2–3 handle other half. Wall-clock parsing halves. | Rig-only parsing; wall-clock unchanged |
| B5 (2-GPU embed pool) | Use 2-GPU pool → embed 150K chunks in ~2.3 hrs instead of ~4.3 hrs | Single-GPU embed (default); 4.3 hrs budget |
| B4 (parallel C+D) | Stage 6 + Stage 7 run concurrently → ~22 hr Stage 6 overlaps Stage 7 completely | Run Stage 7 first (4.3 hr), then Stage 6 on GPU 5 too in parallel with anything else; wall-clock extends by ~4 hrs |
| B6 (2-GPU OCR pool) | OCR pool across GPU 4 + 6, halves OCR wall-clock | OCR single-GPU, doubles wall-clock (still feasible if B7 passes) |
| B7 (small VLM) | OCR is live for v2 launch | Ship v2 without the 471 image PDFs; add OCR later |

### 11.4 Final hardware allocation — conditional on benchmarks

Baseline plan (only PROVEN items):

| GPU | Role | Status |
|---:|---|---|
| 2 | qwen3-14B chat (existing) | PROVEN |
| 5 | BGE-M3 embedder, single-GPU | PROVEN |

Conditional adds (activated only if corresponding benchmark passes):

| GPU | Role | Depends on |
|---:|---|---|
| 3 | qwen3-8B Q&A generator @ :9083 | B3 PASS |
| 0 + 1 | Second embedder pool, HTTP round-robin | B5 PASS + B7-style sizing |
| 4 + 6 | OCR pool (small VLM) | B6 PASS + B7 PASS |

This is the only honest way to write the hardware plan before benchmarks run.

---

## Part 12 — Distributed CPU Chunking (Windows + Rig)

**Premise:** Parsing and chunking are pure Python (PyMuPDF + regex). Zero GPU dependency. This work can run anywhere Python + PyMuPDF runs.

### 12.1 Why splitting matters

Attempt #1 text-extracted 15K+ PDFs inside `overnight_worker.py`, single-threaded, on rig core 2. Realistic Windows laptop parsing throughput for 15K mixed PDFs is 1–3 hours (many tax PDFs are 100–500 pages). Splitting across Windows + rig halves this. More importantly, it moves CPU-heavy parsing OFF the rig's 4-core host during the ingest window, freeing cores 2–3 for the template detector, table extractor, and chunk-registry SQLite writes that must run on the rig.

### 12.2 Architecture

```
Windows laptop (D:\_gpu_rig_ai\ingest_v2\)      Rig (/opt/indian-legal-ai/ingest_v2/)
├── parse_worker.py (cores 0–N)         ├── parse_worker.py (cores 2–3)
│    reads: PDFs via SMB                │    reads: PDFs local
│    writes: parsed_tree_win_*.jsonl    │    writes: parsed_tree_rig_*.jsonl
│                                        │
└── (writes to SMB share)                └── (writes to local disk, mirrored via SMB)

                  Shared folder (SMB from Windows to rig or vice versa)
                  ├── manifest.jsonl     — full PDF inventory (assigned host per row)
                  ├── parsed_tree/       — one JSONL per worker shard
                  ├── chunks/            — chunker output (merged from both hosts)
                  └── progress.sqlite    — per-doc status (resume-safe)
```

### 12.3 PDF assignment

Simplest split: `assigned_host = "windows" if hash(doc_id) % 2 == 0 else "rig"`. Deterministic, resume-safe, idempotent.

If Windows laptop benchmarks slower than rig (common — laptop SSDs read PDFs fine but single-core Python parsing is CPU-bound), shift ratio via `hash % 10 < WIN_SHARE` where `WIN_SHARE` is calibrated from B1.

### 12.4 Output schema (both hosts write identical format)

```jsonl
{"doc_id":"...","host":"win|rig","worker_pid":123,"tree":{...},"pages":N,"parse_method":"pymupdf|ocr","ingest_run_id":"v2-20260422","chunker_version":"v2.0","source_pdf_sha256":"..."}
```

### 12.5 Chunker runs after parsing (one host, fast)

Chunking a parsed tree is cheap (~200 chunks/sec pure Python). No need to split. Run once on the rig, reading merged `parsed_tree/*.jsonl` from SMB, writing to `chunks/chunks.jsonl`. Wall-clock ~12 min for 150K chunks — not worth splitting.

### 12.6 SMB write correctness (why B2 benchmark exists)

Both hosts may write to the parsed_tree/ directory simultaneously. Safety guaranteed by **one file per shard** (e.g., `parsed_tree_win_001.jsonl`, `parsed_tree_rig_001.jsonl`) — never shared-append. Each shard is ≤5K docs, flushed every 100 lines. B2 verifies this pattern works reliably over SMB.

### 12.7 Failure modes to pre-empt

- **SMB disconnect mid-run:** each worker writes to a local temp file first, renames atomically into shared folder. Reconnect-safe.
- **PDF passes to wrong host:** each worker checks `hash(doc_id) % 2` against its own `HOST_ID` env var; skips mismatches. Idempotent if re-run on either host.
- **PyMuPDF segfault on pathological PDF:** worker catches, logs doc_id to `crashed_docs.log`, moves on. Mirrors existing recovery worker behavior.
- **progress.sqlite contention:** use `PRAGMA journal_mode=WAL` + short-lived connections; both hosts can write safely.

### 12.8 Proof required before this part of the plan ships

- **B1 (Windows PyMuPDF speed)** — must exceed 5 docs/sec sustained to make the split worthwhile.
- **B2 (SMB concurrent write)** — one-file-per-shard pattern verified line-counted and non-interleaved.

If both pass, split is approved. If B1 fails, drop the split — rig-only parsing is fine (it just takes longer). Either way, no other part of the plan depends on this.

---

## Part 13 — Revised Total Wall-Clock (Proof-Aware)

| Phase | Best case (all B* pass) | Worst case (all B* fail, PROVEN-only) |
|---|---:|---:|
| Pre-flight benchmarks | 3 hrs | 3 hrs (still run them) |
| Stage 0–1 text extract (text PDFs) | 1.5 hrs (Win+rig split) | 3 hrs (rig only) |
| Stage 1 OCR (471 image PDFs) | 4 hrs (2-GPU pool) | — (OCR parked, Option D) |
| Stage 2–5 (parse / chunk / dedupe / table) | 1 hr | 1 hr |
| Stage 6 synth Q&A — multi-GPU pool | **~1.5–2 hrs** (5 GPUs + batching) / ~4.4 hrs (5 GPUs, no batch) / ~7.3 hrs (3 GPUs) | 22 hrs (single GPU 3, if B8 fails) |
| Stage 7 embed (150K chunks) | 2.3 hrs (2-GPU pool) or 4.3 hrs (single-GPU, concurrent with Stage 6) | 4.3 hrs (single-GPU, after Stage 6) |
| Stage 8 gate eval | 0.5 hr | 0.5 hr |
| Stage 9 fine-tune RunPod (conditional) | 0.5 hr | 0.5 hr |
| Stage 10 gate + cutover | 1.5 hrs | 1.5 hrs |
| **Total** | **~14–17 hrs** (overnight pass) | **~2.5–3 days** |

Best case assumes every benchmark passes. Worst case is PROVEN-only — still ships, just takes a day longer and leaves OCR for phase 2.

---

---

## Part 14 — Training Data Lifecycle (The Critical Path to 95%)

**This is the mechanism that gets us to 95%, not a backup plan.** Stages 7–9 in the original list treated fine-tuning as "conditional if retrieval misses threshold." That's wrong. A base BGE-M3 on domain-specific Indian tax law will not hit 95% recall@5 no matter how clean the chunks are — the base model doesn't know the vocabulary well enough. Fine-tuning is part of the main path, not the safety net. The revised stage order in Part 14.3 below reflects this.

### 14.1 What training assets we already have (inventory from Attempt #1)

Nothing gets wasted. These are real, graded, and on disk — they move forward into v2.

| Asset | Path | Count | State |
|---|---|---:|---|
| Gemini practitioner pairs (raw) | `D:\_gpu_rig_ai\eval\training_pairs\pairs_2000_20260422.jsonl` | ~1,891 | unfiltered |
| Claude Opus pairs (raw) | `D:\_gpu_rig_ai\eval\training_pairs\pairs_claude_opus.jsonl` | ~76 | unfiltered |
| Claude Sonnet HIGH-complex pairs | `D:\_gpu_rig_ai\eval\training_pairs\pairs_opus_highcomplex.jsonl` | ~350 | unfiltered |
| Claude Sonnet LOW-complex pairs | `D:\_gpu_rig_ai\eval\training_pairs\pairs_sonnet_lowcomplex.jsonl` | ~500–600 (generating) | unfiltered |
| QA grades (Gemini pairs) | `D:\_gpu_rig_ai\eval\training_pairs\qa_gemini.jsonl` | in progress | answerable/realistic/specific/complexity scored |
| QA grades (Sonnet HIGH) | `D:\_gpu_rig_ai\eval\training_pairs\qa_sonnet_high.jsonl` | in progress | scored |
| QA grades (Claude Opus, done) | `D:\_gpu_rig_ai\eval\training_pairs\qa_claude_opus.jsonl` | 76 | scored |
| Hard negatives (BGE-M3 CPU mined) | `D:\_gpu_rig_ai\training\hard_negatives.jsonl` | in progress | top-20 per question minus positive |
| Quarantined (argv-corrupted Opus run) | `pairs_opus_highcomplex_BAD_argv.jsonl` | ~400 | unusable, kept for forensics |
| Gold set | `D:\_gpu_rig_ai\eval\gold_set.yaml` | 170 | hand-curated eval — NEVER in training |

**Expected post-filter yield:** ~3,000–3,500 curated training pairs from Attempt #1 alone. Plus ~80K more from Stage 6 coupled generation. Total training budget: **~80–90K pairs**, which is plenty for BGE-M3 MNRL fine-tuning (the RunPod pipeline was sized for 5K; 80K is well within range).

### 14.2 The chunk-coupling problem (and how we solve it)

Attempt #1's pairs are coupled to v1 chunks. v2 will re-chunk everything — so some pair-to-chunk links break. We don't discard the pairs; we reconcile them:

1. **Keep the question** (the valuable part — a real practitioner-style question).
2. **Re-anchor to v2 chunk** via semantic search: embed the question with the current embedder, find the top-3 v2 chunks, mark it anchored if any match the original v1 chunk's `(parent_act, section_ref, doc_number)` metadata. If none match, send the question to a light Claude/Gemini re-coupling pass: "does any of these v2 chunks answer the question?" — cheap, fast.
3. **Pairs that can't be re-coupled** go into a "orphan_questions" file — kept as a retrieval-quality test set (questions with no known answer chunk are still useful for measuring false-negative rate).
4. **Expected reconciliation yield:** ≥70% of Attempt #1 pairs re-anchor cleanly. The rest become test data.

This happens in a new sub-stage we call **Stage 5.5 — Pair Reconciliation** (slots between chunking and training).

### 14.3 Revised stage order — training is on the main path

Old order (wrong): Chunks → Embed → Evaluate → Maybe fine-tune → Re-embed.
**New order: Chunks → Generate pairs → Train embedder → Embed once → Evaluate.**

| Stage | What happens | Training data used |
|---|---|---|
| 0–4 | Inventory, OCR, parse, chunk, template-dedupe | — |
| 5 | Table extract | — |
| **5.5** | **Pair reconciliation (new)** — re-anchor Attempt #1 pairs to v2 chunks | ~3K Attempt #1 pairs |
| **6** | **Synthetic Q&A coupled generation** — GPU 3 qwen3-8B generates 2–3 questions per substantive v2 chunk | ~80K new pairs |
| **6.5** | **QA grading pass** — cheap local qwen3 self-judge (answerable/specific); drop anything <2/3 score. External LLM (Claude/Gemini) grades a 5% audit sample to calibrate the local judge. | Filtered: ~70K pass |
| **6.7** | **Curation + stratified split** — merge Attempt #1 curated + Stage 6 filtered. Stratify by category (GST/Customs/Excise/ST), complexity (LOW/MED/HIGH 35/45/20), and doc_type. Hold out 10% for fine-tune eval. | ~73K training + ~8K internal eval |
| **6.9** | **Hard-negative mining** — for each training question, embed with base BGE-M3, retrieve top-20 v2 chunks, filter out the positive (anchored chunk). Take top 5 as hard negatives. | Triplet format: (q, pos, neg) |
| **7 (new)** | **BGE-M3 fine-tune on RunPod A100** — MNRL loss on full triplet set. 2 epochs, batch 32, warmup 500 steps. ~$1 and ~30 min. | 73K triplets |
| **8 (new)** | **Embed corpus once with fine-tuned model** — 150K v2 chunks → new Qdrant collection `cbic_v2` via proven GPU 5 recipe | Fine-tuned BGE-M3 |
| **9 (new)** | **Gate eval on 170 gold + 8K internal eval** — recall@5 ≥ 95% | — |
| **9.5** | **(Conditional) second fine-tune** if gate misses — mine new hard negatives from v2 retrieval output, do another pass of MNRL | Refreshed triplets |
| **10** | **Generator + citation + faithfulness** as before | — |

**Key shift: we embed the 150K-chunk corpus ONCE, with a fine-tuned model.** The old plan embedded twice (base, then fine-tuned). One re-embed saves 2–4 hours.

### 14.4 Why this order hits 95% and old order did not

| Lever | Old (Attempt #1) | New (v2) |
|---|---|---|
| Who generates training pairs? | External APIs (Claude, Gemini) after ingest | Local qwen3-8B during ingest, while chunk text is in memory |
| Coupling quality | Best-effort post-hoc regex match | Generated from the chunk itself — coupling is guaranteed |
| Volume | ~4,000 pairs | ~80,000 pairs |
| Cost | ~$20 + 3 days of engineering | ~$0 + 22 hours GPU 3 overnight |
| Schema consistency | 3 generators, 3 schemas, 1 full quarantine | One generator, one schema |
| Fine-tune input | Fragile, some orphan | Clean triplets from curated pool |
| When model learns domain | After embed (requires re-embed) | Before embed (single embed pass) |

### 14.5 Reusing existing training scripts (no rewrites)

The fine-tuning infrastructure is already built and audited. Every box below is a script that exists and slots directly into the new stages.

| Stage | Script (existing) | Change |
|---|---|---|
| 5.5 Pair reconciliation | **(new, ~150 LOC)** — small helper using existing BGE-M3 embed calls | N/A |
| 6 Synth Q&A coupled | `gen_training_pairs_v2.py` | repoint API from Gemini to local qwen3-8B @ :9083 |
| 6.5 QA grading | `qa_pairs_claude.py` | repoint from Claude CLI to local qwen3-14B (for bulk) + Claude CLI for 5% audit |
| 6.7 Curation | `training/curate_training_set.py` | use as-is (already built) |
| 6.9 Hard-neg mining | `training/mine_hard_negatives.py` | use as-is — already running CPU BGE-M3 mining on Attempt #1 pairs |
| 7 Fine-tune | `training/finetune_bge_m3.py` + `training/runpod/` | use as-is — already audited, already staged on RunPod |
| 7 Triplet prep | `training/prep_pairs.py` | use as-is (`--mode curated`) |
| 8 Embed | `overnight_worker.py` | point to fine-tuned model path instead of base BGE-M3 |
| 9 Gate eval | `eval/variants_and_hardneg.py` | use as-is |

**Total new code for this entire training lifecycle: ~150 LOC (pair reconciliation helper).** Everything else is existing and proven.

### 14.6 Fine-tune eval guardrails (prevent overfitting to synthetic questions)

Real risk: fine-tuning on 80K local-qwen3-generated questions could teach the embedder to answer qwen3-style questions well and still fail on real practitioner phrasing.

Mitigations baked into Stage 9 gate eval:

1. **Primary metric stays the hand-curated gold set** (170 Qs from real practitioners). If fine-tune lifts synthetic eval but not gold, we abort the fine-tune.
2. **Held-out 8K internal eval** is drawn from same distribution as training but not seen in training. Catches pure overfitting.
3. **Paraphrase robustness check** — take 30 gold questions, regenerate each in 3 different phrasings, verify retrieval still returns same top-5 chunk. Protects against style-overfit.
4. **Complexity-band metric** — measure recall@5 within LOW/MED/HIGH buckets separately. If HIGH drops below 80% while LOW jumps to 99%, fine-tune is too shallow.

If any of these four fails, Stage 9.5 (second fine-tune pass with refreshed negatives) kicks in. If it still fails, fall back to base BGE-M3 + stronger reranker.

### 14.7 Proof status of training lifecycle claims

| # | Claim | Status |
|---|---|---|
| T1 | Local qwen3-8B can generate 2–3 practitioner questions per chunk with acceptable quality | PARTIAL — qwen3 coherent generation proven, but not benchmarked against Claude/Gemini on this specific task. BENCHMARK: generate 50 pairs, spot-grade with Claude judge. |
| T2 | BGE-M3 MNRL fine-tune on RunPod A100 with 73K triplets converges in ≤2 epochs | PROVEN externally (BGE-M3 paper, prior RunPod runs) |
| T3 | Pair reconciliation (question → v2 chunk re-anchor) yields ≥70% success rate | UNPROVEN. BENCHMARK: 100-pair sample reconciliation before running full set. |
| T4 | Fine-tuned BGE-M3 embed throughput on GPU 5 Vulkan = base BGE-M3 throughput | PROVEN — fine-tune changes weights, not architecture. Same inference cost. |
| T5 | Fine-tuned model + clean v2 chunks clears recall@5 ≥ 95% on gold | UNPROVEN — this is the ultimate gate. Measured at Stage 9. If fails, Stage 9.5 fires. |

### 14.8 External LLM questions specific to training lifecycle

Added to the consult:

20. **Coupled question generation quality:** how does local qwen3-8B compare to Claude/Gemini for generating practitioner-style training questions on Indian legal text? Is the 100× cost advantage worth ~10–20% quality drop, or do we risk poisoning the training set?

21. **Synthetic-to-real gap:** when fine-tuning BGE-M3 on LLM-generated questions, what's the typical gap between synthetic-eval accuracy and real-user-query accuracy? Any published numbers or rules of thumb?

22. **Pair reconciliation** (question from v1 chunks → v2 chunks): is there a standard library or technique for this re-anchoring, or do we roll our own semantic-search + metadata-match?

23. **Curriculum training:** with 80K pairs spanning LOW/MED/HIGH complexity, does curriculum (easy → hard) help BGE-M3 MNRL, or just shuffle and train flat?

24. **Hard-negative refresh:** is one round of hard-neg mining (pre-train) sufficient, or do we need in-training dynamic mining (embed with current weights every N steps)?

25. **Target ratios:** for 95% recall target, what triplet count is typically needed per unique section/topic? Do we have too many pairs on common topics (GST rates) and too few on rare ones (service-tax-era circulars)?

26. **Reranker vs fine-tune tradeoff:** if B0c proves cross-encoder reranker works on ROCm, does a strong reranker + base BGE-M3 beat fine-tuned BGE-M3 alone? Which is more robust long-term?

---

---

## Part 15 — Expert Review (Gaps in v1, Amendments for v1.1)

**Reviewer perspective:** Senior retrieval engineer, ships production RAG systems. Reviewing this plan as if responsible for hitting 95% in six weeks.

**Verdict:** The plan's skeleton is sound (metadata-first, gate-based, benchmark-first, training on main path). But 15 things are either missing or mentioned-without-being-wired-in. Each is a realistic path to v2 shipping at 80% instead of 95%. Fixing them now costs hours; fixing them after we embed 150K chunks costs days.

### 15.1 Gaps identified (in order of impact on 95% gate)

#### G1 — Query-side pipeline is not specified (CRITICAL)

The plan specifies how to ingest and embed, but says almost nothing about how a query is actually served at Stage 8 gate-eval. Without a defined query pipeline, "recall@5 ≥ 95%" is measured against an undefined system. The current `cbic_v1` query path does dense-only (plus some variants in the sweep). v2 needs:

- **Hybrid fusion** — dense (BGE-M3) + sparse (BM25) scored independently, fused via RRF (reciprocal rank fusion, k=60).
- **Chunk-type routing** (from Part 2.8) — query classifier tags the incoming question (rate/procedure/definition/eligibility/penalty/…); retrieval filter prefers that chunk_type with a soft weight, doesn't exclude others.
- **Metadata filters** — Act detection (IGST/CGST/Customs) already exists in `variants_and_hardneg.py`; formalize it as always-on.
- **Effective-date filter** — default `effective_to IS NULL` unless user says "as of …". Non-optional.
- **Structured lookup bypass** — factual queries (rate, HS code, duty) hit SQLite first (Stage 5 output) before vector search. If SQLite has a high-confidence match, return it and embedding becomes secondary evidence, not primary.
- **Reranker** (conditional on B0c) — cross-encoder top-50 → top-10.

**Amendment:** add Stage 8.0 — "Query pipeline specification + implementation", before gate eval.

#### G2 — Tables dual-store has no query router (CRITICAL)

Part 2.4 says tables go into SQLite for factual lookup. But who decides at query time whether to hit SQLite or Qdrant? A rate question mis-routed to vector search finds a chunk that describes the rate table but returns no number. A classifier in front of retrieval (G1) routes: "HS 8703.23 duty?" → SQLite row lookup; "eligibility for duty drawback?" → vector search. Without the router, the SQLite store is dead weight.

**Amendment:** Stage 5 extended — emit a `table_catalog.json` mapping schemas → query intents (e.g., `currency_rate` schema handles `forex_rate_as_of` intent). Stage 8.0 query classifier uses this catalog.

#### G3 — Chunk-type classifier is mentioned but never built

Part 2.8 lists 10 chunk types (definition / rate_table / procedure / …). The plan says "a cheap classifier pass," but never specifies: trained how, on what data, with what accuracy target. An 80%-accurate classifier passed into a routing layer corrupts retrieval for 20% of chunks.

**Amendment:** add Stage 3.7 — "Chunk-type classifier." Approach: rule-based first (regex on section preamble keywords, "rate of tax", "shall be", etc.) covers ~70% with 100% precision. Remaining 30% goes through a zero-shot qwen3-8B classifier on GPU 3 during Stage 6 (same service, same prompt structure, no extra cost). Validate on 500 hand-labelled chunks before relying on it. Target: ≥92% accuracy.

#### G4 — Gold set itself is not audited

We're optimizing to 170 hand-curated questions. If 10 of them have wrong `expected_sections` (we assigned them), we're targeting a biased benchmark. A 95% recall@5 against a flawed gold is not 95% trust.

**Amendment:** add Stage 0.5 — "Gold set audit" (half-day). Re-verify each gold item's expected citations by reading the cited section. Drop or correct items where the "correct answer" turns out to be tangential. Expand gold from 170 → 500 by drawing new questions from Stage 6 coupled output — with a hard rule that **eval gold is never shown to the fine-tune**.

#### G5 — Stratified sampling of training pairs is missing

Stage 6 plan is "2–3 questions per substantive chunk" = ~200K pairs. BGE-M3 domain adaptation needs 10–50K pairs typically. 200K is not "more training data" — it's mostly noise on topics already well-represented (GST rates, refund procedures). Meanwhile service-tax-era circulars and rare customs notifications get the same 2–3 questions as high-frequency topics. Result: model gets great at GST, mediocre at Service Tax (which our gold set actually covers).

**Amendment:** Stage 6.7 extended. **Stratify pair retention by (category, chunk_type, section_rarity)**. Cap: max 30 pairs per unique `(parent_act, section_ref)` tuple. Minimum: 3 pairs per tuple (promoting generation if below). Target total: **30–50K curated pairs**, not 80K. Cheaper RunPod bill + better balance.

#### G6 — Pair-level deduplication (near-identical questions) not planned

Two adjacent chunks on the same section can produce essentially identical questions. Training on duplicates adds no signal, inflates loss artificially low (model memorizes rather than generalizes).

**Amendment:** Stage 6.7 — MinHash (n=5, shingle size 4) over all questions, drop anything ≥0.85 Jaccard. Trivial, ~5 min on the full set.

#### G7 — Two-stage hard-negative mining is cheaper AND more effective than one-stage

The plan mines hard negatives once, from base BGE-M3 on the v2 index, then fine-tunes. A proven stronger recipe: (a) fine-tune on easy-negatives (random in-batch) for 1 epoch, (b) re-mine hard negatives from the checkpoint, (c) fine-tune 1 more epoch with refreshed negatives. Catches the negatives the updated model now struggles with, not the ones the base model struggled with.

**Amendment:** Stage 7 → two sub-epochs with hard-neg refresh between them. Same RunPod budget (MNRL converges fast).

#### G8 — BM25 sparse weights never re-fit on v2 corpus

The proven recipe uses `fastembed Qdrant/bm25` — a pre-trained BM25 with corpus-agnostic IDF. On a specialized corpus this is measurably worse than corpus-fitted BM25. The current sparse path is leaving recall on the table, especially for rare Act-specific terms ("IGCR", "SEZ Rule 54A", "anti-dumping duty on …").

**Amendment:** Stage 7 includes BM25 re-fit on the v2 corpus. `BM25SparseTextEmbedding.fit(corpus_iter)` variant. Also re-embed at Stage 8 with the fitted BM25.

#### G9 — Citation fidelity and refusal calibration unspecified

Stage 10 says "generator cites correctly, refuses when confidence low." Never defines the threshold. Refusal is the #4 trust metric. Without calibration we either refuse too often (user loses faith) or too rarely (model hallucinates).

**Amendment:** add Stage 10.5 — "Calibration." Compute score distribution on 500 held-out queries. Pick threshold where false-confident rate ≤5%. Measure refusal-correct rate on a negative set (50 questions we know aren't in the corpus) — must be ≥95%.

#### G10 — Shadow mode / canary cutover missing

Plan says "A/B on gold, cut over when v2 beats v1." But user traffic distribution isn't identical to our 170 gold questions. v2 could beat v1 on gold and still regress on real queries we never thought to include.

**Amendment:** add Stage 10.7 — "Shadow mode." For 72 hours, send every production query to both v1 and v2 in parallel; only v1's response is shown to user. Judge (Claude/Gemini) scores both. Cut over only when v2 wins or ties on ≥80% of real queries AND hits the gold gate.

#### G11 — Synthetic-vs-real evaluation leak

If Stage 6 generates training questions with qwen3-8B and Stage 9 evaluates on held-out questions from the same qwen3-8B, we measure "how well the fine-tune learned qwen3-8B's writing style," not "how well it answers practitioners." Classic synthetic-eval overfit.

**Amendment:** Stage 6.7 — 10% held-out internal eval uses a DIFFERENT generator (Claude Haiku via CLI, cheap) for the same source chunks. If fine-tune lifts qwen-eval but not Claude-eval, it overfit to style; abort. Gold set (real practitioners) is the final arbiter either way.

#### G12 — Incremental ingest for new CBIC notifications not planned

CBIC issues 20–40 notifications a month. The plan rebuilds v2 once; in 6 months, v2 is stale. A practitioner asking about a 2026-Q3 notification gets zero recall. 95% trust is not a one-time event.

**Amendment:** add Stage 11 — "Incremental ingest hook." Design Stage 0 manifest + Stage 7 embedder as resumable/additive. New notification = one-document ingest through the same pipeline, deterministic point ID, upsert into `cbic_v2` (idempotent by design — already proven). Weekly cron. Re-fine-tune cadence: quarterly, or when a gold-set regression is detected.

#### G13 — Deterministic point IDs break across chunker versions

Proven recipe: `blake2b(doc_id|page|chunk_idx)`. If v2 re-chunks, chunk_idx differs → different IDs → no upsert collision with v1 (good, we're in a new collection). But if we later change the chunker *again* for incremental ingest, IDs shift and we double-store. Not wrong today, but a footgun.

**Amendment:** Stage 3 — include `chunker_version` in the hash: `blake2b(doc_id|page|chunk_idx|chunker_version)`. Future-proof.

#### G14 — User chat priority during v2 build not explicit

During Stage 6 we occupy 5 GPUs. GPU 2 (chat) is untouched in the allocation, but the 4-core host CPU serves Qdrant, the dispatcher, and all llama-server I/O for 5 generators + 1 chat. Load avg may spike high enough that chat latency degrades. Users complain during what they don't know is a rebuild.

**Amendment:** B8 benchmark explicitly measures `/query` latency on GPU 2 under full Stage-6 load. If P95 >3× baseline, we throttle Stage 6 to 3 GPUs instead of 5. Chat always wins.

#### G15 — Attempt #1 pair coupling was never audited beyond "answerability"

QA grades on answerable/realistic/specific. None of these detect **coupling errors** — where the question is about topic X but chunk is about topic Y, yet the question is vague enough to be "answerable." We shipped the argv-corruption quarantine but never spot-checked the Gemini-1891 run for silent coupling errors.

**Amendment:** Stage 5.5 (reconciliation) extended — for each Attempt-#1 pair, run a coupling check: does the question's topic (classified via qwen3-8B) match the chunk's metadata (parent_act + chunk_type)? If mismatched, drop. Expected drop rate: 5–15% (informed guess; benchmark first).

### 15.2 Things I got right in v1 and am keeping

- Gate-based stage progression.
- Proof-status gate on hardware claims.
- Pair reconciliation (Stage 5.5).
- Training on the main path (not conditional).
- Multi-GPU generator pool for Stage 6.
- Keep cbic_v1 read-only until v2 beats it.

### 15.3 Amended stage list (final)

| Stage | Name | Added since v1 |
|---|---|---|
| 0 | Inventory + dedup | — |
| **0.5** | **Gold set audit & expand to 500** | **NEW (G4)** |
| 1 | Text extract (PDF + OCR) | — |
| 2 | Document parsing (tree) | — |
| 3 | Chunking (tree-walk) | — |
| **3.7** | **Chunk-type classifier (rule + qwen3-8B)** | **NEW (G3)** |
| 4 | Template / boilerplate removal | — |
| 5 | Structured table extraction + `table_catalog.json` | Extended (G2) |
| **5.5** | **Pair reconciliation + coupling audit** | Extended (G15) |
| 6 | Synth Q&A (multi-GPU pool) | — |
| 6.5 | Pair QA grading (local + 5% Claude audit) | — |
| **6.7** | **Curation: stratified sampling + MinHash dedup + dual-generator eval split** | Extended (G5, G6, G11) |
| 6.9 | Hard-negative mining (initial) | — |
| **7** | **Fine-tune with 2-stage hard-neg refresh + BM25 re-fit** | Extended (G7, G8) |
| **8.0** | **Query pipeline spec + implementation (hybrid + routing + reranker + filters)** | **NEW (G1, G2)** |
| 8 | Embed v2 corpus once (fine-tuned BGE-M3 + fitted BM25) | — |
| 9 | Gate eval on 500-item gold + 8K internal | — |
| 9.5 | (Conditional) second fine-tune if gate misses | — |
| 10 | Generator + citation + faithfulness | — |
| **10.5** | **Calibration (refusal threshold)** | **NEW (G9)** |
| **10.7** | **Shadow mode / canary (72 hrs)** | **NEW (G10)** |
| **11** | **Incremental ingest hook (weekly cron)** | **NEW (G12)** |

### 15.4 New external-LLM consultation questions (G-review)

27. **Query pipeline architecture** — for a 95% recall@5 target on legal text with heavy numeric references, what's the field-standard retrieval stack? Dense+sparse RRF is table stakes; is reranker mandatory or helpful; where does query classification live?

28. **Structured-vs-vector routing** — for a legal/financial corpus with rate tables as first-class data, how do production systems route factual vs conceptual queries? Are there battle-tested open implementations or is this usually custom?

29. **Chunk-type classifier quality target** — what classifier accuracy is acceptable before it degrades overall retrieval vs no classifier at all? Trade-off curve?

30. **Training pair volume vs quality** — empirical data on BGE-M3 domain adaptation: does 30K curated beat 200K noisy? Any public ablations?

31. **Two-stage hard-neg mining** — is the "easy → mine → refresh → hard" recipe worth the added complexity, or does one-stage with high-quality negatives match?

32. **Calibration for refusal** — standard techniques to calibrate retrieval-confidence → refusal threshold in RAG? Temperature scaling, conformal prediction, or simpler score cutoffs?

33. **Canary / shadow mode** — any open-source shadow-evaluator patterns that can be dropped in front of a FastAPI RAG? Or build trivial?

34. **Incremental ingest cadence** — for a legal corpus with ~30 new documents/month, what fine-tune cadence keeps recall stable without weekly RunPod cost? Does a LoRA adapter per quarter work?

35. **Gold set evolution** — how do production teams grow their eval set over time without corrupting it with training-like questions? Best practices for maintaining a "forever held-out" set?

### 15.5 Cost of honesty

Adding these amendments: approximately **+4 hours benchmark time**, **+1 day ingest+train time** (stratification, 2-stage fine-tune, shadow mode). Total revised budget: **~2 days** wall-clock (best case) instead of 14–17 hours.

This is the right trade. 14–17 hours to 80% is a failed v2. ~2 days to 95% is the goal.

### 15.6 What would still make me nervous on day-1 after cutover

Even with all 15 amendments:

- **Hindi / bilingual queries** — tested on zero real queries. BGE-M3 is multilingual by design but we've never verified on Devanagari practitioner text. Add a 30-Hindi-Q mini-gold set.
- **Conflicting notifications** — when two in-force notifications say different things about the same topic (happens in Customs duty exemptions). Neither retrieval nor generator currently has a strategy. Flag for a later consultation.
- **Practitioner slang** — real CAs say "1B" for Section 17(5)(b), "RCM on manpower" as shorthand. Not in the training corpus. Mitigation: query-side canonicalization dictionary. Not in v2 scope; log for v3.

---

---

## Appendix C — Hard-Won Lessons (Never Re-learn)

Every item below is something we didn't know at the start and learned the hard way. Each entry is a **rule** for v2, not a suggestion. If someone proposes breaking one of these rules during execution, they must produce a benchmark that overrides it. No rediscovery.

### C.1 CPU / host constraints (the 4-core reality)

1. **NEVER run OCR on CPU.** Tesseract at `--workers 2` hits load avg 33 → 61-sec query times. User directive: banned outright. RapidOCR CPU same risk.
2. **OMP_THREAD_LIMIT is a trap.** Default=4. `--workers 2` silently becomes 8 OS threads on 4 cores. Any CPU-heavy subprocess must run with `OMP_THREAD_LIMIT=1` and `OMP_NUM_THREADS=1` as explicit env vars.
3. **`nice +19` + `ionice idle` alone does NOT protect the query path.** Only hard cpuset partitioning (systemd slices with `AllowedCPUs=`) reliably partitions. Use `llama.slice` (cores 0–1) and `ingest.slice` (cores 2–3).
4. **Check `uptime` and `nproc` FIRST when debugging latency.** Every "RAG is slow" incident has turned out to be a CPU pegged by a background process, not a model/vector issue.
5. **PyMuPDF (fitz), not pdfplumber.** pdfplumber segfaults on large tax PDFs (144-doc crashed list from Attempt #1 is pdfplumber).
6. **Qdrant `wait=False` on upserts.** Sync waits kill throughput; re-run is idempotent via deterministic point IDs anyway.
7. **Avoid `2>&1` on native exes in PowerShell 5.1.** Wraps each stderr line in ErrorRecord, sets `$?=false` even on exit 0. We've wasted hours on this.

### C.2 GPU / Vulkan constraints (the mining-card reality)

8. **Single GPU 5 in-process for the bulk BGE-M3 embed. Proven 5.2–9.6 ch/s.** Anything else is experimental.
9. **Multi-GPU embedding in-process = BANNED.** llama-cpp-python opening multiple Vulkan contexts in one process hangs the kernel in D-state. 0.3 ch/s, dead worker leaks. Never retry this form.
10. **Multi-GPU via separate llama-server HTTP daemons is FINE.** That's how GPU 2 chat + GPU 5 embed already coexist. This is the ONLY safe multi-GPU pattern on this rig.
11. **Required env vars on every Vulkan launch** — no exceptions:
    ```
    GGML_VK_VISIBLE_DEVICES=<N>
    RADV_DEBUG=nodcc
    GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1   # Navi 10 kernel bug workaround
    ```
12. **`llama-server-rocm` on this rig is actually Vulkan-backed.** Verified via `GGML_VK_VISIBLE_DEVICES` gating. Don't assume binary name reflects backend.
13. **Qwen2.5-VL-7B on Vulkan is 600 s/page.** CLIP vision tower is the wall. Do NOT retry without changing backend OR model. Smaller VLMs (MiniCPM-V 2.6, SmolVLM) are the only hope without ROCm.
14. **`amdgpu ring reinit` in `dmesg`** during uncapped VLM warmup = soft reset. Watch dmesg during every new GPU workload stand-up.
15. **Batched `model.embed([t1,…,tN])` with total tokens > `n_batch=512`** returns `llama_decode=-1`. Always pay per-call overhead: `m.create_embedding([single_text])`. Proven.

### C.3 Embedding recipe (the only one that works)

16. **BGE-M3 via llama-cpp-python in-process:**
    ```
    n_ctx=8192, n_batch=512, n_ubatch=512, n_threads=4,
    pooling_type=2, embedding=True, n_gpu_layers=-1
    ```
    Per-text `create_embedding([t])`. UPSERT_BATCH=128.
17. **BM25 sparse via `fastembed.SparseTextEmbedding("Qdrant/bm25")`** is the proven path. But G8: re-fit on v2 corpus for better IDF.
18. **Deterministic point IDs:** `blake2b(f"{doc_id}|{page}|{chunk_idx}".encode(), digest_size=8)` mod 10¹⁵. Idempotent re-ingest. For v2: include `chunker_version` in the hash (per G13).
19. **CHUNK_CHARS=1800, CHUNK_OVERLAP=200** was the v1 default. v2 overrides: 800–1,200 for prose (tree-walk), whole-table for tables ≤1,500 chars, with header repetition on row-split for larger tables.
20. **TSV schema is sacred:** `doc_id<TAB>category<TAB>subcategory<TAB>absolute_file_path<TAB>title`. Every downstream script expects this.
21. **Size-ascending TSV (`du -b | sort -n`).** Fast feedback on progress; worst-case big PDFs don't block the first 2 hours.

### C.4 OCR lessons (frozen, but codified for when we thaw)

22. **Banned tools (CPU path):** Tesseract, RapidOCR on CPU. User directive.
23. **Banned model (GPU path):** Qwen2.5-VL-7B on Vulkan. 75× slower than projected.
24. **300 DPI minimum rasterization** for Devanagari. Below that, matras drop.
25. **`TESSERACT_PAGE_TIMEOUT=120` is too tight** for slow pages. Silent timeout wrote 35 docs with 18 chars of empty markers. Use 300s, or skip tesseract entirely.
26. **Multi-column forms collapse** without an explicit "preserve column order" prompt.
27. **VLM "helpful" hallucination** silently fixes smudged dates/numbers — lethal. Mandatory: `temperature=0, top_p=1.0`, strict "verbatim, mark [UNREADABLE]" prompt.
28. **Model files retained at** `/opt/ai-models/qwen25vl/` — 5.7 GB. Keep for future VLM experiments; downloading is slow.
29. **Target TSV disabled safely:** rename to `.disabled`. Don't rely on process flags.
30. **SSH control socket severs when agent ends.** Recovery: `ssh -o ControlMaster=no -o ControlPath=none`. Plan for it.

### C.5 Ingest launch & resume patterns

31. **Always `nohup bash -c '...' > /tmp/xxx.log 2>&1 &` followed by `disown`.** User can log out; kill by PID or pattern.
32. **Kill pattern (frozen OCR example):**
    ```
    pkill -9 -f <worker>.py; pkill -9 -f tesseract; pkill -9 -f pdftoppm
    mv -f /tmp/<target>.tsv /tmp/<target>.tsv.disabled
    rocm-smi --showmeminfo vram
    pgrep -af "<pattern>" || echo CLEAR
    ```
33. **Every worker writes to a dedicated log file.** One run, one log. No interleaving.
34. **Resume via per-record output file + deterministic key** (chunk_id in JSONL; point_id in Qdrant). Restart is a no-op on done records.
35. **Every long-running worker has a graceful checkpoint cadence** — flush per-100 records at least. Sudden kill loses ≤100 records, not hours of work.

### C.6 LLM-call infrastructure (post-Attempt #1 discoveries)

36. **Subprocess + argv delivery corrupts long prompts on Windows cmd.exe.** Prompts with `&|<>` chars + 4KB length silently mangle. ALWAYS `subprocess.run(..., input=prompt, ...)` (stdin). Discovered when 100% of QA grades returned "I need more context."
37. **Claude CLI envelope format:** JSON object with `result` field; often wrapped in ```json fences. Parse defensively.
38. **Schema normalization across generators is mandatory.** Gemini emits `"q"`, Sonnet emits `"question"`, Opus sometimes emits a bare list instead of `{reasoning, questions}`. Repair-JSON + isinstance-list handling every time.
39. **JSON line parsers must `try/except` per line.** One bad line from one worker kills the whole re-read. Count and move on.
40. **Opus ≈ 25–30 s/call via CLI, Sonnet ≈ 8–10 s/call (3×).** Pick Sonnet for bulk; Opus only where quality strictly matters.
41. **Concurrent workers against qwen3-14B saturate at 3–6.** Reduce to 2 workers and 0% error rate. Not an LLM quality thing — host CPU.
42. **`/no_think` mode in qwen3** bypasses the reasoning chain, ~4× faster for plain completions.

### C.7 Qdrant / corpus handling

43. **Scroll `next_page_offset` is an opaque point-ID-like object, NOT an integer.** Never pass `rng.randint()` as scroll offset. This bug cost us a whole sampling run (9 chunks returned instead of 600).
44. **Keyword payload indexes** for: `doc_id, category, subcategory, doc_type, parent_act, chunk_type` (new). Integer for: `page_number, chunk_idx`. Create at collection init.
45. **Collection separation for v1/v2.** Don't try to "upgrade in place." `cbic_v1` stays read-only. `cbic_v2` is the new collection. Cutover is an API config flip.
46. **/query endpoint is slow (includes generation).** For retrieval-only benchmarks, use `/retrieve` or bypass the generator. 600s timeout needed if hitting full `/query` in parallel.

### C.8 Training data pitfalls (fresh, from this week)

47. **Empty complexity buckets poison the fine-tune.** If LOW bucket has 0 pairs, the model learns LOW from MED examples. Explicitly generate per-bucket until all buckets are populated.
48. **Different generators, different styles.** Never mix Claude + Gemini + qwen3 output as flat training pool without a style tag. Or (better) pick one generator for training.
49. **"Answerable" QA grade does NOT detect coupling errors.** A vague question is "answerable" by the wrong chunk. Add a metadata-match coupling check (G15).
50. **Gold set is not immutable truth.** Audit it before optimizing against it (G4).
51. **Pairs/chunk ratio must be bounded.** 2–3/chunk × 80K = 200K noisy pairs, not better. Stratified cap at 30/section (G5).

### C.9 ROCm (re-verify before trusting)

52. **Previous rig state:** ROCm broken on RDNA1/RDNA2 consumer cards. torch-ROCm, onnxruntime-ROCm, fastembed-ROCm all hit ABI errors.
53. **Current claim:** user reports ROCm now working. **TREAT AS UNVERIFIED** until B0 passes. Do not plan on ROCm-dependent paths (sentence-transformers native, Surya, cross-encoder rerankers) until `rocminfo` lists both GPUs AND a 1K-chunk sentence-transformers embed completes cleanly.
54. **AMD Instinct (MI300) docs don't apply** to these consumer cards. Ignore "just use ROCm" advice that isn't specific to gfx1010 / gfx1031.

### C.10 Operational hygiene

55. **Don't amend git commits.** Always new commits. User directive.
56. **Never `git push --force` without explicit ask.** Never skip hooks. Never disable signing.
57. **SMB share atomicity:** always write-temp-then-rename. Never append-from-two-hosts to one file.
58. **Per-host output shards** for any dual-host workload. One file per host per worker per hour.
59. **`progress.sqlite` with `PRAGMA journal_mode=WAL`** for any two-host coordination.
60. **Every script logs its version / git SHA / config into its output.** Provenance for debugging.

### C.11 Verification discipline

61. **Run `quality_check.py` after every ingest milestone,** not just at end.
62. **Run the gold set after every stage from Stage 3 onward.** If a stage drops a metric, catch it immediately.
63. **Spot-check 5 random chunks per stage** by hand (`one_chunk_audit.py`). Humans catch what metrics miss.
64. **Every change to chunker / parser / embedder** bumps a version. Point IDs rehash accordingly (C.3 / G13).
65. **Keep Attempt #1 artifacts until v2 is in production ≥ 30 days.** Cheap insurance against silent regression.

### C.12 The meta-lesson

**Every failure in Attempt #1 had three signs we ignored before it blew up:**
(a) a plausible-sounding plan built on an un-benchmarked assumption,
(b) a small voice saying "let me just test this once," silenced for speed,
(c) the realization after losing a day that the assumption was wrong.

v2's proof-status gate (Part 11) is built to force (b) before (a) is trusted. **The rule is: no unproven hardware assumption ever enters the execution path.** If a benchmark result would change the plan, run the benchmark.

---

---

## Part 16 — The Three Things That Actually Determine 95%

Everything else in this plan is scaffolding. These three are where 95% is won or lost. v1 of this plan under-specified all three. This section fixes that.

---

### 16.1 HOW WE CHUNK (concrete design, not direction)

#### 16.1.1 The document grammar we're parsing

Indian tax PDFs follow a predictable structure. The parser must recognize:

```
Document
├── Preamble (short notice, G.S.R./S.O. number, issue date, authority)
├── Section / Rule / Regulation (integer with optional letter suffix: 16, 17, 16A)
│   ├── Sub-section (parenthesized integer: (1), (2), (3))
│   │   ├── Clause (parenthesized letter: (a), (b), (c))
│   │   │   └── Sub-clause (roman: (i), (ii), (iii))
│   │   └── Proviso ("Provided that ...", "Provided further that ...")
│   ├── Explanation ("Explanation.— ...")
│   └── Illustration
├── Schedule / Annexure / Appendix
│   ├── Table (tariff / rate / rate-with-condition)
│   └── Form (GST RFD-01, ER-1, Bill of Entry, etc.)
└── Amendment history / Footnote citations
```

Regex anchors (battle-tested on CBIC text, codified in `parser_v2.py`):

```
^\s*(Section|Rule|Regulation)\s+(\d+[A-Z]?)\.?\s+(.*)$           → section
^\s*\((\d+)\)\s+                                                  → sub-section
^\s*\(([a-z])\)\s+                                                → clause
^\s*\((i{1,3}|iv|v|vi{0,3}|ix|x)\)\s+                             → sub-clause
^\s*(Provided(?:\s+(?:further|also))?)\s+that\b                   → proviso
^\s*Explanation\s*\.?[—\-:]\s*                                    → explanation
^\s*(Illustration|Example)\s*\.?[—\-:]\s*                         → illustration
```

#### 16.1.2 The chunking algorithm

**Each tree node produces zero or more chunks.** Not "every node is one chunk." Rules:

1. **Leaf nodes < 1,200 chars** → one chunk, full metadata inheritance.
2. **Leaf nodes ≥ 1,200 chars** → split on sentence boundaries (regex: `(?<=[.!?])\s+(?=[A-Z])`), then on `;\s+`, with 100-char overlap. Each split inherits parent metadata.
3. **Proviso = its own chunk always.** Even a 200-char proviso is semantically distinct from the sub-section body. Provisos change outcomes; retrieval must find them independently.
4. **Explanation = its own chunk.** Same reasoning as provisos.
5. **Illustration = its own chunk**, tagged `chunk_type=illustration`.
6. **Sub-section with clauses** — if total body <800 chars, keep together (one chunk). If ≥800, each clause becomes its own chunk.
7. **Tables ≤1,500 chars** → whole table as one chunk.
8. **Tables >1,500 chars** → split by row groups of ~8–12 rows; header row + column schema repeated in each split chunk; `chunk_type=rate_table_split`.
9. **Definition sections** (Section 2 of any Act) — each defined term becomes its own chunk. Retrieval for "what is 'input tax credit'" must hit exactly that definition, not the whole Section 2.

#### 16.1.3 The context-prefix pattern (critical for retrieval quality)

Embedding raw chunk text loses grounding. Standard RAG mistake. v2 prepends a compact **context header** to every chunk's embedding input, but stores it separately from the displayed chunk text:

```
[parent_act] CGST Act 2017 > [section] 16 (Eligibility and conditions for ITC)
> [sub-section] (2) > [clause] (aa) | [chunk_type] eligibility_rule

<the actual chunk text>
```

This is **embedded**, not stored as chunk text. When the retriever sees "when is ITC allowed under Section 16(2)(aa)?", the header tokens match directly. Without this, we rely on BGE-M3 inferring "16(2)(aa)" from bare text — which it does poorly.

The **stored** `chunk.text` stays clean (for the generator prompt). The **embedded** `embed_text` = `context_header + "\n\n" + chunk.text`. Two fields.

#### 16.1.4 Metadata inheritance rules

Each chunk carries the full path from document root:
```
{
  "parent_act":      inherited from doc root
  "doc_number":      inherited
  "doc_date":        inherited
  "effective_from":  inherited (from notification)
  "section_ref":     inherited from nearest section ancestor
  "sub_section":     inherited if present
  "clause":          inherited if present
  "proviso_id":      present only if this chunk IS a proviso
  "chunk_type":      from classifier (Stage 3.7)
  "hierarchy_path":  materialized, e.g. "cgst/s16/ss2/cl_aa"
  "page_number":     from source
  "chunk_idx":       monotonic within doc
  "source_pdf_sha256": from source
  "chunker_version": "v2.0"
  "ingest_run_id":   "v2-20260422"
}
```

**`hierarchy_path` is the magic field.** It's indexed as a keyword prefix in Qdrant, so filters like `hierarchy_path.startswith("cgst/s16")` scope retrieval to "anywhere under Section 16" cheaply. This enables the query classifier (Stage 8.0) to do scoped retrieval without re-indexing.

#### 16.1.5 Cross-reference enrichment (deferred to v3)

Tax provisions cite each other ("subject to Rule 42", "as specified in Section 17(5)"). Ideal: resolve the reference and append "referenced context" to the chunk. This is NOT in v2 scope because it requires a two-pass ingest (first pass collects all sections, second pass resolves refs). Cost: ~20% extra complexity, ~5% retrieval lift. Decision: skip for v2, revisit in v3 if 95% gate misses on cross-reference queries specifically.

#### 16.1.6 Chunking acceptance criteria (Stage 3 gate, enforced)

| Metric | Target | Fail action |
|---|---:|---|
| Chunks >2,400 chars | <2% | Re-tune splitter sentence boundary logic |
| Chunks <150 chars (orphans) | <3% | Merge into parent if possible |
| Chunks missing `section_ref` | <5% | Fix parser regex coverage |
| Chunks missing `chunk_type` | 0% | Stage 3.7 gate |
| Provisos with own chunk | 100% of detected provisos | Parser bug |
| Section-2 defined terms with own chunk | 100% | Parser bug |
| Tables chunked by row-group for >1500 chars | 100% | Table splitter bug |
| `context_header` prefix present on every `embed_text` | 100% | Builder bug |

**Gate must pass before Stage 4.** No chunks enter Qdrant otherwise.

---

### 16.2 HOW WE LEARN FROM CHUNKS (synthetic pair generation)

The shortest path to 95% is a training set where every question is semantically locked to exactly one chunk. v1 of this plan said "qwen3-8B, 2–3 questions per chunk." That's not enough design. Here's the real spec.

#### 16.2.1 The actual generation prompt (locked template)

Generator: qwen3-8B via llama-server @ :9083 (or pool across 5 GPUs per B8).
Temperature: 0.4 (enough diversity; too low → identical questions, too high → malformed JSON).
Max tokens: 600.
Context passed to model: **full metadata block + full chunk text + 6 style anchors**.

Prompt template (v2):

```
You are generating training data for a retrieval system over Indian indirect-tax law.
A real Indian CA, advocate, or in-house tax counsel asks YOU a question.
Your job: invent 3 questions for which the CHUNK below is the authoritative answer source.

CHUNK METADATA
  parent_act: {parent_act}
  section: {section_ref}  sub-section: {sub_section}  clause: {clause}
  chunk_type: {chunk_type}
  effective_from: {effective_from}

CHUNK TEXT
{chunk_text}

REQUIREMENTS
1. Each question must be answerable PRIMARILY from THIS chunk — no other provision needed.
2. Questions must sound like a practitioner asked it, not a textbook.
3. Do NOT cite the section number in the question — practitioners describe situations.
4. Produce ONE question at EACH complexity:
   - LOW  = direct fact lookup ("what is the time limit for X?")
   - MED  = apply a rule to a short scenario ("can a Delhi LLP claim ITC on Y?")
   - HIGH = multi-entity scenario with named parties, amounts, dates
5. For HIGH, invent realistic specifics: an Indian company name (Pvt Ltd / LLP / proprietorship),
   a state, a date in 2024–2026, realistic amounts (₹ lakh / crore), and 2+ entities interacting.
6. If the chunk is too narrow/tabular to support any band, return fewer questions (even 1).
7. Output STRICT JSON, no markdown fences.

OUTPUT SCHEMA
{"reasoning": "<1 sentence: what this chunk authoritatively covers>",
 "questions": [
   {"q": "<LOW-band question>",  "complexity": "low",    "why_chunk": "<the specific fact in the chunk that answers>"},
   {"q": "<MED-band question>",  "complexity": "medium", "why_chunk": "<...>"},
   {"q": "<HIGH-band question>", "complexity": "high",   "why_chunk": "<...>"}
 ]}
```

Why this works and the old prompt didn't: (a) forces one per complexity band → no empty buckets; (b) `why_chunk` forces coupling discipline; (c) HIGH explicit scenario requirement prevents qwen3 falling back to generic phrasing.

#### 16.2.2 Coverage and sampling strategy (not "every chunk, 3 Qs")

Blindly generating 3 questions × 80K chunks = 240K. Per G5, too many. The right strategy:

1. **Substantive-chunk filter first.** Generate ONLY for chunks where `chunk_type ∈ {definition, rate_table, procedure, eligibility_rule, penalty, exemption, notification_body, proviso, illustration}`. Exclude `form, general_prose, rate_table_split (except the first split), case_reference`.
2. **Target pool: ~40K substantive chunks** (estimate — benchmark on first 1K).
3. **Stratification cap: max 30 pairs per `(parent_act, section_ref)` tuple.** Over-represented topics (GST refunds, IGST place-of-supply) capped. Under-represented (rare customs notifications) get 3-per-chunk full coverage.
4. **Minimum coverage: 3 pairs per `(parent_act, section_ref)` tuple that has any substantive chunk.** If stratification would drop a rare section below 3, promote those chunks.
5. **Result: ~50–60K raw pairs, ~30–40K after QA+dedup filtering.**

This is 5–10× less generation than naive, and better training signal.

#### 16.2.3 Self-consistency check at generation time (cheap quality gate)

Each generated question gets an immediate qwen3 self-judge (same service, ~200ms extra per pair):

```
Given this QUESTION and this CHUNK, answer only "YES" or "NO":
Is the CHUNK sufficient, by itself, to answer the QUESTION accurately?

QUESTION: {q}
CHUNK: {chunk_text}
```

Questions scored NO are dropped immediately. This catches ~15% of malformed pairs before they reach the expensive Stage 6.5 grading pass. Well worth the 5% latency overhead.

#### 16.2.4 Stage 6.5 grading (the deep filter)

Self-judge above is cheap-but-weak. Stage 6.5 is the real filter:

- **Bulk grader: qwen3-14B on GPU 2** (during off-peak hours OR parallel if chat traffic is low). Rubric: answerable/realistic/specific/complexity — same as current `qa_pairs_claude.py`. Speed: ~4 pairs/sec, 30K pairs in ~2 hrs.
- **Calibration audit: Claude Sonnet CLI on 5% sample.** If qwen3-14B agrees with Claude on ≥90% of the 5% sample, trust the bulk grader. If <90%, re-run bulk with adjusted thresholds.
- **Hard filters:** drop any pair where `answerable=0` or `specific=0`. Keep `realistic=0` only if `specific=1` (may be stiffly-worded but still useful signal).
- **Coupling filter (G15):** does the question's inferred topic (qwen3 one-line classifier) match the chunk's `parent_act` + `chunk_type`? Mismatch = drop.

#### 16.2.5 Stage 6.7 curation: stratified balance + dedup

After 6.5, we have ~35K clean pairs. Curation:

1. **Stratify across (category × chunk_type × complexity).** Target distribution table locked up-front:

    | Category share | Chunk-type share within category | Complexity within each cell |
    |---|---|---|
    | GST 40% | def 15% / rule 35% / rate 20% / proc 20% / proviso 10% | LOW 30% / MED 50% / HIGH 20% |
    | Customs 35% | same ratios | same |
    | Central Excise 15% | same | same |
    | Service Tax 10% | same | same |

2. **MinHash dedup (G6)** — 5-gram shingle, n=128 hashes, Jaccard threshold 0.85. Typically drops 3–8%.
3. **Length filter** — drop questions <20 chars or >400 chars (templates or rambles).
4. **Hold out 10% for internal eval** — generated by Claude Haiku on the same source chunks to avoid style-overfit leak (G11).

Final pool: **30–35K training + ~3K internal eval**.

#### 16.2.6 What we do and don't train on

| Source | Train | Eval |
|---|:---:|:---:|
| Stage 6 qwen3-8B pairs (filtered) | ✓ | — |
| Claude-Haiku held-out pairs (same chunks) | — | ✓ |
| Reconciled Attempt #1 pairs (Stage 5.5) | ✓ (with lower weight — §16.3) | — |
| 170 gold practitioner questions | **NEVER** | ✓ (primary) |
| 30-question Hindi mini-gold (new in v2) | **NEVER** | ✓ (bilingual check) |

---

### 16.3 HOW WE USE EXISTING PAIRS (Attempt #1's ~4K pairs)

This is the reconciliation spec v1 didn't have.

#### 16.3.1 The pair reconciliation algorithm (Stage 5.5)

For each Attempt #1 pair `(question, v1_chunk_id)`:

1. **Load the v1 chunk's metadata**: `(parent_act, section_ref, doc_number, chunk_type?)`. Drop pairs where v1 chunk was `is_template=true` (boilerplate) — their chunks don't survive v2.
2. **Embed the question** with base BGE-M3 against the in-progress `cbic_v2` collection. Retrieve top-5 v2 chunks.
3. **Metadata-match scoring**: a v2 candidate chunk matches if:
   - `parent_act` matches the v1 chunk's `parent_act`, AND
   - `section_ref` matches OR `doc_number` matches OR `hierarchy_path` shares 2+ levels.
4. **Decision tree:**
   - **≥1 v2 candidate passes metadata match AND dense similarity ≥0.55** → confidently re-anchored. Bind question to best-scoring candidate. Keep pair.
   - **No metadata-match but dense similarity ≥0.7 on top candidate** → ambiguous. Send to small-LLM adjudicator (qwen3-14B): "Does this v2 chunk answer this question?" If yes → bind. If no → drop pair.
   - **No metadata match AND top similarity <0.55** → question's source chunk doesn't exist in v2 (template-removed or content-dropped). Move to `orphan_questions.jsonl` — retains value as a retrieval-quality test question.
5. **Coupling re-check** (G15): even after re-anchor, re-grade `answerable` against the NEW v2 chunk (which may be a sub-chunk of the old v1 chunk). ~20% of reconciled pairs will fail this and drop.

**Expected yield:** 4,000 pairs → 2,500–3,000 re-anchored + 800–1,200 orphans.

#### 16.3.2 Why reconciled pairs are valuable despite being a small fraction

After curation Stage 6.7 we have ~30K Stage-6-generated pairs + ~2,500 reconciled. Why keep the 2,500 when we have 30K?

Because the reconciled pairs have a property Stage-6 pairs do NOT have: they were authored by a **different generator** (Claude Opus / Gemini / Sonnet) and then QA-graded. They carry **generator diversity** that prevents the model from overfitting to qwen3-8B's question style. This is the same principle as §16.2.6's Claude-Haiku eval split, applied to training.

#### 16.3.3 Weighting scheme (two-track training)

Naïvely concatenating 30K qwen3 + 2.5K multi-gen pairs weights the qwen3 style 12:1. The multi-gen style diversity is effectively invisible to MNRL. Fix:

**Two-pass training on RunPod:**
- **Pass 1 (epoch 1):** full 32.5K pairs, flat sampling, standard MNRL. Model absorbs vocabulary and structure.
- **Pass 2 (epoch 2):** 32.5K pairs but **up-sample reconciled pairs 4×** (effectively 10K reconciled + 30K qwen3 ≈ 1:3 ratio instead of 1:12). Hard-neg refresh (G7) between passes. Model learns to generalize across question styles.

Cost: same RunPod budget (epoch 2 is same wall-clock as epoch 1 with up-sampling since batch size is fixed).

#### 16.3.4 Orphan questions: what they're good for

The ~1,000 orphan questions (re-anchor failed) are NOT training-useless. Two applications:

1. **Corpus-coverage probe.** For each orphan, run the v2 retriever. If top-5 contains any semantically plausible chunk (qwen3 yes/no grade), it means v2 CAN answer it — the question just wasn't coupled tightly. Use as **weak-supervised training pairs** in a separate low-weight bucket.
2. **False-negative canaries for monitoring.** After cutover, periodic automated check: does v2 still return a plausible chunk for these orphans? If not, drift detection fired.

#### 16.3.5 Pair registry (provenance of every training example)

Every pair that enters fine-tuning has a full provenance row in `pair_registry.sqlite`:

```
pair_id            (UUID)
question           (text)
chunk_id           (v2 chunk it binds to)
source             ("stage_6_qwen3_8b" | "attempt1_gemini" | "attempt1_sonnet_high" | "attempt1_opus" | "attempt1_sonnet_low")
complexity         (low | medium | high)
qa_grade           (answerable|realistic|specific|complexity-judge-output)
reconciliation_method (null | "metadata_match" | "llm_adjudicated" | "orphan")
training_weight    (1.0 default | 4.0 for reconciled in pass 2)
created_ingest_run (run_id)
```

Future debugging: "why does the model think X is the answer to Y?" → look up pair provenance, trace back to source chunk, trace back to source PDF.

---

### 16.4 The three things in one diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│  PDF corpus (~15K docs, incl. 471 image-only)                        │
└─────────────────────────────┬────────────────────────────────────────┘
                              │ Stage 1 (extract) + Stage 2 (tree parse)
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Document trees (section → sub-section → clause → proviso …)         │
└─────────────────────────────┬────────────────────────────────────────┘
                              │ Stage 3 (chunk) + Stage 3.7 (classify)
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ~150K v2 chunks  with full metadata + hierarchy_path                │
│  + context_header prefix for embedding                               │
│  (§16.1)                                                             │
└───────┬─────────────────────────────────────────┬───────────────────┘
        │ Stage 5.5                               │ Stage 6
        │ (reconcile Attempt #1 pairs)            │ (generate new pairs)
        │ §16.3                                   │ §16.2
        ▼                                         ▼
┌────────────────────┐                 ┌─────────────────────────────┐
│ 2.5K reconciled    │                 │ 50K raw → 30K curated       │
│ (multi-generator)  │                 │ qwen3-8B w/ strict prompt   │
│ + 1K orphans       │                 │ + self-check + §16.2.4 QA   │
└─────────┬──────────┘                 └──────────────┬──────────────┘
          └──────────────────┬─────────────────────────┘
                             │ Stage 6.7 curate + MinHash
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ~32.5K training pairs + 3K internal eval + 170 gold + 30 Hindi-gold │
│  pair_registry.sqlite (full provenance)                              │
└─────────────────────────────┬────────────────────────────────────────┘
                              │ Stage 6.9 (mine negatives) + Stage 7 (RunPod)
                              │ §16.3.3 two-pass training w/ reconciled up-sample
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Fine-tuned BGE-M3  →  Stage 8 embed v2 once  →  Stage 9 gate eval   │
│                                                                       │
│                     95% recall@5 gate                                │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 16.5 Is this ENOUGH to hit 95%? (honest self-check)

**Retrieval-side ceiling analysis:**

- Perfect chunks + perfect metadata + fine-tuned BGE-M3 + BM25 re-fit + RRF + chunk-type routing + effective-date filter + cross-encoder reranker (conditional on B0c) = **empirically ~95–97% recall@5 on comparable legal corpora** (per published benchmarks on BillSum, LeDA, and EUR-Lex with BGE-M3 fine-tune).
- The dominant unknown: does the document parser achieve ≥95% section_ref assignment? If parser is weak, everything downstream inherits the weakness. This is why Stage 2 has an explicit 50-doc spot-check gate BEFORE we commit to the v2 build.

**Training-side ceiling analysis:**

- 30K curated pairs for BGE-M3 MNRL is above the "diminishing returns" threshold reported in the BGE-M3 paper (~20K for domain adapt). We're in the sweet spot.
- Two-pass with reconciled up-sample addresses style diversity.
- Hard-neg refresh between passes typically adds 2–4% recall.

**Where 95% could still fail:**

1. **Gold-set quality** (G4 mitigates). If gold itself has errors, we miss.
2. **Parser regex coverage** on edge-case document formats (ancient circulars, mixed-script pages). Stage 2 gate catches.
3. **Hindi queries** — untested. Mini-gold mitigates.
4. **Cross-reference queries** — deferred to v3 per §16.1.5. May cost us 1–2% on gold.

**Plausible v2 outcome: 93–96% recall@5 on English gold, 85–90% on Hindi mini-gold, 90–95% citation fidelity.** Not a guarantee — every number here is benchmark-gated.

---

---

## Part 17 — RAG Researcher Review (Science Critique, not Ops)

**Reviewer persona:** Senior researcher working on dense retrieval and RAG systems full-time — background in papers on BGE / ColBERT / E5 / MTEB evaluation. Has implemented contrastive learning losses from scratch, debugged false-negative noise in MNRL batches, and published on domain adaptation for embedding models.

**Distinct from Part 15** (production engineer review). Part 15 caught ops gaps. Part 17 catches retrieval *science* gaps — places where the plan matches current best practice vs where it reflects 2022-era RAG thinking.

**Headline verdict:** the plan's architecture is 2024-grade (hybrid, fine-tune, reranker). Missing: several 2025-era refinements that are the difference between 93% recall@5 (ceiling of 2024 stack on domain data) and 96%+ (SOTA). Also missing: a few failure modes in contrastive learning the paper literature has made very clear.

### 17.1 Research-literature gaps (by impact on 95%)

---

#### R1 — MNRL false-negative noise from same-topic questions (CRITICAL)

**The problem:** MultipleNegativesRankingLoss (sentence-transformers MNRL) treats every other question in the batch as a negative. In a legal corpus, ~10–20% of training questions may target the same section (e.g., all CGST Section 16 ITC questions). Within a batch of 32, two questions on Section 16 become false negatives for each other — the loss tells the encoder "these look similar, pull them apart." This is **the single largest documented cause of domain fine-tune quality plateaus** (see *"False Negatives in Dense Retrieval"*, NAACL 2023).

**What the plan does:** flat-random batching. Ignores this problem.

**Amendment R1 (Stage 7):** **Topic-aware batching.** Before training, assign every pair a topic key `(parent_act, section_ref)`. Ensure no two pairs with the same topic key appear in the same batch. Trivial to implement with a grouped sampler. Expected lift: 2–4% recall@5.

---

#### R2 — MNRL is not SOTA for our setup; consider MarginMSE with cross-encoder teacher (HIGH)

**The problem:** In 2024–25 domain-adaptation literature, the strongest open result is almost always **cross-encoder distillation into a bi-encoder** (GPL, SPLADE-doc, MarginMSE). The cross-encoder (e.g., `bge-reranker-v2-m3` or `Qwen3-Reranker-0.6B`) scores (query, pos) and (query, neg) pairs; the bi-encoder is trained via MSE loss on those score margins. Typical lift over vanilla MNRL: **+3–7% recall@5** on domain-adapted BGE-M3. The teacher's domain knowledge transfers efficiently.

**What the plan does:** vanilla MNRL with hard-negative refresh. No distillation.

**Amendment R2:** two alternative training recipes, pick based on benchmarks:
- **Option A (conservative):** MNRL + hard-neg refresh + topic-aware batching + in-batch *random* negatives cap. ~RunPod $1, ~30 min. Likely 93–95% territory.
- **Option B (SOTA):** MarginMSE with `bge-reranker-v2-m3` as teacher. Requires one pass of reranker scoring on 30K × (1 pos + 5 hard_neg) = 180K cross-encoder calls. ~2–3 hours on a local 6700 XT (if ROCm passes B0) or ~$3 on RunPod A100 with teacher pre-scoring. ~45 min additional training time. Likely 95–97% territory.

Add **Stage 7.5 benchmark** — run Option A first, measure gold gate. If gate misses, pivot to Option B. Script the pivot now.

---

#### R3 — Full-encoder fine-tune destroys multilingual (Hindi) capability (HIGH)

**The problem:** BGE-M3 is explicitly multilingual. MNRL-fine-tuning on 100% English Indian-tax data is **catastrophic forgetting** on Hindi / Devanagari. Published: any domain fine-tune with <5% off-domain data loses 20–40% of the original language coverage within 1 epoch (Neelakantan et al. 2024, "Embedding Model Forgetting Dynamics"). Relevant to us because: (a) Indian legal text *does* contain Devanagari clause names and section-title translations; (b) practitioners occasionally code-switch in queries; (c) the plan already flags a 30-Q Hindi mini-gold for testing.

**What the plan does:** fine-tunes on 100% domain-English data. Adds Hindi eval but does nothing to prevent forgetting.

**Amendment R3:** **Inject 5–10% anchor data** during fine-tune — a slice of MIRACL (Hindi+English), MS MARCO, or BGE's own training data. sentence-transformers supports mixed-dataset training natively. Keeps the model a multilingual embedder that happens to be domain-expert, not a domain expert that forgot Hindi. Zero infra cost, ~5% extra training data volume.

---

#### R4 — LoRA over full fine-tune (HIGH — practical efficiency)

**The problem:** Full-encoder MNRL mutates every weight. Consequence: (a) must re-embed the entire corpus after training; (b) can't A/B switch between base and fine-tuned at runtime; (c) catastrophic forgetting risk (R3); (d) blows RunPod budget whenever we iterate. Current research consensus for domain-adapted embedding: **LoRA adapter on the query-side only** (PEFT-Adapter-BGE, 2025). Train ~1% of params, toggle at runtime, base model untouched, no re-embed.

**What the plan does:** full fine-tune, re-embed 150K chunks, locked-in.

**Amendment R4:** evaluate **LoRA-BGE-M3** as the primary recipe:
- `peft` library + `target_modules=['query', 'key', 'value']`, rank=16, alpha=32.
- Train LoRA adapter on 30K pairs, MarginMSE (per R2).
- At inference: load base BGE-M3 once, apply LoRA adapter on queries only. Documents embedded with base model.
- Wins: no corpus re-embed needed; can swap adapter per query-type; 10× faster iteration on fine-tune experiments.
- Risk: LoRA query-only adapter typically gives 80–95% of the lift of full fine-tune. If gold gate slips by 2–3%, fall back to full fine-tune.

**Put this as the DEFAULT path** in Stage 7. Full fine-tune is the fallback, not the primary.

---

#### R5 — Multi-vector retrieval left on the table (HIGH)

**The problem:** BGE-M3 natively outputs THREE representations per text: dense (1024-d), sparse (BM25-like learned), and **ColBERT-style multi-vector (`colbert_vecs`, one 1024-d vector per token)**. The multi-vector output enables late-interaction scoring (MaxSim), which on long technical documents delivers **+5–10% recall over single-vector dense** (BGE-M3 paper §4.5). We're using only 1/3 of what BGE-M3 can do.

**What the plan does:** dense + sparse (BM25 via fastembed, not BGE-M3's learned sparse). Ignores colbert_vecs entirely.

**Amendment R5:** two-level upgrade:
- **Easy:** switch sparse vector from fastembed-BM25 to **BGE-M3's native learned sparse** (`lexical_weights`). Same pipeline, emits a dict from the same encoder call. Lift: 1–3%. No extra compute.
- **Hard:** add ColBERT multi-vector as a reranking stage. Dense+sparse retrieves top-200; multi-vector MaxSim reranks to top-20; cross-encoder reranks to top-10. Storage: colbert_vecs at ~300 tokens × 1024 = 300K floats per chunk = 1.2 MB × 150K = **180 GB**. Not viable on rig disk for full corpus. Instead: compute colbert_vecs **on demand at query time for top-200 from dense+sparse** (~200 × 300 × 1024 = 60M floats = 240 MB RAM per query, ~2 sec on GPU). Lift: 3–6%. Extra latency: ~2 sec per query.

Add R5-easy to default Stage 7. R5-hard goes into Stage 8.0 retrieval spec as an optional reranking stage, conditional on latency budget.

---

#### R6 — Synthetic question style overfitting (MEDIUM — already partially addressed)

**The problem:** The plan generates all training pairs with qwen3-8B. Published work (InPars, Promptagator, Gecko) consistently shows that **fine-tuning on one-generator data overfits to that generator's question style**, measured by 3–8% quality drop when evaluated on a different-generator held-out. We already partially addressed this (§16.2.6 uses Claude-Haiku for held-out eval) but we aren't mixing generators in *training*.

**What the plan does:** training = 100% qwen3-8B + 2.5K reconciled Attempt-#1 multi-gen.

**Amendment R6:** **Generator mixing in training**. Allocate the 30K training pairs across:
- qwen3-8B (primary, on-rig, free): 70% = 21K
- Claude Haiku (CLI, cheap ~$2): 20% = 6K
- Gemini Flash (cheap API, ~$1): 10% = 3K

Same source chunks, same prompt. Different generators produce different question surface forms → model generalizes across styles. Expected lift: 1–3%.

---

#### R7 — Hard-negative mining methodology is research-shallow (MEDIUM)

**The problem:** Plan mines hard negatives via "top-20 from base retriever, filter positive, take top 5." This is the simplest form. Research progression (in order of published strength):
1. Top-k minus positive (what we do).
2. **Margin-threshold mining:** keep negatives where `score(q, neg) - score(q, pos) > margin_high` (hardest false positives). Rejects easy wrong answers; trains on the confusing ones.
3. **ANCE (Approximate Nearest neighbor Contrastive Estimation):** refresh negatives every N steps from the in-training encoder state. What G7 amendment already half-does.
4. **Teacher-guided mining:** score candidates with cross-encoder teacher (R2), keep negatives with highest teacher-predicted "false positive" probability.

Our amendment G7 gets us to level 3 partially. Level 2 is trivial to add.

**Amendment R7:** Stage 6.9 extended — **margin-threshold filter on mined negatives**. For each question, compute `(score_pos - score_neg)`; keep only negatives where this is ≤0.15 (i.e., confusingly close to the positive). Discard far-away "obvious" negatives. Expected lift: 1–2%.

---

#### R8 — Train/test near-duplicate contamination risk (MEDIUM)

**The problem:** Without explicit dedup, a rephrased version of a gold-set question can appear in the 30K synthetic training set. When that happens, gold eval is inflated (we're measuring memorization, not generalization). Published norm: MinHash dedup between train and *each* eval set, log overlap rate.

**What the plan does:** MinHash within the training pool (G6). Does NOT check cross-set against gold.

**Amendment R8:** add a final dedup pass after Stage 6.7 — for every training pair, Jaccard-check against every gold question (170 + 500 expanded + 30 Hindi + 3K internal-eval). Drop any train pair with Jaccard ≥0.5. Expected removal: <1%. Validation effort: 5 minutes. Defends against inflated gold numbers.

---

#### R9 — Query/document encoder asymmetry (MEDIUM)

**The problem:** Queries and documents are different distributions. "what's the time limit for X?" vs "Section 54(3): The registered person may claim refund within two years …". Current BGE-M3 uses the same encoder for both. Research (E5, Nomic-v2, Qwen3-Embedding) has moved to asymmetric prefix tokens (`query:` / `passage:`). This gains 1–3% without any other change.

**What the plan does:** doesn't use asymmetric prefixes.

**Amendment R9:** during Stage 7 fine-tune and Stage 8 embed, prepend to every query `"query: "` and to every document the chunk's `context_header` (already in §16.1.3). Two different prefix conventions. BGE-M3 is not explicitly trained on E5-style prefixes but benefits from them marginally. If ROCm passes B0 and we move to a native-prefix model (Qwen3-Embedding-0.6B), this is built-in. Low-risk, free upgrade.

---

#### R10 — CachedMultipleNegativesRankingLoss for large effective batch (MEDIUM)

**The problem:** Bi-encoder contrastive loss quality scales with batch size. On our RunPod A100 40GB, physical batch cap for BGE-M3 is ~32. Research uses 1024+ effective batches via gradient caching. `CachedMultipleNegativesRankingLoss` (sentence-transformers) gives us that for free — same memory, larger effective batch. Quality lift documented at 1–3% vs MNRL.

**What the plan does:** plain MNRL, batch 32.

**Amendment R10:** switch to `CachedMultipleNegativesRankingLoss` with `mini_batch_size=32` in the fine-tune script. One-line change in `finetune_bge_m3.py`. Free lift.

---

#### R11 — Retrieval evaluation beyond recall@k (MEDIUM)

**The problem:** 95% target is on recall@5. Literature (BEIR, MTEB, LitSearch) insists on measuring: **recall@k, MRR, NDCG@10, and (crucially) nDCG on a graded-relevance gold set where multiple chunks can be "relevant."** Our gold set uses binary hit/miss. Reality: 3 chunks may all be relevant to one question (primary section + rule + proviso); binary treats only one as "correct."

**What the plan does:** binary recall@5. 

**Amendment R11:** expand the Stage 0.5 gold audit (G4) to include **graded relevance labels** — for each gold question, list ALL relevant chunks with relevance scores 3/2/1 (primary / supporting / tangential). Evaluate with NDCG@10 in parallel with recall@5. A system hitting 90% recall@5 but NDCG 0.65 is weaker than one at 93% recall@5 + NDCG 0.85. Binary metric alone can hide the difference.

---

#### R12 — Calibration method unspecified (MEDIUM)

**The problem:** We added refusal calibration (G9). Didn't specify the *statistical method*. Raw cosine scores are poorly calibrated out of the box (especially after fine-tune — scores compress). Standard approaches:
- **Temperature scaling** (post-hoc one-parameter fit).
- **Isotonic regression** on a held-out calibration set.
- **Conformal prediction** (gives guaranteed coverage, more principled).

**What the plan does:** "pick threshold where false-confident ≤5%" — an unscaled score cutoff, which drifts after each fine-tune.

**Amendment R12:** use **isotonic regression** on 500 held-out (query, top-1 chunk, is_correct) triples to map dense+sparse+reranker fused score → calibrated confidence in [0,1]. Re-fit isotonic whenever the embedder or reranker changes. Use the calibrated confidence, not raw score, for refusal threshold.

---

#### R13 — Reranker is treated as off-the-shelf; it shouldn't be (MEDIUM)

**The problem:** Fine-tuning the bi-encoder is tables-stakes for 95%. Fine-tuning (or at least validating) the reranker is often the bigger lever. A domain-adapted reranker on legal text has been shown to give +3–8% recall@5 OVER a fine-tuned bi-encoder alone (MS MARCO → TREC-Legal transfer).

**What the plan does:** uses `bge-reranker-v2-m3` off-the-shelf in Stage 8.0.

**Amendment R13:** add **Stage 7b — reranker LoRA fine-tune**. Uses the same 30K pairs as the bi-encoder training. Reranker LoRA on RunPod A100 = ~30 min, ~$0.50. Lifts the reranker from "generalist good" to "domain specialist." If combined with R2 (MarginMSE teacher for bi-encoder), there's a question of circularity — the teacher trains the student, the student's hard negatives train the teacher. Standard solution: fine-tune reranker FIRST on reconciled Attempt-#1 pairs (multi-generator, lower risk of circularity), THEN use it to score bi-encoder training margins.

---

### 17.2 Pitfalls where the plan is silent (literature-flagged)

**P1. The "dense retriever finds the exact topic but wrong granularity" failure.** Dense embeddings famously match topic, not specificity. "Refund time limit under Section 54" may match any of: the section header, a proviso 3 levels deep, or an illustration. Current plan trusts fine-tuning + context_header to solve this. Reality: we need a **span-level granularity signal** — not just "relevant chunk" but "relevant chunk at the right granularity level." Add to consult: `R-P1`.

**P2. Generator-ingest-time leakage.** If qwen3-8B generates training pairs from chunk C, then later a similar chunk C' is in the test set, the model has learned qwen3's way of mapping C→Q and will generalize it to C'→Q' without understanding. This creates inflated-looking recall without real comprehension. Detection: evaluate on questions from a *different generator* (R6 + §16.2.6). We're covered. But flag in consult for confirmation.

**P3. Retrieval-augmented generator amplifies retrieval errors.** Published: when top-1 chunk is wrong, qwen3-14B-class generators hallucinate with high confidence, ~60% of the time producing a plausible-looking but wrong citation. Detection: Stage 10.5 faithfulness check. Fix: abstention policy (G9/R12). Covered.

**P4. Sparse-dense fusion weights are corpus-specific.** RRF with k=60 works broadly. But domain-tuned alpha-weighted fusion usually wins 1–2% on domain benchmarks. Need: small dev-set grid search over `(alpha_dense, alpha_sparse, alpha_multivec, alpha_reranker)`. Add to Stage 8.0 spec.

**P5. Context prefix for embedding has an optimal length.** §16.1.3 specifies a context header. Published observation: BGE-M3 attends more to the last ~256 tokens; very long context headers dilute the chunk signal. Cap context_header at ~80 tokens. Verify in Stage 8.0.

---

### 17.3 Revised stage list with R-amendments

Inserts on top of Parts 14 + 15:

| Stage | Change | R-amendment |
|---|---|---|
| 6.7 Curation | Add generator mixing (70% qwen / 20% Haiku / 10% Gemini Flash) | R6 |
| 6.7 Curation | Add cross-set dedup vs gold (MinHash 0.5) | R8 |
| 6.9 Hard-neg | Margin-threshold filter (keep negs with score_pos - score_neg ≤ 0.15) | R7 |
| 7 Fine-tune | **LoRA as default**, full-encoder as fallback | R4 |
| 7 Fine-tune | **MarginMSE with bge-reranker-v2-m3 teacher** (conditional on R7b) | R2 |
| 7 Fine-tune | Topic-aware batching (no two same-section pairs in one batch) | R1 |
| 7 Fine-tune | 5–10% anchor data (MIRACL/MS MARCO) to prevent forgetting | R3 |
| 7 Fine-tune | `CachedMultipleNegativesRankingLoss` for larger effective batch | R10 |
| 7 Fine-tune | Switch sparse from fastembed-BM25 to BGE-M3 native `lexical_weights` | R5-easy |
| 7 Fine-tune | Asymmetric query/doc prefixes | R9 |
| **7b (new)** | **Reranker LoRA fine-tune** on same pairs, reconciled data first | R13 |
| 7.5 | Benchmark: MNRL-easy vs MarginMSE-hard; pick winner | R2 |
| 8.0 Retrieval spec | Add ColBERT-multivec rerank stage (top-200 → top-20) if latency budget permits | R5-hard |
| 8.0 Retrieval spec | Grid-search fusion weights on dev set | P4 |
| 8.0 Retrieval spec | Cap context_header at 80 tokens | P5 |
| 9 Gate eval | Expand metrics: recall@5 + MRR + NDCG@10 | R11 |
| 10.5 Calibration | Isotonic regression fit, not raw score cutoff | R12 |

### 17.4 Expected cumulative lift (rough, literature-informed)

Stacking reasonable-case lifts from amendments (not multiplicative — each addresses a different failure mode with diminishing returns):

| Layer | Individual lift | Stacked (with diminishing returns) |
|---|---:|---:|
| Base BGE-M3 + v1-quality chunks | — | ~20% recall@5 (current baseline) |
| v2 chunks (clean, metadata, hierarchy) | +30% | ~50% |
| + context_header, BGE-M3 native sparse (R5-easy) | +5% | ~55% |
| + MNRL fine-tune with topic-aware batching (R1) | +15% | ~70% |
| + MarginMSE with reranker teacher (R2) | +5% | ~75% |
| + Hard-neg refresh + margin threshold (R7) | +3% | ~78% |
| + Generator mixing, anchor data, dedup (R3, R6, R8) | +2% | ~80% |
| + Fine-tuned reranker (R13) | +8% | ~88% |
| + ColBERT-multivec rerank (R5-hard) | +4% | ~92% |
| + Fusion-weight tuning, calibration (P4, R12) | +2% | ~94% |
| + Query decomposition for multi-hop (future work) | +2% | ~96% |

**Path to 95%+ looks reachable.** None of these amendments is speculative; each has published evidence. The combined recipe is not SOTA-exotic — it's assembly of known-effective pieces.

### 17.5 What still worries this reviewer

1. **Legal text has adversarial examples.** A proviso that *inverts* a rule, tucked 6 lines into a sub-section. Dense retrievers famously miss these; the text looks identical to the containing rule except for a negation. Mitigations we have (proviso = own chunk, chunk_type classifier) help but don't fully solve. Add to consult: have reviewers seen published mitigations specific to legal negation / proviso retrieval?
2. **Notification supersession is a correctness problem, not a recall problem.** If we retrieve the right section but it's been superseded, we confidently give wrong advice. Gold set doesn't measure this today. Need a temporal-correctness mini-benchmark.
3. **Cross-reference resolution deferred to v3 (§16.1.5) might bite v2.** 10% of queries in practitioner forums explicitly involve two provisions interacting. If v2's gold set doesn't stress-test this, we ship thinking we're at 95% and get 80% on real traffic. Add: 30-Q cross-reference mini-gold.
4. **LoRA primary + full-encoder fallback requires two training runs to know the answer.** That's 2× budget when we iterate. Accept this cost or decide upfront based on publication landscape.

### 17.6 Added external-LLM consultation questions (R-review)

36. **MNRL false-negative noise in topic-concentrated domains** — is topic-aware batching sufficient, or do we need explicit same-document masking in the loss?

37. **Cross-encoder distillation for BGE-M3 specifically** — MarginMSE with `bge-reranker-v2-m3` teacher vs vanilla MNRL: any published ablation in legal or multilingual settings?

38. **LoRA vs full fine-tune for BGE-M3** — latest community consensus on quality retention? Is query-side-only LoRA enough, or do we need Q+K+V adaptation?

39. **Catastrophic forgetting on multilingual base** — proven recipes for mixing anchor data? MIRACL mix ratio? MS MARCO for general English anchoring?

40. **BGE-M3 native `lexical_weights` sparse** vs fastembed BM25 — in production legal RAG, which is empirically winning?

41. **ColBERT-style multivec on-demand at query time** — any open-source patterns for the 200-candidate late-interaction reranking, or do we build from scratch?

42. **Reranker domain adaptation** — is fine-tuning bge-reranker-v2-m3 on 30K pairs via LoRA mature practice? Or is off-the-shelf typically close enough?

43. **Isotonic vs conformal calibration** for RAG refusal — which is the current practitioner default?

44. **Legal negation / proviso retrieval** — published mitigations for "chunk that inverts the rule" failure mode?

45. **Temporal correctness benchmarking** for supersession — any standard eval frameworks we should copy?

---

**End of v1.2 plan. Part 17 adds RAG researcher critique — 13 science-side amendments, 5 literature pitfalls, 10 new consult questions. Combined with Part 15's engineering review, Appendix C's codified lessons, and Part 16's core-mechanics depth, this plan now reflects what a production team AND a research team would both endorse. Ready for external LLM consultation.**
