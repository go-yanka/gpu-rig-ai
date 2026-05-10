# CBIC RAG — Dual Plan for External LLM Evaluation

**Date:** 2026-04-21
**Version:** v2 (incorporates critique from consult round #3)
**Context:** Three consultation rounds returned. Round #3 pushed back on sequencing, hard filters, GBNF vs json_schema, span lengths, two-collection architecture, and flagged Table Extraction as a missing dimension. This v2 reflects their corrections. Changes from v1 marked **[v2]**.

This doc lays out (A) how we'll fix the *current* system end-to-end, and (B) how we'd rebuild from scratch. Please critique both. Flag anything wrong, risky, or under-specified.

---

## PART A — Fix the current system (CBIC RAG, live at `b11b12a8_v1`)

### A0. Baseline (before any change)

**Action:** Run the 50-Q eval harness at `D:\_gpu_rig_ai\eval\run_eval.py` against live backend. Save `runs/baseline_<ts>/`.
**Metrics captured:**
- Section-coverage %
- Expected-keyword %
- Forbidden-word violations
- Verbatim-quote gate pass rate (per-answer, aggregate)
- LLM-as-judge 0-3 (qwen3-14b temp=0)
- Per-category breakdown (gst / customs / central_excise / service_tax / others)
- Per-difficulty breakdown (basic / intermediate / complex)

**Gate for downstream deploys:** any P-step must not regress aggregate by >5 pct-pts, nor any individual query by >20 pct-pts.

**Time:** 20-30 min.

---

### A1. P1 — Retrieval boost + payload-filter forced routing

**Goal:** Fix the "Acts lose semantic contest to circulars" root cause without touching prompts.

**Design:**
1. **Query classifier (regex, zero-LLM):**
   - `STATUTE_REGEX = r'\b(?:Section|Sec\.?|s\.|Rule|Article)\s+(\d+[A-Z]?)(?:\s*\(\s*(\d+[A-Z]?)\s*\))?(?:\s*\(\s*([a-z])\s*\))?'` — catches `Section 10(1)(b)`, `Rule 86B`, `s. 16`, etc.
   - `LEGAL_PHRASE_SET` — ['composite supply', 'mixed supply', 'place of supply', 'time of supply', 'bill-to-ship-to', 'input tax credit', 'reverse charge', 'zero-rated', 'export of services', 'DRC-01B', 'GSTR-1', 'GSTR-3B', 'GSTR-2B', …]

2. **Two-tier routing [v2: no hard filters]:**
   - **Hard match** (STATUTE_REGEX hit with explicit Section/Rule number): large BM25 score multiplier (×3-4) on `doc_type IN ('act','rules')` chunks at fusion time. **No Qdrant payload hard-filter** — blinds the LLM to clarifying circulars (e.g., "Section 12 + vouchers" needs the circular with the formula). Unfiltered retrieval continues; the boost just re-ranks.
   - **Soft match** (phrase hit only, no numeric anchor): doc_id multiplier ×1.8 on Act/Rules chunks at RRF fusion time.
   - **No match**: unchanged behavior (existing hybrid + ColBERT + MMR).

3. **BM25 weight bump on parenthetical refs:** when query contains `\d+\([0-9a-z]+\)`, increase sparse weight from 0.4 → 0.6 in RRF fusion.

4. **Preserve diversity cap + ColBERT rerank** untouched.

**Files touched:**
- `retriever.py`: new `classify_query(question) -> {'tier': 'hard'|'soft'|'none', 'matches': [...]}`; integrate in `augment_section_aware`.
- `api.py`: feed classification into the retrieve block; log `retrieval_tier`, `statute_refs_matched` to timings.

**Sentinel:** `p1_v1`.
**Backup/patched/patch-script:** standard 3-location (rig `/opt/indian-legal-ai/patches/p1_<ts>/`, laptop `D:\_gpu_rig_ai\patches\p1_<ts>\`, pre/post mirrors next to live).
**Risk:** low — retrieval-side only, prompt untouched.
**Ship condition:** eval aggregate up, no individual regression >20%.
**Time:** 1-1.5 hr code + 20 min eval.

---

### A2. Re-ingest updated CGST Rules + refresh IGST Act

**Gap found in audit:** CGST Rules PDF is December 2022 — missing Rule 88D (inserted Aug-2023). IGST Act PDF is March 2020 — missing Finance Act 2023 amendments.

**Action:**
1. Download latest consolidated CGST Rules and IGST Act from CBIC portal.
2. Reuse existing ingestion recipe (single-GPU Vulkan BGE-M3 on GPU 5, 5.2-9.6 ch/s) — see `ingest_playbook_cbic.md`.
3. Delta-ingest only the two documents. Mark old chunk IDs as `superseded: true` in payload rather than deleting (so feedback on old answers still traces).
4. Verify Rule 88D + Finance Act 2023 sections retrieve cleanly.

**Time:** 30-45 min.

---

### A3. ~~P2 — Parent-document retrieval~~ **[v2: DROPPED]**

**Rationale for dropping:** Part B commits to AST chunking at rebuild. Spending 3 hrs on a concatenation work-around that gets replaced wholesale is waste. Accept mid-clause splits on the current corpus until greenfield ships. Two-pass extraction (A4) handles the symptom: if the fragment doesn't contain a complete quotable clause, the extractor will cite a different chunk.

### ~~A3 (original)~~ — Parent-document retrieval (chunk-boundary fix) — NOT DEPLOYING

**Problem found in audit:** IGST s.11 and Rule 86B provisions cross chunk boundaries. Child chunks contain fragments.

**Design:**
1. At ingestion time (or as a one-off pass over `cbic_v1`), add payload fields: `parent_section_id`, `parent_section_text_hash`, `subsection_order`.
2. At retrieval time: after ColBERT rerank, group hits by `parent_section_id`. For each unique parent, concatenate sibling chunks in `subsection_order` up to a token budget (~1500 tokens per parent). Pass the stitched parent to the LLM instead of the child fragments.
3. Fall back to child chunks when no parent grouping is available.

**Sentinel:** `p2_v1`.
**Risk:** medium — changes what the LLM sees. Requires careful eval.
**Time:** 2-3 hr code + re-ingest metadata (~30 min) + 20 min eval.

---

### A4. Option A — Two-pass structured extraction (THE structural fix)

Both external LLMs converged on this as the permanent fix for verbatim quoting.

**Design:**

**Pass 1 — Extraction (qwen3-14B, JSON-schema constrained via llama.cpp GBNF):**

Prompt:
```
SYSTEM: You are a CBIC legal-text extractor. For each sub-question below, find the single best supporting chunk from the provided context and copy its most relevant verbatim span. Return strict JSON only.

SCHEMA:
{
  "sub_answers": [
    {
      "sub_question": "<one of the provided sub-questions>",
      "cited_chunk_id": "<exact chunk ID from CONTEXT>",
      "verbatim_span": "<character-for-character copy from that chunk, 20-400 chars>",
      "conclusion": "<one-sentence answer derived from the span>"
    }
  ]
}

CONTEXT:
<chunks with IDs>

SUB-QUESTIONS (decomposed from user query):
1. ...
2. ...
```

**Gate (Python, no LLM):** For each `sub_answer`:
1. Verify `cited_chunk_id` exists in provided context.
2. Verify `verbatim_span` is a substring of `chunks[cited_chunk_id].text` (after NFKC + whitespace canon).
3. Fallback: 6-gram Jaccard ≥0.80 between span and chunk.
4. Fallback: BGE-M3 cosine ≥0.92 between span and chunk (catches canonicalization edge cases).
5. Drop failures; log them to `suspicious_spans[]` for the UI.

**Pass 2 — Synthesis (qwen3-14B, /no_think, temp=0):**

Prompt (chunks NOT re-fed):
```
SYSTEM: You are rendering verified legal facts into a professional advisory. Rules:
- Copy each verbatim_span EXACTLY as given. Do not paraphrase.
- Cite [S#] immediately after each quote.
- Do not introduce any legal claim not present in the verified facts.
- Format: one paragraph per sub-question. Final paragraph = overall conclusion stitched from per-sub conclusions only.

VERIFIED FACTS:
<validated JSON from Pass 1>

USER QUESTION:
<original question>
```

**Why this works:**
- Pass 1 is a narrow copy-task (easier than "reason + cite simultaneously").
- Gate is mechanical — bypasses LLM's paraphrase instinct structurally.
- Pass 2 has nothing to paraphrase from; chunks aren't in context.
- Context overhead: chunks tokenized once (Pass 1 only); Pass 2 sees ~500-1500 tokens of JSON.

**Files touched:**
- `api.py`: new `two_pass_generate(question, sub_questions, chunks)` replacing `_b17_multi_retrieve` → generate path.
- `storyformat.py`: two new prompts (`EXTRACTION_SYS`, `SYNTHESIS_SYS`), each <300 tokens.
- New `validator.py`: centralize substring + Jaccard + BGE-M3 cosine check logic.

**Sentinel:** `two_pass_v1`.
**Risk:** medium. Pass 2 drift (model inventing connective tissue between verified quotes) is the named risk; mitigation is strict SYS_PROMPT + eval harness gate on forbidden-words and invented sections.
**GBNF fallback:** if JSON mode is unreliable on qwen3-14B Vulkan, use a strict GBNF grammar that constrains output to the schema.
**Time:** 3-4 hr code + 30 min eval.

---

### A5. P3 — Query-class routing

**Goal:** Different prompts for definitional vs scenario vs reconciliation queries. Reduces prompt overhead per query class.

**Classes (detected by regex + keyword set):**
1. **Definitional** ("what does Section X say", "define composite supply"): skip decomposition, retrieve Act/Rules with hard filter, return quote-heavy single-paragraph answer.
2. **Scenario** (Quantum-Tech-style multi-fact): full decomposition + two-pass extraction.
3. **Reconciliation** (keyword-gated: GSTR-1, GSTR-3B, Rule 88C/D, DRC-01B, Section 75(12), Rule 86B): add the 8 calibration rules from `rag_reasoning_calibrations.md` to SYS_PROMPT for *this class only*. Use retrieval-side keyword-aware augmentation.
4. **Rate/Classification** (HSN, SAC, notification-specific): different retrieval (notifications collection, date-aware filter).

**Sentinel:** `p3_v1`.
**Risk:** low-medium — prompts are per-class, smaller than current combined SYS_PROMPT.
**Time:** 2 hr code + 20 min eval.

---

### A6. Embedding contrastive fine-tune (optional, later)

Only if P1+P2+A4 leaves retrieval gaps.

**Design:**
- Harvest (query, relevant-chunk) positive pairs from E10 feedback (upvoted answers) and from gold set.
- Harvest hard negatives: for each positive, top-5 retrieved chunks that weren't the target, minus any that are also relevant.
- Contrastive fine-tune BGE-M3 via sentence-transformers `MultipleNegativesRankingLoss`, 1 epoch, batch size 16.
- Target: 2-3k positive pairs minimum for meaningful shift.
- Hardware: rent Nvidia A100 for 2-4 hrs (AMD Vulkan is not practical for training).

**Time:** 1-2 weeks including data harvesting and off-rig training.
**Deferred:** not needed for professional-grade if A4 lands cleanly.

---

### A7. Feedback loop

**E10 downvote → eval corpus:**
- Every downvoted answer (with user-provided "why") becomes an eval case in `gold_set.yaml` after one round of expert review.
- Weekly eval run against accumulated gold + downvote cases.
- Regressions gate the next deploy.

**Time:** ongoing, ~2 hr/wk curation.

---

### A8. Deploy cadence (one dimension per deploy, each gated by eval)

| # | Feature | Dimension | Risk | Revert if eval shows |
|---|---|---|---|---|
| 0 | Baseline eval | measurement | none | — |
| 1 | P1: retrieval boost + forced routing | retrieval | low | aggregate down OR any category down >10 pct-pts |
| 2 | Re-ingest updated Rules + IGST Act | corpus | none | — |
| 3 | P2: parent-doc retrieval | retrieval | medium | context bloat, latency >15s p50, any regression |
| 4 | Option A: two-pass extraction | generation | medium | Pass-2 drift (invented refs), latency >20s p50 |
| 5 | P3: query-class routing | prompt+retrieval | low-medium | any class-specific regression |
| 6 | Embedding fine-tune | retrieval | medium | only if 1-5 didn't close the gap |

**Never mix dimensions in a single deploy.**

---

### A9. Infrastructure guarantees (already established, restating for completeness)

- Every deploy writes: `*.bak.<feat>.<ts>`, `*.patched.<feat>.<ts>`, `/opt/indian-legal-ai/patches/<feat>_<ts>/apply.py`, laptop mirror at `D:\_gpu_rig_ai\patches\<feat>_<ts>\`.
- systemd unit `cbic-rag-api.service` auto-restarts on crash, survives reboot, logs to journal.
- SSH flags: `-o ControlMaster=no -o ControlPath=none` (Lesson 8).
- Sub-agent prompts include all source code verbatim (Lesson 3, Lesson 10).

---

## PART B — From-scratch rebuild (greenfield, next corpus e.g. Income Tax Act + Rules + Circulars + Case Law)

Assume same hardware (4-core consumer rig, 3× AMD Vulkan GPUs, no ROCm, Qdrant, llama.cpp, no cloud). Assume we start 2026-05-01 with zero prior code.

### B0. Goals explicit from day 1

1. Verbatim-citation trust (every legal claim → substring-verified quote).
2. Handle definitional + scenario + reconciliation + rate queries equally well.
3. Temporal correctness (Section X as of year Y).
4. Reproducible ingestion (same recipe every time, zero rediscovery).
5. Eval harness exists before the first answer is generated.

### B1. Corpus design

**Source inventory:**
- Acts (consolidated, amended-to-date from MoF portal)
- Rules (same)
- Notifications (dated, with supersession chains)
- Circulars (dated)
- Case law (ITAT / HC / SC judgments; optional v2)
- Forms (for reference)

**Document versioning:**
- Every document carries `effective_from`, `effective_to` (null = current), `supersedes: [doc_id]`, `superseded_by: [doc_id]`.
- Historical versions are ingested (not replaced). Query-time temporal filter selects the version applicable as-of a user-specified date (default: today).

### B2. Parser

**Docling** (IBM) as primary. Structure-aware, handles tables, preserves hierarchy as JSON.
Fallback for scanned PDFs: **RapidOCR on GPU 4/6 Vulkan** (or rent Nvidia-hour for bulk OCR at ingest time).
NEVER: CPU tesseract, CPU RapidOCR (see Lesson 7).

### B3. Chunking (AST-style, section-aware)

**Rule:** a chunk is a single Subsection OR Clause OR Sub-clause, never smaller, never crossing a boundary. If a subsection is >2000 tokens, split at the nearest clause boundary.

**Hierarchy payload (mandatory, every chunk):**
```json
{
  "hierarchy_path": ["Income-tax Act, 1961", "Chapter IV", "Section 14", "Subsection 1", "Clause (a)"],
  "section_ref": "14(1)(a)",
  "section_canonical": "S.14(1)(a)",
  "doc_type": "act" | "rules" | "notification" | "circular" | "case",
  "doc_id": "itact:1961:14_1_a",
  "effective_from": "1962-04-01",
  "effective_to": null,
  "amendment_history": [{"by": "Finance Act 2005", "effective_from": "2005-04-01"}],
  "parent_section_id": "itact:1961:14",
  "subsection_order": 3,
  "cross_refs_outbound": ["S.10(13A)", "R.2A"],
  "page": 27,
  "pdf_path": "…"
}
```

**Chunk size target:** 400-800 tokens. Upper cap 1500.
**Overlap:** 0 (AST-based, overlap not needed at boundaries).
**Stitch on retrieval:** at query time, we can re-stitch siblings by `parent_section_id` + `subsection_order` when LLM needs a full section.

### B4. Embedding + indexing

**Two collections (per external LLM #2's advice):**
1. `statutes_primary` — Acts + Rules only. Optimized for exact-match: sparse weight 0.6, dense weight 0.4 in RRF. Smaller (thousands of chunks).
2. `regulatory_secondary` — Notifications + Circulars + (later) case law. Optimized for semantic: sparse 0.3, dense 0.7. Larger (tens of thousands).

**Embedder:** BGE-M3 (unchanged, validated for legal text).
**Sparse:** BM25 at ingestion, stored as Qdrant sparse vectors.
**Query flow:** route query-class → hit one or both collections → RRF across collections with per-collection weights → rerank (ColBERT) → MMR diversity cap → return.

### B5. Query-class router (built day 1, not bolted on)

Four classes, detected by regex + small keyword classifier:
1. **Definitional** → `statutes_primary` only, k=5, minimal rerank.
2. **Scenario** → both collections, decomposition on, k=12 each, full pipeline.
3. **Reconciliation** → both collections with reconciliation keyword augmentation, k=10.
4. **Rate/Classification** → `regulatory_secondary` with date filter, k=8.

### B6. Generation (two-pass from day 1)

Option A (extraction → validate → synthesize) is the default, not an afterthought.

### B7. Verifier (v1 includes all tiers)

Exact substring + NFKC canon + 6-gram Jaccard ≥0.80 + BGE-M3 cosine ≥0.92. All three; drop on any failure.

### B8. Eval harness (built before first answer)

- 100-Q gold set for the new corpus (20 per class, 50 basic / 30 intermediate / 20 complex).
- Runner, diff, regression gate — same shape as current `D:\_gpu_rig_ai\eval\`.
- **Hard rule: no deploy without green eval run.**

### B9. Infrastructure (zero re-learning)

- Patch paths persistent (never /tmp, never AppData\Local\Temp).
- systemd API with `Restart=on-failure`, journal logging, `MemoryMax`.
- SSH flags fixed.
- Sentinel-based version tracking on every patch.
- Backup + patched-mirror + patch-script for every deploy.

### B10. Observability (day 1)

- Every query logs: `retrieval_tier`, `class`, `sub_query_count`, `chunks_retrieved`, `quotes_emitted`, `quotes_verified`, `gate_pass_rate`, `latencies_ms`, `llm_tokens_in/out`.
- `/v1/meta` endpoint surfaces live sentinel, collection sizes, effective-as-of date.
- Downvote feedback (`E10`) captures `{query, answer, which_sub_answer_wrong, why}` → feeds back into eval.

### B11. What we'd NOT do differently

- BGE-M3 on Vulkan GPU 5 — still the right embedder.
- qwen3-14B-hermes generator — still appropriate for 14B-class trust-critical work, assuming two-pass.
- Qdrant — still the right vector DB.
- ColBERT + MMR — still the right rerank stack.

### B12. What we WOULD do differently vs CBIC

- Eval harness FIRST, not after 3 failed deploys.
- AST chunking from the start, not regex-retrofit.
- Two-pass extraction from the start, not after blind prompt iteration.
- Two collections, not one.
- Versioned documents with effective dates in payload, not flat text.
- Query-class router on day 1, not bolted on.
- `doc_type` payload field from day 1 (enables forced routing for free).
- Persistent patch paths from day 1 (no lost work).

---

## Questions for external LLM evaluators

### On Part A (fix-current)

**A-Q1.** Is the P1 → (re-ingest) → P2 → A4 → P3 order correct? Or should two-pass (A4) come BEFORE parent-doc retrieval (P2), since A4 is the structural fix and P2 is a retrieval quality bump?

**A-Q2.** The hard-filter approach in P1 (force `doc_type IN ('act','rules')` on explicit Section/Rule queries) — is there a risk of missing clarifying circulars that the user actually needs? Should the union with the unfiltered pass have a minimum representation of non-Act sources?

**A-Q3.** For Pass 2 of Option A: chunks are NOT re-fed. Is the risk of "Pass 2 invents connective tissue between verified quotes" real enough that we should re-feed chunks with a strict "cite from this set only" instruction? Or does re-feeding re-introduce the paraphrase instinct?

**A-Q4.** GBNF for JSON-schema enforcement in Pass 1 — is this reliable on qwen3-14B-hermes via llama.cpp Vulkan? Any known failure modes (infinite loops, mode collapse)?

**A-Q5.** For the `verbatim_span` length (we propose 20-400 chars): too short and the LLM might copy trivial phrases that pass the gate but don't constitute legal support. Too long and the LLM struggles to copy exactly. What's the sweet spot you've seen in production?

### On Part B (from-scratch)

**B-Q1.** Two collections (`statutes_primary`, `regulatory_secondary`) vs a single collection with strong `doc_type` filters — which is actually simpler in practice? We lean two-collection because it lets us set different RRF weights per collection, but it doubles operational surface area.

**B-Q2.** For Indian regulatory corpora with high amendment churn (especially Income Tax), is versioned-chunk-with-effective-dates the right model, or is it better to maintain a single "current" collection + a "historical" archive collection? Users rarely ask "as of 2018", but compliance scenarios sometimes need it.

**B-Q3.** AST chunking requires a section-regex parser per corpus. For Indian Acts these follow a fairly consistent pattern, but Notifications and Circulars don't. What's the right fallback? Semantic chunking for unstructured circulars + AST for Acts/Rules?

**B-Q4.** 100-Q gold set per new corpus — is that enough to gate deploys? Too many? How do you balance gold-set creation cost vs deploy-confidence coverage?

**B-Q5.** Case law (ITAT/HC/SC judgments) — we're leaving it out of v1. Is that the right call, or is case law foundational enough for professional-grade advisory that it has to be in from day 1?

### Meta

**M-Q1.** Anything *structural* missing from either plan?
**M-Q2.** Anything in Part A that can be dropped because it's redundant once Part B's philosophy is applied (i.e., if we rebuild eventually, what Part A work is wasted)?
**M-Q3.** Honest probability we hit "professional-grade trustworthy" (90%+ verbatim-gate pass rate on the 50-Q gold set, <2% hallucinated citations) after A0 → A4 ships cleanly?

---

## Appendix — Current asset inventory

- Live backend: `b11b12a8_v1` at `http://192.168.1.107:9500`
- Eval harness: `D:\_gpu_rig_ai\eval\` (50 Q/A, runner, diff, README)
- Gold set: `D:\_gpu_rig_ai\eval\gold_set.yaml`
- Memory files: `D:\...\.claude\projects\D---gpu-rig-ai\memory\{project_cbic_rag, ingest_playbook_cbic, optimization_playbook_cbic, optimization_plan_cbic, ocr_research_cbic, rag_reasoning_calibrations, lessons_learned}.md`
- Consultation briefs: `D:\_gpu_rig_ai\consults\cbic_rag_external_consult.md` (with training section), this file
- Patches: `D:\_gpu_rig_ai\patches\` (inline_cite_v2 shipped; b22 + b22b reverted, preserved for forensics)
- Rig persistent patches: `/opt/indian-legal-ai/patches/`
- Corpus: 108,802 chunks in `cbic_v1` Qdrant collection
- Models: qwen3-14B-hermes (GPU 2), BGE-M3 (GPU 5), OCR models parked at `/opt/ai-models/qwen25vl/`

---

*Please evaluate. Push back hard on anything that's wrong, hand-wavy, or optimistic.*
