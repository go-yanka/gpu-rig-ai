# CBIC RAG Re-Ingestion v2 — Plan for External Review

**Audience:** external reviewer LLMs (Claude Sonnet/Opus, Gemini 2.5 Pro, GPT-5) asked to critique this plan before we execute.
**Status:** Plan frozen 2026-04-23 **with amendments from two external reviews (Gemini, Grok) merged inline**. All blocking probes passed or have accepted fallbacks. Ready to build chunker v2 and execute.
**What we want from you:** see §11 "Questions for the reviewer" at the end. Focus your critique there. (Note: Round 1 + Round 2 already complete — amendments merged into §3, §4, §5, §7, §9. Future reviewers should flag *new* concerns, not re-address already-adopted items.)

---

## 1. Context

### The system
- **Project:** CBIC RAG — question-answering over the Central Board of Indirect Taxes & Customs (India) legal corpus: GST/Customs/Central Excise Acts, rules, notifications, circulars, forms, FAQs, judgments, schedules.
- **Corpus:** **15,776 PDFs** (v1 manifest total, `_manifest.sqlite` docs table), of which **851 are image-only and tagged `text_source=ocr`** — they're the OCR queue, ingested in a later pass once the OCR pipeline is thawed. This pass ingests the **~14,925 born-digital docs**. Bilingual (English primary + Hindi twins for many docs). Corrected 2026-04-23 — earlier drafts of this file quoted 851 as the whole corpus; that was wrong.
- **Current prod:** `/query` endpoint → Qdrant collection `cbic_v1` with 114,626 chunks. Was built in a rushed first pass that produced known defects (orphaned provisos, split tables, boilerplate duplication across monthly filings).

### The hardware
- GPU rig: 7-GPU (RX 6700 XT × 7) mining rig repurposed for AI. 4 CPU cores (confirmed bottleneck for IO-heavy ops). Vulkan, not ROCm, for inference (`llama-server-b8840` with `RADV_DEBUG=nodcc`).
- Models on rig (port-per-GPU): qwen3-14b Q4_K_M on GPU 2 port 9082 (our workhorse), qwen2.5-1.5b, qwen3.5-9b, qwen3.5-4b, llama-3.1-8b, qwen2.5-coder-7b, gemma-4-E4B-vision. BGE-M3 embedder pool on GPUs 0/1/4/5/6 (Vulkan, 30+ ch/s measured).
- External LLMs: Claude CLI (Max plan, unmetered) = primary extraction LLM. Gemini 2.0 Flash = judge + backup extractor. Paid tier budget ≤$10.

### The non-negotiable
**95% trust across 4 gates** (G1 Accuracy, G2 Reasoning, G3 Evidence, G4 Refusal — defined below). One gate <95% = spec is broken, fix spec and re-run from failed phase. No iterative patching.

### Why this doc exists
The v1 ingest was chunker-by-size only. That's how we ended up with "provided that..." clauses orphaned from their section context, tables mangled across boundaries, and ~30% duplicate chunks from shared form headers across monthly filings. v2 is a structure-aware rebuild with probe-validated decisions. We want external LLMs to tell us what we've missed before we burn 18–30 hours rebuilding the index.

---

## 2. Target architecture

### Ingest pipeline (one-way, resumable at phase boundaries)
```
PDFs (15,776 total; 851 OCR-deferred → 14,925 born-digital this pass)
  │
  ├─ [Phase 0] Pre-flight: snapshot cbic_v1, freeze code, verify OCR cache
  │
  ├─ [Phase 1] Manifest + bilingual twin linking
  │         → ingest_manifest_v2.sqlite
  │
  ├─ [Phase 2] Chunker v2 (TWO-PASS, see §4)
  │     ├─ Pass 1: Claude CLI classifies each doc → chunking_plan.json
  │     └─ Pass 2: rule-driven splits (R1–R7), emits canonical chunks
  │         → ChunkDeduper (SHA256 NFKC+WS+lowercase) → ~78k canonical
  │
  ├─ [Phase 3] Dense embed: BGE-M3 pool on 5 GPUs, 1024-d, batch 32
  ├─ [Phase 4] Sparse embed: fastembed BM25 (Qdrant/bm25 model)
  ├─ [Phase 5] Upsert to new collection cbic_v2 (keep cbic_v1 as rollback)
  │
  ├─ [Phase 6] Shadow-mode cutover: /query → cbic_v1, /query_v2 → cbic_v2,
  │           dual-writer logs diffs
  │
  ├─ [Phase 7] 4-gate validation on 350–400 gold + 50 adversarial OOC
  │
  └─ [Phase 8] Promote (flip /query to cbic_v2) OR amend spec, re-run from
              failed phase
```

### Query pipeline
```
Query
  │
  ├─ Router (keyword-first; miss → qwen3-14b LLM router; miss → "gst" default)
  │
  ├─ Filter: topic_tags + category + lang
  │
  ├─ Hybrid retrieval: dense (BGE-M3) + sparse (BM25) + Qdrant RRF fusion,
  │  k=20
  │
  ├─ Rerank: BGE-reranker-v2-m3 cross-encoder → top-10
  │
  ├─ Refusal check: if top-10 max score < θ_retrieve → refuse (G4)
  │
  ├─ Selective context compression (§3.1): for each top-10 chunk, keep top-3
  │  sentences by dense-sim to query + parent-hierarchy breadcrumb. Hindi
  │  twin IDs surface in metadata only; LLM sees English text. 40–60%
  │  prompt reduction.
  │
  └─ Answer gen: qwen3-14b /no_think, must_cite_verbatim enforced (G3),
     UPL/arithmetic refusal rule enforced (G4)
```

---

## 3. Decisions already locked (D1–D15)

Each decision cites the probe or explicit user directive that locked it. External reviewers should **not** re-debate these — only flag if you see a decision contradicting its evidence.

| # | Decision | Answer | Evidence |
|---|----------|--------|----------|
| D1 | LLM for extraction & Pass-1 classification | **Claude CLI primary** (V18 PASS: 50/50 JSON parse, p50=3.38s, p95=4.58s); qwen3-14b fallback (V2b: 30/30 parse, p50≈6s, p95≈7s — slower but viable at 85 min total) | V18, V2b |
| D2 | Chunk size policy | target 3500 / cap 5500 / ceiling 8000 (tables only) / floor 500 / overlap 700 mid-section, 0 at boundaries | confirmed |
| D3 | Reranker | BGE-reranker-v2-m3 cross-encoder (current, works) | user directive |
| D4 | Hindi handling | Retrieve across Eng + Hindi twins; **LLM prompt receives English chunks only** (Hindi twin IDs surface in response metadata for UI citation); §3.1 selective context compression | user + review amendment 2026-04-23 |
| D5 | Gemini budget cap | ~$6.56 paid tier (removes free-tier RPM throttle) | V9 PASS |
| D6 | Out-of-corpus behavior | Refuse (per §4 inc. UPL/arithmetic rule) | user directive + review 2026-04-23 |
| D7 | Router fallback chain | keyword → qwen3 LLM router → `gst` default; **query sanitization prefixed** (strip ctrl chars, length cap 2000, drop injection markers) | user + review |
| D8 | Gold eval set size | **350–400 queries** (5–7 per topic × 60 tags) | review adoption 2026-04-23 (was 170) |
| D9 | Evaluation frequency | every phase gate + continuous shadow-mode | user directive |
| D10 | Cleanup policy | Never delete during project; write to `cleanup_backlog.md` | user directive |
| D13 | Cutover strategy | Shadow mode with real testers + dual-writer log | user directive |
| D14 | Bilingual citation format | Cite both English + Hindi twin in UI/response metadata; **LLM reasoning context uses English chunks only** (see D4, §3.1) | user + review amendment 2026-04-23 |
| D15 | Formal doc versioning | Not needed — git tags suffice | user directive |

D11/D12 intentionally skipped (consolidated into D1/D5).

---

## 4. Chunking strategy (the heart of v2)

Full spec: `~/.claude/projects/D---gpu-rig-ai/memory/chunking_strategy_cbic_v2.md`. Summary here.

### Principle
A chunk that's "the right size" but semantically incomplete is worse than a larger one. **Never slice a document without first understanding what it is.**

### Pass 1 — Document understanding (LLM-assisted, per doc)
One LLM call per doc (Claude CLI primary, qwen3 fallback, Gemini second-opinion on low-confidence). Produces `chunking_plan.json`:
```json
{
  "doc_type": "act|rules|notification|circular|form|faq|judgment|schedule|press_release|mixed",
  "structure": "hierarchical_sections|flat_paragraphs|tabular|form_fields|list_of_items|mixed",
  "primary_splitter": "section|rule|chapter|heading|paragraph|table_row|page",
  "critical_units": ["section","proviso","explanation","definition","table","schedule","annexure","form_field_block","footnote"],
  "hard_boundaries": [{"regex_or_marker":"...","never_cross":true}],
  "table_regions": [{"page_start":5,"page_end":7,"reason":"Rate schedule","confidence":0.0-1.0}],
  "has_amendments": true|false,
  "hierarchy_depth": 1-6,
  "language": "en|hi|bilingual",
  "confidence": 0.0-1.0,
  "notes": "free text for edge cases"
}
```
Budget: **14,925 docs × ~6s qwen3-14b ≈ 24.9 h** (D1 flipped to qwen3-14b primary 2026-04-23 after Claude CLI hit 40% JSON-fail rate on production smoke). ~2% low-confidence docs may get Gemini second-opinion (~300 calls, <$3).

### Pass 2 — Rule-driven splits (code-enforced, unit-tested)
- **R1 Tables atomic:** never split mid-row; table = one chunk up to 8000 chars. If >8000: **(1) prefer sub-table detection via repeated header patterns** (e.g. Chapter 84 vs 85 blocks in a GST rate schedule); **(2) else row-boundary split**. Every split chunk prepends column-header row + last-seen hierarchical parent row to preserve context. Tagged `is_table=true, table_part=k/N`.
- **R2 Critical units whole:** never split inside proviso/explanation/definition/form_field_block/footnote. If target size would cut inside, extend up to ceiling (5500), else split the *parent* section before the unit starts.
- **R3 "Unusable cut" validator (post-split):** reject any chunk whose first non-whitespace token is an English connector (`Provided|Except|However|For the purposes of this|Explanation|Illustration|Notwithstanding|Subject to|Where|In case`) or a Hindi connector (`बशर्ते कि|परंतु|किंतु|तथापि|स्पष्टीकरण|व्याख्या|परिभाषा|उदाहरणार्थ|दृष्टांत|अपवाद|इसके बावजूद|परंतुक`) or a bare lowercase verb. On reject, pull split back 200 chars, retry max 3×, else merge into previous.
- **R4 Hierarchy-aware splits:** prefer chapter/part/schedule > section/rule/clause > sub-clause > numbered para > sentence > word (last resort). Primary splitter = `chunking_plan.primary_splitter`.
- **R5 Overlap policy:** 700 chars (20%) for mid-section narrative splits; **zero** at section boundaries; **zero** at table↔non-table transitions.
- **R6 Size targets:** target 3500 / cap 5500 (soft) / ceiling 8000 (hard — atomic tables or merged-to-preserve-unit only) / floor 500 (final sub-500 chunk merges into previous).
- **R7 Payload additions:** `chunking_plan_used: bool`, `chunking_rule_triggered: list[str]` (for audit), `is_table`, `table_part`.

### Per-chunk payload
`chunk_id` (sha256), `doc_id`, `sha256`, `source`, `category`, `subcategory`, `lang`, `text`, `embed_text` (text + parent hierarchy prefix), `section_ref`, `parent_hierarchy_text` (breadcrumb), `chunk_type`, `is_table`, `page_range`, `effective_date`, `text_source` (born|ocr), `hindi_twin_chunk_ids`, `topic_tags` (multi-label, 60 topics / 5 categories from rule-based tagger — V20 PASS), `also_appears_in` (canonical absorbs duplicates), `dup_of_chunk_id`, chunker audit fields.

### Self-tests (must pass before chunker v2 declared ready)
- T1: Table region never split mid-row
- T2: Proviso block stays inside parent
- T3: Explanation block stays inside parent
- T4: "Provided that" never appears as chunk-start token
- T5: Section-start split has zero overlap
- T6: Mid-section split has 700 overlap
- T7: Final chunk <500 chars merges into previous
- T8: Bilingual linker matches ≥90% on a 10-doc sample

---

## 5. Four gates (95% trust definition)

| Gate | What | Metric | Threshold |
|------|------|--------|-----------|
| **G1 Accuracy** | Right chunks retrieved | recall@10 on **350–400 query gold set** (5–7 per topic × 60 tags) | ≥95% |
| **G2 Reasoning** | Answer logic sound | **Dual-judge ensemble** (Gemini 2.0 Flash + Claude CLI); disagreement >1 pt flagged for human review | avg ≥4.5; ≥95% of queries ≥4 on both judges |
| **G3 Evidence** | Citations verifiable | `must_cite_verbatim` substring match → if miss, normalized Levenshtein ≥0.95 fallback (NFKC + ws-collapse + lowercase) | ≥95% queries citation found |
| **G4 Refusal** | No hallucination when OOC | **50 adversarial OOC queries** across 6 attack classes + **UPL/arithmetic refusal rule** (system refuses numeric calculations on user figures; returns rule/rate/method only) | 100% must refuse |

---

## 6. Probe evidence (V1–V24)

Full matrix: `reingest_spec/PROBES.md`. High-level: **10 PASS, 3 recovered-post-reboot (V1 PASS; V2 retired → V2b qwen3 fallback confirmed; V10 unchecked), V17 Gemini-judge stability blocked on rate-limits but no longer on critical path (D1 = Claude CLI).** No probe failure blocks Phase 1.

Key PASS probes:
- **V4** BM25 Hindi: Hindi nonzero count within 30% of English — Devanagari-safe.
- **V14** `must_cite_verbatim`: gold set valid, substring check works end-to-end.
- **V18** Claude CLI latency+parse: 50/50 @ p50=3.38s p95=4.58s — D1 locked.
- **V20** Topic tagger coverage: 57/57 gold-referenced topics have ≥10 chunks in corpus.
- **V21** ChunkDeduper: 31.7% reduction on real sample, NFKC+WS+lowercase SHA256 stable.
- **V2b** qwen3 Pass-1 fallback: 30/30 JSON parse, p50≈6s (viable fallback, not primary).

---

## 7. Risks (R1–R15)

| # | Risk | Mitigation | Status |
|---|------|------------|--------|
| R1 | OCR table structure lost in flat text | V8 table-aware re-OCR with Gemini; Pass 1 `table_regions` flags re-OCR candidates | Planned |
| R2 | Section-ref regex misses on OCR noise | D1 LLM backstop (Claude CLI); regex stays primary | Locked |
| R3 | Multi-GPU embedder pool hangs long-run | V5 1-hr stability soak before Phase 3 | Pending |
| R4 | Qdrant disk exhaustion during dual-collection | V7 headroom check ≥2× | Pending |
| R5 | qwen3 extraction too slow | Claude CLI primary per D1; qwen3 only as fallback (V2b p50≈6s viable) | Resolved |
| R6 | θ_retrieve threshold brittle | V16 distribution analysis → per-category threshold | Pending |
| R7 | Adversarial set too easy | 50 adversarial OOC across 6 attack classes (direct-tax traps, fake sections, real-section-fake-context, encoding attacks, prompt injection, UPL/arithmetic) | Pending |
| R8 | Cross-doc boilerplate pollutes recall | V21 ChunkDeduper with `also_appears_in` audit trail | Locked |
| R9 | Gold queries reference topics not discoverable | V20 rule-based topic_tagger 60 topics / 5 cats | Locked |
| R10 | qwen3 RADV driver degradation over time | Known-good configs locked; reboot-recovery recipe + VRAM/throughput watchdog planned | Locked |
| R11 | Judge sycophancy (G2 single-judge bias) | Dual-judge ensemble Gemini + Claude CLI; >1pt disagreement → human review | Locked |
| R12 | OCR noise corrupts citation verbatim match | G3 Levenshtein ≥0.95 fallback after NFKC+ws-collapse+lowercase | Locked |
| R13 | UPL exposure (user sues on numeric answer) | G4 UPL rule: refuse arithmetic on user figures; return rule/rate/method only | Locked |
| R14 | Shadow-mode cache coherency across /query and /query_v2 | Per-endpoint cache namespace + version key; dual-writer divergence >2% = kill switch | Locked |
| R15 | Context bloat from Hindi twins blows prompt budget | D4/D14 English-only into LLM + §3.1 selective context compression (top-3 sentences/chunk) | Locked |

---

## 8. Components to build (B-1 through B-7)

| ID | Component | Acceptance criteria | Owner |
|----|-----------|--------------------|-------|
| B-1 | `chunker_v2.py` | Implements R1–R7; passes T1–T8 self-tests | us |
| B-2 | `ingest_v2.py` orchestrator | Drives Phase 1→5; resumable at phase boundaries | us |
| B-3 | Post-split validator (R3) | Standalone callable; logs `chunking_rule_triggered` | us |
| B-4 | `/query_v2` endpoint | Same contract as `/query`, backed by `cbic_v2` | us |
| B-5 | Dual-writer middleware | Shadow mode logs diffs to `shadow_log.sqlite` | us |
| B-6 | θ_retrieve tuner | V16 distribution analysis → per-category threshold | us |
| B-7 | cbic_v2 Qdrant snapshot script | Automated post-Phase-5 backup | us |

---

## 9. Out of scope (do not suggest these)

- **Agentic retrieval loops** (retrieve→reflect→re-retrieve). Scope too large for v2; latency cost not justified.
- **Graph RAG / Knowledge Graph.** CBIC corpus is hierarchical (Acts/Rules/Notifications), not relational. No upside for the work involved.
- **HyDE (Hypothetical Document Embeddings).** Redundant with our hybrid + RRF + CE rerank stack.
- **ColBERT / ColPali / multi-vector indexing.** Cross-encoder rerank is already strong; added complexity not justified.
- **Ekimetrics adaptive-chunking full framework** (4 splitters per doc + winner selection). Too expensive at 14,925 docs; our Pass-1 LLM judgement is domain-better. We *do* adopt their 5-metric post-ingest sample audit (see §10).
- **Re-chunking for post-v2 enhancements.** All post-v2 items (§10) are query-time; none require re-ingest.
- **New model downloads.** We use what's already on the rig.
- **ROCm instead of Vulkan.** Tried; Vulkan is faster and stable. Don't suggest.

---

## 10. Post-v2 backlog (not blocking, evaluated and parked)

Advanced-RAG techniques we considered and where they land:

| # | Item | When we'd build it | Ingest dep |
|---|------|--------------------|------------|
| B-post-1 | Multi-Query / Query Rewriting (Claude CLI expands query → 3 paraphrases → RRF union) | If G1 recall@10 misses θ after cutover | None |
| B-post-2 | Step-Back prompting + Query Decomposition | Compound-query failures observed in real logs | None |
| B-post-3 | Self-RAG-lite: qwen3 emits `{answer, confidence, missing_evidence}` → route low-confidence to G4 refusal | Fold into G4 refusal logic post-cutover | None |
| B-post-4 | Query cache (SHA256 normalized_query → answer, TTL 24h) | Any time post-cutover | None |
| B-post-5 | Observability dashboard (per-stage latency, RRF weights, CE scores, refusal rate) | Any time | None |
| B-post-6 | Ekimetrics 5-metric sample audit (SC/BI/RC/DCC/ICC on 200-chunk sample post-ingest) | Post-Phase-5 validation | Consumes ingest output, doesn't gate it |
| B-post-7 | Amendment-graph overlay (link `has_amendments=true` chunks to parent + temporal edges) | If users hit stale-amendment answers in shadow logs | None — payload flag already emitted |
| B-post-8 | Feedback rate-limit + abuse guard on `/query_v2` thumbs-up/down | Pre-GA if public | None |
| B-post-9 | BM25 Hinglish tokenizer tuning (transliterated Hindi in English queries) | If V4 Hindi-recall gap reappears on real traffic | None |
| B-post-10 | TTFT (time-to-first-token) streaming for long answers | User-experience pass post-cutover | None |
| B-post-11 | Per-difficulty θ_retrieve (factual vs. analytical query types) | If V16 per-category still too brittle | None |

---

## 11. Questions for the reviewer

Please focus critique on these. Generic feedback on architecture is less useful than specific answers to these questions.

### On chunking
1. **Pass-1 schema (§4).** Is the `chunking_plan` schema complete for the doc-type list shown? Any field you'd need to see to chunk correctly that we're missing? Example edge case: a mixed-doc PDF (circular + annexed form + FAQ).
2. **R3 unusable-cut token list** (`Provided|Except|However|For the purposes|Explanation|Illustration|Notwithstanding`). Given Indian legal drafting conventions, are we missing any connectors that would create orphaned clauses? Specifically, any Hindi equivalents we should pattern-match on Hindi twin docs?
3. **Table handling (R1).** We treat tables as atomic up to 8000 chars, then split on row boundaries. Are there Indian tax doc table patterns (e.g., multi-level rate schedules with merged cells) where row-boundary splits still break meaning?
4. **Hindi twin linking (§4, §SPEC Phase 2).** We match by hierarchy path after chunking each side independently. Target: ≥90% match on 10-doc sample (T8). Is this reliable enough, or should we use cross-lingual embedding similarity as a fallback matcher?

### On retrieval
5. **Hybrid RRF (dense 1024 + BM25 sparse).** For an Indian legal corpus with heavy English + transliterated Hindi + Sanskrit legal terms, is BGE-M3 + fastembed BM25 sufficient? Any reason to prefer e5-mistral, jina-v3, or a domain-specific embedder?
6. **Cross-encoder reranker (BGE-reranker-v2-m3).** Works on our current system. Any reason this underperforms on bilingual CBIC content vs. alternatives?
7. **θ_retrieve refusal threshold** is planned per-category (V16 pending). Reasonable, or should we use per-query-type (factual vs. analytical)?

### On the 4 gates
8. **G1 recall@10 ≥95% on 350–400 gold queries** (5–7 per topic × 60 tags, per review adoption). Is this distribution right, or should certain high-risk topics (e.g. ITC, e-way bill, anti-profiteering) be over-sampled?
9. **G2 dual-judge ensemble (Gemini 2.0 Flash + Claude CLI), ≥4.5 avg, >1pt disagreement flagged for human review.** Any failure mode of this ensemble you'd predict for legal reasoning traces?
10. **G3 `must_cite_verbatim` + Levenshtein ≥0.95 fallback (NFKC+ws-collapse+lowercase).** Acceptable, or is there a failure mode of Levenshtein at 0.95 on legal text we should anticipate?
11. **G4 refusal on 50 adversarial OOC across 6 attack classes** (direct-tax traps, fake sections, real-section-fake-context, encoding attacks, prompt injection, UPL/arithmetic-bait). Any 7th class we're missing?

### On operations
12. **Shadow-mode cutover (D13) + dual-writer.** Any failure modes of running both `/query` and `/query_v2` that we haven't anticipated? Specifically, are there cache coherency issues we should handle?
13. **RADV driver degradation (R10).** We've locked a known-good qwen3-14b systemd unit and a reboot-recovery recipe. Is there a preventive watchdog (VRAM metric, throughput metric) we should add to catch degradation before it causes incidents?
14. **No-re-ingest guarantee.** We claim all post-v2 items (§10) are query-time and don't require re-ingest. Do you see anything in that list that actually would?

### On anything we've missed
15. **Attacks / safety gaps.** For an Indian tax advisory system, what safety concerns should we address that aren't in the 4 gates? (User is a practicing tax consultant; end-consumers are taxpayers.)
16. **Any decision in §3 (D1–D15) that you think contradicts its cited evidence?**

---

## 12. Reference files (for deeper reading)

- `reingest_spec/SPEC.md` — frozen spec (this doc's source of truth for phases/payloads)
- `reingest_spec/READINESS.md` — runtime execution playbook with exact commands + failure response matrix
- `reingest_spec/PROBES.md` — V1–V24 probe specs and pass/fail status
- `reingest_spec/JOURNAL.md` — decision log / chronological
- `~/.claude/projects/D---gpu-rig-ai/memory/chunking_strategy_cbic_v2.md` — full non-negotiable chunking rules
- `~/.claude/projects/D---gpu-rig-ai/memory/project_cbic_reingest_v2.md` — proven-components registry + backlog
- `~/.claude/projects/D---gpu-rig-ai/memory/known_good_configs.md` — model ports, flags, systemd units, recovery recipes

---

## 13. Amendments — 2026-04-24 session (for reviewer awareness)

Three hard decisions were codified this session. All are additive; none reverse prior commitments.

### 13.1 — Intermediate GST50 scale test (RUNBOOK Stage M)

We inserted a 50-document GST-category-only scale test between the 10-doc smoke and the full 14,925 ingest. Purpose: prove the recipe (chunker_v2 + BGE-M3 Vulkan pool + dense-only retrieval + unified pair corpus) holds at 5× smoke scale with realistic category diversity, before committing full-corpus compute.

**Rationale:** 10-doc smoke passed G1/G3 at 0.9833, but θ came back INFEASIBLE because the adversarial set was authored for 14,925 scope. Scaling to 50 GST docs widens the in-scope coverage so gold vs adversarial score distributions separate.

**Deliverable:** all 5 gates at ≥0.95 on `cbic_v2_gst50` collection = Go for full 14,925 ingest. Any gate <0.95 = No-Go, iterate.

### 13.2 — No-CPU invariant (SPEC §11, RUNBOOK Stage 0.6)

Explicit rule: no ML inference on CPU anywhere in the active pipeline. Enforced by (a) env vars (`DENSE_ONLY=1`, `SKIP_SPARSE=1`), (b) hard-guards in `ingest_v2.py` preflight and `recovery_worker.py` import-time, (c) systemd Environment= for the API.

**Rationale:** 2026-04-24 discovered via failed 50-doc launch that `ingest.py:embed_batch` called fastembed (CPU) unconditionally. User hard rule: GPU only for ML. Independent agent audit confirmed clean after patches.

**Implication for reviewer:** sparse BM25 is no longer in the default pipeline. Dense-only retrieval hit G1=0.9833 on the smoke — sparse may be permanently dropped or reintroduced only on GPU.

### 13.3 — Unified training-pair schema (SPEC §10, RUNBOOK Stage C.5)

Single append-only `cbic_pairs_v2.jsonl` superseding the two pre-existing legacy formats. Every pair generator + every gate evaluator writes/updates in this shape. Legacy files never deleted — migrator reads, canonical file accumulates.

**Rationale:** inventory discovered two divergent formats (Format A flat 3,137 rows, Format B chunk-envelope 2,644 rows). Claude was about to invent a third. User directive: reconcile, don't fragment. The corpus is now a training-dataset asset independent of the RAG system.

**New review question (for reviewer):** does the schema at `memory/pair_schema_cbic_v2.md` carry enough provenance to support (a) BGE-M3 contrastive fine-tune, (b) bge-reranker fine-tune, (c) qwen3-14b LoRA SFT, (d) DPO pair training, when the corpus reaches ~150k rows at full scale? Missing fields?

