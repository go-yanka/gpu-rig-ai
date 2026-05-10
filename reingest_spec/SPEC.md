# CBIC Re-Ingestion Spec v2 — Frozen Execution Plan

**Goal:** Single-pass re-ingestion of **15,776 CBIC PDFs** (v1 manifest total; 851 are image-only/OCR and will be tagged `text_source=ocr` and deferred until OCR thaw per LESSONS_APPLIED row 9 — net ingestable now ≈14,925 born-digital docs) that achieves **≥95% trust** across four gates: Accuracy, Reasoning, Evidence, Refusal. No iterations. No "we'll fix it next pass."

**Source-of-truth for corpus size:** `sqlite3 /opt/indian-legal-ai/data/scraped/cbic/_manifest.sqlite "SELECT COUNT(*) FROM docs"` → 15,776 (verified 2026-04-23). The `851` number that appeared in earlier drafts was the **OCR queue** (`/opt/indian-legal-ai/data/ocr_targets_851.tsv`), not the ingest target. Corrected 2026-04-23; see JOURNAL entry "CRITICAL CORRECTION: corpus size was wrong".

**Status:** v2 frozen 2026-04-23 after probe wave + two external reviews (Gemini, Grok). 10 PASS, V1/V2b/V10 recovered post-reboot (qwen3 viable as fallback only; D1 primary = Claude CLI per V18 PASS), V17 de-prioritized (Gemini rate-limit; not on critical path since D1 resolved). Review-driven amendments locked: §1 gate thresholds, §3.1 selective context compression, §4 UPL rule, chunking Hindi R3 tokens, §5 D14 amended. See JOURNAL.md § 2026-04-23 evening + review adoption log.

---

## 1. Definition of 95% Trust (Four Gates)

| Gate | What it measures | Metric | Pass threshold |
|------|------------------|--------|----------------|
| **G1 Accuracy** | Right chunks retrieved | recall@10 on **350–400 query gold set** (5–7 per topic across 60 tags) | ≥95% |
| **G2 Reasoning** | Answer logic sound | **Dual-judge ensemble** (Gemini 2.0 Flash + Claude CLI), 1–5 on reasoning trace; disagreement >1 pt → flag for human review | avg ≥4.5; ≥95% queries ≥4 on both judges |
| **G3 Evidence** | Citations verifiable | `must_cite_verbatim` **substring match primary → normalized Levenshtein ≥0.95 fallback** (OCR-tolerant) | ≥95% queries citation found |
| **G4 Refusal** | No hallucination when OOC | **50 adversarial queries** (direct-tax traps + fake sections + real-section-fake-context + encoding attacks + prompt injection) → must refuse. **Plus UPL/arithmetic rule:** system refuses all numeric calculations on user-provided financial figures; returns rule/rate/method only | 100% refused |
| **G5 Latency/Cost** *(added 2026-04-26, supplementary)* | Per-query wall-clock + cost | grounded `/query` p95 latency; avg cost/query | **p95 ≤ 15s** *(amended 2026-04-26 from 8s — qwen3-14b single-slot floor + groundedness 3.3s; multi-GPU helps throughput not latency)*; cost ≤ $0.01/q (local qwen3 = $0) |

All four 95%-trust gates (G1–G4) must pass on first ingestion pass. One fail = the spec is broken and we fix the spec before re-running. G5 is operational, not a 95%-trust gate.

---

## 2. Data Pipeline (Phase 0 → 8)

### Phase 0 — Pre-flight (non-destructive, <30 min)
- Snapshot current Qdrant `cbic_v1` (keep as rollback; do NOT delete per D10)
- Write `/opt/indian-legal-ai/cleanup_backlog.md` listing everything to delete *after* project complete
- Freeze code: git tag `reingest-v1-start`
- Confirm 851 OCR `.txt` files present in `/opt/indian-legal-ai/data/ocr_cache/` (OCR queue — these are the image-only PDFs, they will be ingested *later* when OCR pipeline is live; this pass covers the ~14,925 born-digital docs)
- Confirm all 24 probes (V1–V24) pass or have accepted workarounds

### Phase 1 — Ingestion input prep
- Build unified doc registry: `/opt/indian-legal-ai/data/ingest_manifest_v2.sqlite`
  - Columns: `doc_id, sha256, source, category, subcategory, lang, path, text_source (born/ocr), pages, title, effective_date, hindi_twin_sha256`
- Link bilingual pairs (D4): for each English doc, scan manifest for Hindi twin by (category, subcategory, title-match) → store in `hindi_twin_sha256`
- Input text file per doc: `/opt/indian-legal-ai/data/ingest_text/{sha256}.txt` (symlink born-digital `.txt` extract; copy OCR `.txt` verbatim)
- **Chunk-level deduplication (NEW, per V21):** after chunker emits all chunks, run `reingest_spec/dedupe_chunks.py :: ChunkDeduper`
  - Canonical form: Unicode NFKC → collapse whitespace → lowercase → SHA256
  - On collision, keep first occurrence; append `{doc_id, source_chunk_id}` to its `also_appears_in` list; set `dup_of_chunk_id` on the skipped chunk (not upserted)
  - Expected: ~31.7% reduction (114,626 → ~78,291) — real boilerplate like shared CBIC form headers

### Phase 2 — Chunking (new chunker v2, TWO-PASS, structure-aware)

Full non-negotiable spec: see `~/.claude/projects/D---gpu-rig-ai/memory/chunking_strategy_cbic_v2.md` (rules R1–R7, failure modes, self-tests T1–T8). Summary below.

**Pass 1 — Document understanding (per doc, LLM-assisted)**
- Call Claude CLI (D1) with doc metadata + first 2000 chars + last 1500 chars + TOC
- Emit `chunking_plan` JSON: `doc_type, structure, primary_splitter, critical_units, hard_boundaries, table_regions (with per-region confidence), has_amendments, hierarchy_depth, language, confidence`
- Ambiguous cases (confidence <0.6): queue for Gemini second-opinion; if they disagree on `doc_type`, flag for manual review
- Budget: **14,925 × ~6s (qwen3-14b primary as of D1 flip) ≈ 24.9 h** wall (D1 flipped from Claude CLI to qwen3-14b on 2026-04-23 after Claude CLI delivered 40% invalid-JSON on production smoke; qwen3-14b is local on GPU 2 port 9082, proven by V2b)

**Pass 2 — Actual chunking (rule-driven, size-bounded)**
- **Target** 3500, **cap** 5500 (soft), **ceiling** 8000 (hard — atomic tables/merged-to-preserve-unit only), **floor** 500
- **R1: Tables atomic** — never split mid-row; table region = one chunk up to 8000 chars. If >8000: prefer sub-table detection via repeated header patterns; else row-boundary split. Every split chunk prepends column-header row + last-seen hierarchical parent row (e.g. "Chapter 84") to preserve context
- **R2: Critical units whole** — never split inside proviso/explanation/definition/form_field_block/footnote
- **R3: "Unusable cut" validator** — reject if chunk starts with `Provided|Except|However|For the purposes|Explanation|Illustration|Notwithstanding|bare lowercase verb`; pull split back 200 chars, retry max 3×, then merge
- **R4: Hierarchy-aware splits** — chapter/schedule > section/rule > sub-section > numbered para > sentence > word (last resort)
- **R5: Overlap policy** — mid-section narrative 700 chars (20%); section-start splits ZERO overlap; table transitions ZERO overlap
- **R6: Size targets** — as above
- **R7: Payload additions** — `chunking_plan_used: bool`, `chunking_rule_triggered: list[str]`, `is_table`, `table_part`
- **Emit fields per chunk:**
  - `chunk_id` (sha256 of text+doc_id+offset)
  - `doc_id`, `sha256`, `source`, `category`, `subcategory`, `lang`
  - `text` (raw chunk)
  - `embed_text` (chunk + parent hierarchy header prefix — V11 must confirm format)
  - `section_ref` (e.g. "Section 16(2)(c)") — regex-extracted or LLM-backstopped (D1)
  - `parent_hierarchy_text` (breadcrumb: "Chapter V > Section 16 > Sub-section 2 > Clause c")
  - `chunk_type` (section|proviso|table|schedule|preamble|notification|circular_para)
  - `is_table` (bool)
  - `page_range` (e.g. "12-13")
  - `effective_date` (if parseable from doc)
  - `text_source` (born|ocr)
  - `hindi_twin_chunk_ids` (list, if bilingual pair exists at same hierarchy)
  - **NEW** `topic_tags` (list[str], e.g. `["gst:input_tax_credit", "gst:reverse_charge"]`) — multi-label from `reingest_spec/topic_tagger.py` (60 topics across 5 categories; V20 verifies 57/57 gold-referenced topics have ≥10 chunks).
  - **NEW** `also_appears_in` (list[{doc_id, source_chunk_id}]) — populated by ChunkDeduper for canonical chunks that absorb duplicates
  - **NEW** `dup_of_chunk_id` (str | null) — only set on skipped duplicates (these are NOT upserted; retained in manifest for audit)
- **OCR table handling (V8):** Gemini re-OCR of table pages with table-aware prompt → preserve as markdown tables, mark `is_table=true`, do NOT split across chunks if <5500 chars

### Phase 3 — Dense embedding
- Multi-GPU BGE-M3 pool via `embedder_direct.py`, GPUs 0,1,4,5,6 (per V5 stability probe)
- Batch size 32, dim 1024
- Re-verify rate from last session (30+ ch/s achieved)
- **Ingest-time dense embedder MUST be `llama-cpp-python` Vulkan, in-process, via the `embedder.py` facade carrying the `_FACADE_VERSION = "direct-v1"` sentinel.** Ollama is FORBIDDEN for ingest (silent CPU fallback → zero-vectors on this rig; see 2026-04-23 incident in JOURNAL). Preflight script (`reingest_spec/preflight.sh`) greps the sentinel + runs a real 1-doc end-to-end dry-run before any phase. Rationale codified in §3.5 below.

### Phase 4 — Sparse embedding
- `fastembed` Qdrant/bm25 — same as current
- V4 confirms Hindi handling

### Phase 5 — Qdrant upsert
- New collection: `cbic_v2` (keep `cbic_v1` as rollback)
- Payload schema locked in Phase 2; Qdrant payload indexes on:
  - `category`, `subcategory`, `source`, `lang`, `section_ref`, `text_source`, `doc_id`, `chunk_type`, **`topic_tags`** (keyword index for multi-label filter)
- Hybrid config: dense 1024 + sparse bm25

### Phase 6 — API cutover (shadow mode — D13)
- Keep current `/query` → `cbic_v1` unchanged
- Add `/query_v2` → `cbic_v2`, same contract
- Dual-writer middleware: every inbound to `/query` also fires `/query_v2` async
- Log both responses + diffs to `/opt/indian-legal-ai/data/shadow_log.sqlite`
- Real testers exercise `/query_v2` directly; automated dual-write gathers corpus

### Phase 7 — 4-gate validation (see §1 for gate thresholds)
- **G1:** run **350–400 gold queries** (5–7 per topic × 60 topic_tags) through `/query_v2`, measure recall@10 against expected chunk IDs
- **G2:** **dual-judge ensemble** (Gemini 2.0 Flash + Claude CLI) on all answers; flag any query where judges disagree >1 pt for human review
- **G3:** `must_cite_verbatim` substring match → if miss, normalized Levenshtein ≥0.95 fallback (strip whitespace + lowercase + NFKC before distance)
- **G4:** **50 adversarial OOC queries** + UPL/arithmetic refusal rule enforcement — all must refuse

### Phase 8 — Promotion or roll-back
- **All 4 gates ≥95%:** flip `/query` to `cbic_v2`, `cbic_v1` kept (D10)
- **Any gate <95%:** no promotion, diagnose root cause, amend spec, re-run from failed phase

---

## 3. Router & Retrieval

- **Router:** keyword-first (existing `_KW` patterns). On miss → qwen3-14b LLM router (V10 latency check) → fallback `gst` (B15).
- **Query sanitization** (before router): strip control chars, cap length at 2000, drop prompt-injection markers (`<|`, `[INST]`, `### System`).
- **Retrieval:** hybrid RRF, k=20 initial, CE rerank to top-10 (keep current). **Devanagari-query boost:** if query contains any Devanagari codepoint, post-rerank score boost (+0.1) on chunks tagged `lang=hi` or with populated `hindi_twin_chunk_ids`.
- **Hindi handling (D4 amended 2026-04-23):** English queries retrieve across English + Hindi twin chunks. **LLM prompt receives English chunks only** (Hindi twin chunk IDs surface in response metadata for UI citation, not in the reasoning context) — prevents 16k context bloat and attention dilution on qwen3-14b. See §3.1.

### 3.5 Embed-stack invariants (added 2026-04-23)

These rules are INVARIANT — any deviation requires a SPEC amendment + JOURNAL entry before running.

1. **Dense ingest embedder = `llama-cpp-python` Vulkan, in-process.** Facade: `cbic_rag/embedder.py` (exports `_FACADE_VERSION = "direct-v1"` + re-exports from `embedder_direct.py`). Pool: GPUs 0,1,4,5,6 only.
2. **Ollama is FORBIDDEN for ingest-time dense embedding.** On this rig Ollama falls back to CPU silently (no VRAM allocation, no error, but returns vectors with near-zero variance). Ollama is acceptable for query-time, low-volume work — never for ingest.
3. **GPU 2 reserved for qwen3-14b llama-server** (port 9082, Pass-1 classifier). Must NOT appear in `EMBED_GPUS`.
4. **GPU 3 forbidden** — SMU-faulted hardware, hangs under load. Must NOT appear in `EMBED_GPUS`.
5. **Preflight is mandatory before every production run.** `reingest_spec/preflight.sh` runs 13 checks including a real 1-doc E2E dry-run that writes to a throwaway Qdrant collection and drops it. If preflight exits non-zero, the real run is aborted.
6. **`ingest_v2.py` runs its own hard preflight on startup** (run_preflight()): verifies facade sentinel, EMBED_GPUS exclusions, qwen3 reachability, and runs a hello-world embed with variance check. `--no-preflight` is component-smoke only; never production.
7. **Silent-success guard:** phase3-5 raises if `qdrant.points_count < chunks_submitted`. No more green "DONE" logs masking zero-point collections.

### 3.1 Selective context compression (post-rerank, pre-LLM)
After CE rerank to top-10, before LLM prompt assembly:
- For each chunk, emit only: **top-3 most relevant sentences** (scored against query via BGE-M3 sentence-level cosine) + `parent_hierarchy_text` breadcrumb.
- Drops prompt size 40–60%; improves qwen3-14b attention and G2 reasoning scores; resolves D14 context-bloat contradiction.
- Full chunk text + Hindi twin IDs carried through separately in response payload for G3 verification + UI.

---

## 4. Refusal Logic (Gate 4)

- If top-10 reranked max score < θ_retrieve (V16 determines, per-category) → refuse with "no basis in indexed CBIC corpus"
- If retrieved chunks all lack topical terms from query → refuse
- **UPL / arithmetic refusal rule (added per external review 2026-04-23):** system MUST refuse any query that asks to compute a final numeric liability on user-provided financial figures (detect via query pattern: numeric amount + question word + tax-compute verb). Response must return the applicable rule, rate, and computation method, but NOT execute the arithmetic. This protects against LLM arithmetic hallucination and Unauthorized-Practice-of-Law liability. Example: "I earned ₹1,00,000, how much GST?" → returns applicable section + rate + method; does NOT return ₹18,000.
- **Adversarial set of 50 OOC queries** maintained in `eval_set_adversarial.json`, spanning:
  - Direct-tax traps (Income Tax Act questions — CBIC-adjacent but OOC)
  - Fake provisions ("Section 999 of CGST Act")
  - Real-section / fake-context (Customs Act §14 applied to domestic restaurant bill)
  - Encoding attacks (base64, unicode lookalikes)
  - Prompt injection via query text
  - Arithmetic-calculation requests (UPL rule enforcement)

---

## 5. Locked Decisions (D1–D15)

| # | Decision | Answer | Source |
|---|----------|--------|--------|
| D1 | LLM for extraction backstop | **Claude CLI primary** (V18 PASS: 50/50 parse, p50 3.38s, p95 4.58s); **qwen3-14b viable fallback only** (V1 PASS post-reboot parse_rate=1.0 p50=2.5s; V2b PASS 30/30 parse, p50≈6s — acceptable for fallback path at ~85 min total) | V1, V2b, V18 |
| D2 | Chunk size policy | 3500/5500/700 | confirmed |
| D3 | Reranker | CE (current) | user: "your recommendation" |
| D4 | Hindi | Retrieve across Eng + Hindi twins; **LLM prompt receives English chunks only** (Hindi IDs in response metadata); see §3.1 context compression | user + review amendment 2026-04-23 |
| D5 | Gemini budget cap | ~$6.56 worst-case (paid tier removes free-tier RPM cap); V9 PASS | V9 |
| D6 | Out-of-corpus | Refuse (per §4 inc. UPL/arithmetic rule) | user explicit + review 2026-04-23 |
| D7 | Router fallback | Keyword → LLM → gst (B15); query sanitization prefixed (§3) | user: "your rec" |
| D8 | Gold set size | **350–400** (5–7 per topic × 60 tags) | review adoption 2026-04-23 (was 170) |
| D9 | Eval frequency | Every phase gate, plus shadow continuous | user: "your rec" |
| D10 | Old data cleanup | Never delete during project; maintain cleanup_backlog.md | user explicit |
| D11 | n/a | — | — |
| D12 | n/a | — | — |
| D13 | Cutover mode | Shadow with real testers + dual-write log | user explicit |
| D14 | Bilingual citation format | Cite both English + Hindi twin in UI/response metadata; LLM reasoning context uses English chunks only (see D4, §3.1) | user + review amendment 2026-04-23 |
| D15 | Formal doc versioning | Not necessary | user explicit |

---

## 6. Validation Probes (V1–V24)

See `PROBES.md` for full probe specs + run commands. Gate: all must pass or have accepted workaround before Phase 1 starts.

---

## 7. Known Risks

| # | Risk | Mitigation |
|---|------|------------|
| R1 | OCR table structure lost in flat text | V8 table-aware re-OCR prompt |
| R2 | Section ref regex misses on OCR noise | V22 OCR-tolerant variants + D1 LLM backstop |
| R3 | Multi-GPU pool hangs long-run | V5 1-hr stability; watchdog restart |
| R4 | Qdrant disk exhaustion during dual-collection | V7 confirms headroom ≥2× |
| R5 | qwen3-14b extraction too slow | Resolved — D1 primary = Claude CLI (V18 PASS 3.38s); qwen3 fallback only (V2b 30/30 parse, ~6s acceptable) |
| R6 | θ_retrieve threshold brittle | V16 distribution → pick per-category; per-difficulty tuning deferred to post-v2 backlog |
| R7 | Adversarial set captures only easy OOC | **50-query curated set** across 6 attack classes (direct-tax traps, fake sections, real-section-fake-context, encoding attacks, prompt injection, UPL/arithmetic) |
| R8 | Cross-doc duplicate boilerplate pollutes recall | V21 ChunkDeduper — SHA256 over NFKC-normalized + whitespace-collapsed + lowercased text; `also_appears_in` preserves audit trail |
| R9 | Gold queries reference topics not discoverable in manifest | V20 rule-based `topic_tagger.py` — multi-label over 60 content-bearing topics; `topic_tags` payload field + Qdrant index |
| R10 | qwen3-14b RADV driver degradation (observed 2026-04-23) | Known-good configs locked in `~/.claude/projects/.../known_good_configs.md`; D1 defaults to Claude CLI; **throughput-drop watchdog** in systemd unit — if embed/generation rate falls >20% below baseline for 3 consecutive batches, log warning and trigger controlled restart of the affected llama-server instance |
| R11 | G2 single-judge sycophancy bias inflates scores | Dual-judge ensemble (Gemini Flash + Claude CLI); disagreement >1 pt flagged for human review |
| R12 | G3 OCR noise (1↔l, whitespace) causes false negatives on strict substring match | Normalized Levenshtein ≥0.95 fallback after substring miss (lowercase + NFKC + ws-collapse) |
| R13 | LLM arithmetic hallucination → UPL liability on "calculate my tax" queries | §4 UPL rule — refuse numeric calculations on user figures; return rule/rate/method only |
| R14 | Shadow-mode cache coherency between v1 and v2 | Dual-writer cache keys include `collection_version`; divergence monitor >2% = kill switch on `/query_v2` |
| R15 | qwen3-14b 16k context bloat with bilingual citations → G2 attention dilution | §3.1 selective context compression (top-3 sentences + hierarchy) + D14 amended (Eng-only to LLM) |

---

## 8. Execution Order (once probes green)

Phase 0 → 1 → 2 → 3 → 4 → 5 → 6 (shadow starts) → 7 (gates) → 8 (promote or iterate spec)

Total wall-clock estimate: 18–30 hours pipeline + shadow soak period (user-determined).

See `RUNBOOK.md` for the full A→Z execution sequence with per-stage stop-gates.

---

## 9. Post-v2 Backlog (not blocking ingest)

Full backlog lives in `~/.claude/projects/D---gpu-rig-ai/memory/project_cbic_reingest_v2.md`. Summary:

| ID | Item | Trigger |
|----|------|---------|
| B-post-1 | Multi-Query / Query Rewriting (3 paraphrases → RRF) | G1 recall@10 misses θ post-cutover |
| B-post-2 | Step-Back + Query Decomposition | Compound-query failures in real logs |
| B-post-3 | Self-RAG-lite (qwen3 emits confidence + missing_evidence) | Fold into G4 post-cutover |
| B-post-4 | Query cache (SHA256 normalized → answer, TTL 24h) | Any time |
| B-post-5 | Observability dashboard (per-stage latency, RRF, CE scores, refusal rate) | Any time |
| B-post-6 | Ekimetrics 5-metric sample audit (SC/BI/RC/DCC/ICC on 200-chunk sample) | Post-ingest validation |
| B-post-7 | Amendment-graph SQLite sidecar (temporal queries "rate as of date X") | User asks temporal questions |
| B-post-8 | BM25 Hinglish / transliteration char-n-gram | Hindi query recall drops post-cutover |
| B-post-9 | TTFT heartbeat watchdog (RADV degradation early detection) | Post-cutover ops hardening |
| B-post-10 | E-feedback rate-limit per IP/session (anti-poisoning) | User feedback channel opens |
| B-post-11 | Per-difficulty θ_retrieve | G4 false-positive rate >5% post-cutover |

**Rejected (explicit):** agentic retrieval loops, Graph RAG, HyDE, ColBERT/ColPali, Ekimetrics full-framework, ROCm swap. See PLAN_FOR_REVIEW.md §9.

---

## 10. Training-Corpus Accumulator (added 2026-04-24, D18)

**Purpose:** every gold query, LLM answer, judge verdict, hard negative, and gate outcome generated during the entire v2 program is written **once, in a unified schema**, to a single append-only file. This makes the corpus a reusable dataset (BGE-M3 contrastive fine-tune, bge-reranker fine-tune, qwen3-14b CBIC-LoRA SFT, DPO pairs) **independent of the RAG system that produced it**.

### Schema reference
- `~/.claude/projects/D---gpu-rig-ai/memory/pair_schema_cbic_v2.md` — frozen canonical spec (field definitions, type contracts, mapping from legacy Format A / Format B)

### Files
| File | Mode | Purpose |
|---|---|---|
| `/opt/indian-legal-ai/data/training_corpus/cbic_pairs_v2.jsonl` | **append-only**, never rewritten | single source of truth |
| `cbic_pairs_v2_<scope>_<yyyymmdd>.jsonl` | write-once | scope snapshots (smoke, gst50, …, full14925) |
| `cbic_hardneg_v2.jsonl` | append-only | denormalized hard negatives for contrastive training |
| `eval/training_pairs/qa_*.jsonl` + `pairs_*.jsonl` | frozen, never deleted | legacy inputs to migrator |

### Invariants
1. **Never delete legacy files.** Migrator reads them; canonical file accumulates.
2. **Append-only on the canonical file.** Gate re-runs append to `gate_verdicts.history[]`, not overwrite.
3. **`(chunk_id, doc_id, question_id)` is the joint primary key.** `pair_id` is derived.
4. **Every pair carries full provenance** — scope, `generated_ts`, `gold_source`, `chunker_version`, `retriever_config`, `source_file` if migrated.
5. **All generators + all gate evaluators write/update in this shape.** No new formats without a SPEC amendment + JOURNAL entry.

### Projected scale
- Today: 5,781 legacy pairs (Format A + B) + 120 10-doc synthetic → migrator unifies → ~5,900 rows
- After GST50: +500-600 → ~6,500
- After full 14,925: +150,000 (est., 10 queries/doc)

**This is the training dataset asset of the program.**

---

## 11. No-CPU Invariant (added 2026-04-24, D17)

**Rule:** no ML inference runs on CPU anywhere in the active ingest/retrieval pipeline. See RUNBOOK Stage 0.6 for enforcement. Pure-Python orchestration code (regex, JSON, SQL) is explicitly **out of scope** — that is unavoidable CPU execution and not what this rule addresses.

**Gated offenders (currently disabled / frozen):**
- fastembed SparseTextEmbedding — gated by `SKIP_SPARSE=1` / `DENSE_ONLY=1` in `ingest.embed_batch` and `retriever`
- FlagReranker(device='cpu') — gated by `RERANK=none` (default in systemd drop-in)
- colbert_rerank fastembed ONNX — gated by `RERANK=none`
- tesseract OCR in `recovery_worker.py` — hard `sys.exit(2)` unless `RECOVERY_WORKER_UNFREEZE=1`

**Replacement path for sparse + rerank + OCR (post-v2, not blocking):**
- Sparse: either drop (proven unnecessary at 95%+ with dense-only on 10-doc smoke) or re-implement on GPU (SPLADE via llama-cpp-python if feasible)
- Rerank: bge-reranker-v2-m3 GGUF on Vulkan (idle GPU 0/1 have D-state issues — needs retry strategy) OR skip entirely if 95% hits without it
- OCR: Qwen2.5-VL-7B GGUF on llama-server Vulkan (GPU 4 or 6) per `ocr_research_cbic.md`

