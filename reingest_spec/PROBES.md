# Validation Probes V1–V24

Each probe: **purpose, run command, pass criteria, blocks which phase**. Run from rig shell unless noted "from laptop" (Qdrant-only probes work from laptop).

**Status legend:** ⚪ not started · 🟡 running · 🟢 pass · 🔴 fail · ⚠️ workaround

---

## Rig-shell probes

### V1 — qwen3-14b strict JSON schema follow 🔴 BLOCKED
**Why:** Decides D1 (LLM for extraction backstop). If qwen3-14b can't return strict JSON reliably, we fall back to Gemini or Claude CLI.
**Run:**
```bash
cd /opt/indian-legal-ai/rag/cbic_rag
python3 /mnt/d/_gpu_rig_ai/benchmarks/probes/probe_v1_qwen_json.py
```
**Probe script:** sends 50 varied prompts to qwen3-14b at 127.0.0.1:9082 asking for `{"section_ref": "...", "confidence": 0-1}`. Parses response as JSON.
**Pass:** ≥48/50 parse cleanly; ≥45/50 semantically correct on sample review.
**Blocks:** Phase 2 (chunker hierarchy extraction).

### V2 — qwen3-14b extraction latency on 3000+ char inputs ⚰️ RETIRED 2026-04-23
**Why retired:** gate modeled a call pattern we no longer plan to use (full-chunk extraction via qwen3). Actual plan: Pass-1 classification with short head+tail inputs and <150-token JSON output — qwen3 is only the fallback for Claude CLI (D1). See V2b for the replacement probe. Final V2 result (for record): p50=5.9s, p95=13.9s — fails the old gate, irrelevant to the new plan.

### V2b — qwen3-14b Pass-1 classification latency 🟡 RUNNING
**Why:** Real call shape for qwen3 in chunker v2 = Pass-1 classification fallback if Claude CLI (D1) is down. Input = doc head (2000) + tail (1500); output = <150-token JSON.
**Run:**
```bash
python3 /mnt/d/_gpu_rig_ai/benchmarks/probes/probe_v2b_qwen_pass1.py
```
**Pass:** p50 ≤3s, p95 ≤5s, parse_rate ≥0.9 on 30 samples.
**Blocks:** Phase 2 qwen3 fallback confidence (not primary — primary is V18 Claude CLI PASS).

### V3 — langdetect on CBIC samples 🟢
**Why:** D4 requires per-chunk `lang` field. Need a reliable Hindi/English detector.
**Run:** `python3 probes/probe_v3_langdetect.py` — samples 200 chunks across categories, runs `langdetect`, manual spot-check 20.
**Pass:** ≥95% agreement with manual labels on 20-sample check.
**Blocks:** Phase 2 lang field.

### V4 — fastembed BM25 on Hindi 🟢
**Why:** BM25 tokenizer may not handle Devanagari. Broken Hindi sparse → Hindi queries fail retrieval.
**Run:** `python3 probes/probe_v4_bm25_hindi.py` — encode 10 Hindi chunks, 10 English chunks, compare sparse dim/nonzero counts.
**Pass:** Hindi nonzero count within 30% of English.
**Blocks:** Phase 4.

### V5 — Multi-GPU embedder pool 1-hr stability ⚪
**Why:** R3 mitigation. Last session saw VRAM orphan issues when killed.
**Run:** `python3 probes/probe_v5_pool_soak.py` — 1 hour of continuous batches (32 chunks each, sleep 500ms). Monitor `rocm-smi`.
**Pass:** zero failures, zero OOM, VRAM stable ±5%.
**Blocks:** Phase 3 confidence.

### V8 — Gemini table-aware OCR prompt ⚪
**Why:** Current OCR lost table structure; G3 evidence citations on tables will fail.
**Run:** Re-OCR 10 known table pages with new prompt: "Preserve tables as GitHub markdown. Columns aligned. No merged cells flattened." Compare quality.
**Pass:** Manual review — ≥8/10 tables reconstructable.
**Blocks:** Phase 2 table handling.

### V9 — Gemini daily quota 🟢
**Why:** Budget sizing (D5). Re-OCR of tables + V1 fallback may exceed quota.
**Run:** Check current project quota in Google Cloud Console + compute worst-case: (tables_to_reocr × 1 call) + (regex_miss × 1 call) = expected total.
**Pass:** Worst-case ≤80% of daily cap.
**Blocks:** D5 resolution.

### V10 — route_llm latency on qwen3-14b 🔴 BLOCKED
**Why:** Per-query router LLM adds to query latency. If >1.5s, keyword-only with gst fallback.
**Run:** `python3 probes/probe_v10_route_latency.py` — 50 varied queries, time router LLM call.
**Pass:** p50 ≤800ms, p95 ≤1500ms.
**Blocks:** Router config.

### V11 — Chunker parent_hierarchy_text emission ⚪
**Why:** Hierarchy breadcrumb critical for evidence citation display.
**Run:** Chunk 5 known-structure docs (e.g. CGST Act main text, a complex notification). Dump first 20 chunks. Manual verify `parent_hierarchy_text` matches known structure.
**Pass:** 5/5 docs breadcrumb correct on first chunk of each section.
**Blocks:** Phase 2.

### V12 — OCR on 7 empty PDFs 🟡
**Why:** prep_ocr_ingest flagged 7 files <20 bytes — either blank scans or OCR failure.
**Run:** Inspect each; if real content, re-OCR with higher DPI (200) + table-aware prompt.
**Pass:** 7/7 either confirmed blank or re-OCR succeeds.
**Blocks:** Phase 1 manifest.

### V13 — Chunker run-time on 100 typical docs ⚪
**Why:** Budget check; last session didn't measure chunker alone (with/without PyMuPDF + pdfplumber dual-pass).
**Run:** `time python3 -c "from chunker import chunk_many; chunk_many(sample_100_paths)"`.
**Pass:** ≤15 min for 100 docs → 115k docs extrapolated ≤3 hr.
**Blocks:** Phase 2 timing.

### V14 — must_cite_verbatim intent detection 🟢
**Why:** G3 gate requires knowing which queries demand verbatim citation vs paraphrase.
**Run:** Manual audit — check that 170-query gold set `must_cite_verbatim` field is populated correctly on the 40+ queries that need it.
**Pass:** 100% of citation-critical queries have must_cite_verbatim populated.
**Blocks:** G3 eval.

### V17 — Gemini-as-judge consistency 🟡
**Why:** G2 rides on judge stability. If same answer scores 4 vs 2 across runs, gate is invalid.
**Run:** `python3 probes/probe_v17_judge_stability.py` — 20 known answers × 3 runs each. Measure score variance.
**Pass:** ≤0.5 stdev per answer; ≤0.3 mean diff run-to-run.
**Blocks:** G2 validity.

### V18 — Claude CLI for LLM extraction 🟢
**Why:** D1 alternative if qwen3-14b fails V1. Rig has CLI authed under Max plan.
**Run:** `python3 probes/probe_v18_claude_cli.py` — same 50-prompt extraction test, compare parse-rate + latency vs qwen3-14b.
**Pass:** parse-rate ≥qwen3; latency ≤5s/call.
**Blocks:** D1 resolution.

### V19 — api.py stays up during re-ingest ⚪
**Why:** Shadow mode requires `/query_v2` via new collection while `/query` stays live. Must not crash api.py.
**Run:** Start mini-ingest of 100 docs into `cbic_v2_test`, concurrently hammer `/query` at 1 QPS for duration.
**Pass:** zero 5xx on `/query`; zero OOM.
**Blocks:** Phase 6 shadow mode safety.

### V20 — Subcategory taxonomy covers 170 queries 🟢
**Why:** If gold queries reference subcategories that don't exist in manifest, recall@10 can't be measured.
**Run:** Set-diff gold `category/subcategory` vs manifest distinct values.
**Pass:** zero queries reference missing subcategories.
**Blocks:** G1 eval.

### V23 — api.py 4-gate contract refactor safety ⚪
**Why:** Adding `/query_v2` touches api.py. Must not break existing endpoints.
**Run:** Diff + unit-test existing 12 endpoints before/after refactor (requires V19 infra).
**Pass:** 12/12 existing endpoints unchanged behaviour.
**Blocks:** Phase 6.

### V24 — Validator dry-run rejection rate ⚪
**Why:** Phase 5 upsert validates payload schema. If validator rejects >5%, chunker has bugs.
**Run:** Chunk 100 docs, run payload validator without upsert, count rejects.
**Pass:** ≤2% reject rate.
**Blocks:** Phase 5.

---

## Qdrant-only probes (laptop OR rig — laptop sufficient)

### V6 — Qdrant snapshot+restore 🟢
**Why:** Phase 0 rollback safety.
**Run:** From laptop:
```bash
curl -X POST http://192.168.1.107:6343/collections/cbic_v1/snapshots
# check returned snapshot name
curl http://192.168.1.107:6343/collections/cbic_v1/snapshots  # list
```
**Pass:** snapshot succeeds; file size ≥90% of collection size on disk.
**Blocks:** Phase 0.

### V7 — Disk space for dual collections 🟢
**Why:** R4 mitigation. Keep v1 + build v2 = 2× disk.
**Run:** Qdrant `/telemetry` endpoint or shell `du -sh` for storage dir. Compare vs available disk.
**Pass:** free disk ≥1.5× current collection size.
**Blocks:** Phase 5.

### V15 — Qdrant payload update-in-place perf 🟢
**Why:** If we need to backfill a field (e.g. hindi_twin_chunk_ids) after initial ingest, must be fast.
**Run:** Pick 1000 points, update one payload field via batch `/collections/cbic_v1/points/payload`. Time it.
**Pass:** ≤10s for 1000-point update.
**Blocks:** Phase 2 backfill strategy.

### V16 — θ_retrieve threshold stability ⚪
**Why:** Phase 7 refusal logic needs a score threshold that separates "real hit" from "no hit."
**Run:** 350–400 gold queries × retrieve scores + 50 OOC adversarial × retrieve scores (per D8 review adoption). Pick θ that separates them.
**Pass:** Clean separation — OOC max < gold min × 0.9.
**Blocks:** G4.

### V21 — sha256 dedup at 114k scale 🟢
**Why:** Duplicate `sha256` chunks pollute recall.
**Run:** Scroll all 114k points, count unique `doc_id` + flag chunks with identical `text` hash across different `doc_id`.
**Pass:** ≤0.5% duplicate text bodies.
**Blocks:** Phase 1 manifest dedup logic.

### V22 — OCR-tolerant regex variants on 844 OCR'd PDFs 🟢
**Why:** OCR introduces noise (e.g. "Sec tion 16(2)(c)", "Sectlon", zero-vs-O). Current regex misses these.
**Run:** Scroll OCR chunks (`text_source=ocr` — currently missing payload field, use path match). Test 5 regex variants, count hits.
**Pass:** Best variant recalls ≥80% of manually-identified section refs on 50-chunk sample.
**Blocks:** Phase 2 section_ref extraction.

---

## Run Order (parallel-friendly)

**Wave A (Qdrant-only, from laptop, no rig shell needed):** V6, V7, V15, V16, V21, V22 — can run right now if rig Qdrant up.

**Wave B (rig shell, LLM-dependent):** V1, V2, V10, V17, V18 — these resolve D1.

**Wave C (rig shell, chunker-dependent):** V3, V4, V11, V13, V20, V24

**Wave D (rig shell, infra):** V5, V8, V9, V12, V19, V23

**Wave E (gated on D1 decision):** V14

---

## Deliverables from probe run
Each probe emits `/opt/indian-legal-ai/data/probes/v{N}_result.json` with:
```json
{"probe": "V1", "status": "pass|fail|workaround", "metrics": {...}, "notes": "..."}
```
Rollup: `probes_summary.json` + this file updated with ⚪→🟢/🔴.
