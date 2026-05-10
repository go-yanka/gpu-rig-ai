# CBIC RAG v2 — Total Runbook (A→Z)

**Purpose:** Single authoritative sequence from current state → "v2 live at 95% trust". Every step: owner, action, time, pass criteria, stop-gate. Ask "status" at any step — answerable without context.

**Hard rule:** any step that fails QA → HALT, escalate to user. Do not proceed with known defects. 95% trust is non-negotiable.

**Legend:** ✅ done · 🟡 in progress · ⚪ pending · 🔴 blocked · ⏸️ paused for user

---

## Stage 0 — MANDATORY INVENTORY (before ANY planning step in ANY stage)

**The rule:** before proposing a plan for any step that involves drafting new data, new queries, new labels, new prompts, or new code that might already exist, **run `bash D:/_gpu_rig_ai/inventory.sh` first** and paste relevant lines into the conversation.

**Why:** on 2026-04-23 night we planned "draft 180–230 new gold queries from scratch" for Stage C without knowing `eval/training_pairs/` had 5,781 pre-generated pairs. Would have wasted 6+ hours. Enforced via `D:/_gpu_rig_ai/CLAUDE.md` which loads on every session start.

**Exit check:** if you are about to type "I'll draft X," stop and run inventory first. No exceptions.

---

## Stage A — Finalize Plan & Spec Documents
**Goal:** all 5 source-of-truth docs reflect external-review amendments, internally consistent, zero contradictions.
**Owner:** Claude (me) · **Time:** ~30 min · **Currently:** ✅ COMPLETE (14/14) — 2026-04-23 night. Also swept stale `170/20` references in PROBES.md V16 + READINESS.md B-6 to match new D8 (350–400 gold / 50 OOC).

| # | Step | Action | Pass criteria | Status |
|---|------|--------|---------------|--------|
| A1 | Chunking strategy doc | Patch schema + R1 row-headers + R3 Hindi tokens + LLM prompt | grep confirms 8 new strings in file | ✅ |
| A2 | SPEC §1 gates | G1 350-400, G2 dual-judge, G3 Levenshtein, G4 50+UPL | All 4 gate rows reference new thresholds | ✅ |
| A3 | SPEC §3 retrieval | Sanitization, Devanagari boost, §3.1 context compression | §3.1 subsection exists with top-3-sentences rule | ✅ |
| A4 | SPEC §4 refusal | UPL rule + 50-query set + 6 attack classes | UPL rule paragraph present | ✅ |
| A5 | SPEC Phase 7 sync | Mirror §1 gate thresholds | Phase 7 lists 350-400, dual-judge, Levenshtein, 50 adversarial | ✅ |
| A6 | SPEC decisions D4/D8/D14 + status line | Amended per D14 English-only + D8 expand + status noted | Decisions table reflects amendments | ✅ |
| A7 | SPEC risks R5/R7/R10 updated + R11–R15 new | R11 sycophancy, R12 OCR noise, R13 UPL, R14 cache, R15 context bloat | Risks table has 15 rows | ✅ |
| A8 | SPEC §9 backlog section | §9 with 11 B-post items + rejected list | ✅ |
| A9 | **STOP-GATE** SPEC.md full read | 2 defects found + fixed (Phase 2 R1 summary stale; D1 stale re: qwen3 reboot) | ✅ |
| A10 | PLAN_FOR_REVIEW.md mirror | Update gates, decisions, risks, §3.1, §9 backlog | ✅ |
| A11 | **STOP-GATE** PLAN cross-check | grep of gate thresholds + risks + attack-class wording matches SPEC exactly | ✅ |
| A12 | project_cbic_reingest_v2.md backlog | Sync with SPEC §9 (11 items) | ✅ |
| A13 | JOURNAL.md append | Round-1 Gemini + round-2 Grok + Stage-A completion logged with defects caught | ✅ |
| A14 | **FINAL STOP-GATE** all 4 files | Zero contradictions verified via grep sweep (350–400 consistent in SPEC/PLAN/JOURNAL/RUNBOOK; attack-class wording aligned; 170/20 residuals only in historical JOURNAL notes describing the change — correct) | ✅ |

**Stage A exit:** all 14 items ✅. Then proceed to Stage B.

---

## Stage B — Build Components (B-1 through B-7)
**Goal:** the code that actually implements the spec exists, unit-tested, ready to run.
**Owner:** Claude (me), writing Python · **Time:** ~6–10 hours spread over sessions · **Currently:** ⚪ not started

| # | Component | File | Acceptance | Time | Status |
|---|-----------|------|------------|------|--------|
| B1 | `chunker_v2.py` | `reingest_spec/chunker_v2.py` | Implements Pass 1 LLM call + Pass 2 rules R1–R7; passes T1–T8 self-tests | 3–4h | ⚪ |
| B2 | `chunking_plan_prompt.md` | `reingest_spec/chunking_plan_prompt.md` | Claude CLI system prompt file with schema + all instructions | 20m | ⚪ |
| B3 | `test_chunker.py` | `reingest_spec/test_chunker.py` | T1–T8 all pass on synthetic + real samples | 1h | ⚪ |
| B4 | `ingest_v2.py` orchestrator | `reingest_spec/ingest_v2.py` | Drives Phase 1→5; resumable at phase boundaries; manifest-aware | 2h | ⚪ |
| B5 | `/query_v2` endpoint + dual-writer | `cbic_rag/query_v2.py`, `cbic_rag/shadow_writer.py` | Same contract as `/query`; cache keys versioned; >2% divergence kill switch | 1.5h | ⚪ |
| B6 | θ_retrieve tuner | `reingest_spec/theta_tune.py` | V16 distribution → per-category threshold JSON | 1h | ⚪ |
| B7 | cbic_v2 snapshot script | `reingest_spec/snapshot_v2.sh` | Post-Phase-5 automated Qdrant snapshot | 30m | ⚪ |

**Stage B exit:** all 7 components exist + tests green. STOP-GATE: run T1–T8, any fail = HALT.

---

## Stage C — Curate Eval Sets from Existing Corpus (Gold + Adversarial)
**Goal:** gate data at 95%-trust scale. **Revised 2026-04-23 night** after inventorying `D:/_gpu_rig_ai/eval/` — we already have 5,781 pre-generated QA pairs + 170 curated expansion items + 149 observed failure modes. This is a **curation/filter** task, not drafting from scratch.
**Owner:** Claude (filter + normalize) → USER (sign-off) · **Time:** ~2h automated + ~1h user review · **Currently:** ⚪ not started

**Source inventory:**
- Raw QA pool (~5,781 lines): `eval/training_pairs/qa_gemini.jsonl` (2,559) + `pairs_2000_20260422.jsonl` (1,909) + `qa_sonnet_high.jsonl` (426) + `pairs_opus_highcomplex.jsonl` (213) + `pairs_sonnet_lowcomplex.jsonl` (197) + `qa_claude_opus.jsonl` (152) + `pairs_claude_opus.jsonl` (76)
- Current gold: `eval/gold_set.yaml` (170)
- Curated expansion: `eval/gold_set_expansion/` — 170 items already drafted across 6 buckets
- Adversarial seed: `bucket_2_refusal.yaml` (10) + `failure_modes_20260422.jsonl` (149 observed failures)
- Hard negatives: `variants_results.jsonl` (510)

| # | Step | Action | Target | Status |
|---|------|--------|--------|--------|
| C1 | Consolidate raw pool | Merge 7 JSONL files into one normalized schema; dedup on `chunk_id + q` | Single pool `eval_pool_raw.jsonl`, est. ~4,500 after dedup | ⚪ |
| C2 | Topic-tag pool | Apply `topic_tagger.py` (V20 PASS) to each `text` → primary topic + multi-labels | Each pair has `topic_tags` field | ⚪ |
| C3 | Chunk-ID remap | Post-dedup V21 produced 78,291 canonical chunks; map old `chunk_id` → canonical via SHA256; drop unresolvable | ≥90% retention | ⚪ |
| C4 | Stratified sample 350–400 | 5–7 per topic × 60 tags; prefer high-difficulty + high-LLM-quality pairs | `eval_set_gold.json` with 350–400 items | ⚪ |
| C5 | Adversarial curation | Merge `bucket_2_refusal.yaml` (10) + mine `failure_modes_20260422.jsonl` for OOC/hallucination candidates; add UPL-arithmetic + encoding + injection; expand to 50 across 6 classes | `eval_set_adversarial.json` with 50 items | ⚪ |
| C6 | User sign-off | User reviews C4 + C5 outputs | Approved | ⚪ |

**Stage C exit:** both eval sets committed + checksum'd + <5% topic-tag gaps. STOP-GATE: user signs off on both.

---

## Stage D — Pre-flight Probe Completion
**Goal:** every probe green or has documented accepted-workaround.
**Owner:** Claude · **Time:** ~2 hours · **Currently:** 10 PASS · 3 recovered · 1 deprioritized · 10 pending

| # | Probe | What | Action | Status |
|---|-------|------|--------|--------|
| D1 | V5 pool soak | Multi-GPU BGE-M3 1-hr stability | Run on rig; zero OOM; VRAM ±5% | ⚪ |
| D2 | V7 Qdrant disk headroom | Confirm ≥2× | Quick df check | ⚪ |
| D3 | V8 Gemini table OCR | 20-sample table-aware prompt test | Visual diff v1 vs table-aware | ⚪ |
| D4 | V11 embed_text format | Which of 3 format variants gives best recall | Small A/B on gold subset | ⚪ |
| D5 | V12 topic tagger live test | Tagger accuracy on 100 sample chunks | ≥90% multi-label agreement | ⚪ |
| D6 | V13 chunker speed | Run chunker_v2 on 50 docs; extrapolate | Projects <3h for 115k chunks | ⚪ |
| D7 | V15 refusal-keyword detector | Works on adversarial subset | 100% detection | ⚪ |
| D8 | V16 θ_retrieve distribution | Score histogram per-category on gold set | Per-category thresholds file | ⚪ |
| D9 | V19 dup-audit trail | `also_appears_in` survives round-trip | Random sample audit | ⚪ |
| D10 | V22 OCR-tolerant section regex | Test against known OCR errors | ≥95% match vs clean | ⚪ |
| D11 | V23 Hindi twin linker T8 | 10-doc sample; hierarchy + embedding fallback | ≥90% link rate | ⚪ |
| D12 | V24 dual-judge agreement | 20-sample G2 with Gemini + Claude | Mean abs diff ≤0.5 | ⚪ |

**Stage D exit:** all 12 probes green OR documented accepted workaround. STOP-GATE: any unresolved red = HALT.

---

## Stage E — Execute Phase 0 (Pre-flight, non-destructive)
**Owner:** Claude on rig · **Time:** <30 min · **Reversible:** yes

| Step | Command | Pass | Status |
|------|---------|------|--------|
| E1 | Snapshot `cbic_v1` | `POST /collections/cbic_v1/snapshots` | snapshot file created | ⚪ |
| E2 | Git tag | `git tag reingest-v1-start && git push --tags` | tag exists | ⚪ |
| E3 | Verify 851 OCR txt files | `ls /opt/indian-legal-ai/data/ocr_cache/*.txt \| wc -l` | 851 | ⚪ |
| E4 | Confirm probes gate | manually tick off D1–D12 | all green | ⚪ |
| E5 | Write cleanup_backlog.md | `touch` + initial entries | file exists | ⚪ |

**Stage E exit:** rollback path verified. STOP-GATE: snapshot must be readable before Phase 1.

---

## Stage F — Execute Phase 1 (Manifest + bilingual linking)
**Owner:** Claude on rig · **Time:** ~30 min · **Reversible:** yes (drop the sqlite)

| Step | Command | Pass | Status |
|------|---------|------|--------|
| F1 | Run ingest_v2.py --phase=1 | emits `ingest_manifest_v2.sqlite` | 851 rows | ⚪ |
| F2 | Bilingual linking | hindi_twin_sha256 populated where applicable | ≥400 pairs linked | ⚪ |
| F3 | Manifest QA | Sample 20 rows, verify category/subcategory/lang | 20/20 correct | ⚪ |

**Stage F exit:** manifest validated. STOP-GATE: any category miscategorization → HALT.

---

## Stage G — Execute Phase 2 (Chunker v2, two-pass)
**Owner:** Claude on rig · **Time:** ~4 hours · **Reversible:** yes (delete chunks dir)

| Step | Command | Pass | Status |
|------|---------|------|--------|
| G1 | Pass 1: Claude CLI classification | 851 docs × ~3.5s | 100% have chunking_plan; ≤2% low-confidence queued for Gemini | ⚪ |
| G2 | Gemini second-opinion on low-confidence | ~20 calls | all resolved or manual-review flagged | ⚪ |
| G3 | Pass 2: rule-driven chunking | CPU-bound ~3h | ~115k raw chunks emitted | ⚪ |
| G4 | ChunkDeduper | SHA256 over normalized text | ~78k canonical (31.7% dup rate ±5%) | ⚪ |
| G5 | Chunker audit | sample 100 chunks; verify R1–R7 rule compliance | zero R3 violations in sample | ⚪ |

**Stage G exit:** ~78k canonical chunks, audit clean. STOP-GATE: >1% R3 unusable cuts → HALT, re-tune.

---

## Stage H — Execute Phase 3–5 (Embed + Sparse + Upsert)
**Owner:** Claude on rig · **Time:** ~3 hours · **Reversible:** yes (drop cbic_v2 collection)

| Step | Command | Pass | Status |
|------|---------|------|--------|
| H1 | Phase 3: BGE-M3 dense embed pool | batch 32, GPUs 0/1/4/5/6 | 30+ ch/s; zero pool failures | ⚪ |
| H2 | Phase 4: BM25 sparse | fastembed | all chunks have sparse vectors | ⚪ |
| H3 | Phase 5: Upsert to `cbic_v2` | payload indexes per §5 spec | count matches ~78k | ⚪ |
| H4 | Post-upsert snapshot | `POST /collections/cbic_v2/snapshots` | snapshot file created | ⚪ |

**Stage H exit:** `cbic_v2` collection live, snapshotted. STOP-GATE: chunk count deviation >5% → HALT.

---

## Stage I — Execute Phase 6 (Shadow mode)
**Owner:** Claude on rig + real testers · **Time:** user-determined (days/weeks) · **Reversible:** yes

| Step | Command | Pass | Status |
|------|---------|------|--------|
| I1 | Deploy `/query_v2` endpoint | B5 code live | endpoint responds | ⚪ |
| I2 | Dual-writer middleware | every `/query` fires `/query_v2` async | log entries in shadow_log.sqlite | ⚪ |
| I3 | Divergence monitor | daily report on v1 vs v2 diff | divergence <2% or investigated | ⚪ |
| I4 | Real-tester soak | user + team exercise `/query_v2` | qualitative feedback collected | ⚪ |

**Stage I exit:** user declares soak done. STOP-GATE: divergence >2% sustained → HALT, investigate.

---

## Stage J — Execute Phase 7 (4-gate validation)
**Owner:** Claude on rig · **Time:** ~4 hours · **Reversible:** n/a (read-only measurement)

| Step | Command | Pass | Status |
|------|---------|------|--------|
| J1 | G1 recall@10 on 350–400 gold | `eval_g1.py` | ≥95% | ⚪ |
| J2 | G2 dual-judge on all answers | `eval_g2.py` Gemini + Claude ensemble | avg ≥4.5, ≥95% ≥4 both | ⚪ |
| J3 | G3 verbatim + Levenshtein fallback | `eval_g3.py` | ≥95% citations found | ⚪ |
| J4 | G4 50 adversarial + UPL | `eval_g4.py` | 100% refused | ⚪ |

**Stage J exit:** all 4 gates ≥ threshold. STOP-GATE: any gate fail → amend spec, re-run from failed phase. **NO PROMOTION** until all 4 green.

---

## Stage K — Execute Phase 8 (Promote or rollback)
**Owner:** Claude on rig + user sign-off · **Time:** ~15 min

| Step | Command | Pass | Status |
|------|---------|------|--------|
| K1 | User sign-off on J1–J4 results | explicit yes | user confirmed | ⚪ |
| K2 | Flip `/query` → `cbic_v2` | config change + reload | `/query` answers from v2 | ⚪ |
| K3 | Keep `cbic_v1` for rollback | D10 — never delete during project | collection intact | ⚪ |
| K4 | Post-promotion smoke test | 10 canary queries | no regressions | ⚪ |

**Stage K exit:** v2 is prod. Clock starts on monitoring.

---

## Stage L — Post-deployment Monitoring & Backlog
**Owner:** Claude + user · **Time:** ongoing

- L1: Weekly G1/G2/G3/G4 re-run on fresh queries (sampled from logs)
- L2: Monitor divergence, refusal rate, CE rerank confidence on Hindi queries
- L3: Execute post-v2 backlog when triggered (Multi-Query Rewriting, Step-Back, Self-RAG-lite, query cache, observability, amendment-graph, BM25 Hinglish, TTFT watchdog, feedback rate-limit)

---

## Hard Rules (apply at every stage)

1. **95% trust is non-negotiable.** No promotion without all 4 gates green.
2. **STOP-GATE protocol.** Any QA defect I cannot cleanly fix → I HALT and escalate.
3. **No destructive ops during project.** Delete goes to `cleanup_backlog.md`, not `rm`.
4. **Spec-amendment discipline.** Any in-flight change → amend SPEC.md + JOURNAL.md *before* re-running failed phase.
5. **Status answerable at any time.** User says "status" → I show this runbook with current state.

---

## Current state (live — updated after every completed step)

**Stage A: 9/14 complete** (64%)
- ✅ A1 chunking strategy patched (schema fields, R1 row-headers, R3 Hindi tokens, LLM prompt instructions)
- ✅ A2–A5 SPEC §1/§3/§4/Phase 7 patched + synced
- ✅ A6 decisions D4/D8/D14 + status line amended
- ✅ A7 risks R5/R7/R10 updated + R11–R15 added
- ✅ A8 §9 backlog section added (11 B-post items)
- ✅ **A9 SPEC STOP-GATE passed** (found + fixed 2 defects: Phase 2 R1 summary stale, D1 stale qwen3 text)
- 🟡 A10 PLAN_FOR_REVIEW mirror — next
- ⚪ A11 PLAN STOP-GATE
- ⚪ A12 memory backlog sync
- ⚪ A13 JOURNAL append
- ⚪ A14 FINAL STOP-GATE

**Stage B–L: not started**

**QA defects found + fixed so far: 3** (chunking prompt gap, SPEC Phase 7 sync ×4, SPEC stale references ×2). Zero defects carried forward.

**Blocking path to first ingestion run:** finish Stage A (4 items left, ~15 min) → Stage B build (6–10h) → Stage C eval expansion (user time ~6h) → Stage D close 12 probes (~2h) → Stage E pre-flight (<30min). Earliest Phase 1 execution: ~2 work sessions.

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


---

## 2026-04-25 ADDENDUM 2 — phase6_pairs added; gold-source rule changed

### Phase order (new)
```
phase1 (manifest) → phase2 (chunk) → phase3_4_5 (embed+upsert) →
  phase6_pairs (per-chunk pair generation) → G1 → G2 → G3 → G4 → G5
```

phase6_pairs is non-optional for any new cohort going forward. Set 5 will
NOT be back-filled — it remains the legacy-gold validation point for the
chunker recipe lock. Set 6 onwards and the full re-ingest run with phase6.

### Gold sourcing for G1 (changed)
Old: "filter legacy pool to scope's doc_ids"  → DEPRECATED.
New: "union of (phase6 pairs for scope, legacy pool filtered to scope)".
Sample 5–8 queries per doc per source for G1 eval gold; retain full pair
files for training-corpus accumulation.

### Per-cohort runner (updated)

```bash
ssh -o ControlPath=none -i ~/.ssh/id_ed25519_rig root@192.168.1.107
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

# 3. phase6_pairs (NEW — per-chunk pair generation with hard negatives)
QDRANT_COLL_V2=cbic_v2_setN \
  GEMINI_API_KEY="$(grep ^GEMINI_API_KEY /root/.cbic_env | cut -d= -f2)" \
  PYTHONPATH=/home/user/.local/lib/python3.10/site-packages:/opt/indian-legal-ai/rag/cbic_rag \
  python3 ingest_v2.py --phase phase6_pairs --scope setN

# 4. Build G1 gold from union
python3 evaluators/build_g1_gold_union.py --scope setN \
  --out eval/scale_sets/setN/v2_gold_setN_union.json

# 5. G1 → G5 in sequence
python3 evaluators/gate_g1_recall.py --collection cbic_v2_setN \
  --gold eval/scale_sets/setN/v2_gold_setN_union.json \
  --retrieve-only --out evaluators/gate_g1_setN.json
python3 evaluators/gate_g2_dual_judge_parallel.py --collection cbic_v2_setN --workers 8 ... # 2026-04-25: parallel adopted as canonical (8x speedup, claude-cli + gemini API)
python3 evaluators/gate_g3_answer_quality.py ...
python3 evaluators/gate_g4_adversarial.py ...
python3 evaluators/gate_g5_latency_cost.py ...
```

### Hard-rule references
- Operational parameters: `reingest_spec/DECISIONS.yaml` (single source of truth)
- Pair schema: `~/.claude/projects/D---gpu-rig-ai/memory/pair_schema_cbic_v2.md`
- Phase6 design: `reingest_spec/PAIR_GEN_SPEC.md`

---

## 2026-04-26 ADDENDUM 3 — FULL RE-INGEST EXECUTION (Stage M)

**Goal:** ingest all 14,925 docs into single shared `cbic_v2` Qdrant collection at ≥95% across all 4 trust gates. Live state tracked in `reingest_spec/INGEST_TRACKER.md` (per-batch row updated as each batch finishes).

**Architecture:** 10 batches × ~1500 docs, deterministic stratified picks (SEED=42 + batch_num). Single collection, no per-batch collection or merge. Manifest at `/opt/indian-legal-ai/data/ingest_manifest_v2_full.sqlite` (separate from set5/set6 manifests).

### Checkpoint policy (3 stops, not 10)

| CP | After batch | Gates | Stop condition (HALT, no patch-and-continue) | Time |
|---|---|---|---|---|
| **CP-1: Pipeline smoke** | 1 (1.5k docs) | G3 + G4 grounded on Set 6 gold subset overlapping batch 1; 5 hand-picked /query smokes across 6 prefix families | G3 < 0.90 OR G4 < 0.95 → HALT | ~10 min |
| **CP-2: Mid-corpus drift** | 5 (~7.5k docs) | G1 + G3 + G4 on union(Set 5, Set 6) gold; reranker score histogram check (gold≥0.7 / adv<0.65) | G1 < 0.95 OR G3 < 0.92 OR G4 < 0.95 → HALT | ~15 min |
| **CP-3: Final acceptance** | 10 (14.9k docs) | G1 + G2-full dual-judge n=380 + G3 + G4 + G5 | ANY gate < 0.95 → HALT, fix spec, full re-run (Hard Rule #1) | ~45 min + ~$4 G2 |

**Why not after every batch:** (a) batches are deterministic disjoint picks — no per-batch quality variance to monitor; what matters is corpus-density effects which surface at scale. (b) G2 cost ~$4/run × 10 = $40 wasted. (c) gates serialize against ingest API.

### Stage M live status

**See `reingest_spec/INGEST_TRACKER.md` for live-updated batch table with per-batch gate values.**

Current state: see top of INGEST_TRACKER.md "Stage tracker" section.

### Stage M exit
All CP-3 gates ≥0.95. Then snapshot `cbic_v2`, promote per Stage K, deprecate `cbic_v1`.

### Per-batch A-Z SOP (every batch follows this exactly — no skipping)

**Pre-flight (once per session, not per batch):**
- A0. SSH to rig: `ssh -i ~/.ssh/id_ed25519_rig root@192.168.1.107`
- A1. Verify services up: `ss -ltn | grep -E '6343|9082|9085'` → Qdrant 6343, qwen3 9082, reranker 9085 all LISTEN
- A2. Verify GPU pool healthy: `nvidia-smi` (or vulkaninfo) → GPUs 4,5,6 idle, GPU 0 reranker, GPU 2 qwen3
- A3. `set -a; source /root/.cbic_env; set +a` (loads GEMINI_API_KEY, OPENROUTER_API_KEY)
- A4. Confirm DENSE_ONLY=1, EMBED_GPUS=4,5,6 (or post-bench expanded set), MANIFEST_V2 path
- A5. Confirm telemetry sampler PID alive: `pgrep -af ingest_telemetry` (writes to `/tmp/ingest_telemetry.csv`)

**Per-batch run (steps B-O):**
- **B. Build batch:** `python3 reingest_spec/build_batch.py --batch N` → writes `/opt/indian-legal-ai/data/batches/batch{N}_doc_ids.csv` + `_audit.json`. Verifies SEED=42+N stratified pick, excludes already-ingested via manifest sqlite.
- **C. Pre-launch sanity:** verify `batch{N}_doc_ids.csv` size ≈ 1500 (last batch ≈ 1425), audit JSON shows category distribution matches corpus.
- **D. Launch ingest:** `python3 reingest_spec/build_batch.py --batch N --ingest` → spawns `ingest_v2.py --phase all --doc-ids ...` PID logged to `/opt/indian-legal-ai/logs/reingest_batch{N}_<ts>.log`.
- **E. Monitor Phase 2 (chunk+dedupe):** target ~117s for 1500 docs at 12.7 doc/s. Watch for `[phase2 DONE] N docs → M raw → K canonical, X% dedupe`. Hard stop if rate < 5 doc/s — likely qwen3 not bypassing.
- **F. Monitor Phase 3-5 (embed+upsert):** target ~190-220s at 22 ch/s. Watch for `[phase3-5 DONE] X chunks submitted, qdrant points_count=Y in Zs` + `RECONCILE all N docs have expected==upserted`.
- **G. Verify completion signals:** (a) ingest process exited 0; (b) reconcile PASS; (c) Qdrant `points_count` increased by ~4500-5000; (d) zero `[embed err]`, zero `nan`/`inf` in log; (e) zero Qdrant 400s (or halve-retry recovered all).
- **H. Record batch row:** update `INGEST_TRACKER.md` batch row: status=DONE, docs, Phase 2 (s), Phase 3-5 (s), Total (min), Pts after.
- **I. Spot-check 3 random doc_ids in Qdrant:** `curl http://127.0.0.1:6343/collections/cbic_v2/points/scroll -d '{"filter":{"must":[{"key":"doc_id","match":{"value":"<id>"}}]},"limit":1}'` → must return ≥1 point with non-empty vector + payload.
- **J. Save manifest snapshot:** `cp /opt/indian-legal-ai/data/ingest_manifest_v2_full.sqlite /opt/indian-legal-ai/data/snapshots/manifest_after_batch{N}.sqlite`
- **K. Snapshot Qdrant collection (cheap, recoverable):** `curl -X POST http://127.0.0.1:6343/collections/cbic_v2/snapshots` → returns snapshot path on rig disk (allows rollback if mid-corpus drift).
- **L. CP gate (only on CP batches: 1, 5, 10):** see Checkpoint policy table above. **HALT if any gate fails — do not launch next batch.**
- **M. Append JOURNAL.md entry:** date-stamped block with: batch N, doc count, chunk count, points after, phase timings, anomalies, errors, decisions made.
- **N. Append optimization findings (if any):** to `INGEST_OPTIMIZATIONS.md` if rate slipped or new bottleneck surfaced.
- **O. Green-light next batch:** all H-N done + (if applicable) CP gate passed → proceed to A0 of batch N+1.

**Stop conditions across all batches (any one → HALT):**
- Phase 2 rate < 5 doc/s
- Phase 3-5 rate < 10 ch/s sustained for 60s+
- Embed pool drops below 3 GPUs ready
- Qdrant 400/500 not recovered by halve-retry
- Reconcile FAIL (expected ≠ upserted canonical chunks)
- CP gate < threshold
- Any G-gate measured between batches drops > 0.03 from prior batch (drift signal)

### Operational refs (frozen 2026-04-26)
- Batch builder: `reingest_spec/build_batch.py` — `python3 build_batch.py --batch N --ingest`
- Patches deployed for batch 1 retry: `rag/cbic_rag/ingest.py:upsert_chunks` NaN/Inf sanitiser; `reingest_spec/ingest_v2.py:_flush_batch` halve-and-retry on Qdrant 400.
- Bypass-prefixes 12-prefix list APPROVED for full re-ingest (DECISIONS.yaml `chunker.bypass_prefixes_status_2026_04_26`).
- Reranker config locked at `-c 32768 --parallel 8` (4096 tokens/slot).
- Qdrant: docker container `qdrant-cbic`, port **6343** (not 6333), mapped 6343→6333.
- Manifest: `/opt/indian-legal-ai/data/ingest_manifest_v2_full.sqlite` (separate from set5/set6).
- Telemetry: `/tmp/ingest_telemetry.csv` (sampler) → snapshot to `INGEST_TELEMETRY.csv` after each batch.
- Per-batch log pattern: `/opt/indian-legal-ai/logs/reingest_batch{N}_<ts>.log`
- Audit JSON pattern: `/opt/indian-legal-ai/data/batches/batch{N}_audit.json`
- Gold subset for CP-1 G3: `reingest_spec/eval/v2_gold_cp1_batch1.json` (38 queries, batch-1 overlap)
- Gold for CP-2 G1+G3+G4: `reingest_spec/eval/v2_gold.json` ∪ `v2_gold_set6.json`
- Adversarial for G4: `reingest_spec/eval/v2_adversarial_clean_v2.json` (201 OOC queries, 7 categories)
- Smoke prefixes for CP-1: notification, circular, act, rule, form, instruction (one /query each + 1 hand-picked)

### Evaluator commands (copy-paste, frozen)

```bash
# CP-1 (after batch 1)
python3 reingest_spec/evaluators/gate_g3_levenshtein.py --collection cbic_v2 \
  --gold reingest_spec/eval/v2_gold_cp1_batch1.json \
  --out data/eval/gate_g3_cp1_batch1.json
python3 reingest_spec/evaluators/gate_g4_grounded.py --collection cbic_v2 \
  --adv reingest_spec/eval/v2_adversarial_clean_v2.json \
  --out data/eval/gate_g4_cp1.json --threshold 0.95 --refuse-on no --workers 4

# CP-2 (after batch 5)
python3 reingest_spec/evaluators/gate_g1_recall.py --collection cbic_v2 --out data/eval/gate_g1_cp2.json
python3 reingest_spec/evaluators/gate_g3_levenshtein.py --collection cbic_v2 --out data/eval/gate_g3_cp2.json
python3 reingest_spec/evaluators/gate_g4_grounded.py --collection cbic_v2 --out data/eval/gate_g4_cp2.json --threshold 0.95

# CP-3 (after batch 10)
# All four parallel-safe except G2 (must run alone — judge contention).
python3 reingest_spec/evaluators/gate_g1_recall.py --collection cbic_v2 --out data/eval/gate_g1_cp3.json
python3 reingest_spec/evaluators/gate_g3_levenshtein.py --collection cbic_v2 --out data/eval/gate_g3_cp3.json
python3 reingest_spec/evaluators/gate_g4_grounded.py --collection cbic_v2 --out data/eval/gate_g4_cp3.json --threshold 0.95
python3 reingest_spec/evaluators/gate_g5_latency_cost.py --collection cbic_v2 --out data/eval/gate_g5_cp3.json
# G2 last, alone:
python3 reingest_spec/evaluators/gate_g2_dual_judge_parallel.py --collection cbic_v2 --n 380 --out data/eval/gate_g2_cp3.json
```
