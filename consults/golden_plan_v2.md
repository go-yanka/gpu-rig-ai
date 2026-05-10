# Golden Plan v2 — CBIC RAG Post-A3-Regression Recovery

**Date:** 2026-04-21
**Baseline to beat:** 33.23% (50 items, A3 OFF)
**Targets:** 55–65% in 2 weeks · 75%+ in 4 weeks
**Basis:** synthesized from two external LLM consults (GPT-class + Gemini-class) + our own measurements

---

## 1. Context snapshot

| Run | Config | Score | p95 latency |
|---|---|---:|---:|
| Pre-A3 baseline | single-pass | **33.23%** (106/319) | 55.8s |
| A3 ON | two-pass (JSON → synth, no chunks) | **21.63%** (69/319) | 79.8s |
| Delta | | **−11.6 pts / −35% rel** | **+24s** |

A3 flipped OFF. `bucket_1_tariff_v3_llm2.yaml` (25 items) verified clean, ready to merge.

---

## 2. Diagnosis (both LLMs agreed independently)

**A3 regression = cascade failure, not a single bug.**

1. `qwen3-14b` in JSON-schema mode truncates / reformats legal spans
2. Our 5-level validator (Jaccard ≥ 0.80 + cosine ≥ 0.92) rejects these slightly mangled spans
3. Pass-2 receives skeletal JSON + **no chunks** → model falls back on pre-trained weights → generic, ungrounded answers
4. JSON-constrained generation on Vulkan is also slower than free-text → latency spike

**Fix (not retirement) has three parts:**
- Re-feed retrieved chunks to pass-2 with "cite only from JSON, use chunks for context" instruction
- **Drop the 6-gram Jaccard rung** — too brittle against LLM formatting
- Loosen cosine to ≥0.85 (from 0.92); 3-level funnel: length+clause → NFKC substring → BGE cosine

---

## 3. Agent-eligibility key

- 🤖 **Auto** — fully delegable to a spawned agent
- 🔧 **Semi** — agent drafts, I deploy to rig (SSH + eval)
- 👤 **Human** — needs your approval (factual checks, production risk)
- ⚙️ **Rig-serial** — requires service restart + eval; only one at a time

---

## 4. Phase P0 — Diagnose & stabilize (Day 1, ~4 hours)

| # | Task | Time | Agent? | Depends on | Status |
|---|---|---|---|---|---|
| P0.1 | Chunker audit (AST vs recursive?) | 30 min | 🤖 | — | ✅ DONE — hierarchy-aware, not a bug |
| P0.2 | Pull pass-1/pass-2 JSON from query_log for 8 A3-zeroed items | 30 min | 🤖 | — | 🏃 running |
| P0.3 | Confirm cascade hypothesis from P0.2 data | 1 hr | 🤖 | P0.2 | pending |
| P0.4 | Fix chunker if needed (hard-section splits, word-boundary) | 2–6 hrs | 🔧 | P0.1 | **SKIPPED — not a priority-zero bug** |

### Chunker audit findings (P0.1)
- Custom `HierarchyTracker` with regex for Chapter/Section/Rule/subsection/clause
- 3500 char target, 5500 max, 700 overlap
- Hierarchy prepended into embed_text
- Tables separately via pdfplumber

**Three weaknesses (downgraded to P3 polish):**
1. Soft hierarchy — labels but doesn't force section splits
2. Mid-word cuts on form-heavy pages (hard-cap fallback)
3. Some tables fall to narrative splitter (pdfplumber miss)

---

## 5. Phase P1 — Quality recovery (Days 2–4)

**Goal:** recover baseline, hit 40–45%.

| # | Task | Time | Agent? | Depends on |
|---|---|---|---|---|
| P1.1 | Deploy A1 P1 retrieval boost (already drafted) | 30 min | 🔧 | P0.4 skipped |
| P1.2 | Simplify verbatim gate: drop Jaccard, cosine ≥0.85 | 1 hr | 🔧 | — |
| P1.3 | Fix A3: re-feed chunks to pass-2 + decomp verifier | 2 hrs | 🔧 | P0.3 |
| P1.4 | Eval run after P1.1 | 45 min | ⚙️ | P1.1 |
| P1.5 | Eval run after P1.2+P1.3 | 45 min | ⚙️ | P1.4 |
| P1.6 | Decision: proceed or revert | 15 min | 👤 | P1.5 |

**Parallelism:** P1.2 drafted while P1.1 deploys; P1.3 drafted while P1.4 runs.
**Critical constraint:** eval runs are serial, ~45 min each.

---

## 6. Phase P2 — Structural lift (Days 5–10)

**Goal:** 55–65% overall, 80%+ on tariff category.

| # | Task | Time | Agent? | Depends on |
|---|---|---|---|---|
| P2.1 | A4 table pipeline — SQLite `tariff.db` (HSN → rate → notif) | 4 hrs | 🤖 build + 🔧 wire | P1 green |
| P2.2 | A4 query router — HSN/rate queries hit SQLite first | 2 hrs | 🔧 | P2.1 |
| P2.3 | Version-pinning: `effective_from/to` in payload | 2 hrs | 🔧 | — |
| P2.4 | Hard payload filter: regex NER → Qdrant `must` filter | 3 hrs | 🔧 | — |
| P2.5 | Query decomposition for multi-part questions | 3 hrs | 🔧 | — |
| P2.6 | Eval runs after each | 45 min × 5 | ⚙️ | each |

**Parallelism:** P2.1/P2.3/P2.4/P2.5 all parallel-drafted by agents.
**Rig cost:** ~4 hrs total eval time.

---

## 7. Phase P3 — Foundations (Days 11–20)

**Goal:** 70–80%.

| # | Task | Time | Agent? | Depends on |
|---|---|---|---|---|
| P3.1 | A2 corpus refresh (CGST Rules + IGST Act latest) | 6–12 hrs (4 rig) | 🔧 | — |
| P3.2 | A5 query-class routing | 4 hrs | 🔧 | P2 green |
| P3.3 | Amendment-chain SQLite sidecar (lightweight notif graph) | 4 hrs | 🤖 + 🔧 | P2.3 |
| P3.4 | Selective context compression post-retrieve | 3 hrs | 🔧 | — |
| P3.5 | Chunker polish (hard section splits, word-boundary) | 2 hrs | 🔧 | — |
| P3.6 | Eval runs | 45 min × 4 | ⚙️ | each |

---

## 8. Phase E — Eval gold-set expansion (parallel to all)

**Goal:** 50 → 175 hand-verified pairs across 6 buckets.

| # | Bucket | Size | Status |
|---|---|---:|---|
| E.1 | 1 — tariff/rate/HSN/SAC/notifs | 25 | ✅ verified, ready to merge |
| E.2 | 2 — no-evidence/refusal | 10 | 🏃 Gemini in progress |
| E.3 | 3 — complex multi-section | 25 | pending |
| E.4 | 4 — ST+IT crossover | 20 | pending |
| E.5 | 5 — customs advanced | 20 | pending |
| E.6 | 6 — others (appeals/GAAR/penalty) | 20 | pending |
| E.7 | Merge all into `gold_set.yaml` | — | pending |

**Runs independently.** Each bucket = ~30 min human loop (paste prompt → Gemini → paste response → spot-check → merge).

---

## 9. Explicit deferrals (rig / cost constraints)

| Item | Why deferred |
|---|---|
| Cross-encoder reranking | 4-core CPU; would crush latency (Gemini agreed) |
| Chain-of-verification | Doubles LLM latency; p95 already 55s |
| Full notification graph DB (Neo4j) | Use SQLite sidecar instead |
| LoRA fine-tuning | Retrieval is bottleneck, not weights |
| Anti-hallucination ledger inline | Too much latency; add as offline diff |
| Judge-grounded classifier | Wait until gold set ≥150 items |

---

## 10. Dependency graph (critical path)

```
P0.1 chunker audit ✅ ──┐
P0.2 A3 log dump 🏃 ────┼──► P0.3 cascade confirm ──► P1.3 fixed A3 ──┐
                        │                                              ├──► P1.4/5 eval ──► P2
P1.1 deploy A1 ────────┤                                              │
P1.2 simplify gate ────┘                                              │
                                                                      ▼
P2.1 tariff.db ──► P2.2 router ──┐
P2.3 version-pin ────────────────┼──► P2.6 evals ──► P3
P2.4 hard filter ────────────────┤
P2.5 query decomp ───────────────┘

P3.1 A2 corpus (independent, long-running)
P3.2 A5 routing (needs P2)
P3.3 amendment graph (needs P2.3)
P3.4 context compression (independent)
P3.5 chunker polish (independent)

Eval expansion E.1–E.7 — independent track, no blockers
```

---

## 11. Time budget

| Phase | Wall-clock | Agent hrs | Rig-serial hrs |
|---|---|---|---|
| P0 | 1 day | 2 | 0 |
| P1 | 2–3 days | 4 | 2 |
| P2 | 1 week | 14 | 4 |
| P3 | 1.5 weeks | 15 | 3 |
| **To 75%** | **~4 weeks** | **~35** | **~9** |
| E (parallel) | 2 weeks | human loop ~6 hrs | 0 |

---

## 12. Gating rules

- Each phase completes only if eval stays ≥ previous checkpoint
- Any patch that regresses >15% → automatic revert
- A3 re-deploy: go/no-go after P1.5 eval
- P2 entry: requires P1 ≥ 40%
- P3 entry: requires P2 ≥ 55%

---

## 13. Immediate next actions

1. ✅ Bucket 1 (25 items) verified — merge into `gold_set.yaml` when user says
2. 🏃 P0.2 A3 log dump — agent running in background
3. 🏃 Bucket 2 (refusal) — Gemini running on user's screen
4. Then: sequential buckets 3–6, interleaved with P1 patches as they land

---

## 14. Source documents

- `D:\_gpu_rig_ai\consults\cbic_rag_status_and_questions_v1.md` — consolidated brief sent to external LLMs
- `D:\_gpu_rig_ai\consults\training_data_generation_plan.md` — earlier training-data consult
- `D:\_gpu_rig_ai\consults\cbic_rag_dual_plan_v3.md` — fix-current vs from-scratch comparison
- `C:\Users\Rahul Goyanka\.claude\projects\D---gpu-rig-ai\memory\rag_quality_playbook.md` — 5-level verifier ladder + anti-patterns
- `D:\_gpu_rig_ai\eval\runs\baseline_a3off_20260421_220214\summary.md` — 33.23% baseline
- `D:\_gpu_rig_ai\eval\runs\a3on_20260421_222520\summary.md` — 21.63% A3-on regression
- `D:\_gpu_rig_ai\eval\gold_set_expansion\bucket_1_tariff_v3_llm2.yaml` — bucket 1 ready to merge
