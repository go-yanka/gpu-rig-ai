# CBIC RAG — Re-Ingestion Plan for External LLM Review

**Reviewer brief:** You are reviewing a production plan to rebuild a legal-RAG system serving Indian tax law (CBIC: GST, Customs, Central Excise, Service Tax). Target: **95% answer trust** on first ingestion pass. System is live; ~115k chunks already indexed but section-ref metadata fill is 17% (root cause of poor evidence citation). We are rebuilding the index with a new chunker and metadata schema.

**What we want from you:**
1. Hole-poke the plan. What will break that we haven't accounted for?
2. Challenge the 4-gate 95% definition. Is it the right contract?
3. Flag any hidden coupling / single-point-of-failure we missed.
4. Propose probe or gate additions if you see under-covered risk.

---

## System Context

**Hardware:**
- Ubuntu mining rig, 4 CPU cores, 7 AMD GPUs (6700 XT / 6600 class)
- Services live: Qdrant (6343), qwen3-14b llama-server (9082, Vulkan), LiteLLM (4444), FastAPI `api.py` (9500)
- Embeddings: BGE-M3 multi-GPU pool across GPUs 0,1,4,5,6 via subprocess-per-GPU (`embedder_direct.py`), ~30 chunks/sec batched

**Data:**
- 851 source PDFs: 7 were unOCRable, 844 usable
- ~500 born-digital (PyMuPDF extract), ~344 scanned (OCR'd with Gemini 2.0 Flash)
- Languages: English (majority), Hindi (bilingual twin documents for major acts)
- Gold eval set: 170 queries with `expected_sections`, `expected_terms`, `must_cite_verbatim`, `must_not_say` fields
- Adversarial set: 20 out-of-corpus queries (G4 refusal test)

**Retrieval stack:**
- Hybrid RRF: BGE-M3 dense (1024d) + fastembed BM25 sparse
- Router: keyword regex → optional qwen3-14b LLM router → fallback `gst`
- Rerank: Cross-encoder top-20 → top-10
- Answer: qwen3-14b, no_think mode, evidence-grounded prompt

---

## Current Baseline (cbic_v1, to be replaced)

| Metric | Value | Target |
|--------|-------|--------|
| Total points | 114,626 | — |
| `text` fill | ~99% | ≥99% ✅ |
| `section_ref` fill | **17%** | ≥80% 🔴 root cause of G3 miss |
| `lang` fill | 0% | ≥98% 🔴 |
| `source` fill | 0% | ≥99% 🔴 |
| `hindi_twin` link | — | new field 🔴 |
| OCR table structure | flattened | preserved as markdown 🔴 |
| Known recall@10 | unmeasured | ≥95% |

---

## The Four Gates (pass/fail contract)

| Gate | Measurement | Pass |
|------|-------------|------|
| **G1 Accuracy** | recall@10 vs gold `expected_sections` | ≥95% queries |
| **G2 Reasoning** | Gemini-judge 1–5 score on answer reasoning trace | avg ≥4.5; ≥95% queries ≥4 |
| **G3 Evidence** | `must_cite_verbatim` substring found in retrieved chunks | ≥95% queries |
| **G4 Refusal** | 20 OOC adversarial queries | 100% refused |

**Why these four:** Previous painful lesson — the system can retrieve right chunks (G1) but still generate hallucinated reasoning (G2 miss), or retrieve right chunks and cite wrong excerpts (G3 miss), or confidently answer things not in corpus (G4 miss). All four must hold simultaneously.

---

## Pipeline (Phase 0 → 8, abridged; full detail in SPEC.md)

- **P0** Snapshot + cleanup backlog + code freeze
- **P1** Unified manifest SQLite with Hindi-twin linking
- **P2** Chunker v2: hierarchy-aware, 3500/5500/700, 17 payload fields
- **P3** Dense embed (BGE-M3 multi-GPU)
- **P4** Sparse embed (BM25)
- **P5** Upsert to new collection `cbic_v2` (keep v1 as rollback)
- **P6** Shadow mode: `/query` (v1) + `/query_v2` (v2) dual-write to `shadow_log.sqlite`; real human testers exercise v2
- **P7** Run 4 gates
- **P8** Promote v2 or amend spec

---

## Key Design Decisions (locked or TBD)

| Decision | Value | Rationale |
|----------|-------|-----------|
| Keep old v1 during transition | yes | rollback safety; no deletes during project |
| Cutover style | shadow with human testers + automated dual-write log | low risk; gathers real diffs before flip |
| Hindi handling | English queries, cite English + Hindi twin chunk when exists | users are English-first; Hindi is for legal primary-source reference |
| Chunk policy | 3500 char target, 5500 cap, 700 overlap, hierarchy-preferred splits | proven from optimization playbook |
| LLM extraction backstop for section_ref | **TBD** — depends on probe V1/V2 (qwen3-14b feasibility) vs V18 (Claude CLI) | cost/latency tradeoff |
| Gemini spend cap | **TBD** — depends on probe V9 quota + V8 table-reOCR volume | budgetable after probe |
| Refusal threshold θ_retrieve | **TBD** — V16 determines from score distributions | empirical |

---

## 24 Validation Probes (gate: all pass or workaround before P1)

See `PROBES.md`. Organised into 5 waves:
- **A (Qdrant HTTP, laptop-runnable):** V6, V7, V15, V16, V21, V22
- **B (LLM evaluation):** V1, V2, V10, V17, V18 — resolve LLM-backstop decision
- **C (Chunker behaviour):** V3, V4, V11, V13, V20, V24
- **D (Infra):** V5, V8, V9, V12, V19, V23
- **E (Gate-validity):** V14

---

## Open Questions We Want Challenged

1. **Is shadow mode enough?** We rely on real human testers for 95% confidence during soak. Should we add synthetic adversarial traffic generator?
2. **θ_retrieve per-category vs global?** Categories may have different score distributions. Worth the complexity?
3. **Hindi twin citation ergonomics.** When we cite both English + Hindi, how do answers present this without noise?
4. **Is 3500/5500/700 still right?** With new hierarchy-aware splits, smaller chunks might win. Probe V13 measures cost, not quality.
5. **G3 substring match is literal.** What if the paraphrase is correct but verbatim doesn't match due to OCR diacritic? Need tolerance rules?
6. **R5 (qwen3-14b extraction too slow).** If 115k chunks × 3s each = 96 hours, is Gemini the only realistic backstop? Is Claude CLI rate-limited in ways we haven't tested?
7. **Are we missing a gate?** E.g. "no-contradiction between answer and retrieved evidence" — currently rolled into G2 reasoning judge but could be its own gate.

---

## What we are NOT doing this pass

- New document sources (Income Tax, MCA, Labour, RBI) — explicitly deferred per user
- Multi-modal (image-in-answer) — no
- Streaming tokens in shadow mode (simpler dual-write first)
- Changing the LLM (qwen3-14b remains; no plan to swap)
- Claude CLI as runtime answer LLM — only considered as extraction backstop (V18)

---

## Ask

Please identify:
1. The **highest-risk thing we've under-specified**
2. One **gate we should add**
3. One **probe we're missing** that would change a Phase decision
4. Any **ordering mistake** in P0→P8
