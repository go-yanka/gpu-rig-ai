# Consultation Brief — CBIC Tax-Law RAG, retrieval recall plateau at 0.84
**For external LLMs / consultants. Self-contained. Last updated: 2026-05-08.**

We need help getting our retrieval recall@10 from **0.84 → ≥0.95** on a 380-query gold set. After ~13 hours of iteration today (3rd full corpus re-ingest), every standard lever we pulled has plateaued at 0.84. Looking for non-obvious approaches. We are open to **anything** including re-chunking, swapping the embedding model, fine-tuning, generating synthetic data, or restructuring the corpus.

---

## 1. Domain & corpus

- **Domain**: Indian indirect-tax law and customs regulations published by the Central Board of Indirect Taxes and Customs (CBIC).
- **Corpus**: ~14,000 ingested documents covering CGST/IGST Acts, allied acts (Customs Act, FEMA, NDPS, Air Cargo), rules, notifications, circulars, instructions, orders, regulations, forms. Many shared-PDF clusters (e.g. one master PDF `CGST-Rules-2017-Part-B-Forms.pdf` is referenced by 176 different doc_ids in the manifest).
- **Languages**: English (primary) + Hindi twins for some docs (handled via separate chunk-id linking).
- **Document types**: Acts (long, hierarchically sectioned, 200+ pages), Rules (hierarchical, 50-150 pages), Notifications (1-10 pages, often amendments to specific schedules), Circulars (1-30 pages, clarifications), Instructions, Forms (templates), Regulations.
- **Source quality**: ~5% of corpus has structural issues — D-2a (no PDF on disk: 292 docs), D-2b (junk content: HTML error page saved as PDF: 7 docs), D-1 (shared-PDF cluster losers: 532 docs across 163 clusters). These are documented carve-outs.

## 2. Vector store + retrieval architecture

- **Qdrant** (Docker, hybrid collection schema)
- **Dense vectors**: BGE-M3 1024-dim, computed via `llama-cpp-python` on AMD GPUs (Vulkan), in-process embedder pool of 6 GPUs (5×RX 5700 XT 8GB + 1×RX 6700 XT 12GB)
- **Sparse vectors**: fastembed `Qdrant/bm25` (CPU)
- **Rerank**: `bge-reranker-v2-m3-Q4_K_M` (Vulkan llama-server, port 9085, --parallel 8)
- **LLM** for HyDE / answer generation / classify: `qwen3-14b-q4_k_m.gguf` (Vulkan llama-server on a 12GB card, single-slot --parallel 1)
- **Retrieval flow** at query time:
  1. `/retrieve` endpoint receives `{question, k, collection}`
  2. (Optional) HyDE: qwen3 rewrites question into hypothetical-answer form (we tested both on/off)
  3. Embed search text via BGE-M3 (dense, 1024d)
  4. Embed query via BM25 (sparse)
  5. Hybrid prefetch: Qdrant RRF over dense + sparse, returns top-K initial (we tried K=20, K=100)
  6. Cross-encoder rerank: bge-reranker scores top-N candidates, keeps top-10
- **Chunker config**:
  - TARGET 3500 chars, CAP 5500, hard CEILING 8000, FLOOR 500, OVERLAP_MID 700
  - Section-aware splitter for hierarchical docs (Act/Rules)
  - Plan-driven (qwen3-14b classifies doc structure first, then deterministic rules)
  - Recently added R9 hard-ceiling enforcement (2026-05-08): no chunk > 8000 chars
  - Final corpus: ~47,000 chunks across 14,000 docs (median 1 chunk/doc, p75=2, max=438)

## 3. Trust criterion (the bar)

Per spec, system must hit **≥95% on each of four trust gates**:

| Gate | What it tests | Threshold |
|---|---|---|
| **G1 Accuracy** | Right doc in top-10 retrieved | recall@10 ≥ 0.95 on **380-query gold** |
| **G2 Reasoning** | LLM answer factually correct | dual-judge (Gemini + Claude) mean ≥ 0.95 |
| **G3 Evidence** | Citations verifiable | substring or Levenshtein ≥0.95 fallback in top-K |
| **G4 Refusal** | Refuses out-of-corpus queries | refusal_rate ≥ 0.95 on 201 adversarial queries |

**Independent auditors will rerun gates as defined**. No "adjusted" metrics, no carve-outs from gold denominator. The 380-query gold is fixed; we cannot prune it.

## 4. Current results

| Gate | Current | Bar | Status |
|---|---|---|---|
| G4 (strict, refuse on no-or-partial) | **0.9851** | 0.95 | ✅ PASS |
| G1 (recall@10, full 380, no adjustment) | **0.8368** (v3 best) | 0.95 | ❌ FAIL by 0.11 |
| G3 | not run | 0.95 | likely 0.85ish |
| G2 | not run (Claude CLI patched, ready) | 0.95 | unknown |

**G1 is the blocker.** G3 builds on G1. G2 depends on whether the LLM can answer faithfully from retrieved evidence (so a good G1 makes G2 easier).

## 5. What we've tried (G1-specific)

| Configuration | Result |
|---|---|
| Dense-only (BGE-M3) | 0.5474 |
| Dense + sparse hybrid (RRF + rerank) | **0.8447** ← biggest single jump (+0.30) |
| Hybrid + HyDE on long-scenario queries | 0.8395 (slight regression) |
| Hybrid + TOP_K_RETR=100 (5× larger initial pool) | wash (~0.85) |
| **Full re-ingest with R9 ceiling enforcement (split chunks > 8000)** | **0.8368** (slight regression) |

**Diagnostic finding (the most important one)**: we sampled 25-30 missed gold queries and ran `/retrieve` with `k=50`. **24-29 of them have the expected doc NOT in top-50** of the hybrid dense+sparse retrieval — at any rank. Only ~1-5 are in the rank 11-50 range where reranker θ tuning could help. The rest are simply unfindable by current dense or sparse signals.

## 6. The fundamental problem (our hypothesis)

Gold queries are **long realistic business scenarios** like:
> "Hindalco-Bharat Copper Works LLP, a manufacturer of copper wires and copper alloy rods in Gujarat, wishes to claim countervailing duty exemption on imported copper-cathode raw material…"

The expected document is a specific notification like `cbic-notification-msts:1002110` whose actual chunk text is something like:
> "The principal notification No. 5/97-Cus dated 13.01.97 is amended in the manner specified in the Schedule annexed hereto, namely…"

There is **no lexical or semantic similarity** between the long company-name scenario and the formal-amendment notification text. BGE-M3 (general-purpose) gives them a low similarity score. BM25 also fails because the keywords (`Hindalco`, `copper-cathode`) don't appear in the notification — the notification just refers to "copper wire" generically.

Hypothesis: the retrieval gap is between **query intent** and **chunk surface form**. A user asks "X company importing Y product wants exemption for Z" but the answer chunk says "schedule entry 84 amended to read 12% rate."

## 7. What the 0.84 plateau represents

We've tested:
- Bigger initial retrieval pool (no help)
- Adding HyDE / question rewriting (slight regression)
- Splitting mega-chunks to expose internal sections (slight regression — fragmenting actually hurt)
- All retrieval-stack tuning levers we know

We're convinced the bottleneck is **the embedding model's ability to map long-scenario queries to formal-statute text**, not chunking, not reranker θ, not initial-pool size.

## 8. Things we have NOT tried (and are open to)

We are **open to any of these** if you think they'll help:

- **Swap embedding model**: try a legal-domain-tuned model (e.g. nlpaueb/legal-bert, custom) instead of BGE-M3
- **Fine-tune BGE-M3** on our domain — we have ~2.5M synthetic Q-chunk pairs from prior sessions in `eval/training_pairs/`
- **Synthetic Q-A enrichment at ingest**: for each chunk, use qwen3 to generate 3-5 paraphrase questions about the chunk content, embed those alongside the chunk text. Match queries to question-embeddings.
- **Two-stage retrieval**: first retrieve "topic clusters" via category/section filters, then dense within cluster
- **Re-chunk with different strategy**: smaller targets (1500), bigger targets (8000), section-only splits, or a completely different chunker
- **Query rewriting**: instead of HyDE, use qwen3 to extract the "essential ask" from the long scenario before retrieval (e.g. "countervailing duty exemption on imported copper")
- **Reranker fine-tuning** on our miss patterns
- **Cascade retrieval**: dense → if low scores, expand with sparse + LLM-generated queries
- **Knowledge-graph augmentation**: build a doc-id → topic graph from notifications (each notification's "amends" reference)
- **Anything else**

We can re-ingest the entire 14K-doc corpus (takes ~3 hours on our rig) any number of times. We can re-embed. We can fine-tune. **No constraint on engineering time, only on the result: 95% on G1.**

## 9. The specific question

**Given the diagnosis above, what's the most promising path to push G1 recall@10 from 0.84 to ≥0.95 on the 380-query gold set?**

Concretely:
1. Is the diagnosis correct (embedding-query semantic mismatch on long-scenario queries)?
2. What single intervention (or sequence) is most likely to close that 11-point gap?
3. If you recommend embedding model swap or fine-tune, which model + what training regime?
4. If you recommend synthetic Q-A enrichment, what's the prompt / generation strategy you'd use to produce questions that match the gold's long-scenario form?
5. Are there any approaches we haven't listed that would be easier and equally effective?

Please be specific and actionable. We will execute the recommendation, even if it takes days.

---

## Appendix A — Sample missed query and expected chunk

**Gold query (full):**
> "Prakash & Associates, a Cost Accounting firm in Chennai, has been representing Suvarna Steels Pvt Ltd, an iron and steel manufacturer who has received a Show Cause Notice for irregular CENVAT credit availment of ₹4.5 crores. The notice was issued under Section 11A(4) of the Central Excise Act citing willful suppression. The firm wants to know the timeline within which a personal hearing must be granted after the demand notice if the assessee requests it under the natural justice principles."

**Expected `expected_doc_id`**: `cbic-notification-msts:1005497`

**Expected `expected_section`**: "The Central Excise (Appea..." (truncated in gold)

**Actual top-3 retrieved with hybrid (rerank scores):**
1. `cbic-notification-msts:1001691` (sec="Seeks to specify jurisdiction…", score 2.204)
2. `cbic-notification-msts:1001771` (sec="Seeks to amend the CENVAT…", score 2.082)
3. `cbic-notification-msts:1001771` (different chunk, score 2.081)

Expected doc not in top-50.

**Why we suspect embedding mismatch**: nothing about "Prakash & Associates", "Cost Accounting firm", "Chennai", "Suvarna Steels", or "₹4.5 crores" appears in the expected notification. The notification is about generic personal-hearing timelines under Section 11A. The semantic gap between "scenario about a specific firm with a specific situation" and "generic procedural rule" is what we cannot bridge with BGE-M3.

---

## Appendix B — Hardware/infrastructure (in case it constrains advice)

- 1 server, 7 AMD GPUs (1×RX 6700 XT 12GB + 6×RX 5700 XT 8GB)
- AMD-only, Vulkan compute (no CUDA)
- llama.cpp / llama-cpp-python with Vulkan backend
- ~256GB system RAM, can re-embed 14K docs in ~80 min on the embed pool
- Have access to Gemini API + Claude CLI (subscription)
- No GPU cluster for training — we can fine-tune small models locally but anything requiring 8×A100 we cannot do
