# Embedder Alternatives vs BGE-M3 — Research Consult

**Date**: 2026-04-22
**Context**: CBIC RAG, Indian indirect-tax law (GST/Customs/Excise/ST), 108K chunks ~1500 chars, baseline recall@5=20% with BGE-M3 oob, 12GB VRAM (RX 6700 XT Vulkan) + CPU fallback, target ~50 qps.
**Question**: Stick with BGE-M3 (with 5K-pair fine-tune) or swap?

---

## 1. Summary Table — Top Open Embedders (late-2025 / early-2026)

| Model | Params | Dim | MTEB (Eng/Multi) | License | Fits 12GB? | FT-friendly | Verdict |
|---|---|---|---|---|---|---|---|
| **Qwen3-Embedding-8B** | 8B | 4096 | 70.58 (Multi, #1 Jun'25) / 80.68 Code | Apache-2.0 | No (fp16 ~16GB) | Yes (InfoNCE, HF/SWIFT) | Too big for our GPU |
| **Qwen3-Embedding-4B** | 4B | 2560 | ~69 (Multi) | Apache-2.0 | Tight (fp16 ~8GB) | Yes | Viable |
| **Qwen3-Embedding-0.6B** | 0.6B | 1024 | ~64 (Multi, "just behind Gemini") | Apache-2.0 | Yes (easily) | Yes, well-documented | **Top candidate** |
| **NV-Embed-v2** | 7.85B (Mistral) | 4096 | 72.31 (Eng #1 Aug'24); 62.65 retrieval | CC-BY-NC (non-commercial) | No | Possible, heavy | License blocks commercial |
| **gte-Qwen2-7B-instruct** | 7B | 3584 | 70.24 (Eng, #1 Jun'24) | Apache-2.0 | No (fp16 ~14GB) | Yes | Too big |
| **Stella-en-1.5B-v5** | 1.5B | 1024/8192 | ~71 (Eng, reported) | MIT | Yes | Yes | Viable; English-only |
| **jina-embeddings-v3** | 570M | 1024 (Matryoshka) | 65.52 (Multi) | CC-BY-NC 4.0 | Yes | Task-LoRA adapters | License blocks commercial |
| **E5-mistral-7b-instruct** | 7B | 4096 | 66.63 | MIT | No | Yes | Older, eclipsed |
| **BGE-M3 (baseline)** | 568M | 1024 | ~59 (Eng) / 69.2 Multi | MIT | Yes | Yes; dense+sparse+colbert | Current, solid |
| **Llama-Embed-Nemotron-8B** | 8B | — | #1 multilingual MTEB v2 (Oct'25) | Open-weight (NVIDIA) | No | Yes, via Llama stack | Too big |

*Scores are MTEB-reported averages; "—" where not published. Exact MTEB Eng-v2 numbers shift weekly.*

## 2. Legal / Domain Benchmarks

- **COLIEE 2025 (legal retrieval)**: Fine-tuned BGE-M3 beats OOB alternatives; F1=0.2262 @top-5, further lift via ensembling. This is direct evidence that BGE-M3 FT is competitive in legal domain.
- **FinMTEB (finance, Feb 2025)**: Shows general-purpose embedders degrade ~15–25% on domain text; fine-tuned domain models (SaulLM-7B legal, BloombergGPT) dominate but are LLMs not embedders.
- **Stanford Legal-Retrieval Benchmark (CS&Law 2025)**: Reasoning-style legal queries — all off-the-shelf dense embedders <40% recall; reranker + FT is the standard recipe.
- **BEIR**: Saturated; community moved to MMTEB/RTEB/CoIR.

Takeaway: **no purpose-built legal embedder dominates**; everyone fine-tunes a general model.

## 3. Fine-tune Lift Data (BGE-M3 and peers)

- **REFINE paper (arXiv 2410.12890)**: Synthetic pairs + model fusion → Recall@3 +5.79% (TOURISM), +6.58% (SQuAD), +0.32% (RAG) over vanilla BGE-M3. Small-data regime.
- **CLP paper (arXiv 2412.17364)**: Contrastive Learning Penalty on 5–10K pairs gives 3–8 pts nDCG@10 lift.
- **Kaggle BGE-M3 FT notebooks**: typical lifts 10–20 points recall@5 on domain-specific synthetic pairs.
- **COLIEE legal**: FT BGE-M3 delivered the best single-model legal retrieval.
- At our baseline of 20% recall@5, a well-curated 5K-pair MNRL FT should get us to ~40–55% recall@5 — this is the biggest single lever available.

## 4. Cost–Performance on our 12GB Vulkan constraint

At ~50 qps with 1500-char chunks on RX 6700 XT Vulkan (no ROCm for embedders in llama-server today for most of these):

| Model | Est. qps (batched) | VRAM fp16 | Notes |
|---|---|---|---|
| BGE-M3 | 80–150 | ~1.5GB | Proven on our rig |
| Qwen3-Emb-0.6B | 60–120 | ~1.5GB | Qwen3 arch, GGUF available |
| Qwen3-Emb-4B | 15–30 | ~8GB | Tight, batch=1–2 |
| Stella-1.5B | 40–80 | ~3GB | English-only risk for Hindi |
| NV-Embed-v2 / gte-Qwen2-7B | <10 | OOM fp16 | Not viable |

## 5. Top 3 Recommendations

### #1 — **Fine-tune BGE-M3 first (stay)**
Rationale: biggest lift is domain adaptation, not base model. 20%→40–55% recall expected. Infra already in place, ingestion playbook frozen, Qdrant index sized for 1024-dim. **Do not swap mid-flight.**

### #2 — **Qwen3-Embedding-0.6B as the swap candidate IF FT plateaus**
- Apache-2.0, 0.6B params, 1024 dim (drop-in replacement for BGE-M3 index-size-wise), MTEB-multi ~64 (+5 pts over BGE-M3 OOB).
- Supports instruction prompts ("Given a GST query, retrieve…") which helps domain recall out-of-box.
- Well-documented FT recipe (SWIFT, HF trainer).
- Multilingual incl. Hindi script.
- **Expected lift vs FT BGE-M3: +2 to +5 pts recall@5** (honest: similar, not transformative).

### #3 — **Qwen3-Embedding-4B (aspirational)**
Only if we prove #2 wins and want more ceiling. Requires careful VRAM management on 12GB. Quantize to Q8 GGUF via llama.cpp.

## 6. Fine-tune Guidance (Qwen3-Emb-0.6B, for when we get there)

From Alibaba blog, SWIFT guide, HF discussions, arXiv 2506.05176:
- **Loss**: InfoNCE with in-batch + hard negatives (match pretraining regime).
- **LR**: 6e-6 (conservative, preserves base) or 1e-4 (aggressive, small data). Start 6e-6.
- **Batch**: per-device 4–8, grad-accum 4 → effective 16–32. Bigger batch = more in-batch negatives = better.
- **Hard negatives**: 7 per query, mined with BM25 + current-model top-20 minus positives.
- **Epochs**: 1–3. Overfits fast on 5K pairs; use eval-set early-stop.
- **Instruction prefix**: "Given a question about Indian indirect tax law, retrieve relevant statutory text:" — consistently +1–3 pts.
- **LoRA** (for 4B/8B variants): r=16, alpha=32, targets q/k/v/o projs.

## Expected Lift vs Fine-tuned BGE-M3

Honest assessment: **similar.** The jump from OOB BGE-M3 → FT BGE-M3 is ~+20 pts recall@5. Jump from FT BGE-M3 → FT Qwen3-0.6B is likely **+2 to +5 pts**, sometimes zero. The 4B/8B variants may give +5–10 pts but don't fit our ops envelope.

## "If we had to swap today, which?"

**Qwen3-Embedding-0.6B** — Apache-2.0, same 1024-dim (index-compatible re-embed), +5 MTEB pts, instruction-aware, actively maintained, tiny enough to re-embed 108K chunks in a few hours on our rig.

---

## Sources
- MTEB leaderboard: https://huggingface.co/spaces/mteb/leaderboard
- March 2026 MTEB rankings: https://awesomeagents.ai/leaderboards/embedding-model-leaderboard-mteb-march-2026/
- Modal blog, top MTEB models: https://modal.com/blog/mteb-leaderboard-article
- Cheney Zhang 2026 embedding benchmark: https://zc277584121.github.io/rag/2026/03/20/embedding-models-benchmark-2026.html
- VentureBeat, Gemini vs Qwen3: https://venturebeat.com/ai/new-embedding-model-leaderboard-shakeup-google-takes-1-while-alibabas-open-source-alternative-closes-gap
- BentoML, best open embedders 2026: https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models
- Qwen3-Embedding paper: https://arxiv.org/abs/2506.05176
- Qwen3-Embedding blog: https://qwenlm.github.io/blog/qwen3-embedding/
- Qwen3-Embedding-0.6B card: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- Qwen3-Embedding-8B card: https://huggingface.co/Qwen/Qwen3-Embedding-8B
- BGE-M3 card: https://huggingface.co/BAAI/bge-m3
- BGE-M3 docs: https://bge-model.com/bge/bge_m3.html
- NV-Embed-v2 card: https://huggingface.co/nvidia/NV-Embed-v2
- NV-Embed paper: https://arxiv.org/abs/2405.17428
- gte-Qwen2-7B-instruct: https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct
- jina-embeddings-v3: https://jina.ai/news/jina-embeddings-v3-a-frontier-multilingual-embedding-model/
- Stella 1.5B v5: https://huggingface.co/NovaSearch/stella_en_1.5B_v5
- E5-mistral-7b: https://huggingface.co/intfloat/e5-mistral-7b-instruct
- REFINE paper (FT small data): https://arxiv.org/html/2410.12890v1
- CLP FT paper: https://arxiv.org/html/2412.17364
- FinMTEB paper: https://arxiv.org/html/2502.10990v1
- Stanford legal retrieval benchmark: https://dho.stanford.edu/wp-content/uploads/Legal_Retrieval.pdf
- Qwen3-Emb FT with SWIFT guide: https://medium.com/@kimdoil1211/fine-tuning-qwen3-embedding-with-swift-and-docker-a-complete-practical-guide-e4107c2781c9
- Qwen3 vs BGE-M3 comparison: https://medium.com/@mrAryanKumar/comparative-analysis-of-qwen-3-and-bge-m3-embedding-models-for-multilingual-information-retrieval-72c0e6895413
- Agentset BGE-M3 vs Qwen3-0.6B: https://agentset.ai/embeddings/compare/baaibge-m3-vs-qwen3-embedding-06b
- BEIR: https://github.com/beir-cellar/beir
