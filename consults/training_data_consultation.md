# Training Data for Qwen LoRA — External Consultation

**Date:** 2026-04-21
**Context:** CBIC RAG project. 4 prior consultation rounds converged on "skip generator fine-tuning for now, ship two-pass structured extraction first." We accept that. This brief is a **parked question**: *if/when* we do fine-tune, what's the fastest + cheapest path to 5,000 high-quality training pairs?

Specifically, **we want to know if public datasets already exist that we can reuse or adapt, before we spend $200-500 and a week synthesizing from scratch.**

---

## 1. What we're trying to train

**Target behavior:** Make qwen3-14B-hermes default to verbatim-copying legal text from a provided context chunk, rather than paraphrasing it.

**Pair shape we need:**
```json
{
  "messages": [
    {"role": "system", "content": "EXTRACTION_SYS: You are a CBIC legal-text extractor. Copy the verbatim span that supports the sub-question. 80-450 chars. Contain one complete clause."},
    {"role": "user", "content": "CONTEXT:\n[S1] {chunk_id}: {chunk_text}\n\nSUB-QUESTIONS:\n1. {q}"},
    {"role": "assistant", "content": "{\"sub_answers\":[{\"sub_question\":\"...\",\"cited_chunk_id\":\"...\",\"verbatim_span\":\"<exact copy>\",\"conclusion\":\"...\"}]}"}
  ]
}
```

**Target: 5,000 pairs.** Stratified: 2,000 GST, 1,000 Customs, 500 Central Excise, 500 Service Tax, 500 Income Tax (forward-looking), 500 Others.

**Alternative format for DPO** (if we go preference-tuning):
```json
{"prompt": "...", "chosen": "<verbatim-quote answer>", "rejected": "<paraphrased answer>"}
```

**Alternative for embedding fine-tune** (BGE-M3 contrastive):
```json
{"anchor": "<query>", "positive": "<relevant chunk>", "negatives": ["<hard negative>", "..."]}
```

---

## 2. What we already have

| Asset | Count | Usable as training data? |
|---|---|---|
| CBIC chunks in Qdrant (`cbic_v1`) | 108,802 | Raw text only, not pairs |
| 50-Q gold eval set | 50 | Too small to train; use as seed only |
| E10 feedback logs (upvote/downvote) | ~0 | Future source once we have traffic |
| B-series failure transcripts (what the model got wrong) | ~30 curated | Could become DPO "rejected" samples |

**Gap:** 5,000 `(context, question, verbatim-copy answer)` triples.

---

## 3. Our default synthesis plan (if no public data exists)

**Method 1 — Section-exhaustive synthesis (4,800 pairs):** For each of ~400 statutory sections in corpus, prompt a bigger LLM (Claude 3.5 / GPT-4o / Gemini 1.5 / qwen3-72B via API) to generate 12 Q/A pairs per section. Auto-filter via our existing verifier (substring + 6-gram Jaccard ≥0.80 + BGE-M3 cosine ≥0.92). Expected 60-70% pass rate → generate 7k candidates, keep ~4.8k.

**Method 2 — Scenario synthesis (+1-2k):** Seed from 5 complex gold-set Qs, generate 40 variations each, filter.

**Method 3 — Query log mining:** Organic, requires traffic.

**Estimated cost:** $200-500 API + $30-80 A100 rental.
**Estimated time:** ~1 week wall clock.

---

## 4. Questions for external LLMs

### Q1 — Do public Indian-tax training datasets exist?

Specifically looking for:
- **Hugging Face datasets** — anything tagged `indian-law`, `tax`, `cbic`, `gst`, `legal-qa-india`?
- **Kaggle competitions or datasets** — Indian regulatory Q/A sets?
- **Academic releases** — IIT/IIM/NLSIU/NLU papers with released datasets? (Nyaya-LLM, InLegalBERT training data?)
- **Government open-data portals** — data.gov.in has CBIC PDFs but do they have Q/A sets?
- **Commercial APIs with permissive terms** — any vendor who's already done this extraction and released data?

If you know of any, please include:
- Dataset name + URL
- License (must be permissive for commercial use)
- Size (number of pairs)
- Format (matches our need?)
- Quality assessment if known

### Q2 — Adjacent datasets we could adapt

Even if no CBIC-specific set exists, what about:
- **General Indian legal corpora** (IndianKanoon, LegalBERT-India training sets) — can they pre-train the verbatim-copy behavior generically?
- **English-language tax Q/A** (US IRS, UK HMRC) — could cross-lingual transfer learning help the verbatim-copy behavior?
- **Academic legal RAG eval sets** (LegalBench, LexGLUE, CaseHOLD) — usable as DPO preference data?
- **Verbatim-copy benchmarks from other domains** (PubMedQA, medical citation datasets) — does verbatim-copy behavior transfer across domains?

### Q3 — Is our synthesis plan right?

- Is **Claude 3.5 / GPT-4o / Gemini 1.5 / qwen3-72B** the right generator for Method 1? Which has the best Indian legal text understanding? Any bias we'd import?
- **60-70% pass rate** after auto-filter — realistic or optimistic?
- Would **distillation from a bigger open model** (qwen3-72B, DeepSeek-V3, Llama-3.3-70B) beat commercial APIs for this?
- **400 sections × 12 questions each** — right stratification? Too many per section (over-fitting)? Too few (under-coverage)?

### Q4 — Should we train the generator at all?

Re-confirming the consensus from 4 prior rounds:
- Given our plan to ship two-pass structured extraction (A3) first, does training still make sense after?
- **Embedding fine-tune** (contrastive BGE-M3) vs **generator LoRA** — which has higher ROI as first-move *if* we train anything?
- Does **DPO on preference pairs** (copied vs paraphrased) work better than **SFT on copy-triples** for our specific goal (forcing verbatim behavior)?

### Q5 — Hardware pragmatism

- Our rig is AMD Vulkan only (no ROCm, no CUDA). Confirm we cannot train locally, even for tiny LoRA?
- Any AMD-compatible training path we've missed (ROCm fork of bitsandbytes, Axolotl with Triton fallback, anything else)?
- **Runpod vs Lambda vs Modal vs Together AI vs Fireworks** — which has qwen3-14B-hermes as a LoRA-trainable base today? Cheapest reliable option for a 2-6 hr job?

### Q6 — The sneaky alternative — distillation

Instead of training on synthesized pairs, what if we:
- Query a much bigger model (Claude 3.5 Sonnet, GPT-4o) on our 50-Q gold set + 4,950 synthesized Qs
- Capture its responses as training data
- Fine-tune qwen3-14B to mimic that response distribution

Is this **behavioral cloning / model distillation** a legitimate shortcut? Legal/TOS concerns aside, does it work for legal-text verbatim-copy behavior? Known success cases in legal domains?

### Q7 — The really sneaky alternative — use a different base

Instead of fighting qwen3-14B's paraphrase instinct:
- **qwen3-32B** — does the larger base have better verbatim-copy behavior out-of-the-box, removing the need to fine-tune?
- **DeepSeek-V3 / V3.1** — 37B active with MoE, strong on instruction following. Legal-domain performance?
- **Llama-3.3-70B-instruct** — known strong at structured output and citation-following?
- **A legal-specialty base** — Nyaya-LLM, SaulLM-7B, LegalMath — any worth starting from?

If a larger or specialty base closes the gap without fine-tuning, **skip training entirely**. Is that the right call?

### Q8 — The actually sneaky alternative — just do retrieval harder

If we fix retrieval perfectly (A1 P1 + perfect chunks surfaced every time):
- Does verbatim-copy become trivial for qwen3-14B?
- Is A3 two-pass plus A1 plus A2 corpus refresh plus A4 tables plus A5 routing **sufficient** to hit 90%+ verbatim-gate without any model training?

The prior consensus said yes (85-94% predicted). Confirm or dissent based on your own production experience with legal RAG?

---

## 5. What we'd like from you

1. **Explicit list of any public datasets you know about** — even partial matches. We'll verify licenses + quality ourselves. A 1k-pair tangentially-relevant set beats synthesizing 5k from scratch.
2. **Verdict on training vs not training** — given our A3-A5 roadmap, is training genuinely necessary or theater?
3. **Concrete recommendation on base model** — stay on qwen3-14B-hermes or switch?
4. **Hardware/service recommendation for training** IF we decide to proceed.
5. **Honest probability estimate:** after A3-A5 ships cleanly, what's P(verbatim-gate pass rate ≥ 90% on our 50-Q gold set) *without* training?

---

## 6. Context — what we've already ruled out

- ❌ **Local training on our rig** — AMD Vulkan, no CUDA.
- ❌ **Paying tax practitioners to hand-label** — too expensive, too slow.
- ❌ **Ignoring the trust problem** — we've tried prompt-only 3 times, all failed. Verbatim citation is the product.
- ❌ **Lowering verifier threshold** — that's lying to users.

---

## 7. Companion files (if useful)

- Full dual-plan spec: `cbic_rag_dual_plan_v3.md`
- Training data generation plan (Methods 1-3 with pseudocode + costs): `training_data_generation_plan.md`
- Lessons learned: `../memory/lessons_learned.md`
- Reasoning calibrations (GSTR-1 vs 3B etc.): `../memory/rag_reasoning_calibrations.md`
- New-corpus playbook (reusable for Income Tax/MCA/Labour/RBI): `../memory/new_corpus_playbook.md`

---

*Please be specific. "Search Hugging Face" isn't an answer — dataset names + URLs are. Thanks.*

---

## 8. Response log — round 1 & round 2

*Added 2026-04-21 after two external LLM consultations responded to this brief.*

### Consult #1 summary
- NyayaAnumana (702k SC/HC/Tribunal cases, arXiv:2412.08385) named as "strongest starting point" for citation-behavior adaptation.
- `msinankhan1/India_Tax_FAQs` flagged as seed for future Income Tax corpus.
- `ninadn/indian-legal` and `MeeraR/legal-qa-dataset` flagged as seed-only (general legal QA).
- AkshatGupta Kaggle dataset (CC0) covers IPC / CrPC / Constitution QA.
- Filter pass rate after auto-verify: 60-70% realistic.
- Claude 3.5 Sonnet / GPT-4o rated best for Indian legal understanding among generators.
- Recommends 20% adversarial pairs (no-evidence + conflicting-evidence) for refusal training.
- Embedding contrastive fine-tune on BGE-M3 = higher ROI than generator LoRA.

### Consult #2 summary
- Named `Techmaestro369/indian-legal-texts-finetuning` on HF (~14,500 QA pairs, IPC/CrPC/Constitution, NOT tax).
- Named `Awaizg/GST_JSON` on HF — raw GST manual text (not pair-formatted).
- **Adapting general Indian legal corpora is "a trap"** — skews vocabulary away from tax schedules.
- **Claude 3.5 / GPT-4o ToS prohibits** using outputs to train competing models → legal risk for commercial product. Recommends permissive open-weights: Llama-3.3-70B-Instruct, DeepSeek-V3, Qwen3-72B.
- **60-70% pass rate is optimistic. Plan ~50%.** Generate 10k candidates to yield 5k survivors.
- Mode-collapse warning on 14B LoRA: fixes quoting behavior, breaks multi-hop reasoning.
- **If A3 insufficient → upgrade base model first** (Qwen2.5-32B-Instruct or DeepSeek-R1-Distill-Qwen-32B), not LoRA. Instruction-following scales ~linearly with parameters.
- Verbatim-gate probability without training: 90-95% if A3 + retrieval fixes land cleanly.

### Divergences table

| Topic | Consult #1 | Consult #2 |
|---|---|---|
| Adapting NyayaAnumana / general legal QA | "Strongest starting point" | "A trap — skews vocabulary away from tax schedules" |
| Claude 3.5 / GPT-4o as generator | Best Indian legal understanding | TOS prohibits training competing models (legal risk) |
| Filter pass rate after auto-verify | 60-70% realistic | 50% realistic — plan 10k candidates for 5k survivors |
| Fallback if A3 insufficient | Embedding fine-tune | Upgrade base to 32B class first, then embedding, then LoRA |
| Verbatim-gate without training | 90%+ likely | 90-95% likely |

### Convergent decisions (locked after 5 rounds total)
1. ❌ No generator LoRA at 14B.
2. ❌ No adapting general Indian legal QA to teach verbatim (domain skew > benefit).
3. ✅ If A3+A1+A4+A5 < 90% verbatim-gate → next move = base upgrade (Qwen2.5-32B / DeepSeek-R1-Distill-32B), not training.
4. ✅ If training ever happens → BGE-M3 embedding contrastive first; distillation from open-weights second.
5. ✅ Synthesis budget: plan 10k candidates → 5k survivors (not 7k → 4.8k).

### Dataset references (for memory, not immediate use)
- NyayaAnumana / INLegalLlama — 702,945 SC/HC/Tribunal cases. arXiv:2412.08385. Research license.
- `msinankhan1/India_Tax_FAQs` — HF, open, Income-Tax FAQs (1-10k).
- `ninadn/indian-legal` — HF, general legal QA.
- `MeeraR/legal-qa-dataset` — HF, general legal QA.
- `AkshatGupta/llm-fine-tuning-dataset-of-indian-legal-texts` — Kaggle, CC0, IPC/CrPC/Constitution.
- `Techmaestro369/indian-legal-texts-finetuning` — HF, ~14,500 pairs, IPC/CrPC/Constitution (NOT tax).
- `Awaizg/GST_JSON` — HF, raw GST manual text (not pairs).

