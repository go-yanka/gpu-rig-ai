# Training Data Generation Plan — Qwen LoRA for CBIC

**Status:** PARKED. Execute only if A3+A1+A2+A4+A5 deploys don't reach 85%+ verbatim-gate pass rate on the 50-Q gold set.

**Consult verdict (4 rounds):** skip generator LoRA for now. Mode collapse risk + data cost > expected lift over two-pass structured extraction.

---

## 1. What we have vs what we need

| Asset | State | Usable? |
|---|---|---|
| 108,802 CBIC chunks in Qdrant | Ingested | Raw source, not training pairs |
| 50-Q gold set | Written | Too small alone (use as seed) |
| E10 feedback logs | Capturing now (~0 real entries) | Future source |
| Expert review bandwidth | Unknown | Needed for final QA (~250 pairs ~ 4-8 hrs) |

**Gap:** Need ~5,000 clean `(context, question, answer_with_verbatim_span)` triples.

---

## 2. Three generation methods

### Method 1 — Section-exhaustive synthesis (PRIMARY, ~4,800 pairs)

For each of ~400 statutory sections/rules already in corpus:

```python
# Pseudocode
for section_chunk in all_act_and_rules_chunks:
    prompt = f"""
    You are a CBIC tax expert generating training data.
    CHUNK (from {section_chunk.citation}):
    ---
    {section_chunk.text}
    ---
    Generate 12 distinct practitioner questions this chunk can answer.
    For each question, write an answer that:
    - Contains a verbatim quote (80-450 chars) from the chunk above
    - The quote must contain at least one complete clause
    - Cite as [S1]
    - Add a brief conclusion AFTER the quote
    Return strict JSON: [{{"question":"...", "answer":"..."}}]
    """
    candidates = call_big_llm(prompt)  # Claude 3.5 / GPT-4 / Gemini 1.5 / qwen3-72B API
    for c in candidates:
        if validate_span(extract_quote(c['answer']), section_chunk.text):
            keep(c, source=section_chunk)
```

- **Generator:** Claude 3.5 Sonnet OR GPT-4o OR Gemini 1.5 Pro OR qwen3-72B-instruct via Fireworks/TogetherAI
  - ⚠️ **Claude 3.5 Sonnet / GPT-4o output terms prohibit training competing models** (per consult #2, 2026-04-21). For a commercial product, use permissive open-weights instead: **Llama-3.3-70B-Instruct, DeepSeek-V3, Qwen3-72B-Instruct** (all allow distillation under their licenses).
- **Filter:** our existing `validator.py` (substring + 6-gram Jaccard + BGE-M3 cosine)
- **Expected yield:** **~50% pass rate** (per consult #2 — consult #1's 60-70% deemed optimistic). **Generate 10,000 candidates to yield ~5,000 survivors.**
- **Cost:** ~$150-400 API spend (400 sections × ~25 Qs × ~500 tokens each at 10k total)
- **Time:** ~24-36 hrs wall clock with rate-limit pacing

#### Method 1b — Adversarial pairs (20% of final set, ~1,000 pairs)

Per consult #1 (2026-04-21), explicit refusal-training is required for trustworthy behavior. Reserve ~20% of the final 5k survivor budget for adversarial examples:

- **No-evidence refusals (~600 pairs):** Question where no retrieved chunk actually answers it. Gold answer = "The provided sources do not address this. I cannot answer without supporting material." + no verbatim span. Trains the model to decline rather than hallucinate.
- **Conflicting-evidence prefer-conservative (~400 pairs):** Two chunks disagree (e.g. a superseded circular vs current rule). Gold answer = cite both, flag the conflict, prefer the more conservative / more recent interpretation, explicitly note the conflict to the user. Trains the model to surface ambiguity instead of silently picking one side.

These pairs bypass the 80-450-char verbatim gate where appropriate (a valid refusal has no verbatim span). Auto-validator must be extended to recognize the two allowed "no-span" answer schemas.

---

### Method 2 — Scenario-synthesis (SECONDARY, +1,000-2,000 pairs)

Seed from the 5 complex Qs in the gold set. For each seed:
```python
prompt = f"""
Original scenario: {seed_q}
Generate 40 scenario variations with different facts (different GSTINs, states, amounts,
goods/services mixes) that each require multi-section reasoning. For each:
- Write the scenario question
- Write the answer using two-pass format: verbatim quotes [S1][S2]... + stitched conclusion
Return JSON list.
"""
# Filter: run each through our own two-pass pipeline (once A3 is live); keep only those where
# ALL verbatim spans pass the verifier.
```

- **Generator:** same as Method 1
- **Expected yield:** 5 × 40 = 200 candidates per round; ~30% pass → ~60 clean per round; iterate on seeds
- **Cost:** ~$50-100

### Method 3 — Query log mining (FUTURE, organic)

Once real traffic flows:
- E10 upvoted answer + its verified quotes → positive pair
- E10 downvoted answer + user's "why wrong" + expert-corrected answer → preference pair (for DPO)
- Mine weekly, add to training set for the next LoRA round

Requires usage volume (~1,000 rated queries before this is meaningful).

---

## 3. Data formats

### Generator LoRA (verbatim-copy behavior)

```jsonl
{"messages": [
  {"role": "system", "content": "EXTRACTION_SYS prompt..."},
  {"role": "user", "content": "CONTEXT:\n[S1] {chunk_id}: {chunk_text}\n\nSUB-QUESTIONS:\n1. {q}"},
  {"role": "assistant", "content": "{\"sub_answers\": [{\"sub_question\": \"...\", \"cited_chunk_id\": \"...\", \"verbatim_span\": \"<exact copy>\", \"conclusion\": \"...\"}]}"}
]}
```

### Embedding contrastive fine-tune (MultipleNegativesRankingLoss)

```jsonl
{"anchor": "What is the place of supply under Section 10(1)(b)?",
 "positive": "<IGST Act Section 10(1)(b) chunk text>",
 "negatives": ["<unrelated chunk>", "<same-section but different clause>", "..."]}
```

### DPO preference (for fixing paraphrase instinct specifically)

```jsonl
{"prompt": "CONTEXT:\n{chunk}\n\nQ: {q}",
 "chosen": "<answer with exact substring quote>",
 "rejected": "<answer with paraphrased quote (even if meaning-equivalent)>"}
```

---

## 4. Validation gates before spending training budget

Before launching LoRA training:
- [ ] 5% sample (~250 pairs) reviewed by tax practitioner — ≥90% accuracy
- [ ] Stratified coverage check: at least 50 pairs per category (GST, Customs, CE, ST, Others)
- [ ] Difficulty mix: 50% basic, 35% intermediate, 15% complex
- [ ] Duplicate/near-duplicate filter (cosine <0.98 between any two pairs)
- [ ] No training pair exactly matches a gold-set Q (prevents eval contamination)
- [ ] All verbatim spans pass our production verifier

---

## 5. Hardware + training execution

**Our rig cannot train.** AMD Vulkan, no ROCm, no CUDA. bitsandbytes/PEFT need CUDA.

**Options:**
1. **Runpod / Lambda Labs / Modal** — rent A100 (40GB or 80GB) for 2-6 hrs. ~$10-30/hr.
2. **Together AI / Fireworks** — some offer LoRA fine-tuning as a service on open-weight models. Check qwen3-14B-hermes support.
3. **Anyscale / Modal custom** — full control, upload data + config.

**Training recipe (starting point):**
- Base: `qwen3-14b-hermes` or whatever we're running
- LoRA rank: 16
- Alpha: 32
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- LR: 2e-4 with cosine decay
- Epochs: 1-2 (more = mode collapse risk)
- Batch: 4 with gradient accumulation 4
- Max seq length: 8192
- Estimated training time: 2-4 hrs on A100

**Deploy:** LoRA adapter merged into base for llama.cpp Vulkan inference. Test in parallel with un-LoRA'd base before flipping live.

---

## 6. Cost + timeline summary

| Item | Cost | Time |
|---|---|---|
| API generation (Method 1, 7k candidates) | $100-300 | 1-2 days |
| API generation (Method 2, iterative) | $50-100 | 1 day |
| Auto-filter via local verifier | $0 | 4 hrs |
| Expert review of 5% sample | 4-8 practitioner hrs | 1 day |
| Rent A100 for LoRA | $20-80 | 2-6 hrs |
| Integration + eval | $0 | 1 day |
| **Total** | **~$200-500** | **~1 week** |

---

## 7. Decision trigger (when to execute)

**Gate:** If after shipping A3, A1, A2, A4, A5, the eval harness shows:
- Verbatim-gate pass rate on complex tier < 85% OR
- Aggregate < 90% OR
- Regression cases pile up in feedback log without a clear non-training fix

THEN **do not jump straight to LoRA.** Execute this fallback ladder in order, stopping at the first step that closes the gate:

1. **Base model upgrade first** — swap qwen3-14B for **Qwen2.5-32B-Instruct** or **DeepSeek-R1-Distill-Qwen-32B**. Per consult #2 (2026-04-21), instruction-following and citation fidelity scale ~linearly with parameters; a 32B base often eliminates the need for training altogether. Both are open-weights and run on the existing rig via llama.cpp Vulkan (one GPU pair each, tested path).
2. **BGE-M3 embedding contrastive fine-tune** (see §8) — cheaper, lower-risk, measurable retrieval lift. Only if base-upgrade didn't close the gate.
3. **Distillation from open-weights generator** (Llama-3.3-70B / DeepSeek-V3 / Qwen3-72B) into our 14B or 32B base — only if steps 1+2 still under-perform.
4. **Generator LoRA at 14B** — last resort. Known mode-collapse risk per consult #2: fixes quoting behavior but can break multi-hop reasoning. Execute Method 1 (above) for data preparation only if steps 1-3 failed.

Otherwise: do not train. The structural fixes (two-pass extraction + retrieval routing + table pipeline) are predicted to suffice per 5 consultations.

---

## 8. Alternative: embedding contrastive fine-tune (cheaper first-move)

If retrieval ranking is still the bottleneck after A1 ships, tune BGE-M3 instead of qwen3:
- Harvest 1,500-3,000 `(query, relevant-chunk)` positives from gold set + E10 upvotes + Method 1 synthesized Qs.
- Hard negatives: for each positive, take top-5 retrieved non-target chunks.
- Train with `MultipleNegativesRankingLoss`, 1 epoch, batch 16.
- Cheaper (~$10-30 rental, 1-2 hrs training), less risk, measurable retrieval lift.

This is a better first-move than generator LoRA per all 4 consult rounds.

---

## Update log

- **2026-04-21:** Created. Parked per consensus to ship structural fixes first.
