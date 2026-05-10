# External LLM Consultation Brief v3 — 2026-04-22

## Objective (READ FIRST)

**The goal is 95% trustworthiness.** Not 50%, not 70% — the system needs to be something a tax practitioner can rely on without verification. That reframes the cost/benefit of every option below. Do not dismiss ideas for being "expensive" — re-ingesting 108k chunks, generating 2000 synthetic Q/A pairs via Gemini API, or renting an A100 for contrastive fine-tuning are all on the table if they move pass-rate meaningfully toward 95%.

We currently sit at 40.44%. Getting to 95% is a >2x gain — single interventions of +3-5pp won't get us there. We need the consulted model to think about this as "what is the stack of changes that compounds to 95%?" not "what's the cheapest next +3pp?"

## Note on a prior Gemini consult

In a separate thread we asked Gemini whether to LoRA-fine-tune the 14B generator on 2000 extraction pairs. Its reply advised against generator fine-tuning (mode-collapse risk, architecturally redundant if two-pass holds, hardware mismatch) and suggested instead **contrastive fine-tuning of BGE-M3 on (Q, correct_chunk) pairs** as the higher-ROI move.

Caveat: Gemini's "deploy two-pass first, it'll hold" was given before we ran the 170-item eval. Two-pass **did not hold** — it regressed −13.4pp (details in Q1). So Gemini's strategic framing is partly stale. The BGE-M3 suggestion stands as an independent idea worth evaluating.

We want the consulted model to:
- React to the BGE-M3 contrastive-tuning suggestion as a potential option under Q3 (or as its own Q4).
- Still answer Q1/Q2/Q3 as posed.
- Keep the 95% target in mind when ranking priorities — small safe wins are less valuable than compounding structural improvements.

---

**System:** CBIC RAG (Indian indirect-tax law over 108k Qdrant chunks).
**Stack:** FastAPI :9500 / Qdrant `indian_legal_t1_v2` / BGE-M3 embed (GPU 5 Vulkan) / qwen3-14B-hermes synth (GPU 2 llama-server :9082) / ColBERT rerank CPU / MMR.
**Gold set:** 170 items across 6 buckets (tariff, refusal, multi-section, ST+IT crossover, customs advanced, appeals/penalty/AAR).
**Current pass rate:** **40.44%** (up from 33.23% baseline after P1.2 validator simplification).
**Judge:** qwen3-14B scoring gold-criteria rubric (sections/rules/notifications cited + conclusion keywords + must-not-say + must-cite-verbatim).

We ran two structural experiments (P1.3 two-pass extract+synth, P2.1 tariff.db router) and hit clear walls. Need outside perspective on three design questions before spending more eval cycles.

---

## Q1. Two-pass synthesis: how to design the soft-fallback?

### What we tried
Two-pass pipeline:
- **Pass 1:** prompt the LLM for a JSON array of *verbatim* spans that answer the question, each grounded in a retrieved chunk. Validator ladder (length + NFKC substring + is_table + BGE cosine ≥0.85) rejects any span that doesn't match verbatim.
- **Pass 2:** re-prompt with the validated JSON as the only citation source; emit a cited answer.
- If pass-1 returns zero validator-accepted spans, emit a canned refusal: *"The retrieved sources did not yield a defensible verbatim span for this question. Conclusion: Cannot answer from corpus."*

### Result
- Single-pass baseline: **40.44%**
- Two-pass (flag on): **27.03%** (−13.41pp)
- Two-pass + chunk refeed in pass-2: **26.62%** (−13.82pp)

### Triage
58/170 items (34%) hit the canned refusal. 55 of those scored positive under single-pass. That one failure mode = the entire regression. An additional ~38 items produced worse-but-non-zero prose (pass-2 hedged more than single-shot).

### The ask
Two options on the table for P1.3-v2:

**R1 — Soft fallback.** If pass-1 yields zero validator-accepted spans, fall through to the single-shot path. Preserves two-pass where it works, avoids the cliff.

**R2 — Loosen validator.** Allow pass-1 to return paraphrased spans with chunk attribution. Higher recall, risks hallucination.

Questions:
1. Which option is more likely to actually gain points vs the 40.44% single-shot baseline — or is the whole two-pass architecture wrong for legal-research-style compositional questions (where the "answer" is a rule synthesized from multiple clauses, not a verbatim quote)?
2. For R1: what's the right trigger for soft-fallback — pass-1 JSON empty? N validated spans < threshold? Judge-model confidence signal?
3. For R2: is there a pattern for "paraphrase + attribution" that doesn't degrade into hallucination in practice? Specifically in a legal domain where citation fidelity matters.
4. Should we abandon two-pass entirely and invest the same eval budget in (a) better retrieval or (b) better single-shot prompting?

---

## Q2. Tariff.db router: how to merge structured facts into narrative?

### What we built
SQLite sidecar (`tariff.db`) with 5 tables (codes, notifications, rates, list_membership, exemptions). Seed: 110 HSN/SAC codes, 5 notifications, 108 rate rows. Router regex-matches queries like "GST rate on HSN 1006" → SQL lookup → returns authoritative rows in 24-51ms vs 25-30s for full RAG.

### Result
Full 170-item eval with router flag on: **39.54%** (−0.90pp vs 40.44%).

Router fired on 7 items. Latency was great. But the answer rendering — bare SQL rows formatted as "Rate: 5% IGST per Notification 01/2017-CT(R) Schedule I S.No.51" — strips the narrative the judge rewards. Net: +1 "correct-citation" point per hit, −3 "conclusion + reasoning" points.

### The ask
Instead of **replacing** the RAG answer with SQL rows, we want to **merge**: let the LLM write narrative prose while anchoring factual claims (rate %, notif number, S.No., effective date) to the SQL rows.

1. What's the right prompt pattern? Candidates we've considered:
   - Pre-retrieval: inject SQL rows as a special "authoritative facts" block in the LLM's context along with retrieved chunks.
   - Post-generation: have the LLM draft freely, then a second pass "verifies and corrects" factual claims against SQL rows.
   - Constrained decoding: force specific rate/number/date tokens to match SQL.
2. Does the approach differ between "pure tariff" queries (where SQL has the full answer) vs "partially tariff" queries (where SQL anchors a rate but the real question is procedural)?
3. How do we prevent the LLM from ignoring or paraphrasing SQL facts inconsistently? (Token-level constraint feels fragile for 14B model.)
4. Is there precedent for this "structured-facts + narrative" pattern in legal/regulatory RAG systems we could mirror?

---

## Q3. Corpus enrichment for entity-filtered retrieval (P2.3/P2.4)

### What we want
When a query names an entity ("Section 16(2)(c)", "Notification 50/2017-Cus", "HSN 9983"), add a Qdrant `must` filter so retrieval returns chunks that **cite** that entity — not just chunks from that section/notification.

### What we have
Qdrant payloads on `indian_legal_t1_v2` carry only scalar fields:
- `section_ref` (17% populated) — the chunk's own internal section in its source doc, not cited sections.
- `doc_number` (82% populated, multiple canonical variants: `50/2017-Cus`, `50/2017-Custom`, `50/2017-CUSTOM`) — the chunk's source notification, not cited notifications.

No arrays for `sections`, `rules`, `notifications`, `hsn`, `sac`. So `must` filters on "chunks that cite Sec 16" return near-zero.

### The ask
Enrichment strategy choices:
1. **Re-ingest:** regex-extract citation arrays from chunk text during fresh ingestion. 108k chunks, ~5-9 ch/s on current pipeline ≈ 3-6 hours. Clean, but blocks other work.
2. **In-place payload update:** scroll Qdrant, regex-extract, write back payload fields without re-embedding. Faster (~1h), but fragile if indexing or schema edits needed.
3. **Second collection:** build enrichment-only sidecar collection, query both, merge results. Preserves the existing collection.
4. **Query-time extract:** skip enrichment; at query time, run regex on top-K retrieved chunks' text and filter/boost. No upfront cost, but every query pays the scan.

Questions:
1. For a 108k-chunk Indian tax corpus, which of these is the proven path? We haven't found a reference implementation.
2. What's the regex/NER quality trade-off — is regex (e.g. `Section\s+(\d+[A-Z]?(\([\w\d]+\))*)`) good enough for Indian legal citation patterns, or do we need a dedicated NER model?
3. Canonicalization: we have variants like `50/2017-Cus` / `50/2017-Custom` / `Notif. No. 50/2017-Customs`. Should we canonicalize aggressively (one form per notif) or index all variants and let MatchAny at query time handle it?
4. If we do (1) re-ingest, is there any reason NOT to also fix other known payload issues (doc_number truncation `Centra`, case variants) in the same pass?

---

## Context the consult may find useful

- Rig is **4-core** (surprising given 6 GPUs — verified via `nproc`). Anything CPU-heavy needs `nice +19 ionice idle --workers 2`.
- qwen3-14B synth model responds to `/no_think` prefix (skips chain-of-thought, ~3x faster).
- Validator ladder was originally 5 rungs (len+clause → NFKC substr → is_table → 6-gram Jaccard → cosine 0.92). We simplified to 3 rungs (dropped Jaccard, loosened cosine to 0.85) and that gained +7pp. Aggressive simplification has paid off once already.
- Gold set is heavy on compositional/multi-section items (~60%) and light on pure-lookup (~15%). So fixing compositional synthesis is higher leverage than fixing lookup.
- Latency budget per query: currently 25-30s p50. Users will tolerate up to ~45s for a better answer.

## What decision this consult feeds

We will pick 1 of Q1/Q2/Q3 as the next implementation target based on: (a) expected pass-rate gain, (b) implementation cost, (c) blocking dependencies for other work. We want the external perspective to either validate our ranking or reshuffle it.

---

**Deliverable requested:**
1. One paragraph per question (Q1/Q2/Q3 + BGE-M3 option from Gemini) with a concrete recommendation + one sentence of why.
2. A **stacked roadmap to 95%** — what's the ordered sequence of changes (not just the next one) that compounds to 95%? Assume resources (API budget, re-ingest time, cloud GPU rental) are available.
3. Any change of frame we're missing — e.g. are we measuring the right thing, is the gold set itself flawed, is a different model architecture (larger synth model, different embedder, multi-query retrieval) a prerequisite we haven't considered?
