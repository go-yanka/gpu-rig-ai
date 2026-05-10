# CBIC RAG — External Consultation Brief

**For:** external LLMs / second-opinion advisors
**Date:** 2026-04-21
**Author:** Claude Code session (sonnet) + Rahul (user)

---

## 1. What the system is

A Retrieval-Augmented Generation system over the Indian Central Board of Indirect Taxes & Customs (CBIC) corpus. Answers practitioner-grade questions on GST, Customs, Central Excise, Service Tax, and related regulation. UI serves a chat interface with an Answer tab (citations + reasoning), Evidence tab (retrieved chunks), verified-vs-suspicious quote split, PDF snapshots, inline citation panels, and a downvote feedback channel.

**Users:** a small set of tax practitioners and the system's author (not a general-audience product).

**Trust bar:** every substantive legal claim must be backed by a verbatim quote from the cited source PDF chunk. A "verified" quote is one where the italic-quoted text is a character-for-character substring of the retrieved chunk (after canonicalization). Quotes that don't pass that test are shown as "suspicious" with a warning.

---

## 2. Stack

| Layer | Component |
|---|---|
| Corpus | 108,802 chunks across 15,559 CBIC PDFs (GST, Customs, Central Excise, Service Tax, "Others" like Finance Acts, Compensation Act, etc.) |
| Vector DB | Qdrant (`cbic_v1` collection), hybrid: dense BGE-M3 (1024-d) + sparse BM25, fused with RRF |
| Embedder | BGE-M3 on llama.cpp Vulkan, pinned to GPU 5 (RX 5700 XT, 8 GB) |
| Generator | qwen3-14B-hermes on llama.cpp Vulkan via LiteLLM gateway (port 9082), pinned to GPU 2. Run in `/no_think` mode at temp=0 for determinism. Context 32k. |
| Reranker | ColBERTv2 on CPU (lazy-loaded), followed by MMR diversity cap (2-3 chunks per `doc_id` prefix) |
| API | FastAPI + uvicorn on port 9500, systemd-managed with auto-restart |
| Router | Keyword classifier for category (gst / customs / central_excise / service_tax / others) → Qdrant payload filter |
| HyDE | qwen3-14B at temp=0.4 generates a hypothetical answer used for dense-embed retrieval. LRU-cached on normalized question |

Hardware is a 4-core consumer rig with 3 AMD GPUs (RX 5700 XT RDNA1 + 2× RX 6700 XT RDNA2). No ROCm support (consumer cards, ABI hell). Vulkan-only on llama.cpp. Running systemd on Ubuntu 22.04.

---

## 3. What's been built

### Working / shipped

- **Hybrid retrieval** (dense + sparse, RRF-fused) with category router
- **ColBERT rerank** + **diversity cap** (MMR, max 2-3 chunks per doc) — prevents single-doc swamping
- **HyDE** with LRU cache (fixed a 5s-per-query cache miss caused by HyDE's temp=0.4 randomness changing the dense-embed cache key)
- **Strict quote verifier** — canonicalization (unicode NFKC, smart-quote normalization, whitespace collapse) + label-prefix strip (handles "(a) …" glue) + 6-gram Jaccard at 0.80 threshold
- **Post-answer validator** — flags answers that contain placeholder leaks (`<exact text>`), mention sections not in the cited chunks, or repeat sentences
- **E-suite UI** — clickable citation badges, inline PDF snapshots, "Verify in PDF" links, date/number badges, downvote feedback, dynamic footer from `/v1/meta`, full-chunk expand/collapse
- **Inline citation panel v2** — click `[S#]` in Answer tab → inline expansion with verbatim chunk text (quoted span highlighted) → Copy/Open PDF/Close buttons → Open PDF loads a 50% right-split iframe at `/pdf/<doc>#page=N` → split toolbar has New-tab / New-window / Close
- **systemd unit** — API auto-restarts on crash, survives reboots
- **Infrastructure hardening** — `faulthandler.enable()`, `threading.Lock` around BGE-M3 embedder for concurrent-query safety, httpx timeouts (180→300s), max_tokens 800→6144 (answers no longer truncate)
- **Prompt hygiene** — removed `<exact text>` placeholder leaks from SYS_PROMPT, added worked examples, reinforced quote-format rules

### Failed / reverted

1. **B17–B21 first attempt** — added multi-sub-query decomposition + section-aware retrieval augmentation + 8 hard reasoning rules + format rules + 3 worked examples. **Reasoning was all correct** (composite supply → IGST, bill-to Karnataka, services advance in January) but only **1 of 6 quote blocks passed verification** — the model quoted statutory language from training memory instead of copying from retrieved chunks. Reverted.

2. **B22 (quote drill)** — added a "Quote Extraction Drill" block + "Anti-paraphrase rule" (with a concrete WRONG/RIGHT example) to the SYS_PROMPT on top of B17-B21. **Worse on both axes**: reasoning collapsed (said "mixed supply" instead of composite, "ship-to Tamil Nadu" for POS, "November" for services advance) AND 0 of 4 quotes verified. One suspicious quote was a verbatim copy of the "RIGHT" example from the anti-paraphrase block — the model was copying from our prompt, not from retrieved chunks. Reverted.

3. **B22b (slim prompt)** — removed the two new blocks from B22, expecting to reproduce the first-attempt result. **Worse still**: same 3 reasoning gates failed AND the model emitted zero quote blocks of any kind (not even suspicious ones). Reverted. This was the third consecutive blind deploy and the data stopped correlating with hypotheses.

### New capabilities built after we stopped iterating blind

4. **Corpus audit** — comprehensive scan of Qdrant for verbatim statutory text. Conclusion below.
5. **Eval harness** — 50 gold-standard Q/A pairs across 5 categories (5 complex multi-part, 19 intermediate, 26 basic), Python runner with section-coverage / keyword / forbidden-word / verbatim-quote gate / LLM-as-judge scoring, diff tool with regression gate. Ready to use as objective measurement.

---

## 4. The critical finding

**We assumed the Act text wasn't in the corpus. It is.**

The audit proved every high-value target (CGST Act sec 2(30), 7, 8, 9, 10, 12, 13, 15(2)(d), 16, 17(5)(h), 31, 34, 39, 49, 50, 54, 73, 74, 75(12); IGST Act sec 5, 7, 8, **10(1)(b)**, 12, 13; CGST Rules 36(4), 37, 42, 86A, 86B, 88C) is **present in Qdrant as verbatim chunks** of the consolidated Act/Rules PDFs. 33 of 34 target references are fully ingested.

**What's actually missing:** only Rule 88D (inserted August 2023; our CGST Rules PDF is December 2022). Possibly stale: IGST Act PDF is March 2020; post-2020 amendments may not be reflected.

**What's actually broken:** retrieval. The Act has 30-204 chunks total. Circulars/notifications/forms total >20,000. In a semantic-similarity contest, the Act consistently loses to circulars *about* the Act. When a user asks about Section 10(1)(b), retrieval returns circulars that mention "bill-to-ship-to" but not the actual Act chunk. The LLM never sees the statutory text, so it paraphrases from training memory, and the verifier correctly flags the paraphrase as suspicious.

This is why all three prompt-level deploys (B17-21, B22, B22b) failed the quote-verification gate: **we were trying to make the model quote text it was never shown**.

---

## 5. Where we stand right now

- **Live backend:** `b11b12a8_v1` (pre-B17 baseline). Verified quotes work on simple queries; fail on complex multi-part scenarios.
- **UI:** inline citation v2 is live (good). User hasn't tested extensively yet.
- **Measurement:** eval harness ready but not yet run for baseline.
- **Retrieval:** ColBERT rerank, MMR diversity, HyDE, category router all working. Missing: bare-statute boost when query names a Section/Rule.
- **Pending deploy:** P1 (retrieval boost for Act/Rules doc_ids when query matches Section-N or Rule-N patterns). Estimated ~1 hour, low-risk.
- **Follow-up:** re-ingest latest CGST Rules (post-Aug-2023) for Rule 88D, refresh IGST Act for Finance Act 2023 amendments.
- **Structural follow-up (not yet attempted):** two-pass structured extraction — first LLM pass returns JSON of `{sub_question, cited_chunk_id, verbatim_span_from_chunk, conclusion}`, post-validator substring-checks each `verbatim_span` against the named chunk and drops/flags failures; second pass renders that JSON into narrative prose. This would *structurally* guarantee verified = emitted, independent of retrieval quality.

---

## 6. Current challenges ranked

1. **Retrieval ranking** — Act chunks lose the semantic popularity contest. Fix: doc_id boost + BM25 weight bump on parenthetical section refs like "10(1)(b)".
2. **Chunk boundary splits** — IGST s.11 and Rule 86B show only fragmentary matches; their provisions cross chunk boundaries. Single chunk should cover a full sub-section.
3. **Model paraphrase instinct** — qwen3-14B is trained to summarize, not verbatim-copy. Prompt rules alone don't override this; the B22 data is definitive.
4. **Prompt bloat threshold** — observed: rule-following breaks down somewhere between 800-1000 tokens of SYS_PROMPT on this 14B model. Stacking rules + examples + drills is counterproductive past that point.
5. **No early feedback loop** — we iterated 3 deploys without an eval harness. All three regressed; we didn't know how badly until the fourth attempt. This is our single biggest process failure.
6. **14B ceiling on 8-sub-question scenarios** — even with perfect retrieval, qwen3-14B may not reliably track all 8 sub-tasks in a query like the Quantum Tech multi-facet scenario. Thinking mode (`/think`) mitigates at a latency cost.
7. **One-corpus fixation** — the ingestion playbook was specialized for CBIC. Other corpora (Income Tax, MCA, Labour, RBI) are queued. If we redesigned from scratch, we'd want a generalizable recipe.

---

## 7. What we did vs what we should have done

### What we did (chronological, honest)
- Shipped UI polish (E-suite, inline popover v1) before stabilizing answer quality
- Fixed individual bugs (B8/B9/P5, B11-B16) one at a time in the same deploy turn, often without isolated verification
- Attempted reasoning/quoting patches (B17-B21, B22, B22b) **without an eval harness to measure change**
- Wrote patch artifacts to `/tmp/` on rig and `%TEMP%` on laptop; lost the B17-B21 forward code when revert destroyed them (had to forensically recover from sub-agent JSONL transcript)
- Assumed the corpus was the bottleneck without auditing; wasted three deploy cycles trying to prompt around a retrieval problem
- Expanded SYS_PROMPT incrementally past the model's reliable-following threshold
- Didn't separate "reasoning rules" from "format rules" in A/B terms — one failed deploy mixed signals

### What we should have done (with hindsight)
- **Built the eval harness FIRST** — before any prompt or retrieval tweak. Even a 10-question gold set would have caught the B22 reasoning regression immediately.
- **Audited the corpus FIRST** — a 30-minute audit would have shown us the problem was retrieval, not ingestion, not prompt. We'd have skipped three bad deploys.
- **Persistent patch paths from day 1** — `/opt/indian-legal-ai/patches/<feature>_<ts>/` and `D:\_gpu_rig_ai\patches\<feature>_<ts>\` plus `*.patched.<feature>.<ts>` mirrors next to live files. Zero patch loss. Zero forensic recovery.
- **One dimension per deploy** — never mix "new retrieval feature" with "new prompt rules" in the same patch. When Quantum Tech fails, we can't tell which dimension broke.
- **Measured baseline before any change** — without a before-number, after-numbers don't mean anything.
- **Feature flags** — every new feature (multi-sub-query, section-aware, quote drill) should be toggleable via env var. Disabling a feature shouldn't require a code revert.
- **Retrieval-side first, prompt-side last** — getting the right chunks to the model is the foundational problem. Teaching the model to quote better assumes the chunks have what to quote. We inverted the priority and paid for it.
- **Don't iterate on a single query as a success metric** — Quantum Tech is one scenario. Sample size 1 = noise. 50-Q eval = signal.

---

## 8. Questions for external LLMs

We'd like a second opinion on the following. Please be direct — tell us where our thinking is wrong.

### Architecture questions

**A.** Given a 14B model on consumer AMD Vulkan (no ROCm), what's the *right* RAG architecture for high-trust legal citations? Specifically:
- Is two-pass structured extraction (JSON-first, prose-second) the right structural fix for verbatim quoting, or is there a better pattern? (function calling? grammar-constrained generation via GBNF? retrieval with sentence-level chunks and multi-chunk concatenation?)
- If you were designing this from scratch, would you use a 14B generator + strict verifier, or a larger generator + looser verifier? What's the trust trade-off?
- For 8-sub-question multi-part legal scenarios, is our decomposition approach (split → per-subquery retrieve → union → rerank → single-shot generate) the right call? Or should each sub-question get its own generation pass and we stitch answers?

### Retrieval questions

**B.** Our Act chunks (30 for IGST, 204 for CGST) are losing semantic similarity contests to 20,000+ circulars/notifications *about* those Acts. What's the right fix?
- Simple doc_id boost at fusion time (`priors` multiplier when query regex-matches Section/Rule)?
- Payload-filter forced routing (if query mentions "Section 10(1)(b) IGST", pre-filter to `doc_id = IGST_Act` then rerank)?
- Parent-document retrieval (chunk→document link, retrieve whole Section as a unit)?
- Tiered retrieval (first pass: is this a definitional question? if yes, restrict to Acts/Rules; else open)?
- Something else?

**C.** BGE-M3 is our dense embedder. For legal text with parenthetical section numbers ("10(1)(b)", "75(12)", "17(5)(h)"), is there a better embedder? Is domain-adapting BGE-M3 via contrastive fine-tuning on CBIC pairs worth it, or is sparse BM25 + boost sufficient?

**D.** Our chunker splits some Rules mid-clause (observed for IGST s.11 and Rule 86B). What's the right chunking strategy for legal text? Sentence windows with parent-doc retrieval? Section-aware (regex-based) boundaries? Semantic chunking?

### Quoting / verification questions

**E.** The verifier uses canonicalization (NFKC, whitespace, smart quotes) + label-prefix strip + 6-gram Jaccard at 0.80. Is this threshold sensible? Too loose? Too strict? For legal text with genuine paraphrasing tolerance, would BGE-M3 cosine similarity on quote-vs-chunk be a better verifier than n-gram overlap (while still being strict)?

**F.** Is there a way to make the model *structurally* unable to hallucinate quotes? Grammar-constrained generation (GBNF) that forces each quote span to match a regex derived from retrieved chunks? Copy-from-context attention biasing? What works in production?

### Reasoning questions

**G.** qwen3-14B paraphrases legal text by default. We've tried: temp=0, strict prompt, worked examples (contaminated), anti-paraphrase rule (worse). What's the state of the art for forcing verbatim copying in 14B-class models? Chain-of-thought scratchpad? Self-consistency voting? Fine-tuning LoRA on verbatim-copy exemplars?

**H.** For multi-part legal reasoning (8 sub-questions), is "reasoning rules in the prompt" a losing game past a certain complexity? Would domain-tuned LoRA be the right move? Function-call decomposition ("classify_supply_type(facts)" → returns composite/mixed/bundled with reasoning)?

### Ingestion questions (design from scratch)

**I.** If we're ingesting a new Indian regulatory corpus (Income Tax Act + Rules + Circulars + Case Law) from scratch, what's the right pipeline? Assume:
- Same hardware (4-core, consumer AMD Vulkan, Qdrant, BGE-M3)
- Mix of native-text PDFs (most) and image-only PDFs (minority, OCR'd)
- Want high-trust verbatim quoting as Day-1 capability
- Want the system to handle definitional queries ("what does Section X say?") and scenario queries ("given facts F1, F2, F3, what's the tax treatment?") equally well

What we'd like opinions on:
- Chunking strategy: sentence-window? section-aware? semantic? hybrid with parent-doc?
- Embedding: BGE-M3 sufficient, or consider alternatives (Jina v3, Voyage legal, bge-en-legal if available)?
- Sparse: BM25 vs SPLADE vs hybrid?
- Metadata schema: what payload fields are essential beyond `doc_id`, `page`, `title`, `category`, `text`?
- Should we ingest provisions as *structured* objects (Section ID, subsection tree, cross-refs) in parallel with flat text? Would that help retrieval?
- Should Act/Rules get separate collections from circulars/notifications, queried differently?
- How to handle historical versions (Section X as of 2017, as amended 2019, current 2023)? Versioned collection?
- Evaluation: what gold-set size per corpus? Who produces it?

### Training / fine-tuning questions

**User asked directly:** *"Is training an LLM now a bottleneck? If we did that would we get overall better results — reasoning, citations, accuracy, trustworthy answers a professional can use?"*

Our current view (want your verdict):

- **Base-model ceiling hypothesis.** qwen3-14B paraphrases statutory text by default. Prompt rules, worked examples, and anti-paraphrase drills have all failed to override this instinct (B17-21, B22, B22b data). Is this a *fundamental* 14B-class limitation, or still a prompt/retrieval problem in disguise?
- **What fine-tuning *could* fix, if it's the real bottleneck:**
  - Verbatim-copy behavior — train on `(retrieved_chunk, question, answer_with_exact_span_copied)` triples so the model's default is to copy, not paraphrase.
  - Domain vocabulary — "bill-to-ship-to", "composite supply", "10(1)(b)", "DRC-01B" etc. get first-class representations rather than being assembled from general tokens.
  - Multi-sub-question decomposition — train on scenario queries with 8 sub-facets so the model doesn't drop any in a single pass.
  - Cite-before-claim discipline — train so that every conclusion token is preceded by a citation token referencing a provided chunk.
- **What fine-tuning probably *won't* fix:**
  - Missing text in retrieved context (retrieval bug).
  - Stale corpus (ingestion bug).
  - Chunk boundary splits mid-clause (chunker bug).
  - A 14B model's raw legal-reasoning ceiling on 8-sub-question scenarios — more likely needs a bigger model or decomposition, not LoRA.
- **Hardware constraints.** Consumer AMD Vulkan, no ROCm. LoRA training on the rig itself is not realistically supported (bitsandbytes / PEFT expect CUDA). Practical paths: (a) rent an Nvidia A100/H100 for a few hours, train LoRA offline, deploy adapter; (b) use a cloud fine-tuning API; (c) skip training entirely and lean harder on retrieval + two-pass extraction.
- **Data shape question.** To produce measurably better verbatim citation behavior, how many training examples do we need? Guesses: 500 for LoRA sanity, 2-5k for meaningful behavior shift, 10k+ for professional-grade trust. Producing 10k hand-verified CBIC Q/A pairs is a major sub-project on its own.
- **Embedding fine-tuning vs generator fine-tuning.** Contrastive fine-tuning of BGE-M3 on `(query, relevant-chunk)` pairs to improve retrieval ranking is *much cheaper* than generator LoRA and might solve the "Acts lose to circulars" problem more directly. Is this the higher-ROI training investment?

**Questions to you:**

**M1.** Is fine-tuning the generator (qwen3-14B LoRA on verbatim-copy exemplars) likely to produce a measurable improvement in our verbatim-quote gate pass rate, or is the base model's paraphrase instinct fundamentally unfixable below a 32B scale?

**M2.** Given our constraints (consumer AMD Vulkan, no local training, bottlenecked professional time for producing gold data), what's the right *order* of investment: (a) embedding fine-tune first, (b) generator LoRA first, (c) skip training and double-down on retrieval + two-pass extraction, (d) upgrade base model (qwen3-32B or similar) without any fine-tuning?

**M3.** For verbatim-copy training data, what's the right shape? `(chunk, question, answer)` with explicit copy markers? RLHF-style with a copy-faithfulness reward? DPO with `(copied_answer, paraphrased_answer)` preference pairs?

**M4.** Is there evidence that fine-tuning a 14B legal model on Indian tax law has been done successfully anywhere? Any public datasets, open-weight starting points (e.g., LexGLUE, InLegalBERT, Nyaya-LLM), or commercial APIs whose output we could distill from?

**M5.** If we *don't* fine-tune and instead stack (structured extraction + retrieval boost + bigger model qwen3-32B), can we realistically hit "professional-grade trustworthy" without any training? What's your honest probability estimate?

### Process questions

**J.** We spent ~3 deploy cycles iterating without an eval harness. Our rule going forward is "eval harness first, always". Is there a lightweight way to keep this discipline in a fast-moving dev loop? Any pattern you've seen work?

**K.** When do you call a deploy "done"? Our current rule: all hard-gates pass on the target query + eval harness doesn't regress by >5%. Is that right, or too loose/strict?

**L.** We ship prompt changes and retrieval changes in the same patch today. That's bad (can't attribute regressions). What's the right deploy cadence? One dimension at a time, each gated by eval?

---

## 9. What we're about to do

**Immediate (next 1-2 hours):**
1. Run eval harness against current live `b11b12a8_v1` to get baseline.
2. Deploy P1: retrieval boost for bare-statute doc_ids when query regex-matches `\b(Section|Sec|s\.|Rule)\s*\d+` or legal phrases (composite supply, place of supply, etc.). Retrieve-time only, zero prompt change.
3. Re-run eval. Diff vs baseline. Ship if aggregate improves and no individual query regresses >20%. Revert if not.

**Next phase (1-2 days):**
4. If P1 passes but quote-verification still fails on some queries: deploy Option A two-pass with substring validator.
5. Re-ingest updated CGST Rules (post-Aug-2023) for Rule 88D. Consider refreshing IGST Act.

**Near-term (1-2 weeks):**
6. Query-class routing — different prompts for definitional vs scenario queries. Reconciliation-class queries get the GSTR rules we cataloged separately.
7. Feedback mining — E10 downvotes → eval harness regression cases.

**Open:**
8. Option A vs alternatives (grammar-constrained generation? function calling?). This is partly what we're asking you about.

---

## 10. What we'd specifically like from you

1. **Direct criticism of our architecture** — anything we're missing?
2. **Verdict on our immediate plan (section 9)** — right order? Wrong order? Missing anything?
3. **Answers to the questions in section 8** — especially A, B, F, I, J. Be specific; "it depends" isn't useful here.
4. **Ingestion-from-scratch recipe** for the next corpus (Income Tax) — what would you do differently vs what we did for CBIC?

Thank you.
