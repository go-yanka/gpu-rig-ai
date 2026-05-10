# Consult Brief: CBIC RAG — Trustability Architecture

**Audience:** Senior RAG / retrieval / LLM-ops engineer with domain awareness of legal/regulatory text.
**Context:** OOM-safe ingest pipeline for the Indian CBIC (Central Board of Indirect Taxes & Customs) corpus is validated. I am about to do a `--reset` re-ingest to fix an unrelated Qdrant unicode panic. Before I do, I want to bundle payload/chunking decisions that are only reversible via another 45-min re-ingest, and lock in the architecture for the *downstream* retrieval + answer layers.

## Project Objective (hard requirement)

> **99% trustworthy search over the CBIC corpus — including complex queries — with clearly stated reasoning for how the answer was derived, inline citations, proofs, and evidence drawn only from the supplied data.**

"Trustworthy" means:
- Every factual claim in the answer is traceable to a specific retrieved passage.
- The system refuses / hedges when evidence is weak rather than fabricating.
- Citations are specific enough to verify (Section/sub-section number, notification number + date, page, source file).
- Complex queries (multi-hop, comparative, temporal, amendment-chain) are handled, not just single-fact lookups.
- No hallucinated case law, section numbers, or notification references.

Corpus: ~15,559 PDFs, 164,974 chunks currently in Qdrant (status `red` pending re-ingest). Categories: `gst` (99k pts), `customs` (48k), `central_excise` (10k), `service_tax` (3k), `others` (5k). Content is mostly English, some Hindi/Devanagari, heavy on section references, notification numbers, circular numbers, dated amendments.

## Expected Query → Answer → Evidence UX

The end-user interface pattern is **two-panel**:
- **Left panel:** the question and the generated answer, with inline citation markers `[1]`, `[2]`, ... each clickable.
- **Right panel:** when a citation is clicked (or hovered), the exact retrieved passage is pinned on the right with source metadata (file, page, section, notification number, date, clickable URL back to cbic.gov.in). Every factual claim in the answer must have a corresponding evidence card on the right. If you remove the right panel, the left panel loses its trust basis.

Reasoning trace is also clickable: the system shows its chain (what sub-queries it ran, what it retrieved, what it chose to cite) as an expandable section.

### Example 1 — Simple factual

**Query:** "What is the GST rate on handicrafts?"

**Expected answer (left panel):**
> Handmade handicraft items attract a GST rate of **12%** under the principal notification for GST rates, subject to the conditions specified for the category `[1]`. Certain specific handicraft items may be taxed at 5% if they fall under the concessional entry for handmade goods `[2]`. The Ministry has periodically issued clarifications on what qualifies as "handicraft" for this purpose `[3]`.
>
> **Confidence:** High for the 12% general rate; Medium for the 5% carve-out (depends on item classification).

**Expected evidence cards (right panel, one per citation):**
```
[1] Notification No. 01/2017-Central Tax (Rate), Schedule II, Entry 199
    Date: 28-Jun-2017 (as amended)
    Page: 14
    Source: cbic.gov.in/notifications/01-2017-ct-rate.pdf
    Quoted passage: "[exact retrieved chunk text, verbatim, ~3 lines]"
    [Open source PDF] [Copy citation]

[2] Notification No. 21/2018-Central Tax (Rate)
    Date: 26-Jul-2018
    Page: 3
    Source: cbic.gov.in/notifications/21-2018-ct-rate.pdf
    Quoted passage: "..."
    [Open source PDF] [Copy citation]

[3] Circular No. 52/26/2018-GST
    Date: 09-Aug-2018
    Quoted passage: "..."
    [Open source] [Copy citation]
```

### Example 2 — Multi-hop / conditional

**Query:** "If I am a registered person in a Special Economic Zone supplying goods to a customer in the Domestic Tariff Area, is IGST applicable, and who pays it?"

**Expected answer:**
> Yes, IGST is applicable on supplies from an SEZ unit to the Domestic Tariff Area (DTA). Such a supply is treated as an **import into India** `[1]`, and under the IGST Act the recipient in DTA is liable to pay IGST along with applicable customs duty at the time of clearance `[2]`. The SEZ supplier issues an invoice without charging IGST `[3]`; the liability crystallises at the DTA end.
>
> **Reasoning trace (expandable):**
> 1. Sub-query: "SEZ to DTA treatment under IGST" → retrieved 8 chunks, 4 cited
> 2. Sub-query: "who is liable IGST on SEZ-to-DTA supply" → retrieved 6 chunks, 2 cited
> 3. Cross-checked with SEZ Act reference in CGST context
>
> **Confidence:** High — multiple independent sources align.

**Expected evidence cards:** one per citation, same format as Example 1, each quoting the exact retrieved passage.

### Example 3 — Temporal / amendment

**Query:** "What was the GST rate on restaurant services in January 2019, and how has it changed since?"

**Expected answer:**
> In January 2019, restaurant services were taxed at **5% without ITC** for standalone restaurants, a rate that had been effective since 15-Nov-2017 via Notification 46/2017-CT(R) `[1]`. Restaurants in hotels with room tariff above ₹7,500 continued at **18% with ITC** `[2]`.
>
> Since then, the principal change was the inclusion of cloud kitchens and restaurant services supplied through e-commerce operators under Section 9(5) from 01-Jan-2022 per Notification 17/2021-CT(R) `[3]`, shifting the tax liability to the e-commerce operator while retaining the 5% rate.
>
> **Confidence:** High for historic rate; High for the Jan-2022 ECO change.

**Expected evidence cards:** with `doc_date` visible so user can verify temporal claims.

### Example 4 — Adversarial / "refuse-or-hedge"

**Query:** "What is the GST on cryptocurrency trading?"

**Expected answer (correct behavior = hedge + flag):**
> The corpus does not contain a definitive statutory provision setting a GST rate specifically for cryptocurrency trading. Circulars and notifications up to the corpus cutoff treat virtual digital assets primarily under the income tax framework (Section 115BBH, Finance Act 2022), not as a separate GST category `[1]`. Industry and press reporting suggests 18% GST on exchange services (platform fees), but **I cannot confirm this from the supplied CBIC data** — the corpus lacks a specific notification or circular setting this rate.
>
> **Confidence:** Low. **Recommend:** confirm with the latest CBIC circulars post-corpus-cutoff before relying on this.

**Expected behavior:** system must NOT fabricate a notification number. It must surface the gap.

### What the UI must enforce

- **No orphan claims.** If a sentence in the left panel has no citation marker, that is a bug.
- **Citation markers must resolve.** Clicking `[1]` scrolls/highlights the evidence card on the right.
- **Evidence must be verbatim.** The quoted passage on the right is the exact retrieved chunk text, not paraphrased.
- **Source link must open the original PDF at the cited page** (PDF.js anchor or similar).
- **Confidence must be visible** and must be calibrated (if the system says "High", empirical accuracy should be >95% on the eval set).
- **Reasoning trace must be inspectable** — user should be able to see which sub-queries ran, what was retrieved at each step, and what was thrown away.

This UX design is the forcing function for the architecture: the payload schema, retrieval stack, and answer prompt must all be designed to make this UI possible, not the other way around.

## Current Technical State

### Infrastructure
- **Rig:** 6× AMD GPUs (mining-rig repurpose), 31 GB RAM, Linux + systemd
- **Vector DB:** Qdrant (Docker, local)
- **Embedding:** `llama-cpp-python` Vulkan, 6× `mp.Process` workers (one per GPU). BGE-M3 1024d dense + sparse (BM25-style). Direct in-process — zero HTTP intermediaries. Rate: ~60 items/s steady.
- **Chunker:** PyMuPDF (`fitz`) CPU-only, 4 `ProcessPoolExecutor` workers, bounded in-flight window.
- **Ingest stability:** validated. Peak RSS 9.1 GB / 31 GB, zero swap, PSI avg10 0.00%. 48-min full run. OOM is structurally solved.
- **Constraint: fully local.** No cloud APIs. No paid services. Any LLM used for query/answer stage must run on this rig (GPUs available).

### Current chunker output (payload fields I believe exist — to be verified)
- `text` (chunk content)
- `source` (file path or manifest ID)
- `category` (one of 5)
- possibly `chunk_idx`
- **Likely missing:** page_number, section heading, notification number, issue date, amendment chain, source URL back to cbic.gov.in

### Current retrieval stack
- Hybrid dense + sparse via Qdrant native support
- No reranker yet
- No query decomposition
- No answer-generation layer yet (this consult will shape it)

## Open Architectural Questions — Need Specific Opinions

### Q1. Chunking strategy for legal/regulatory PDFs

Tax law text has strong hierarchical structure: **Act → Chapter → Section → Sub-section → Clause → Proviso → Explanation**. Notifications reference sections. Circulars reference notifications. Fixed-size chunking (e.g., 512 tokens) will shred this structure and lose the link between a proviso and its parent section.

**Ask:** Concrete chunking strategy. Specifically:
- Fixed-size with overlap, or semantic / section-aware?
- If section-aware: how to reliably detect section boundaries in CBIC PDFs? These vary wildly in layout — some are text-PDFs with clean structure, some are scanned+OCR'd, some are tabular. Heuristics I'm considering: regex for `^Section \d+[A-Z]?\.?`, `^Rule \d+`, `^\d+\.\s+[A-Z]`, font-size jumps via `fitz` block metadata.
- Should each chunk carry its hierarchical path as a payload field (e.g., `"hierarchy": "CGST Act > Chapter V > Section 17 > (5) > (b)"`)? Does that improve retrieval, or just bloat the payload?
- Chunk size target? Legal text tends to have longer coherent units than general prose.
- Overlap strategy? For legal text, I've seen arguments for 15–25% overlap.

### Q2. Payload schema for legal provenance

What's the minimum-viable payload schema such that the answer can cite like a human lawyer would? Target citation example:
> "Input tax credit is blocked on motor vehicles for personal use — **per Section 17(5)(a), CGST Act 2017, as amended by Notification 14/2022-CT dated 5-Jul-2022, p.43 of gazette** [source: cbic.gov.in/notification/14-2022-ct.pdf]"

**Ask:** Concrete payload schema. My draft:
```json
{
  "text": "...",
  "source_file": "notifications/2022/14-2022-ct.pdf",
  "source_url": "https://cbic.gov.in/...",
  "category": "gst",
  "doc_type": "notification|circular|act|rule|order|instruction",
  "doc_number": "14/2022-CT",
  "doc_date": "2022-07-05",
  "page_number": 43,
  "section_ref": "17(5)(a)",
  "parent_act": "CGST Act 2017",
  "hierarchy": "CGST Act > Chapter V > Section 17 > (5) > (a)",
  "amends": ["13/2021-CT", "..."],
  "superseded_by": null,
  "chunk_idx": 7,
  "chunk_total": 24,
  "ingest_ts": "2026-04-20T15:00:00Z"
}
```

Questions:
- Is this over-engineered or under-engineered?
- Which fields are extractable from the PDF itself vs. which need a separate metadata enrichment pass (e.g., an LLM pass over each doc to extract doc_number/doc_date)?
- `amends`/`superseded_by` ideally requires a legal knowledge graph. Is that in-scope or a v2 problem?

### Q3. Retrieval stack for complex queries

Top-k hybrid search on 164k chunks will handle simple factual queries. It will fail on:
- **Multi-hop:** "What is the tax rate on X under the most recent amendment?"
- **Comparative:** "How did ITC rules change between 2017 and 2023?"
- **Temporal:** "What was the GST rate on handicrafts in July 2021?"
- **Conditional:** "If I'm a registered person in SEZ, is IGST applicable on exports to DTA?"

**Ask:** Concrete retrieval architecture. Specifically:
- Query decomposition: use an LLM to break complex queries into sub-queries? If yes, which local model (we have 6 GPUs, can run 13B–70B)? Prompt template?
- HyDE (Hypothetical Document Embeddings) — worth it for legal text or gimmick?
- Reranker: cross-encoder (e.g., BGE-reranker-v2-m3) on retrieved top-50 → top-10? Worth the added latency?
- Metadata pre-filtering: for temporal queries, filter on `doc_date` before retrieval?
- Iterative retrieval (retrieve → read → retrieve again) for multi-hop?
- Should amendment chains be resolved pre-retrieval (given notification N, find all amendments to N) or post-retrieval (present all and let LLM reason)?

### Q4. Grounded-answer prompt architecture

The answer LLM must:
1. Refuse / hedge when retrieved evidence doesn't support a confident answer
2. Quote verbatim from retrieved chunks with inline citation markers
3. Structure output: claim → evidence → citation
4. Output a confidence score that actually correlates with correctness
5. Never invent section numbers, notifications, or dates not present in retrieved context

**Ask:** Concrete prompt architecture. Specifically:
- System prompt template (full text, not a sketch)
- Structured output format — JSON schema? XML tags? Citation markers like `[1]` with a reference table?
- Should we force function-calling / structured output at the model level, or regex-validate after generation?
- Local model choice for the answer stage — we have 6 GPUs. Candidates we can run: Qwen 2.5 72B, Llama 3.3 70B, Mixtral 8x22B, Qwen 2.5 Coder 32B, Gemma 2 27B. Which for legal reasoning with citation discipline? Any we should specifically avoid?
- Temperature / sampling params for factual-grounded generation?
- Anti-hallucination patterns: self-consistency, citation-verification pass, grounded-ness scoring?

### Q5. Evaluation methodology

"99% trustworthy" is unmeasurable without a framework.

**Ask:** Concrete eval plan. Specifically:
- How to build the gold set? Size? 50 / 200 / 500 questions? Who writes them (me, or LLM-generated and human-reviewed)?
- Distribution across complexity (simple factual, multi-hop, comparative, temporal, conditional, adversarial)
- Metrics:
  - Citation accuracy (cited passage actually contains the claimed fact)
  - Citation completeness (no unsupported claims)
  - Refusal calibration (refuses when it should, doesn't refuse when it shouldn't)
  - Answer correctness vs gold
- Automated eval: LLM-as-judge — which judge, what prompt, how to avoid same-family-bias if judge is Qwen/Llama and answerer is same?
- Regression suite: run on every change, block changes that regress any category?

### Q6. Domain-specific failure modes to defend against

Legal AI has known failure patterns. Which matter most for Indian tax law, and what are the mitigations?
- Hallucinated case citations (fake case names / citations)
- Cross-Act confusion (CGST vs IGST vs SGST vs UTGST vs Customs — all have Section 17s, all different)
- Stale notifications (retrieved text doesn't reflect later amendment)
- Territorial confusion (Central vs State GST vs Union Territory)
- Scope creep (query about customs, answered with GST law)
- Confidence-hallucination (confident tone on wrong answer)

**Ask:** Ranked list of top 3–5 failure modes most likely to bite us on CBIC corpus, and concrete mitigations (not "be careful" — specific prompt patterns, retrieval filters, or post-gen checks).

## Constraints

- **Local-only.** No OpenAI / Anthropic / Gemini APIs in the production path. (External LLM consults like this one are fine for *design*, not runtime.)
- **Hardware committed.** 6 AMD GPUs + 31 GB RAM rig. Vulkan via `llama-cpp-python`. ROCm possible for some workloads.
- **Non-developer operator.** I can write/debug scripts but prefer composing existing tools over bespoke code. Prefer proven open-source components.
- **One re-ingest budget.** Whatever chunker + payload decisions we lock in now will run. Changing chunker later = another 45-min re-ingest. Retrieval/answer layer can iterate freely without re-ingesting.

## What I am NOT asking

- How to run Qdrant (covered)
- How to run llama-cpp-python (covered)
- Ingest memory management (solved)
- "General RAG best practices" — I want opinions specific to Indian legal text, complex queries, and citation discipline, not "try chunking with overlap".

## Deliverable I'd like back

A single consult response that:
1. Answers Q1–Q6 with concrete, implementable decisions (not menus of options — a pick, with reasoning)
2. Evaluates whether the Expected UX examples above are achievable with the proposed stack, and flags specific examples where the stack will struggle
3. Flags anything I missed that is critical for a trustworthy legal RAG
4. Proposes a phased rollout: what to lock in pre-re-ingest (chunker + payload) vs. post-re-ingest (retrieval tuning, answer prompt, eval)
5. Identifies the single highest-leverage decision in this design — the one thing that, if I get wrong, drops us from 99% to 70% — and why

Thanks.
