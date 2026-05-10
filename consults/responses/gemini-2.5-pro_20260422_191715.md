Excellent. This is one ofr the most thorough and self-aware RAG rebuild plans I've seen. The honesty in Part 1, the detailed hardware constraints, and the self-critiques in Parts 15 & 17 show a mature team that has learned from painful experience.

However, the plan mistakes exhaustive process for iterative progress. It designs a perfect 3-day waterfall and risks a "big bang" failure where a flaw in Stage 2 is only discovered 48 hours later. It's a B+ plan that can become an A+ by embracing speed, focusing more on the query, and adopting SOTA training from day one.

My critique is organized into three core challenges, followed by direct answers to your questions.

---

### Challenge 1: The "One Big Rebuild" Fallacy

Your plan is a multi-day, monolithic batch job. This is incredibly risky. A single flaw in your parser (Stage 2) could poison the entire 150K-chunk dataset, a mistake you'd only catch during the gate eval in Stage 9, days later. The feedback loop is too long.

**Concrete Alternative: The 2-Hour Minimum Viable Corpus (MVC)**

Before you process 15,000 PDFs, prove your entire pipeline on 150.

1.  **Create an MVC:** Select 1% of your corpus (~150 PDFs). Don't pick them randomly. Stratify them:
    *   50 GST (notifications, circulars, rules)
    *   50 Customs (tariff notifications, drawback schedules)
    *   20 Central Excise (legacy, complex formats)
    *   10 Service Tax (even older formats)
    *   20 known-problematic PDFs (image-only, weird layouts, 500+ pages).
2.  **Create a Mini-Gold Set:** Curate 25-50 questions whose answers are *exclusively* within that 150-PDF set. This is your rapid-eval benchmark.
3.  **Timebox the End-to-End Run:** Your entire v2 pipeline—from Stage 0 to Stage 10—must run on the MVC in **under 2 hours**. This is your new critical KPI. If it takes longer, your tools are too slow for rapid iteration.
4.  **Iterate on the MVC:** Run the full pipeline. Does the parser hit 95% coverage? Does the fine-tuned model hit 95% recall on the mini-gold set? If not, fix the broken stage and re-run the *entire* 2-hour pipeline. Repeat until the MVC pipeline is perfect.

Only after you have a proven 2-hour factory that reliably produces a 95%-quality system on the MVC do you "flip the switch" and run it on the full 15K-PDF corpus. This de-risks the entire 3-day build.

### Challenge 2: Ingestion-Obsessed, Query-Agnostic

The plan dedicates 16 parts to perfecting the artifact (the chunks) and barely one to the process (the query). Perfect chunks are useless if the query is misunderstood or the retrieval logic is naive. Your own self-critique (G1) correctly identifies this, but it's an afterthought. It needs to be a core design principle.

**Concrete Alternatives: Build the Brain Before the Library**

1.  **Query Classifier is Non-Negotiable:** A query for `"HSN 8703 duty"` should **never** do a vector search first. It's a structured data lookup. Your query pipeline (Stage 8.0) must start with a classifier (a cheap qwen3-8B call) that routes the query:
    *   **Type A (Factual Lookup):** Rate, date, number, HSN code -> **SQLite first**.
    *   **Type B (Semantic/Procedural):** "Eligibility for...", "Procedure to...", "Conditions for..." -> **Vector search first**.
    *   **Type C (Multi-Hop):** "GST on service X by SEZ to DTA unit" -> **Decomposition step**.
    This router alone will fix a huge class of failures that no amount of chunking can solve.

2.  **Mandate Query Decomposition:** Your gold set is 6.5% multi-hop. That's a hard 6.5% ceiling on your 95% target. Use a local LLM to rewrite complex questions into sub-queries that can be answered by individual chunks.
    *   *Input:* "What is the time limit for claiming ITC on an invoice from last year if the GSTR-3B was filed late?"
    *   *Sub-query 1:* "time limit for claiming ITC on invoice"
    *   *Sub-query 2:* "impact of late GSTR-3B filing on ITC claim"
    Retrieve for both, then synthesize.

3.  **Specify the Fusion:** "Hybrid search" is a category, not a plan. Mandate **Reciprocal Rank Fusion (RRF)** with `k=60` as the default for combining dense, sparse, and keyword search results. It's simple, requires no tuning, and consistently outperforms weighted scoring.

### Challenge 3: Your Training Strategy is Good, but Not Aggressive Enough

Your self-critiques in Parts 14 and 17 are excellent and correctly identify the SOTA techniques. My critique is that you frame them as "amendments" or "options." For a 95% target, they are not optional. They are the plan.

**Concrete Alternatives: Adopt the SOTA Recipe from Day One**

1.  **LoRA is the Default, Full Fine-Tune is the Fallback:** Never re-embed 150K chunks if you can avoid it. The massive operational cost and iteration delay are brutal.
    *   **Your Plan:** Fine-tune the full BGE-M3 encoder, then re-embed everything (4.3 hours).
    *   **New Plan:** Train a **query-side LoRA adapter** on BGE-M3. The 150K document embeddings are created once with the base model and never change. At query time, you load the base model and apply the lightweight LoRA adapter. Iterating on the fine-tune now takes 30 minutes on RunPod, not 4.3 hours of re-embedding. This makes experimentation cheap and fast. You only escalate to a full fine-tune if LoRA fails to bridge the last 2-3% recall gap.

2.  **Use a Teacher Model (MarginMSE):** Vanilla MNRL is leaving performance on the table. You have a powerful cross-encoder (`bge-reranker-v2-m3`). Use it as a teacher.
    *   **Your Plan:** MNRL with hard negatives.
    *   **New Plan:** Use **MarginMSE loss**. Score your training pairs with the cross-encoder first. Then, train the bi-encoder (with its LoRA adapter) to mimic the cross-encoder's score margins. This is a proven 3-7% recall lift over MNRL because it transfers the reranker's nuanced understanding to the much faster bi-encoder.

3.  **Mandate Generator Mixing:** Don't just *evaluate* on a different generator's style; *train* on it. Your plan to use a 70/20/10 split of qwen/Haiku/Flash is correct. This is not optional; it's critical for robustness.

---

### Direct Answers to Your Questions

#### On the ROCm Pivot (Highest Priority):

*   **0a (ROCm Stack):** If ROCm now works on RDNA1/2, this is a **game-changer**. Your highest priority is to run Benchmark B0. If it passes, the stack is `PyTorch-ROCm 6.x` (latest stable), `bitsandbytes-rocm`, and FlashAttention via the official Triton path. The community Discord for ROCm is the best source for gfx10xx-specific gotchas.
*   **0b (What ROCm Unlocks):** Everything Vulkan can't do. Native `sentence-transformers` (for LoRA/MarginMSE), native cross-encoders (for the teacher model), and GPU-accelerated OCR/layout parsers like **Surya and Marker**. This is the path to SOTA.
*   **0c (GPU Chunking):** Don't bother. Your regex-and-tree parser on CPU is simple, debuggable, and fast enough (~17 mins for the whole corpus). Semantic chunking adds complexity and is often worse on structured legal text where boundaries are explicit. The quality lift is minimal for a huge engineering cost. Stick to your tree-walk.
*   **0d (Nougat/Surya/Marker on ROCm):** Yes, they should run. **Marker** is your best bet. It's designed for technical PDFs with tables and has shown excellent results. If ROCm works, your first experiment should be to replace your Stage 1/2/5 with a single pass from Marker. It could obsolete your entire custom parser.
*   **0e (Cross-encoder on ROCm):** `bge-reranker-v2-m3` is the strongest generalist. Start there. It will run fine on PyTorch-ROCm.

#### On the Original Plan:

1.  **Overengineering?** No, it's thorough. The weakness isn't overengineering, it's the monolithic process. Break it down with the MVC approach.
2.  **Parser: Scratch or Existing?** Build from scratch. Your regex-and-tree approach is correct. Indian legal parsers are for case law, not the specific structure of tax notifications.
3.  **Chunking Strategy:** Tree-walk with a size cap and context-prefix (§16.1.3) is the right strategy. It's better than propositional or small2big for this domain.
4.  **Table Handling:** Dual-store (Qdrant+SQLite) is **absolutely correct**. The key is the query router you need to build.
5.  **Embedding Model:** Start with **BGE-M3**. It's a known, powerful baseline with a matching reranker. If ROCm works, benchmark against `Qwen3-Embedding-0.6B` on your MVC, as it may have better domain-specific tokenization.
6.  **Synthetic Q&A:** **Local qwen3-8B is the right call.** The 100x cost advantage and tight coupling to chunks is a massive win. Mitigate quality concerns by mixing in 30% from other cheap generators (Haiku/Flash).
7.  **Supersession Tracking:** Roll your own. It's a graph problem (`(doc_id, supersedes_doc_id)`). No off-the-shelf library will understand the nuances of CBIC notifications.
8.  **95% Definition:** Your 5 points are good. Calibration (#4) is critical and often ignored. Keep it.
9.  **Query Decomposition:** Yes. For 6.5% of your gold set, it's not optional. Add it to your query classifier's routing logic.
10. **Qdrant Scaling:** 200K chunks is trivial for Qdrant on 64GB RAM. No concerns there.

### Final Verdict

You have a plan that can plausibly hit 90% recall. The self-critiques correctly identify the path to 95%, but frame them as optional amendments. They are not.

**My recommendation:**
1.  **Prioritize the ROCm benchmark.** It changes the entire landscape of available tools.
2.  **Adopt the MVC workflow immediately.** De-risk your process and shorten your feedback loop from days to hours.
3.  **Re-architect your training plan around LoRA + MarginMSE as the default.**
4.  **Build the query classifier and decomposer** as a first-class component, not an afterthought.

This plan is a testament to learning from failure. A few strategic pivots will turn that learning into a decisive success.