# P2.5 — Query Decomposition Design

**Goal.** Multi-part bucket-3 queries ("In scenario X: is A taxable? what rate? which form?") get split into 2–4 self-contained sub-queries; each retrieves independently; chunks are deduplicated and fed jointly to the synthesizer. Fixes the failure mode where a single chunk cannot cover all facets and MMR diversity alone is insufficient.

## 1. Trigger conditions

Decomposition runs only when at least one holds:

- **Explicit multi-part:** query contains ≥2 `?`, or the pattern `\b(and|also|additionally|furthermore)\b` joining two interrogatives, or bullet/numbered list syntax (`^\s*[-*\d.]`).
- **Length heuristic:** token count > 40 (empirical cutoff from gold_set bucket-3 items).
- **Entity multiplicity:** P2.4 NER returns ≥2 entities of different types (e.g. a section AND a notification AND a form) — signals distinct lookups.
- **Router class:** a lightweight classifier tags each query `lookup | explain | scenario`. `scenario` → always decompose. `lookup` → never (one answer chunk). `explain` → heuristic above.

If none fire, skip decomp (keeps p95 unchanged for ~70% of traffic).

## 2. Decomposition prompt (qwen3-14B `/no_think`)

```
You split Indian-tax questions into atomic sub-questions.
Rules:
- Each sub-question must be answerable from a single statute/circular chunk.
- Preserve all entities (section numbers, notifications, HSN, dates).
- Output strict JSON: {"subqueries": ["...", "..."]} with 2-4 items.
- If the input is already atomic, return {"subqueries": [<original>]}.

Query: <user_query>
```

**Failure handling:** parse JSON; if malformed, 1-item list, or >4 items → fall back to single-query pipeline. Log parse-failure rate; target <3%.

## 3. Retrieval per sub-query

- Run each sub-query through the **full** existing pipeline: P2.4 NER filter → dense+BM25 RRF top-50 → ColBERT rerank top-8 → MMR.
- Run sub-queries in parallel (asyncio.gather) — retrieval is I/O-bound against Qdrant + CPU-bound rerank; with 2–4 concurrent, wall time ≈ 1.3× single.
- Dedup returned chunks by `chunk_id`; keep highest rerank score per chunk.
- Cap combined set at top-12 (was top-8); rerank-score sorted.

## 4. Synthesis prompt

```
Answer the ORIGINAL question using ONLY the context chunks.
Each factual claim must cite [chunk_id].
If a sub-question is unanswerable from context, say so explicitly rather than inventing.

Original question: <raw_query>
Sub-questions identified: <list>
Context chunks:
[id=...] <text>
...
```

Single qwen3-14B call with full `/think` (not `/no_think`) for synthesis — coherence matters more than latency here.

## 5. Latency budget

| Stage | Single-query p95 | Decomp p95 |
|---|---|---|
| Decomp LLM call | — | +8 s (qwen3-14B `/no_think`, ~60 tok out) |
| Retrieval | 5 s | ~6 s (3 sub-queries parallel, Qdrant serial bottleneck) |
| Rerank | 4 s | ~8 s (3× ColBERT passes, partial parallel) |
| Synthesis | 40 s | ~45 s (more context → more tokens) |
| **Total** | **55 s** | **~70 s** |

Decomp adds ~15s. Acceptable only for multi-part queries — hence strict triggering.

## 6. Test cases (gold_set.yaml bucket_3)

1. "A hotel in Goa charges ₹8000/night — is GST applicable, what rate, and which HSN/SAC?"
 → [GST applicability on hotel services], [GST rate for room tariff ₹8000], [SAC code for hotel accommodation]
2. "ITC on motor vehicle: when allowed under Section 17(5), what exceptions, how to claim in GSTR-3B?"
 → [Section 17(5) motor vehicle ITC blocking], [exceptions to Section 17(5)(a)], [ITC claim procedure in GSTR-3B]
3. "Exporter with LUT — documentation required, refund timeline, Rule reference?"
 → [LUT documentation for exports], [GST refund timeline for exporters], [Rule governing LUT exports]
4. "Composition scheme turnover limit, eligible persons, quarterly return form?"
 → [composition scheme turnover threshold], [eligibility Section 10], [CMP-08 quarterly return]
5. "TDS under Section 51: who deducts, what rate, when to file GSTR-7?"
 → [Section 51 deductor categories], [TDS rate under Section 51], [GSTR-7 filing due date]

## 7. Implementation sketch

```python
async def maybe_decompose(q: str, entities: dict, cls: str) -> list[str]:
    if not should_decompose(q, entities, cls):
        return [q]
    try:
        subs = await llm_decompose(q)
        if 2 <= len(subs) <= 4:
            return subs
    except Exception: pass
    return [q]

async def query_v2(q):
    ents = extract_entities(q)            # P2.4
    cls  = classify(q)
    subs = await maybe_decompose(q, ents, cls)
    chunk_sets = await asyncio.gather(*[retrieve(s) for s in subs])
    chunks = dedup_merge(chunk_sets, k=12)
    return await synthesize(q, subs, chunks)
```

Wraps current `/v1/query`; gated behind `?decomp=auto|on|off` flag for A/B.

## 8. Risks

- **Over-decomposition.** Simple lookups split unnecessarily → +15s for no gain. Mitigation: strict trigger + log decomp rate, target <25% of traffic.
- **Sub-query latency blowout.** 4 sub-queries with serial Qdrant = 4× retrieval. Enforce parallelism + hard cap at 4 subs.
- **Synthesis incoherence.** 12 chunks across 3 topics can produce stitched-together prose. Mitigation: explicit "answer each sub-question in order" instruction; eval on bucket-3 faithfulness.
- **Entity loss in decomp.** LLM may drop "Section 16(2)(c)" when splitting. Mitigation: post-process — if any sub-query lacks entities present in original, append them verbatim.
- **Cost.** +1 LLM call per multi-part query. Acceptable locally; track decomp-call count.
