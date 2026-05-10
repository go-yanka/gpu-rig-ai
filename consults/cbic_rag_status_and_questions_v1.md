# CBIC RAG — Consolidated Status + Open Questions for External LLM Review

**Date:** 2026-04-21
**Purpose:** single-document brief for external frontier LLMs (Claude Opus, GPT-5, Gemini 2.5 Pro, DeepSeek R1) to review progress on our CBIC (Indian indirect-tax) RAG system and advise on next steps.
**What we want from you:** (a) diagnose the A3 regression, (b) rank the 19 open questions by expected lift-per-effort for our constrained rig, (c) flag anything we haven't considered.

---

## 1. System in one paragraph

FastAPI RAG over **108,802 Qdrant chunks** of Indian indirect-tax law (CGST Act, IGST Act, CGST Rules, Customs Act, Central Excise, Service Tax archive, CBIC notifications). Retrieval: BGE-M3 dense on GPU 5 (Vulkan) + BM25 + ColBERT late-interaction (CPU) + MMR + HyDE. Generator: `qwen3-14b-hermes` via `llama-server` Vulkan on GPU 2, port 9082. Orchestrator: FastAPI on `:9500` (UI `/ui`, admin `/admin`). Rig is 4-core (CPU-bound), 8 AMD Vulkan GPUs, Ubuntu boot from 1TB USB, models on Windows SMB. Judge: external LLM scoring against hand-verified gold set.

---

## 2. Baseline & regression numbers

| Run | Config | Score | Latency median | p95 | max |
|---|---|---:|---:|---:|---:|
| Pre-A3 baseline | single-pass extract+synth | **33.23%** (106/319) | 51.3s | 55.8s | 59.6s |
| A3 ON | two-pass (extract JSON → synthesize from JSON) | **21.63%** (69/319) | 55.5s | **79.8s** | 92.7s |
| **Delta (A3 ON − OFF)** | | **−11.6 pts / −35% relative** | +4.2s | **+24s** | +33s |

### By category (A3 ON vs OFF)
| Category | N | OFF % | ON % | Δ |
|---|---:|---:|---:|---:|
| gst | 20 | 38.7 | 21.0 | **−17.7** |
| customs | 10 | 25.8 | 21.2 | −4.6 |
| central_excise | 8 | 38.3 | 27.7 | **−10.6** |
| service_tax | 6 | 35.9 | 17.9 | **−18.0** |
| others | 6 | 20.9 | 20.9 | 0.0 |

### Items that zeroed out under A3
`gst_pos_001` 3→0, `gst_cs_001` 3→0, `gst_inv_001` 3→0, `gst_rcm_002` 3→0, `st_nl_001` 4→0, `st_val_001` 4→0, `exc_cen_001` 4→0, `cus_val_001` 2→0.

**Action taken:** A3 flag flipped OFF on rig (`TWO_PASS_ENABLED=0`), service restarted, confirmed active.

---

## 3. A3 two-pass architecture (regressed — please diagnose)

**Design intent:** force verbatim fidelity by separating extraction from synthesis.

```
Pass 1: query + chunks → LLM (json_schema mode) → structured spans
   ↓
Python validate_span() → 5-level ladder:
   1. length + clause check
   2. NFKC substring exact match
   3. is_table bypass
   4. 6-gram Jaccard ≥ 0.80
   5. BGE-M3 cosine ≥ 0.92
   ↓
Pass 2: query + validated-JSON-only (chunks NOT re-fed) → LLM → final answer
```

**Observed regressions:**
- Answers became shorter and more generic
- Keyword-hit rate dropped (must_not_say phrases started appearing)
- Latency p95 +24s — suggests pass-1 is slow AND pass-2 isn't short-circuiting

**Hypotheses (rank / add your own):**
1. Pass-1 JSON extraction producing skeletal/partial spans → pass-2 synthesizes from insufficient context
2. Validator ladder too strict — rejecting valid extractions → generic fallback
3. Not re-feeding chunks to pass-2 starves the synthesizer of surrounding context that single-pass had
4. Prompt-cache misses double the token cost without warming

**Diagnostic data we can provide:** raw pass-1 JSON + pass-2 output for the 8 zero-scored items.

---

## 4. Structural patches — status

| Patch | Target problem | Status |
|---|---|---|
| **A1** BM25 ×3.5 hard / ×1.8 soft boost on act/rules, RRF client-side rank-shift, min-2-non-Act backfill | retrieval precision | drafted, not deployed |
| **A2** Corpus refresh — CGST Rules + IGST Act re-ingest | recall (known gaps) | pending |
| **A3** Two-pass JSON extraction | verbatim gate | DEPLOYED → REGRESSED → OFF |
| **A4** Tariff / rate routing pipeline (table-first, HSN-aware) | rate/HSN queries specifically | drafted, not deployed |
| **A5** Query-class routing (classify first, route to specialized pipeline) | cross-cutting | pending |

---

## 5. Infrastructure notes

- **Systemd ExecStart migrated** `/tmp/start_api.sh` → `/opt/indian-legal-ai/rag/cbic_rag/bin/start_api.sh` (survived reboot)
- **Feature-flag pattern** via systemd Environment drop-in (`/etc/systemd/system/cbic-rag-api.service.d/two_pass.conf`) — lets us A/B A-patches without editing the unit file
- **Rig RO-filesystem incident** — ext4 journal abort forced emergency remount-read-only; recovered via full backup to Windows (870MB) + reboot + fsck
- **RAG quality playbook** saved to memory (264 lines, documents 5-level verifier ladder, retrieval config, table bypass, deploy discipline, failed experiments)

---

## 6. Gold-set expansion — we are building Q&A pairs for the RAG

Current gold set = 50 items. Target = **150–200 hand-verified pairs** across 6 buckets.

| Bucket | Theme | Target | Status |
|---|---|---:|---|
| 1 | Tariff / rate / HSN / SAC / notifications | 25 | v3 draft ready, high quality |
| 2 | No-evidence / refusal queries | 10 | pending |
| 3 | Complex multi-section reasoning | 25 | pending |
| 4 | Service Tax + Income Tax crossover | 20 | pending |
| 5 | Customs advanced (valuation, classification, drawback) | 20 | pending |
| 6 | Others (appeals, GAAR, penalties, advance ruling) | 20 | pending |

### Pair schema
```yaml
- id: <cat>_<subcat>_<NNN>          # cat ∈ {gst, customs, central_excise, service_tax, others}
  category: gst
  subcategory: rate
  difficulty: basic | intermediate | complex
  question: "practitioner-style query"
  expected_sections: ["Section 9(2) CGST Act"]       # real refs only
  expected_rules: ["Rule 89(5)"]
  expected_notifications: ["01/2017-CT(R) Schedule III"]
  expected_conclusion_keywords: ["18%", "HSN 8471"]  # literal strings
  must_not_say: ["12%", "exempt"]                    # concrete wrong answers
  must_cite_verbatim: true|false
  notes: "provenance / flag for reviewer"
```

### Pair-generation workflow
1. Human-authored prompt with factual-trap callouts + HSN chapter-spread rules + notification whitelist
2. Frontier LLM (Claude / GPT-5 / Gemini 2.5) emits 25 items as strict YAML
3. Human spot-checks 5 random items against CBIC PDFs before merging into `gold_set.yaml`
4. Eval runner scores against these pairs using LLM-judge + keyword match + verbatim-gate

### Why this matters for trust/evidence/citation work
- Hard-negatives for cross-encoder tuning mined from failed pairs
- Verbatim-gate threshold calibration uses judge outputs from these pairs
- Refusal discipline validated via bucket 2 (no-evidence) pairs
- Gold set becomes the seed for a potential "canonical answers" curated layer

---

## 7. Open questions — rank these 1–19 by lift-per-effort on a 4-core rig with Vulkan GPUs, assuming A1 + A2 are done

### Search trustworthiness
1. **Cross-encoder re-ranking** — BGE-reranker-v2-m3 or Jina-reranker-v2 on top-20 → top-5. Is it meaningfully better than our current ColBERT late-interaction on legal queries?
2. **Query rewriting / decomposition** — split multi-part queries ("rate AND cess AND notification?") into parallel retrievals and fuse.
3. **HyDE tuning** — we use HyDE always-on. Is per-query-class HyDE (e.g. only for rate-lookup queries) better, measurable?
4. **Hard-negative mining from eval failures** — use A3's 8 zero-scored items to mine hard negatives, then (a) fine-tune embedder or (b) tune BM25 stop-words / boosts.
5. **Metadata filters + hybrid score weighting** — "query mentions CGST section X → chunks tagged CGST+section=X get score × 1.5" rule.

### Evidence solidity
6. **Multi-chunk evidence requirement** — for "complex" queries, require ≥2 independent chunks supporting the same conclusion (N-of-M rule).
7. **Chunk-level provenance scoring** — each cited chunk carries a per-claim alignment score visible to the user.
8. **Table-aware extraction** — pre-index tables as structured rows (HSN, rate, notification) and query by exact match first, embedding-search as fallback. A4 partly addresses this.
9. **Notification / section graph** — build cross-reference graph (e.g. `01/2017-CT(R)` amended by `18/2021-CT(R)` superseded by `03/2022-CT(R)`). When retrieving an old notification, auto-pull the latest amendment and flag staleness.
10. **Version-pinned corpus** — timestamp each chunk with "effective date range" so RAG can say "as of 2026-04, rate is 18%" instead of citing obsolete text.

### Citation accuracy
11. **Verbatim-gate threshold calibration** — 6-gram Jaccard 0.80 / BGE cosine 0.92 were picked heuristically. Calibrate per query-class (rate lookups stricter, reasoning looser).
12. **Citation span-level mapping** — currently chunk-level. Push to exact sentence / table-row so users / judges can verify without reading the full chunk.
13. **Anti-hallucination ledger** — every emitted citation cross-checked against retrieved pool; if cited notification doesn't appear in any retrieved chunk, reject answer and retry. Worth the latency cost?
14. **"I don't know" discipline** — tune system to refuse when retrieval confidence < threshold. How to pick that threshold from eval data?
15. **Judge-grounded feedback loop** — failed judgments feed into a small classifier that flags "likely wrong" at inference time.

### Meta / process
16. **Golden corpus of canonical answers** — hand-curate answers for top 200 practitioner queries; use RAG only as fallback when query doesn't match a canonical one.
17. **Dual-retrieval disagreement flag** — run BM25 and dense retrieval; if top-5 disagree heavily, flag query as ambiguous and show both candidate sets instead of forcing one answer.
18. **Chain-of-verification** — after answer synthesis, re-query the RAG with "Is it true that [answer]?" and verify retrieval returns the same supporting chunks. Cheap post-hoc check.
19. **Gold-set-as-training-signal** — once we have 150–200 verified pairs, use them for (a) eval only, (b) distill into a small "answer canonicalizer" classifier, (c) LoRA-tune the generator on (question, ideal-answer), (d) embed ideal answers and do answer-retrieval before chunk-retrieval. Rank these four.

---

## 8. Specific asks

1. **Diagnose A3 regression** — given the architecture in §3 and the zeroed-item list, what's your best hypothesis? What diagnostic would you run first if you had raw pass-1 / pass-2 outputs in hand?
2. **Deploy order** — do we deploy A1 (retrieval boost) first, A2 (corpus refresh) first, or parallel? Any reason to re-attempt A3 with a fix (e.g. feed chunks to pass-2, relax validator thresholds) vs retiring A3 entirely?
3. **Rank questions 1–19** by lift-per-effort for our rig. Flag anything theoretically attractive but not worth it at this scale.
4. **Gaps** — what structural fix are we missing that a production legal-RAG would have and we don't?
5. **Verbatim gate** — is a 5-level ladder too aggressive? Is there a cleaner formulation (e.g. single learned span-alignment model instead of 5 heuristics)?

---

## 9. Constraints we cannot change

- Rig is 4 physical cores, CPU-bound on any non-GPU work
- Vulkan on `llama-server`, not ROCm (not real ROCm despite the name)
- No fine-tuning budget for the generator short-term (qwen3-14b-hermes is fixed)
- Embedder is BGE-M3 on GPU 5 (can swap, but requires full re-embed of 108k chunks)
- User is not a developer — solutions must be operable via existing tools / simple scripts, not heavy infra

---

*End of brief. Please return: (a) A3 diagnosis, (b) ranked list 1–19, (c) deploy-order recommendation, (d) anything we missed.*
