# Indian Legal AI — Master Plan

**Goal:** Fine-tuned ("weighted") Indian-legal LLM backed by a RAG system over authoritative Indian legal text. Open-sourced per dataset licenses.

---

## Architecture — The Holy Grail

```
  User question
        │
        ▼
  ┌─────────────────────────────────┐
  │  RAG retrieval (Qdrant)         │  ← authoritative text, updatable, citable
  │  Top-k Indian law passages      │
  └─────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────┐
  │  Fine-tuned base LLM            │  ← "weighted" — trained on Indian legal
  │  (Llama-3.1-8B or Qwen2.5-7B    │     reasoning, citation style, tone
  │   + LoRA on our 335K Q&A)       │
  └─────────────────────────────────┘
        │
        ▼
  Answer with cited Indian law
```

- RAG alone → generic model parroting retrieved text, wrong citation format, hedges.
- Fine-tune alone → confident but hallucinates section numbers, stale, can't cite.
- **Both → model that thinks in Indian legal terms, grounded in real passages.**

---

## Current Assets (as of 2026-04-16)

### Tier 1 — Authoritative statutory text (RAG gold)
| Dataset | Rows | Coverage |
|---|---|---|
| `indian_laws` (mratanusarkar) | 34,244 | **1,021 Central Acts** with act_title/section/law |
| `constitution_s` (Sharathhebbar24) | 454 | All Constitution articles |
| `ipc_sections` | 512 | Full IPC (needs `[INST]` unwrap) |
| `indian_law_complete` | 13,133 | TBD inspection |

**Key-act coverage inside `indian_laws`:** Income Tax Act 283 secs • Companies Act 2013: 470 secs • CGST Act 2017: 174 secs • NI Act: 142 secs • CrPC: 484 secs • Motor Vehicles: 217 secs • Contract Act: 75 secs • Arbitration: 259 secs • Consumer Protection: 138 secs • IPC: only 12 (gap filled by `ipc_sections`) • Constitution: only 35 (gap filled by `constitution_s`) • FEMA: partial via Foreign Exchange (97 secs).

### Tier 2 — Case law (retrieval, some OCR noise)
| Dataset | Rows |
|---|---|
| `courts_cases` | 28,816 |
| `sc_chunked` | 10,605 |
| `judgment_summaries` | 6,944 |
| `legal_abs` | 3,599 |
| `sc_en_ta` | 20,000 (EN+TA) |

### Tier 3 — Q&A SFT data (future fine-tune)
| Dataset | Rows |
|---|---|
| `prarabdha_sft` | 200,000 |
| `kshitij_law` | 25,601 |
| `indian_law` | 24,607 |
| `indian_law_9b` | 24,601 |
| `deshmukh_law` | 24,607 (`[INST]` wrap) |
| `tech_legal` | 14,544 |
| `jizzu_law_v4` | 13,013 (`[INST]` wrap) |
| `ipc_insights` | 5,175 |
| `varma_law` | 3,311 |
| Small sets (shreyas, lawyer, traffic, …) | ~3,000 |

### Tier 4 — Excluded
- `income_tax` (ITR form data, not tax law)
- `court_cases_m` (synthetic fake cases)
- `constitution_k` (corrupted / movie-script contamination)
- `legal_ner` (NER tags, not text)
- `legal_corpus` (abbreviation lists)
- 10 empty datasets that failed download

### Gaps
1. Rules & Regulations (IT Rules 1962, CGST Rules 2017, Companies Rules)
2. CBDT / CBIC circulars & notifications
3. Recent amendments (post-2023 Finance Acts, GST Council)
4. State Acts
5. BNS 2023 (replaces IPC)
6. Labour Codes 2020
7. Recent case law (post-2023)
8. Authoritative PDF backup of key acts

---

## Phases

### ✅ Phase 0 — POC (done)
TF-IDF + Qwen3.5 + chat UI at http://192.168.1.107:7000. 5/5 test questions passed.

### ▶ Phase 1 — RAG working on full curated corpus (current)
1. Unwrap `[INST]` tags from `ipc_sections`, `deshmukh_law`, `jizzu_law_v4`
2. Clean OCR noise in `courts_cases` (e.g. `companyducted → conducted`)
3. Drop Tier 4
4. Build Qdrant index from Tiers 1+2 with source-tier metadata (retrieval prefers Tier 1)
5. Switch port 7000 to `rag_api_qdrant.py`
6. Smoke-test 8 real legal questions
7. Commit all scripts

### Phase 2 — Training data prep
- Dedup + quality filter 335K Q&A pairs (Tier 3)
- Convert Tier 1 sections into synthetic (question, cited-answer) pairs using base LLM
- Train/val/test split + 100-Q held-out eval set
- Output: `legal_sft.jsonl`

### Phase 3 — Fine-tune (the "weighted" model)
- Base model: **Llama-3.1-8B** (downloaded) or **Qwen2.5-7B** (TBD)
- Method: QLoRA
- Hardware: RTX 2070/2060 (TBD if present) or rent 1 GPU-day on vast.ai (~$10–30)
- Eval: compare against base on held-out 100-Q set

### Phase 4 — Merge + serve the holy grail
- Merge LoRA adapters → GGUF for llama-server
- Point `rag_api_qdrant.py` at the fine-tuned model
- Release weights + dataset on HuggingFace per licenses

### Phase 5 — Fill data gaps
- Scrape Rules, circulars, BNS, Labour Codes from `indiacode.nic.in`
- Indian Kanoon API for recent judgments
- Targeted PDF backup of 7 key acts
- Re-index (RAG improves without retraining)

---

## Decisions pending (before Phase 3)
1. Base model: Llama-3.1-8B vs Qwen2.5-7B
2. Training hardware: actual RTX 2070/2060 availability vs cloud rental
3. Fine-tune depth: 1 epoch / 50K filtered (fast, ~3–6 GPU-h) vs 3 epochs / 335K (deep, ~20–40 GPU-h)

---

## Files on rig (`/opt/indian-legal-ai/`)
- `datasets/` — 1.8 GB, 32 JSONL datasets, 480K rows
- `models/meta-llama-3.1-8b-instruct.Q4_K_M.gguf` — 4.9 GB
- `rag/tfidf_index.pkl` — POC index
- `scripts/` — poc_setup, rag_api (TF-IDF, running), rag_api_qdrant, build_rag_index_v2, download_datasets_v2
- Qdrant — Docker `localhost:6333`, collection `indian_legal_full` (empty until Phase 1 build)

## Local copies (`D:\_gpu_rig_ai\indian_legal_ai\`)
All scripts synced back from rig for git.
