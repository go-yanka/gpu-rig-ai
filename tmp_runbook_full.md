# RUNBOOK — CBIC v2 Full Re-ingest (15,776 docs)

**Status:** READY (chunker locked, pair-gen pipeline built, all gates defined)
**Owner:** rig @ `192.168.1.107`
**Authoritative parameters:** `reingest_spec/DECISIONS.yaml` (every script reads this)
**Last updated:** 2026-04-25

---

## Corpus

| metric | count |
|---|---|
| Total docs in scrape manifest | **15,776** |
| Born-digital (this re-ingest scope) | **14,925** |
| OCR-deferred (separate pipeline, already complete) | 851 |
| Shared-source-PDF doc_ids (Defect D) | 391 (2.5%) — cleared <5% threshold, NO manifest enrichment needed |

---

## Hardware fleet (rig 192.168.1.107)

| GPU | Role | Service |
|---|---|---|
| GPU 0, 1 | **EXCLUDED** for Vulkan compute (60× slowdown documented 2026-04-24) | — |
| GPU 2 | qwen3-14b LLM | `qwen3-14b.service` on `:9082` (Pass-1 classifier + pair-gen primary + answer LLM) |
| GPU 3 | Verify per-boot, candidate for embed | smoke first |
| GPU 4, 5, 6 | **BGE-M3 embedder pool** (Vulkan llama-cpp-python) | `embedder_direct.py` |
| CPU | **OFF-LIMIT** for embed/rerank/OCR | — |

**Mandatory env on every phase that embeds or chunks:**
```
DENSE_ONLY=1
EMBED_GPUS=4,5,6
RADV_DEBUG=nodcc
GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1
PYTHONPATH=/home/user/.local/lib/python3.10/site-packages:/opt/indian-legal-ai/rag/cbic_rag
```

**Mandatory key sourcing for phase6/G2:**
```
set -a; source /root/.cbic_env; set +a   # GEMINI_API_KEY, OPENROUTER_API_KEY
```
**Forbidden flags:** `--flash-attn`, `--mlock`, KV quant.

---

## Decision references (locked)

All operational params live in `reingest_spec/DECISIONS.yaml`. Highlights:

| decision | value | locked |
|---|---|---|
| Chunker recipe | qwen3-14b Pass-1 + rule splitter Pass-2 | 2026-04-25 |
| Defect C bypass for forms | `_BYPASS_QWEN_PREFIXES=("cbic-form-msts",)` + `_DEFAULT_PLANS_BY_PREFIX` | 2026-04-25 |
| Pair gen cardinality | 12 queries / chunk | 2026-04-25 |
| Generator mix | qwen3-14b (100%) + Gemini Flash (20%, hash%5==0) + Claude CLI (10%, hash%10==0) | 2026-04-25 |
| Hard negatives | inline, cosine band 0.60–0.85, k=5, same-doc | 2026-04-25 |
| Gold sourcing for G1 | union(generated, legacy-filtered) | 2026-04-25 |
| Recall threshold | adjusted recall@10 ≥ 0.95 (excl. shared-PDF Defect D) | non-negotiable |
| Per-doc reporting | required at end of every phase6 run | 2026-04-25 |

---

## Phase summary table (full corpus 14,925 docs)

| phase | what | hardware | LLM | per-doc | total | parallel? |
|---:|---|---|---|---:|---:|---|
| 0 | Pre-flight (manifest, Defect D check, services up, keys, GPUs) | rig | — | — | ~10 min | — |
| 1 | Manifest copy/build to `ingest_manifest_v2.sqlite` | rig | — | <1s | ~5 min | no |
| 2 | Chunk (two-pass) | GPU 2 (qwen3) + CPU file I/O | qwen3-14b | ~28s | **~116h sequential** / **~30h with 4 parallel workers** | yes |
| 3-4-5 | Embed + upsert to Qdrant | GPUs 4,5,6 | BGE-M3 | ~0.5s | **~2.5h** | inherent (3-GPU pool) |
| 6 | Per-chunk pair generation + hard-neg mining | GPU 2 (qwen3) + Gemini API + Claude CLI + GPUs 4,5,6 (hard-neg embed) | qwen3-14b + gemini-2.0-flash + claude (Max) | ~30s/chunk | **~1500h sequential / ~150h with 10× parallel workers** | yes — biggest bottleneck |
| 7 | Build G1 union gold | rig | — | <1s | ~5 min | no |
| 8 | G1 — recall@10 | rig + GPUs 4,5,6 | BGE-M3 | <1s | ~30 min | yes |
| 9 | G2 — dual-judge faithfulness | rig + Gemini + Claude CLI | qwen3 (answer) + Gemini + Claude (judge) | ~10s/q | ~10–15h | yes |
| 10 | G3 — answer quality | rig | qwen3-14b + Gemini judge | ~5s | ~3h | yes |
| 11 | G4 — adversarial / refusal | rig | qwen3 + Claude judge | ~5s | ~2h | yes |
| 12 | G5 — latency + cost benchmark | rig | full stack | ~3s | ~1h | yes |
| 13 | Cutover (cbic_v2 → cbic_v1 alias) + codification | rig | — | — | ~30 min | no |

**Critical path total at recommended parallelism:** **~210 hours ≈ 9 days** (dominated by Phase 6).
**Sequential worst case:** ~1700h (~70 days) — DO NOT run sequentially.

---

## Phase 0 — Pre-flight (~10 min)

**Goal:** verify nothing is broken before committing 9 days of compute.

### Steps
1. **Manifest sanity:**
   ```bash
   sqlite3 /opt/indian-legal-ai/data/scraped/cbic/_manifest.sqlite \
     "SELECT COUNT(*), text_source, COUNT(DISTINCT path_en) FROM docs GROUP BY text_source;"
   ```
   Expect 15,776 total / 14,925 born / 851 ocr.

2. **Defect D pre-flight (already cleared 2026-04-25):**
   ```sql
   SELECT path_en, COUNT(*) AS n FROM docs WHERE phase1_done=1
   GROUP BY path_en HAVING n>1 ORDER BY n DESC LIMIT 5;
   ```
   Confirmed 2.5% < 5% threshold → proceed.

3. **GPU 2 — qwen3-14b alive:**
   ```bash
   curl -sm 5 http://127.0.0.1:9082/v1/models | grep qwen3
   ```

4. **GPUs 4,5,6 — Vulkan smoke:**
   ```bash
   for g in 4 5 6; do GGML_VK_VISIBLE_DEVICES=$g \
     /opt/llama-server-b8840/llama-server --model /opt/ai-models/bge-m3.gguf \
     --embedding -ngl 99 -t 1 -c 1024 --port $((9000+g)) -e "test" --batch-size 1 --no-warmup; done
   ```

5. **Keys present:**
   ```bash
   grep -c '^GEMINI_API_KEY\|^OPENROUTER_API_KEY' /root/.cbic_env
   ```
   Expect 2.

6. **Disk space:**
   ```bash
   df -h /opt /var/lib/qdrant
   ```
   Need ≥ 50 GB free for full Qdrant collection + 10 GB for pair corpus.

7. **DECISIONS.yaml sha snapshot** (stamp every artifact in this run):
   ```bash
   sha256sum /opt/indian-legal-ai/reingest_spec/DECISIONS.yaml | cut -c1-16
   ```

### Precautions
- **NEVER skip preflight.** `--no-preflight` flag exists for component smoke only.
- If qwen3 returns 503 "Loading model", wait — first load is ~60s after restart.
- If GPU 3 looks healthy, document but DO NOT add to embed pool without throughput benchmark.

---

## Phase 1 — Manifest build (~5 min)

**Goal:** copy/refresh `ingest_manifest_v2.sqlite` from scrape manifest, set initial `phase1_done=1` for born-digital docs.

```bash
cd /opt/indian-legal-ai/reingest_spec
python3 ingest_v2.py --phase phase1 --no-preflight
```

- **No LLM. CPU only.** sqlite copy + Hindi-twin index build.
- Output: `/opt/indian-legal-ai/data/ingest_manifest_v2.sqlite`
- Skip if already populated (`--resume` default).

### Precautions
- Sanity check after: `SELECT COUNT(*) FROM docs WHERE phase1_done=1;` should be ~14,925 (born-digital only).

---

## Phase 2 — Chunking, two-pass (~30h with 4 parallel workers)

**Goal:** for each doc, qwen3-14b classifies structure + rule-driven splitter cuts chunks honoring section/proviso/explanation boundaries. Defect C fallback covers all known prefix traps.

### Hardware
- **GPU 2** runs qwen3-14b on `:9082` (Pass-1 classifier).
- CPU does file I/O (PDF read via PyMuPDF, regex splits, dedup).

### Per-doc cost
- Set 5 measured: 99 docs / 47.4 min sequential = **~28s/doc** (one Pass-1 call + Pass-2 split).
- Scaling: 14,925 / 99 × 47.4 min = ~119h sequential.
- **With 4 parallel workers** (qwen3-14b server `--parallel 4` already supports): ~30h.

### Command (single worker — sequential)
```bash
cd /opt/indian-legal-ai/reingest_spec
DENSE_ONLY=1 EMBED_GPUS=4,5,6 RADV_DEBUG=nodcc GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1 \
  PYTHONPATH=/home/user/.local/lib/python3.10/site-packages:/opt/indian-legal-ai/rag/cbic_rag \
  nohup python3 ingest_v2.py --phase phase2 \
  --allow-phase2-failures 750 \
  > /tmp/phase2_full.log 2>&1 &
```

### Command (4 parallel workers — RECOMMENDED)
Split doc_id list into 4 shards, run 4 instances pointing at the same manifest:
```bash
sqlite3 /opt/indian-legal-ai/data/ingest_manifest_v2.sqlite \
  "SELECT doc_id FROM docs WHERE phase1_done=1 AND phase2_done=0" > /tmp/all_pending.txt
split -n l/4 -d /tmp/all_pending.txt /tmp/p2_shard_
for s in 0 1 2 3; do
  IDS=$(paste -sd, /tmp/p2_shard_0$s)
  DENSE_ONLY=1 EMBED_GPUS=4,5,6 RADV_DEBUG=nodcc GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1 \
    PYTHONPATH=/home/user/.local/lib/python3.10/site-packages:/opt/indian-legal-ai/rag/cbic_rag \
    nohup python3 ingest_v2.py --phase phase2 --doc-ids "$IDS" \
    --allow-phase2-failures 200 > /tmp/p2_w${s}.log 2>&1 &
done
```

### Parameters (all locked)
- qwen3 max_tokens=4096, timeout=180s, retries=3
- `_BYPASS_QWEN_PREFIXES=("cbic-form-msts",)` (Defect C)
- `_DEFAULT_PLANS_BY_PREFIX` (10 prefix templates + `_GENERIC_`)
- `_detect_repetition()` early-break
- `--allow-phase2-failures 750` — at full corpus scale, allow ≤5% phase2 raises before halting (single-doc edge cases shouldn't block the run)

### Precautions
- **PyMuPDF (`fitz`) MUST be installed** — without it phase2 silently falls back to pdfminer and poisons table_regions. Verified by preflight `_preflight_fitz()`.
- Watch `/tmp/p2_w*.log | grep -E 'FAIL|raise|RuntimeError'` periodically.
- If qwen3 service crashes (OOM): `systemctl restart qwen3-14b && wait 60s for model load`. Phase2 has resume.
- DO NOT run Phase 2 and Phase 3-4-5 concurrently against the same manifest.

### Expected outcome
- ~14,925 docs chunked, ~180,000 chunks emitted, written to manifest's chunks table.
- Failure rate target: 0 raises in steady state (Defect C generic fallback covers all observed cases). Hard ceiling 5%.

---

## Phase 3-4-5 — Embed + upsert (~2.5h)

**Goal:** BGE-M3 dense embedding of all chunks, upsert to fresh Qdrant collection `cbic_v2_full`.

### Hardware
- **GPUs 4, 5, 6** in parallel via `embedder_direct.py` `_Pool` (3-process pool, one per GPU).
- Throughput: **22.4 ch/s** measured on Set 5 (1210 chunks in 54s). Budget conservatively at 20 ch/s for 180K chunks → 9000s ≈ **2.5h**.

### Command
```bash
DENSE_ONLY=1 EMBED_GPUS=4,5,6 RADV_DEBUG=nodcc GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1 \
  QDRANT_COLL_V2=cbic_v2_full \
  PYTHONPATH=/home/user/.local/lib/python3.10/site-packages:/opt/indian-legal-ai/rag/cbic_rag \
  nohup python3 ingest_v2.py --phase phase3_4_5 \
  --allow-upsert-drift 50 \
  > /tmp/p345_full.log 2>&1 &
```

### Parameters
- `QDRANT_COLL_V2=cbic_v2_full` (fresh collection — never overwrite existing)
- Vector name: `dense` (size 1024, cosine)
- `--allow-upsert-drift 50` — minor count drift tolerated; >50 triggers raise

### Precautions
- **Use a NEW collection** per project rule (`cbic_v2_full`) — do not touch `cbic_v1` or `cbic_v2_set5`.
- If GPUs 0/1 sneak into the pool, ABORT — they crash to 0.24 ch/s.
- Embedder warmup: first 3-5 minutes show 9 ch/s, ramps to ~14, then 22+ as KV warms. Patience.
- End-of-phase asserts `qdrant.points_count == chunks_submitted` (within `--allow-upsert-drift`).

---

## Phase 6 — Per-chunk pair generation (~150h with 10 parallel workers)

**Goal:** for every chunk in `cbic_v2_full`, generate ~12 qwen3 queries + 4 gemini queries (20% sample) + 2 claude adversarial queries (10% sample) + mine 5 hard negatives per query. Write append-only to `/opt/indian-legal-ai/data/training_corpus/cbic_pairs_v2.jsonl`.

### Hardware
- **GPU 2** — qwen3-14b for bulk generation
- **Gemini API** — 20% diversity slice ($100 budget cap, fall back to qwen3-only on hit)
- **Claude CLI (`claude -p`)** — 10% adversarial slice (FREE, Max plan, codified in DECISIONS.yaml)
- **GPUs 4,5,6** — BGE-M3 for hard-neg query embedding (concurrent with generation)

### Volume + time
- ~180,000 chunks
- Per-chunk: qwen3 emits 12 q (~25-30s on GPU 2 with `--parallel 4`), gemini 4 q (~1s API), claude 2 q (~11s CLI)
- Sequential single-worker: 180,000 × 30s = **1500h** (62 days — UNACCEPTABLE)
- **Parallelism strategy:** run **N=10 worker processes**, each scrolling a shard of chunks. qwen3 server `--parallel 4` handles ~4 concurrent. Gemini API + Claude CLI also serve in parallel. Effective throughput ~5×: **~300h ≈ 13 days.**

### Throughput options if 13 days too slow
- **(a) Add OpenRouter as bulk path** — user has `OPENROUTER_API_KEY`; route 50% of qwen3 calls through openrouter (kimi-k2 / qwen2.5 / etc.) at ~$0.10/1M tokens. Costs ~$50-100, cuts time to ~7 days.
- **(b) Spin up second qwen3 instance on GPU 3** if GPU 3 health-check passes — doubles bulk throughput, ~7 days.
- **(c) Reduce cardinality from 12 to 8 q/chunk** (still hits 1.4M pairs) — 67% time, ~9 days.

### Command (10 parallel workers via shards)
```bash
# Generate shard files (one chunk_id list per worker)
python3 -c "
from qdrant_client import QdrantClient
c = QdrantClient(host='127.0.0.1', port=6343)
ids = []
off = None
while True:
    pts, off = c.scroll('cbic_v2_full', limit=512, offset=off, with_payload=False, with_vectors=False)
    ids += [p.id for p in pts]
    if off is None: break
print(len(ids), 'total chunks')
import os, math
n = 10
sz = math.ceil(len(ids)/n)
for i in range(n):
    open(f'/tmp/p6_shard_{i}.txt','w').write('\n'.join(map(str, ids[i*sz:(i+1)*sz])))
"

# Launch workers
set -a; source /root/.cbic_env; set +a
for s in 0 1 2 3 4 5 6 7 8 9; do
  DENSE_ONLY=1 EMBED_GPUS=4,5,6 RADV_DEBUG=nodcc GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1 \
    PYTHONPATH=/home/user/.local/lib/python3.10/site-packages:/opt/indian-legal-ai/rag/cbic_rag \
    nohup python3 /opt/indian-legal-ai/reingest_spec/phase6_pairs.py \
    --scope full --collection cbic_v2_full --shard /tmp/p6_shard_${s}.txt \
    > /tmp/p6_w${s}.log 2>&1 &
done
```
*(Note: `--shard` arg needs to be added to phase6_pairs.py — currently it scrolls the full collection. One-line patch.)*

### Parameters (locked)
- 12 q/chunk from qwen3 (factual=4, scenario=3, definition=2, procedural=2, multi_hop=1)
- 4 q/chunk from Gemini Flash on hash(chunk_id)%5==0
- 2 q/chunk from Claude CLI on hash(chunk_id)%10==0
- Hard negatives: cosine band [0.60, 0.85], k=5, same-doc only
- Resumability: skips (chunk_id, generator) already in `cbic_pairs_v2.jsonl` for this scope
- Schema validation: `pair_schema_cbic_v2.md`; rejects → `cbic_pairs_v2_rejects.jsonl`

### Cost
- Gemini Flash: 36,000 chunks × 4 q × ~300 tokens/q = ~$30-50
- Claude CLI: 18,000 chunks × 2 q = FREE (Max plan)
- qwen3 + BGE-M3: rig power only

### Precautions
- **Scale up gradually** — start with 1 worker for 30 min, verify pairs/min rate matches projection, then add workers.
- **Watch Gemini budget** — phase6 logs cumulative cost; on $100 cap, switches to qwen3-only.
- **qwen3 server stability** — at `--parallel 4` × 10 workers = up to 40 in-flight. Set `--parallel 8` to give buffer; monitor for OOM. If qwen3 OOMs: restart, phase6 resumes from canonical file.
- **Hard-neg miner can hammer Qdrant.** Throttle to ≤20 concurrent search() calls. Configurable.

### Expected output
- **~2.15M pairs** in `cbic_pairs_v2.jsonl` (append-only canonical)
- Per-scope snapshot `cbic_pairs_v2_full_<yyyymmdd>.jsonl`
- Summary file `cbic_pairs_v2_full_<yyyymmdd>.summary.json` with **per-doc + per-generator counts** (user directive 2026-04-25)
- `cbic_pairs_v2_rejects.jsonl` should be <1% of attempts

---

## Phase 7 — Build G1 union gold (~5 min)

**Goal:** sample 5–8 queries per doc per generator from phase6 output ∪ legacy pool → eval gold for G1.

```bash
python3 evaluators/build_g1_gold_union.py \
  --scope full --pair-file /opt/indian-legal-ai/data/training_corpus/cbic_pairs_v2.jsonl \
  --legacy-dir /opt/indian-legal-ai/eval/training_pairs \
  --sample-per-doc 6 \
  --out eval/scale_sets/full/v2_gold_full.json
```

Expected size: ~6 queries/doc × 14,925 docs ≈ 90K eval queries (sampled, not full corpus).

---

## Phase 8 — G1 recall@10 (~30 min)

**Goal:** verify retrieval recall ≥ 0.95 (adjusted, excl. shared-PDF Defect D).

```bash
python3 evaluators/gate_g1_recall.py \
  --collection cbic_v2_full \
  --gold eval/scale_sets/full/v2_gold_full.json \
  --retrieve-only \
  --out evaluators/gate_g1_full.json
```

### Hardware
- GPUs 4,5,6 for query embedding
- Qdrant for retrieval

### Pass criteria
- **Adjusted recall@10 ≥ 0.95** (exclude form prefix Defect D queries from denominator)
- Per-prefix recall ≥ 0.94 for all non-form prefixes
- Errors = 0

### If FAIL
- **HALT — fix spec, do not patch-and-continue** (hard rule).
- Inspect `gate_g1_full.misses.json` — look for new prefix patterns or chunker regressions.

---

## Phase 9 — G2 dual-judge faithfulness (~10–15h)

**Goal:** sample ~2,000 queries, generate qwen3 answer, judge faithfulness with BOTH Gemini Flash + Claude CLI.

```bash
set -a; source /root/.cbic_env; set +a
python3 evaluators/gate_g2_dual_judge.py \
  --collection cbic_v2_full \
  --gold eval/scale_sets/full/v2_gold_full.json \
  --sample 2000 \
  --judges gemini,claude \
  --out evaluators/gate_g2_full.json
```

### Pass criteria
- Faithful (per judge) ≥ 0.95
- Inter-judge agreement ≥ 0.85
- Hallucination rate < 0.02

### Cost
- Gemini Flash judge: ~$15
- Claude judge: FREE (CLI Max plan)

---

## Phase 10 — G3 answer quality (~3h)

**Goal:** answer quality on a 1,000-query sample, scored against reference (chunk text + judge).

```bash
python3 evaluators/gate_g3_answer_quality.py \
  --collection cbic_v2_full --sample 1000 \
  --out evaluators/gate_g3_full.json
```

### Pass criteria: composite ≥ 0.95.

---

## Phase 11 — G4 adversarial / refusal (~2h)

**Goal:** verify correct refusal/flagging on adversarial queries (the 10% Claude-CLI slice from phase6 has these).

```bash
python3 evaluators/gate_g4_adversarial.py \
  --collection cbic_v2_full \
  --gold eval/scale_sets/full/v2_gold_full_adversarial.json \
  --out evaluators/gate_g4_full.json
```

### Pass criteria
- Correct refusal on `query_type=refusal_bait` ≥ 0.90
- No-hallucination on `query_type=out_of_scope` ≥ 0.90

---

## Phase 12 — G5 latency + cost (~1h)

```bash
python3 evaluators/gate_g5_latency_cost.py \
  --collection cbic_v2_full --sample 200 \
  --out evaluators/gate_g5_full.json
```

### Pass criteria
- p95 latency ≤ 8s end-to-end
- cost ≤ $0.01/query

---

## Phase 13 — Cutover + codification (~30 min)

1. **Snapshot Qdrant collection:**
   ```bash
   curl -X POST http://127.0.0.1:6343/collections/cbic_v2_full/snapshots
   ```

2. **Update API service to point at `cbic_v2_full`:**
   Edit `/etc/systemd/system/cbic-rag-api.service.d/full_collection.conf`:
   ```
   [Service]
   Environment=QDRANT_COLL=cbic_v2_full
   ```
   ```bash
   systemctl daemon-reload && systemctl restart cbic-rag-api
   ```
   Verify with a /retrieve call. Keep old `cbic_v2_set5` drop-in for revert.

3. **Codify results:**
   - Append G1-G5 results block to `reingest_spec/JOURNAL.md`
   - Update `MEMORY.md` top entry: "FULL CORPUS PASSED 2026-XX-XX"
   - Update `project_cbic_reingest_v2.md` with final per-prefix recall + per-gate scores

4. **Backup pair corpus:**
   ```bash
   cp /opt/indian-legal-ai/data/training_corpus/cbic_pairs_v2.jsonl \
      /opt/indian-legal-ai/data/training_corpus/backups/cbic_pairs_v2.full.<yyyymmdd>.jsonl
   ```

---

## Failure matrix (what to do when X breaks)

| symptom | cause | action |
|---|---|---|
| Phase 2 raises >5% | qwen3 trap not in fallback table | Add prefix to `_DEFAULT_PLANS_BY_PREFIX`, restart |
| Phase 2 silent fallback to pdfminer | PyMuPDF missing | `pip install pymupdf` then re-run |
| Phase 3-4-5 throughput < 5 ch/s | GPU 0/1 in pool | check `EMBED_GPUS`, kill, restart |
| qwen3 service in "Loading model" | service restarted | wait 60s, retry |
| Gemini Flash 429 rate-limit | budget burning | phase6 auto-falls-back to qwen3-only |
| Claude CLI hangs | rate limit on Max plan | phase6 timeout=60s, marks generator skip |
| Qdrant points_count drift > 50 | upsert race | re-run phase3_4_5 with `--resume`, manifest tracks |
| G1 raw < 0.85 | new doc-type breakage | check misses, NOT patch — back to chunker |
| G1 adjusted < 0.95 | something WORSE than Defect D | HALT, escalate, do not proceed |
| G2 faithfulness < 0.90 | retrieval+answer mismatch | inspect rerank, reranker config |

---

## Standing rules (NEVER violated)

1. **95% non-negotiable on every gate.** No "acceptable" compromises.
2. **Never delete during project.** Append to `cleanup_backlog.md`.
3. **CPU off-limit for embed/rerank/OCR.**
4. **New collection per scope.** Never overwrite `cbic_v1` or other scope collections.
5. **Inventory before planning.** Every new task starts with `bash D:/_gpu_rig_ai/inventory.sh`.
6. **Read DECISIONS.yaml first.** PreToolUse hook enforces this on protected paths.
7. **No silent assumption-inheritance.** Verify every reused component (embedder, ingest, chunker) is covered by playbook.

---

## End-state (when this runbook completes)

- ✅ Qdrant `cbic_v2_full` with ~180K chunks, recall@10 ≥ 0.95
- ✅ `cbic_pairs_v2.jsonl` with ~2.15M training pairs (append-only canonical)
- ✅ `cbic_hardneg_v2.jsonl` with mined hard negatives
- ✅ G1-G5 gates all green
- ✅ API service pointing at new collection
- ✅ JOURNAL + MEMORY codified
- ✅ Pair corpus ready for downstream BGE-M3 contrastive, bge-reranker fine-tune, qwen3-14b LoRA

---

## What this runbook does NOT cover

- Hindi corpus (separate cohort, post-cutover)
- Reranker fine-tune (consumes `cbic_pairs_v2.jsonl` after phase6 completes)
- LoRA training (downstream, separate runbook)
- OCR pipeline (851 docs already processed in separate flow; not part of this scope)
- Active learning loop (post-deployment)
