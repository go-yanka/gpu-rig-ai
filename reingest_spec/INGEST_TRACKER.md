# Full Re-Ingest Tracker — cbic_v2 (14,925 docs in 10 batches)

Live source-of-truth for the full re-ingest. Updated after each batch + each
checkpoint. Read this first when asked "where are we?".

---

## Stage tracker (where we are right now)

| Field | Value |
|---|---|
| **Current stage** | Stage M (full re-ingest execution) |
| **Active batch** | none — batch 1 (try 2) DONE, awaiting CP-1 |
| **Last completed batch** | Batch 1 (try 2) ✅ 2026-04-26 17:41 IST — 1500 docs, 4847 chunks, pts=7823, reconcile PASS |
| **Next checkpoint** | **CP-1 (run NOW before batch 2)** |
| **Hard rule** | 95% on all gates non-negotiable. Any gate < threshold = HALT. |

---

## Batch table (per-batch gate values fill in as we go)

Columns: `Pts` = qdrant points_count after batch; `G1/G3/G4/G2/G5` = gate scores measured at the checkpoint that follows that batch (blank = no checkpoint there); thresholds = G1≥0.95, G3≥0.95, G4≥0.95, G2≥0.95, G5 p95 ≤ 12-15s (per spec amendment).

| Batch | Status | Docs | Phase 2 (s) | Phase 3-5 (s) | Total (min) | Pts after | G1 | G3 | G4 | G2 | G5 p95 | CP | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 (try 1) | **FAILED** | 1500 (1484 eff) | 116.8 | died ~131s @ 2880/4242 ch | ~4.1 before crash | 2976 (partial) | – | – | – | – | – | – | Qdrant 400 NaN/Inf in vector. See post-mortem. |
| 1 (try 2) | ✅ DONE | 1500 | 142.9 | 221.7 | 6.1 | 7823 | – | – | – | – | – | **CP-1 pending** | 4847 chunks, reconcile PASS, 21.8 ch/s steady, 0 NaN trips, 0 Qdrant 400s |
| 2 | pending | ~1500 | – | – | – | – | – | – | – | – | – | – | – |
| 3 | pending | ~1500 | – | – | – | – | – | – | – | – | – | – | – |
| 4 | pending | ~1500 | – | – | – | – | – | – | – | – | – | – | – |
| 5 | pending | ~1500 | – | – | – | – | – | – | – | – | – | **CP-2** | Mid-corpus drift check |
| 6 | pending | ~1500 | – | – | – | – | – | – | – | – | – | – | – |
| 7 | pending | ~1500 | – | – | – | – | – | – | – | – | – | – | – |
| 8 | pending | ~1500 | – | – | – | – | – | – | – | – | – | – | – |
| 9 | pending | ~1500 | – | – | – | – | – | – | – | – | – | – | – |
| 10 | pending | ~1425 | – | – | – | – | – | – | – | – | – | **CP-3** | Final acceptance, full gate panel |

---

## Checkpoint policy + STOP CONDITIONS (hard, no patch-and-continue)

### CP-1 — Pipeline smoke (after batch 1)
- **Runs:** G3 + G4 grounded on Set 6 gold queries that overlap batch-1 doc_ids; 5 hand-picked /query smokes across notification/circular/act/rule/form/instruction.
- **Pass:** G3 ≥ 0.90 AND G4 ≥ 0.95 AND all 5 /query smokes return correct top-1 doc_id.
- **🛑 STOP if:** G3 < 0.90 OR G4 < 0.95 → halt batches 2-10, root-cause, fix, restart from batch 1 if chunker/reranker change.
- **Time budget:** ~10 min.

### CP-2 — Mid-corpus drift (after batch 5)
- **Runs:** G1 + G3 + G4 on union(Set 5, Set 6) gold against partial cbic_v2; reranker histogram check (gold p50 ≥ 0.7, adv max < 0.65 — codified separation).
- **Pass:** G1 ≥ 0.95 AND G3 ≥ 0.92 AND G4 ≥ 0.95 AND histogram bands cleanly separated.
- **🛑 STOP if:** any band fails → halt before wasting batches 6-10. Likely root causes ranked: chunker density bug, reranker context bleed, theta drift.
- **Time budget:** ~15 min.

### CP-3 — Final acceptance (after batch 10)
- **Runs:** G1 + **G2-full dual-judge n=380** + G3 + G4 + G5 latency.
- **Pass:** ALL of {G1, G2, G3, G4} ≥ 0.95 AND G5 p95 within current spec (12-15s post-amendment).
- **🛑 STOP if:** ANY gate < 0.95 → halt. **No patch-and-continue.** Per Hard Rule #1: fix spec, re-run from the failed phase. Promotion to prod is BLOCKED until all green.
- **Time budget:** ~45 min + G2 cost ~$4.

---

## Steps + observed rates

| Step | Rate (Set 6) | Rate (Batch 1 partial) | Bottleneck |
|---|---|---|---|
| Phase 2 chunk+dedupe | 2.55 doc/s | **5.37 / 12.7 doc/s** (warm) | None — 12-prefix bypass eliminates qwen3 |
| Phase 3 OCR cache | instant on cache hit | instant | None — files already OCR'd in scraper |
| Phase 4 embed (BGE-M3) | ~20 ch/s on 3-GPU pool | 22 ch/s | 3-GPU pool {4,5,6}; GPU 4 saturates first (fan-out skew, codified deferred) |
| Phase 5 upsert (Qdrant) | sub-ms per chunk | TBD post-fix | None |
| Phase 6 pair-gen | – | NOT IN ingest path for full re-ingest | qwen3 single-slot, deferred to post-corpus |

## Optimization findings (live)

A read-only ingest-optimization-hunter agent ran 2026-04-26 17:43 EDT during the gap between batch 1 retry (completed) and batch 2 launch. Findings:

- **Top rec (H, ~13 min savings):** add GPUs 1+3 to embed pool (3-card → 5-card) — see `INGEST_OPTIMIZATIONS.md` rec #1. Requires per-GPU sequential bench first per codified rule.
- **Combo rec (M, +3 min):** bump `EMBED_BATCH` 48 → 96 alongside the 5-card pool.
- **Pipeline rec (M, ~17 min):** run phase 2 of batch N+1 concurrent with phase 3-5 of batch N (manifest WAL needed).
- **Correctness fix (L):** `build_batch.py:get_already_ingested` flags partial docs as done. Fix in rec #9.
- **Stale-lesson finding:** the "GPU 4 99% / 5,6 idle" fan-out skew codified in MEMORY.md appears to no longer apply — current `_Pool.embed()` does explicit weighted shard split. Empirical telemetry (live CSV) will confirm.

Telemetry CSV: `INGEST_TELEMETRY.csv` (rig: `/tmp/ingest_telemetry.csv`, sampler PID 863202).
Full report + ranked recs + code diffs + hard-stops: `INGEST_OPTIMIZATIONS.md`.

---

## Bottlenecks identified (live punch list)

### CONFIRMED (in scope of ingest pipeline)
1. **Embed pool fan-out skew (codified, deferred)** — under load GPU 4 saturates 99%, GPUs 5/6 idle. RR scheduler is equal-weight; needs weighted-deficit. Symptom: ingest single-GPU bound, not 3-GPU.
2. **Phase 3-5 single-process embed** — embedder_direct.py uses ProcessPoolExecutor across GPUs but fan-out is the bottleneck.
3. **NaN/Inf in vectors → Qdrant 400 (FIXED 2026-04-26)** — `upsert_chunks` had no sanitiser; one bad BGE-M3 dense or sparse value killed entire flush. Patch deployed: clamp non-finite → 0.0 + per-point logging + halve-and-retry on Qdrant 400 in `_flush_batch`.

### CONFIRMED (out of ingest scope, separate fix path)
4. **G3 retrieval ceiling 0.921** — 3-pt gap to 0.95 is hybrid-BM25 + gold-quality work, NOT chunker. Doesn't block ingest.
5. **Reranker -c 8192 was wrong** — fixed to `-c 32768 --parallel 8` (4096 tokens/slot). Eval-time only, not ingest.

### POTENTIAL (watch for)
6. **Manifest sqlite write contention** — multiple batches share manifest. Should be fine since batches are serial.
7. **Qdrant points_count drift** — phase3_4_5 reconcile asserts but may miss edge cases at 270k points.
8. **VRAM creep on embed pool** — long-running BGE-M3 instances may leak; watch GPU 4 mem usage across batches.
9. **OCR cache misses on rare doc types** — `cbic-attachment-dtls` (19 docs total) may not have OCR'd content.

## Decisions to fine-tune speed

### Already applied
- `bypass_prefixes` 12-prefix → 231× phase-2 speedup
- `_DEFAULT_PLANS_BY_PREFIX` deterministic templates → no qwen3 calls in phase 2
- Reranker `-c 32768` (post-ingest fix; doesn't affect ingest itself)
- **NaN/Inf sanitiser in upsert_chunks** (2026-04-26)
- **Halve-and-retry on Qdrant 400** in `_flush_batch` (2026-04-26)

### To try if rate slips
- Bump `EMBED_BATCH` from 48 to 96 (test on batch 2)
- Add GPU 3 to embed pool (was reset 2026-04-25 but never benched under load — bench first per codified rule)
- Parallelize batches (run batch N+1's phase 2 while batch N's phase 3-5 runs) — would need careful manifest locking

### Hard exceptions (do NOT change)
- qwen3-14b on GPU 2 stays `--parallel 1` (codified)
- Embed pool cold-load stays sequential (codified — concurrent load = 16× slowdown)
- Per-card PCIe reset is BANNED (codified — caused regression 2026-04-25)

---

## Batch 1 (attempt #1) post-mortem — 2026-04-26 17:22 IST

**Outcome: FAILED.** ingest_v2.py crashed in phase3_4_5 → `_flush_batch` → `upsert_chunks` with Qdrant `400 Bad Request: Format error in JSON body: expected ',' or ']' at line 1 column 219103`. Last clean checkpoint `done=2880/4242 rate=22.0 ch/s`; Qdrant `points_count=2976`.

### Phase timings (partial)

| Phase | Wall clock | Rate | Result |
|---|---|---|---|
| Phase 2 (chunk+dedupe) | **116.8s** | 12.7 doc/s | DONE — 1484 docs → 6967 raw → 4242 canonical, 39.1% dedupe savings, 0 failed, 4 OCR-deferred |
| Phase 3-5 (embed+upsert) | ~131s before crash | **22.0 ch/s** | CRASHED — Qdrant 400 mid-flush |

### Root cause (high confidence, codified)

`upsert_chunks` built JSON body with no NaN/Inf guard. Python's JSON encoder emits bare `NaN`/`Infinity` tokens which Qdrant's strict parser rejects with exactly the observed `expected ',' or ']'` error. Most likely: BM25 sparse fastembed returning NaN on a normalize-to-zero chunk, or a single bad BGE-M3 dense vector.

### Patches deployed (2026-04-26, ready for retry)

1. **`rag/cbic_rag/ingest.py:upsert_chunks`** — `_finite()` helper + per-point loop clamps non-finite floats (dense + sparse) to 0.0, logs first 10 occurrences with pid/field/idx/raw value + summary count.
2. **`reingest_spec/ingest_v2.py:_flush_batch`** — recursive halve-and-retry on `UnexpectedResponse(status=400)`. Isolates the single bad point, logs `chunk_id` + `doc_id`, marks rest as upserted, continues.

### Errors / warnings observed

- 1× `qdrant_client.http.exceptions.UnexpectedResponse: 400` (the fatal one).
- ~50× `init: embeddings required but some input tokens were not marked as outputs -> overriding` — benign llama.cpp embed-init notice from BGE-M3 worker spawn, not an error.
- 0× `[embed err]`, 0× `[rerank err]`, 0× `nan`/`inf` (the NaN was inside a vector array, not a string the regex would catch).

### Final state at attempt-1 report time

- Qdrant `cbic_v2.points_count = 2976` (partial batch 1, status=green).
- ingest_v2.py: not running.
- Batches 2-10: BLOCKED until retry succeeds + CP-1 passes.

### Re-launch plan (post-patch)

- Re-run is idempotent on Qdrant side — pid hash `(doc_id, page, char_start)` overwrites the partial 2976 points cleanly. No reset needed.
- Expected total wall-clock: phase 2 ~117s + phase 3-5 ~190s = **~5 min**.
- Then CP-1 (~10 min).
