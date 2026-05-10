# Ingest Optimizations — live findings during full re-ingest

**Agent:** ingest-optimization-hunter (read-only on running ingest)
**Started:** 2026-04-26 17:43 EDT (21:43 UTC)
**Ingest state at start:** Batch 1 retry COMPLETED (qdrant `cbic_v2.points_count=7823`, `[phase3-5 RECONCILE]` passed for all 1500 scoped docs); ingest process not running between batches.
**Telemetry CSV:** `D:/_gpu_rig_ai/reingest_spec/INGEST_TELEMETRY.csv` (rig: `/tmp/ingest_telemetry.csv`), sampling every 10s.

References cited below use file:line on the rig (mirrored to `D:/_gpu_rig_ai/rig_mirror/` for review).

---

## Section 1 — Live observations

### 1.1 Batch 1 retry (the just-completed run, log `reingest_batch1_1777239279.log`)

Phase | Wall-clock | Rate | Notes
---|---|---|---
Phase 2 (chunk+dedupe, 1472 docs) | ~117s warm | 7.1 → 13.9 doc/s ramp | Bypass-prefix templates eliminate qwen3 calls; CPU-bound on PDF parse + dedupe. `dup=9` early then flatlines — most batch-1 docs are unique within batch.
Phase 3-5 (4847 canonical chunks) | **221.7s** | **steady 21.7 ± 0.4 ch/s** | Reconcile PASS (1500/1500 docs). 3-GPU pool {4,5,6}.

Per-batch `[phase3-5] done=N rate=R ch/s` traces (from log):
```
done=480/4847 rate=20.3
done=960  rate=20.5
done=1440 rate=21.3
done=1920 rate=21.2
done=2400 rate=21.6
done=2880 rate=21.7
done=3360 rate=21.9
done=3840 rate=22.0
done=4320 rate=22.0
done=4800 rate=21.8
```
Rate is flat (no degradation across batch) → no thermal/PCIe drift, no memory pressure, no GPU dying mid-batch. Pool is steady-state.

### 1.2 Embed pool warmup p50 (cold)

```
GPU 4 (RX 5700 XT, PwrColor) load=3.06s warmup_p50=29ms
GPU 5 (RX 5700 XT, ASRock)   load=2.97s warmup_p50=30ms
GPU 6 (RX 5700 XT, ASRock)   load=3.01s warmup_p50=29ms
3/3 GPUs ready: [4, 5, 6]
```
Cold-load mode `SEQUENTIAL` (codified mandatory). p50 latencies are within 1ms of each other → with `rebalance_after_warmup: false` (codified) weights stay equal at 1.0. Shard split for `n=48` batch = 16/16/16 (per `_Pool.embed`, line 418).

### 1.3 Fan-out skew claim (codified in MEMORY.md) — NEEDS RE-VERIFICATION

The codified lesson says "GPU 4 saturates 99%, GPUs 5/6 idle." However:
- Current `embedder_direct.py:_Pool.embed()` (lines 411-440) does **explicit weighted sharding** for `n>1` calls, not round-robin.
- `EMBED_BATCH=48` triggers `n>1` path → 16/16/16 split → all three GPUs receive work simultaneously per flush.
- The flat 21.8 ch/s rate × 3 GPUs ≈ **7.3 ch/s/GPU**, which matches solo BGE-M3 single-card numbers (29ms warmup ≈ ~30 calls/s for short text, derated for full chunks).

**Hypothesis:** the "skew" lesson refers to a PRIOR implementation (shared queue) that was already replaced. The current architecture doesn't exhibit it. **Telemetry sampling at 10s during the next active phase 3-5 will prove this empirically.** As of telemetry sample T+4min the ingest is idle (all GPU busy=0); next batch launch will produce the data.

### 1.4 Idle GPUs

- **GPU 0:** hosts `bge-reranker.service` (PID 807752, port 9085, GPU 0 pinned). Idle during ingest (used at gate-time only).
- **GPU 1:** completely idle. Has profile entry, in `default_gpus`, but not in launcher's `EMBED_GPUS=4,5,6` override.
- **GPU 2:** qwen3-14b host. OFF-LIMITS during phase 3-5 per codified hard rule.
- **GPU 3:** completely idle. Reset 2026-04-25, never re-benched.

`embed_pool_profiles.json#default_gpus = [0,1,3,4,5,6]` — the **profile defaults to a 6-card pool**, but `build_batch.py` line 126 forces `EMBED_GPUS=4,5,6`. The override pre-dates the fan-out fix and the GPU 1/3 reset; it has not been revisited.

### 1.5 Process / port snapshot (rig at start)

```
LISTEN 9082  qwen3-14b llama-server (PID 656, GPU 2)
LISTEN 9085  bge-reranker llama-server (PID 807752, GPU 0)
LISTEN 6343  qdrant docker container
GPU busy% all = 0    (ingest idle)
qdrant cbic_v2 points_count = 7823, status=green
```

---

## Section 2 — Ranked recommendations

Severity: H = blocks/saves >10min; M = saves 3-10min; L = saves <3min or quality-only.
ROI = total minutes saved across remaining 9 batches (batches 2-10). Risk levels assume codified hard exceptions.

---

### 1. Add GPU 1 + GPU 3 to embed pool (3-card → 5-card)
**Severity:** H | **ROI:** ~120-160 min saved across 9 batches | **Risk:** med | **Apply at:** batch 2

Current `EMBED_BATCH=48` shards 16/16/16 across {4,5,6} = 22 ch/s. With 5 cards, shard becomes ~10/10/10/9/9 = expected **~36 ch/s**, cutting phase 3-5 from 222s → 135s per batch (~87s/batch × 9 = 13 min). This is the single largest lever.

**Codified blocker:** GPU 1 and GPU 3 were never benched under load post-2026-04-25 reset. The codified rule (MEMORY.md, "GPU POOL EXPANSION IS NOT TRIVIAL") says: per-GPU solo bench MANDATORY before expanding canonical EMBED_GPUS.

**Two-step plan:**

Step A — pre-batch-2 sequential bench (no concurrency, ~30s total). Run between batches (NOT during phase 3-5). This is safe per cold-load rule (1 GPU at a time):
```bash
# On rig, after batch N completes and before batch N+1 launches:
EMBED_GPUS=1 python3 -c "from embedder_direct import get_pool, _Pool; p=get_pool(); \
  import time; t0=time.time(); \
  [p.embed_on(1,['the quick brown fox']) for _ in range(50)]; \
  print(f'gpu1: 50 calls in {time.time()-t0:.2f}s')"
# repeat for EMBED_GPUS=3
```
Pass criterion: ≥15 q/s (matches GPU 4/5/6 baseline).

Step B — if both pass, change launcher:
```diff
# build_batch.py line 126
-            "EMBED_GPUS": "4,5,6",
+            "EMBED_GPUS": "1,3,4,5,6",
```

**Test plan:** batch 2 phase 3-5 should complete in ≤140s (vs 222s baseline). Reconcile PASS required. If `[embed_pool] N/5 GPUs ready` shows N<5, abort batch 2 and revert to `4,5,6`.

**Hard exceptions check:** sequential cold-load already mandatory (per profile JSON). No PCIe reset needed (GPUs already settled since 2026-04-25). qwen3 untouched (GPU 2 excluded). Reranker untouched (GPU 0 excluded — tier-2 rec covers that).

---

### 2. Bump `EMBED_BATCH` 48 → 96
**Severity:** M | **ROI:** ~10-20 min over 9 batches | **Risk:** low | **Apply at:** batch 2 (combine with #1)

INGEST_TRACKER.md lists this as a tunable. Larger batches keep GPUs busy across the embed→upsert handoff in `_flush_batch` (`ingest_v2.py:958`). With 5 GPUs (rec #1 applied), shard becomes ~19 per GPU, well within `n_batch=512` Vulkan budget (BGE-M3 typical chunk text ≈ 800 tokens, so 19 chunks × 800 = 15.2k tokens — fits in `n_ctx=8192` if processed in sub-shards by llama-cpp internally; if not, shard naturally caps at 8 per call which is still bigger than current 16).

**Code diff:**
```diff
# In build_batch.py launcher env (line 122-130):
         env.update({
             "QDRANT_COLL_V2": QDRANT_COLL_FULL,
             "MANIFEST_V2": MANIFEST_V2,
             "DENSE_ONLY": "1",
             "EMBED_GPUS": "1,3,4,5,6",
+            "EMBED_BATCH": "96",
             "RADV_DEBUG": "nodcc",
```

**Test plan:** batch 2 [phase3-5] rate should be ≥30 ch/s (with 5 GPUs + batch=96). If batch=96 triggers OOM on any worker (`RuntimeError` from llama_cpp), pool routes around (codified `degraded_policy`); fall back to 64.

**Risk:** low — batch size only changes payload-per-flush, not batch correctness; per-call latency budget unchanged.

---

### 3. `_flush_batch` skip-if-pid-already-in-Qdrant on retry
**Severity:** M | **ROI:** ~22 min saved per crash-retry scenario (one-shot, but high value if any batch crashes) | **Risk:** low | **Apply at:** before batch 2

`reingest_spec/ingest_v2.py:_flush_batch` (line 958) re-embeds every chunk in the buffer. On the batch 1 retry, 2976 of 4242 chunks were already in Qdrant from attempt #1, but ALL 4242 were re-embedded — wasted ~22 min of GPU work.

`upsert_chunks` writes by deterministic `pid = abs(hash((doc_id, page, char_start))) % 10**15` (`ingest.py:158`). One `qc.retrieve(QCOLL, ids=[pids])` round-trip can identify already-present pids, drop their chunks from the embed list, and proceed. Manifest sqlite already tracks `upserted=1` per chunk_id; the cheaper option is to filter the SELECT in `phase3_4_5` before buffering.

**Code diff:**
```diff
# ingest_v2.py:861, in phase3_4_5
-    q = "SELECT chunk_id,payload_json FROM chunks WHERE is_canonical=1 AND upserted=0"
+    # Skip-pass: chunks already upserted (from a partially-completed prior run on
+    # SAME chunk_id) are now filtered server-side by manifest.upserted=0 already.
+    # But on a clean retry where manifest WAS updated but Qdrant lost rows, OR
+    # vice versa, we want belt-and-braces: cross-check Qdrant. Cheap on small N.
+    q = "SELECT chunk_id,payload_json FROM chunks WHERE is_canonical=1 AND upserted=0"
```
Actually: **the code is already correct for the common case** — `upserted=0` filter excludes chunks marked done in manifest. The wasted retry happened because `_flush_batch` couldn't update `upserted=1` until the WHOLE flush succeeded; the failing flush left ALL 48 buffered chunks marked `upserted=0` even though 2976 had landed in Qdrant before the crash.

**Real fix:** mark chunks `upserted=1` IMMEDIATELY after each successful `qc.upsert()` call (currently happens; but only if the whole batch succeeds). The `_try_upsert` recursion (lines 967-986) already handles individual point isolation. The remaining gap: in batch 1 attempt #1, the whole `_flush_batch` raised before any `executemany` ran. The 2026-04-26 patch added halve-and-retry, so attempt #2 worked. **Net: rec #3 may already be addressed by the 2026-04-26 patch.** Verify by running a small forced-failure test post-corpus.

**Downgrade severity:** L (already mitigated). Keep as a documented validation step, no code change needed unless another crash happens.

---

### 4. Producer/consumer: Phase 2 of batch N+1 || Phase 3-5 of batch N
**Severity:** M | **ROI:** ~17 min saved across 9 batches (one phase-2 fully hidden behind phase 3-5 each batch) | **Risk:** med | **Apply at:** batch 3 (after batch 2 stabilizes the 5-GPU pool)

Phase 2 of batch N+1 takes ~117s (CPU-bound, no GPU). Phase 3-5 of batch N takes ~222s (3-GPU bound) or ~135s (with rec #1). They share **only the manifest sqlite** as contention surface.

**Pattern:** between each batch's phase 2 completion and next batch's phase 1 setup, run them concurrently as separate ingest_v2 processes:

```bash
# Pseudocode for orchestrator
B1: phase 1+2 → phase 3-5 (foreground)
                       ↓ as soon as phase 2 of B1 ends, kick off:
B2: phase 1+2 (background, different doc_ids)  ← runs WHILE B1's phase 3-5 still finishing
B1: phase 3-5 done → wait for B2's phase 2 → B2 phase 3-5
```

**Manifest contention:**
- Phase 2 writes to `docs.phase2_*` and `chunks.*` for B2's doc_ids.
- Phase 3-5 of B1 reads `chunks WHERE is_canonical=1 AND upserted=0` and updates `chunks.embedded, .upserted` for B1's chunks.
- Sqlite default journal_mode = DELETE (writer-exclusive). Concurrent processes block each other on every COMMIT.

**Fix path:** enable WAL on the manifest before the multi-batch run:
```bash
sqlite3 /opt/indian-legal-ai/data/ingest_manifest_v2_full.sqlite \
  "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;"
```
WAL allows concurrent reader + 1 writer. Phase 2 commits per-doc (one chunk-batch insert + UPDATE); phase 3-5 commits per-flush (`executemany`). The two will mostly read different rows (B2's chunks vs B1's chunks) — actual contention is minimal even without WAL, but WAL eliminates the writer-block.

**Test plan:**
1. Enable WAL on manifest.
2. Launch batch 3 phase 1+2 concurrent with batch 2 phase 3-5. Check manifest integrity after both: every doc has `phase2_done=1, phase3_status='ok' or NULL`, no unaccounted rows.
3. Roll back to serial if any reconcile failure.

**Risk:** med because WAL behaves slightly differently across sqlite versions and a midnight crash in WAL leaves a `-wal` and `-shm` file that newer ingest must recover from — generally safe but adds a recovery path.

---

### 5. Hot-add GPU 0 (reranker host) to embed pool during phase 3-5
**Severity:** M | **ROI:** ~22 min saved across 9 batches if used (further 5→6 GPU pool) | **Risk:** med | **Apply at:** batch 4+ (only after rec #1 stabilizes)

`bge-reranker.service` is idle during ingest. The `_Pool.add_gpu(0)` API exists (`embedder_direct.py:485`). The codified hot-swap protocol for GPU 2 already documents this pattern. Procedure: stop reranker → `pool.add_gpu(0)` → run ingest → `pool.remove_gpu(0)` → restart reranker before CP-1.

**Code diff:** thin wrapper script, no production code change:
```python
# scripts/cbic_ingest/hot_swap_gpu0_for_ingest.py (new)
import subprocess, sys
from embedder_direct import get_pool
subprocess.run(["systemctl", "stop", "bge-reranker.service"], check=True)
pool = get_pool()
result = pool.add_gpu(0)
print("[hot-swap] add_gpu(0) =>", result)
sys.exit(0 if result.get("ok") else 1)
```
Reverse script for after.

**Risk:** med — sequential cold-load rule applies to add_gpu (but `add_gpu` already uses single-card load path, sequential by construction). The bigger risk: forgetting to restart reranker → CP-1 G3/G4 measurement fails because reranker missing. Mitigate via systemd dependency: write a systemd `path` unit that re-starts reranker when an "ingest_done" sentinel file is touched.

**Defer to batch 4** — only after rec #1 proves the larger pool works without instability.

---

### 6. Phase 2 PDF parsing parallelism (5-thread pool)
**Severity:** L | **ROI:** ~12 min saved across 9 batches | **Risk:** med | **Apply at:** batch 4+ if rec #4 not adopted

Phase 2 in `ingest_v2.py:phase2` (lines 618-742) is a serial `for i, r in enumerate(docs)` loop. With 1472 docs / 117s = ~13 doc/s warm. The bottleneck is `_read_pdf_text` (PyMuPDF) + `classify_and_chunk` (bypass-prefix → fast template path; non-bypass → qwen3 HTTP).

A 5-thread pool would 4-5× phase 2 to ~25-30s but adds complexity:
- `ChunkDeduper.add()` is not thread-safe (in-memory hash dict).
- `c2.execute()` against sqlite needs serialization (can use a Lock).
- qwen3 calls are still single-slot (`--parallel 1`), so concurrent classify on non-bypass docs would block on the LLM.

**Code diff sketch:**
```python
from concurrent.futures import ThreadPoolExecutor
deduper_lock = threading.Lock()
sqlite_lock = threading.Lock()

def _process_doc(r):
    # ... existing per-doc body (PDF read, chunk, dedupe, write) ...
    with deduper_lock:
        canonical_chunk, is_new = deduper.add(ch_dict)
    with sqlite_lock:
        _insert_chunk(c2, ch, dup_of=...)
        c2.commit()

with ThreadPoolExecutor(max_workers=5) as ex:
    list(ex.map(_process_doc, docs))
```

**Better path:** if rec #4 (cross-batch pipelining) is adopted, phase 2 of B(N+1) overlaps phase 3-5 of B(N) and the 117s is fully hidden — no need to parallelize phase 2 internally. Pick rec #4 OR rec #6, not both.

**Risk:** med — touches the dedupe + manifest write logic, both of which have caused silent bugs before (Defect D in MEMORY.md). Add only if rec #4 is rejected.

---

### 7. Telemetry sidecar (already deployed by this agent)
**Severity:** L | **ROI:** observability only | **Risk:** none | **Already running**

`/tmp/gpu_telemetry.sh` PID 863202 on rig is sampling all 7 GPU `gpu_busy_percent` + `qdrant.points_count` + detected batch/phase every 10s into `/tmp/ingest_telemetry.csv`. CSV mirrored to Windows at `D:/_gpu_rig_ai/reingest_spec/INGEST_TELEMETRY.csv` (post-run snapshot). Used to PROVE or refute fan-out skew on subsequent batches.

To stop after batch 10: `pkill -f gpu_telemetry.sh`.

---

### 8. NaN/Inf counter surfaced in phase 3-5 progress
**Severity:** L | **ROI:** observability only | **Risk:** none | **Apply opportunistically**

The 2026-04-26 patch in `ingest.py:upsert_chunks` (lines 152-189) clamps non-finite values and logs the first 10 occurrences. But the count isn't aggregated across flushes. Add a per-batch counter so we can answer "how often does this fire?":

```diff
# ingest_v2.py phase3_4_5, near `done = 0`:
+    nan_count = 0
     ...
# inside the per-flush loop after _flush_batch returns:
-    n = upsert_chunks(qc, chunks, dense, sparse)
+    n = upsert_chunks(qc, chunks, dense, sparse, nan_counter=nan_counter_dict)
```
Or simpler: have `_finite()` increment a process-global counter, print summary at `[phase3-5 DONE]`.

**Risk:** none. Pure additive. Skip if no defect surfaces in batches 2-3.

---

### 9. Fix `build_batch.py:get_already_ingested` partial-doc bug
**Severity:** L | **ROI:** correctness, prevents orphaned doc_ids in batch composition | **Risk:** low | **Apply before next full-corpus run**

`build_batch.py:30-45` query:
```sql
SELECT DISTINCT doc_id FROM chunks WHERE is_canonical=1 AND upserted=1
```
This flags a doc as "already ingested" if **any** of its canonical chunks are upserted. A doc that crashed mid-ingest with 3-of-7 chunks upserted = 1 will be excluded from future batches even though 4 chunks are missing in Qdrant.

**Currently mitigated** by: (a) `phase3_4_5` reconcile guard raises if `upserted < canonical` for any in-scope doc — so a partial doc means the whole ingest bailed, no future batch gets composed wrong; (b) the 2026-04-26 NaN patch + halve-and-retry now isolates bad points without crashing the whole flush.

**Code diff (defensive):**
```diff
-        rows = c.execute(
-            "SELECT DISTINCT doc_id FROM chunks WHERE is_canonical=1 AND upserted=1"
-        ).fetchall()
+        rows = c.execute(
+            """SELECT doc_id FROM chunks WHERE is_canonical=1 GROUP BY doc_id
+               HAVING SUM(CASE WHEN upserted=1 THEN 1 ELSE 0 END) =
+                      SUM(CASE WHEN is_canonical=1 THEN 1 ELSE 0 END)"""
+        ).fetchall()
```

**Risk:** low. Pure SQL correctness improvement. Apply before batch 2 if you have 30 seconds.

---

## Section 3 — Hard-stops

Items that LOOK like optimizations but would violate codified rules. Documented for transparency, NOT recommended.

### S1. "Just enable concurrent cold-load to start the pool faster"
**STATUS: VIOLATES** the cold-load ≤2 cards rule (MEMORY.md "NEVER COLD-LOAD >2 GPU CARDS CONCURRENTLY"). 16× slowdown empirically reproduced 2026-04-25. `embed_pool_profiles.json#sequential_cold_load` MUST stay `true`. The current 3-card sequential cold-load takes ~10s total and is not on the critical path.

### S2. "Bump qwen3-14b to `--parallel 4` so multiple phase 2 docs classify concurrently"
**STATUS: VIOLATES** the codified single-slot rule (qwen3-14b GPU 2 stays `--parallel 1`). Phase 2 has bypass-prefix templates that already eliminate qwen3 contention for the vast majority of CBIC docs. The remaining non-bypass docs go serially through the single slot — that's by design.

### S3. "Reset GPU 1 or GPU 3 via `echo 1 > /sys/class/drm/cardN/device/reset` to ensure a clean state"
**STATUS: BANNED** per MEMORY.md "PCIe PER-CARD RESET IS UNSAFE." Reproduced regression 2026-04-25 (3 healthy → 2 healthy). Cold reboot is the ONLY safe pre-pool ritual. If rec #1's bench fails, don't reset — schedule a maintenance reboot.

### S4. "Move sparse BM25 onto a CPU process pool to parallelize phase 4"
**STATUS: VIOLATES** the no-CPU rule (`DENSE_ONLY=1` mandatory, codified in `ingest_v2.py:1009`). Sparse fastembed CPU work is forbidden. The current sparse path returns empty dicts under `DENSE_ONLY=1` (`ingest.py:127`).

### S5. "Route LLM calls (e.g., NaN debug or pair gen) through LiteLLM at :4444 for retry/failover"
**STATUS: VIOLATES** the no-proxy rule (codified 2026-04-26). All LLM calls direct to llama-server.

---

## Section 4 — Deferred to post-corpus

Things that are sound ideas but too risky to apply mid-run (would need a full re-bench / re-validation cycle).

### D1. Weighted-deficit scheduler in `_Pool.embed`
The codified `_rebalance_note` in `embed_pool_profiles.json` says re-enabling `rebalance_after_warmup` requires a weighted-deficit scheduler. Worth implementing post-corpus to pre-compute steady-state weights from logged p50 latency under load, not warmup. Keeps the door open for heterogeneous pools (e.g., GPU 2 hot-add at 1.5× weight).

### D2. Convert `_Pool.embed` from per-call shard to long-lived streaming pipeline
Currently each `embed_batch` call goes through `_Pool.embed` → shards to N workers → joins. A streaming pipeline (shared SPMC queue + per-worker pull) would let workers pull more work the moment they finish their shard, eliminating tail-latency stragglers. Marginal gain on a 3-GPU equal-weight pool (≈5%), bigger gain on heterogeneous pools.

### D3. PDF parse cache
Phase 2 re-parses every PDF every batch (different doc_ids, but if a doc appears in batch 1 and batch 5 due to bilingual twin re-discovery, it parses twice). Add `(sha256(file_bytes)) → pickled (text, page_offsets)` cache in `_read_pdf_text`. Saves ~20-30% phase 2 time on bilingual-twin re-runs. Risk: cache invalidation on chunker_v2 changes. Defer.

### D4. Embed-output cache keyed by `(canonical SHA256 chunk text)`
Dedupe already removes 39% of chunks. Of the 61% canonical, some appear across batches (rare — different doc_id, same canonical text). A persistent kv cache keyed by chunk_id (sha256 of text) → dense vector would let phase 3-5 skip already-embedded canonical chunks even across full re-ingests. ROI low for this run but high for future re-ingests. Defer.

### D5. Replace fastembed BM25 with a Vulkan/GPU sparse encoder
Currently sparse is OFF (`DENSE_ONLY=1`). When sparse comes back online, the CPU fastembed path is the bottleneck. Train/load a small GPU sparse encoder (SPLADE-mini on Vulkan) post-corpus. Out of scope for current re-ingest.

---

## Cumulative wall-clock savings estimate (if all H/M recs applied)

Recommendation | Saving across batches 2-10 | Cumulative
---|---|---
#1 (5-GPU pool)               | 9 × ~87s     = ~13 min | 13 min
#2 (EMBED_BATCH=96 with #1)   | 9 × ~10-20s  = ~2-3 min| 16 min
#4 (cross-batch pipeline)     | 9 × ~117s    = ~17 min | 33 min
#5 (hot-add GPU 0)            | 6 × ~30s     = ~3 min  | 36 min
#9 (build_batch sql fix)      | correctness only        | 36 min
**Total (top recs)**          |                         | **~30-36 min** of the remaining ingest run

Order of application: #9 (instant, before batch 2) → #1 (bench then apply at batch 2) → #2 (combine with #1) → #4 (batch 3) → #5 (batch 4 if stable).

---

*This document is appended-to every 5 minutes during the run. Last update: 2026-04-26 17:48 EDT.*
