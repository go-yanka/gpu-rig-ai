# 🔴 LIVE INGEST DASHBOARD — cbic_v2

> **Keep open in preview pane.** Updated after every step.
> **Last update:** 2026-04-26 **18:25 IST**  ·  **Stage:** M  ·  **Active:** Batch 1 → CP-1 (G3 retry + G4 running)

---

## At-a-glance

| Metric | Value |
|---|---|
| Batches done | **1 / 10** (10%) |
| Points in cbic_v2 | **7,823** / ~67,000 expected |
| Wall clock spent on ingest | **6.1 min** (batch-1 try-2) + **~14 min** debug/CP-1 setup |
| Wall clock projected remaining | **~95 min ingest** + **~70 min CP gates** = **~2h 45m** |
| HALTs / failures | 1 (batch-1 try-1 NaN crash, recovered) |
| ETA full re-ingest done | **2026-04-26 ~21:10 IST** if no further halts |

```
Batch:  [✅][🟡][⬜][⬜][⬜][⬜][⬜][⬜][⬜][⬜]   1/10
        1   2   3   4   5   6   7   8   9   10
                            ↑               ↑
                          CP-2            CP-3
```

---

## Current activity (real-time)

| What | Started | Elapsed | ETA | Status |
|---|---|---|---|---|
| Batch 1 try-2 ingest | 17:35 | 6.1 min (done 17:41) | – | ✅ DONE |
| CP-1 G3 (retry, abs-path fix) | 18:24 | <1 min | ~3 min | 🟡 RUNNING (PID 907679) |
| CP-1 G4 grounded | 18:07 | ~17 min | ~3 min more | 🟡 RUNNING (PID 888727) |
| CP-1 5× /query smokes | – | – | ~2 min | ⬜ queued after G3/G4 |

**What I'm waiting on right now:** G3 retry + G4 to finish, then I run smokes + parse + decide pass/fail.

---

## Per-batch detailed timeline

### Batch 1 — try 1 ❌ FAILED

| When | Event | Duration |
|---|---|---|
| 17:17 | Launched | – |
| 17:19 | Phase 2 done (1484 docs → 4242 chunks) | 116.8s |
| 17:22 | Phase 3-5 crashed at chunk 2880/4242 (Qdrant 400 NaN/Inf) | 131s before crash |
| 17:22 | Total wasted | **~5.0 min** |

### Batch 1 — try 2 ✅ DONE

| When | Event | Duration | Cumulative |
|---|---|---|---|
| 17:35 | Launched (PID 852987) | – | 0:00 |
| 17:38 | Phase 2 done (1500 docs → 4242 chunks, 39.1% dedupe) | 142.9s | 2:23 |
| 17:41 | Phase 3-5 done (4847 chunks → pts=7823, 21.8 ch/s) | 221.7s | 6:05 |
| 17:41 | Reconcile PASS, 0 NaN, 0 Qdrant 400s | – | 6:05 |

| Step | Status | Time | Detail |
|---|---|---|---|
| B. Build batch | ✅ 17:34 | 0:01 | 1500 doc_ids, SEED=43 |
| C. Pre-launch sanity | ✅ 17:35 | 0:01 | category dist OK |
| D. Launch ingest | ✅ 17:35 | 0:00 | PID 852987 |
| E. Phase 2 chunk+dedupe | ✅ 17:38 | **2:23** (target ~2:00) | 4242 canonical, 39.1% savings |
| F. Phase 3-5 embed+upsert | ✅ 17:41 | **3:42** (target ~3:10) | 21.8 ch/s on 3-GPU pool |
| G. Verify completion | ✅ 17:41 | 0:05 | reconcile PASS, 0 errors |
| H. Record batch row | ✅ 17:50 | 0:02 | INGEST_TRACKER.md updated |
| I. Spot-check 3 doc_ids | ⬜ | – | pending (run after CP-1) |
| J. Snapshot manifest | ⬜ | – | pending |
| K. Snapshot Qdrant | ⬜ | – | pending |
| L. **CP-1 gate** | 🟡 18:07 | **~18 min so far** | G3 retry running, G4 running |
| M. JOURNAL entry | ⬜ | – | after CP-1 |
| N. Optimization log | ⬜ | – | after CP-1 |
| O. Green-light → Batch 2 | ⬜ | – | blocked on CP-1 |

**CP-1 sub-checklist:**

| Check | Threshold | Started | Result | Status |
|---|---|---|---|---|
| G3 Levenshtein on 38 batch-1 gold | ≥ 0.90 | 18:07 (re-launched 18:24) | TBD | 🟡 |
| G4 grounded on 201 clean adv | ≥ 0.95 | 18:07 | TBD | 🟡 |
| 5× hand-picked /query smokes | 5/5 top-1 correct | – | – | ⬜ |

---

### Batch 2 ⬜ PENDING (blocked on CP-1)

| Step | ETA | Status |
|---|---|---|
| B. Build | – | ⬜ |
| C. Sanity | – | ⬜ |
| D. Launch | – | ⬜ |
| E. Phase 2 (~2:30) | – | ⬜ |
| F. Phase 3-5 (~3:45) | – | ⬜ |
| G. Verify | – | ⬜ |
| H. Record | – | ⬜ |
| I. Spot-check | – | ⬜ |
| J. Snapshot manifest | – | ⬜ |
| K. Snapshot Qdrant | – | ⬜ |
| L. CP gate | ➖ | (no CP after batch 2) |
| M. JOURNAL | – | ⬜ |
| N. Optimization | – | ⬜ |
| O. Green-light → batch 3 | – | ⬜ |

**Projected runtime:** ~6.5 min · **Blocked:** waiting CP-1 pass

---

### Batches 3, 4 ⬜ PENDING — same shape, no CP, ~6.5 min each

### Batch 5 ⬜ PENDING — **CP-2 after this batch**

CP-2 gates: G1 ≥ 0.95 · G3 ≥ 0.92 · G4 ≥ 0.95 · reranker histogram (gold p50 ≥ 0.7, adv max < 0.65). Time budget ~15 min + ingest 6.5 min.

### Batches 6, 7, 8, 9 ⬜ PENDING — same shape, no CP, ~6.5 min each

### Batch 10 ⬜ PENDING — **CP-3 final acceptance**

CP-3 gates: G1 ≥ 0.95 · G2-full dual-judge n=380 ≥ 0.95 · G3 ≥ 0.95 · G4 ≥ 0.95 · G5 p95 ≤ 12-15s. Time budget ~45 min + ~$4 G2.

---

## Gate values per batch (filled as we measure)

| Batch | Pts after | G1 | G3 | G4 | G2 | G5 p95 | CP | Time spent |
|---|---|---|---|---|---|---|---|---|
| 1 | 7,823 | – | TBD | TBD | – | – | CP-1 | 6.1 min ingest + ~? CP-1 |
| 2 | – | – | – | – | – | – | – | – |
| 3 | – | – | – | – | – | – | – | – |
| 4 | – | – | – | – | – | – | – | – |
| 5 | – | TBD | TBD | TBD | – | – | CP-2 | – |
| 6 | – | – | – | – | – | – | – | – |
| 7 | – | – | – | – | – | – | – | – |
| 8 | – | – | – | – | – | – | – | – |
| 9 | – | – | – | – | – | – | – | – |
| 10 | – | TBD | TBD | TBD | TBD | TBD | CP-3 | – |

Thresholds: G1≥0.95 · G3≥0.95 (CP-1≥0.90, CP-2≥0.92) · G4≥0.95 · G2≥0.95 · G5 p95≤12-15s

---

## Resource utilization snapshot (last sample 18:25)

| GPU | Role | State | Utilization |
|---|---|---|---|
| 0 | bge-reranker (port 9085) | active | idle (no traffic) |
| 1 | unused | idle | – (candidate for embed pool expansion) |
| 2 | qwen3-14b (port 9082) | active | low (G4 grounded calls) |
| 3 | unused | idle | – (candidate for embed pool expansion) |
| 4 | BGE-M3 embed | idle | last batch peaked 99% briefly |
| 5 | BGE-M3 embed | idle | last batch ~7.3 ch/s/GPU steady |
| 6 | BGE-M3 embed | idle | last batch ~7.3 ch/s/GPU steady |

**Pool plan:** after CP-1 pass, bench GPU 1 + GPU 3 solo, then expand pool to 5 GPUs → estimated ~3 min/batch saving × 9 remaining batches = ~27 min.

---

## Halts / errors / fixes log

| Time | What | Fix | Status |
|---|---|---|---|
| 17:22 | Batch 1 try-1 Qdrant 400 NaN/Inf in vector | Sanitiser + halve-retry patches deployed | ✅ verified by try-2 |
| 18:07 | CP-1 G3 wrong gold schema | Reformatted to `{"queries":[]}` | ✅ |
| 18:07 | CP-1 G4 wrong arg name (`--gold` vs `--adv`) | Switched to clean adv set + `--adv` | ✅ |
| 18:12 | CP-1 G3 output write FileNotFoundError (relative path, cwd) | Re-launched with absolute output path | 🟡 retry running |

---

## Next 5 actions (in order)

1. ⏳ Wait G3 retry (~3 min) + G4 (~3 min) to finish
2. ▶ Run 5× /query smokes (notification, circular, act, rule, form, instruction prefixes)
3. ▶ Parse all CP-1 results → fill gate table → flip step L checkbox
4. ▶ If pass: Batch 1 steps I, J, K, M, N, O → green-light Batch 2
5. ▶ Launch Batch 2 (parallel: GPU 1+3 solo bench in background)
