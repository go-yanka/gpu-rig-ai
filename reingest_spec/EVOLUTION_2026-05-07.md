# Evolution — 2026-05-07 session

Full chronology + permanent fixes from today's session. Append-only style. Companion to JOURNAL.md.

## Part 1: GPU power-state-on-boot bug — root cause + permanent fix

**Symptom (yesterday 2026-04-26):** qwen3-14b on GPU 2 timing out at 30s for 5-10 tokens; HTTP llama-server unusable; even direct in-process llama-cpp-python (no server, no proxy) measured 104s for 10 tokens.

**Test today:** is it the GPU power state, not architecture?

1. Booted rig, stopped all GPU services.
2. Manually `echo high > /sys/class/drm/card{0,2}/device/power_dpm_force_performance_level`.
3. Ran direct in-process bench: **27.4 tok/s** (warm avg 364ms / 10 tokens).
4. Started qwen3-14b service via systemd; HTTP bench: **36.14 tok/s** (180-352ms / 5 tokens).

Both paths fast. **GPU 2 is fine.** The issue was 100% the power state when the model loaded.

**Why it reproduced every reboot:**
- `qwen3-14b.service` has `WantedBy=multi-user.target` → auto-starts on boot.
- After boot, `power_dpm_force_performance_level` = `auto` (low when idle).
- llama-server invokes the GPU before any compute → GPU stays low → model loads at low clocks → never escapes.
- Direct in-process had the same bug because `llama-cpp-python` calls the same Vulkan layer hitting the same idle GPU.

**Permanent fix codified:** `ExecStartPre` drop-ins
- `/etc/systemd/system/qwen3-14b.service.d/power_dpm.conf` forces card2 high BEFORE model load.
- `/etc/systemd/system/bge-reranker.service.d/power_dpm.conf` forces card0 high BEFORE model load.
- Both mirrored in Windows tree at `D:/_gpu_rig_ai/reingest_spec/systemd/{qwen3-14b,bge-reranker}.service.d/power_dpm.conf` so they survive a rig wipe.

**Standing rule:** any new GPU-using systemd service must add the equivalent `ExecStartPre`.

**Naviano22 sysfs quirk:** `pp_dpm_sclk` shows `0Mhz *` even when working — kernel quirk on RX 6700 XT, not degradation. Trust `pp_dpm_mclk` (shows `1000Mhz *` correctly) and the actual inference benchmark.

## Part 2: Low-density tier — sparse-text PDFs no longer silently dropped

**Symptom:** while diagnosing D-DEFECT, found 4 large customs PDFs (cs-ins-07-2022, cs-ins-03-2024, cs-ins-14-2025, csnt-odr-2k9) had real extractable text (1000-2400 chars) but were silently dropped. Log: `PHASE2-DEFER ocr — ocr_deferred: text_density below 200/pg, no ocr_cache hit`.

**Root cause:** `detect_text_source()` returned only `"born"` (≥200 chars/page) or `"ocr"` (which silently `continue`d if no OCR cache hit). The OCR cache is empty for these docs and there's no live OCR pipeline — so they fell into a hole.

**Fix at `ingest_v2.py:594-606`:** new third tier `"low_density"`.
- `density >= 200/pg` → `"born"` (full chunking)
- `density < 200/pg` AND `total >= LOW_DENSITY_TOTAL_MIN=500` → `"low_density"` (chunk anyway, set `ocr_pending=True` in payload for future supersede)
- `total < 500` → `"ocr"` (still defers — truly image-only)

**Smoke test 2026-05-07:** the 4 victim docs ingested cleanly post-fix.

**Standing rule:** when introducing any "defer to OCR / external pipeline" path, verify the pipeline actually exists; fall back to ingesting available text rather than silently dropping.

## Part 3: D-DEFECT classified into 3 distinct defects + corpus-wide carve-out

Until today, `post_batch_lint.py` reported every missing doc as one "D-DEFECT" P0 — undifferentiated. Today's audit (`generate_carveouts.py`) classified all 157 missing docs in batches 1-5 and then the full 15,776-doc manifest:

| Subtype | Count | Description | Fixable how |
|---|---|---|---|
| D-2a NO_PDF | 292 | manifest has no `path_en` (scrape never delivered or empty body) | rescrape needed |
| D-2b JUNK_CONTENT | 7 | PDF on disk but pdftotext < 500 chars (HTML page saved as PDF, 1.2KB placeholders, image-only forms) | rescrape / OCR |
| D-1 SHARED_PDF | 532 | multiple doc_ids referencing same sha256_en (dominant cluster: `CGST-Rules-2017-Part-B-Forms.pdf` shared by 176 doc_ids) | chunker-v3 per-form structural splitting |
| **Total unique** | **831** | ~5.3% of corpus | — |

**Effective expected for full-corpus QA = 14,945 docs**.

**Output file:** `/opt/indian-legal-ai/data/eval/known_d_defect_carveouts.json`

**Lint integration at `post_batch_lint.py:56-71`:** subtracts carved doc_ids from `expected` before D-DEFECT P0 check. Anything missing AND outside carve-out → real defect → exit 2 → `run_batch_loop.sh:run_lint()` HALTs.

**Verification 2026-05-07:** re-ran lint on batches 3-5 — all CLEAN, exit 0.

**Re-run when:** manifest changes (new scrape, recovered PDFs) or chunker handling of D-1/D-2 changes:
```bash
python3 /opt/indian-legal-ai/reingest_spec/scripts/generate_carveouts.py
```

## Part 4: run_batch_loop.sh — two script-level Hard Rule violations

Found while CP-2 was launching from yesterday's deployed version.

### Bug 1 — Lint args mangled

The literal line in the script was:
```
post_batch_lint.py \ \/batches/batch\_doc_ids.csv >> \ 2>&1 || log "[lint] post_batch_lint exit= — see report"
```

Escaped backslashes, missing `$N` and `$LOG`. argv[1] was the literal string ` /batches/batch_doc_ids.csv`. `int()` on that → `ValueError: invalid literal for int() with base 10: ' /batches/batch_doc_ids.csv'`. Lint crashed silently every batch; loop ignored via `|| true`.

**Means batches 3, 4, 5 had ZERO lint coverage** before today's intervention.

### Bug 2 — Gate concurrency baked into the script (Hard Rule #10 violation)

Original `run_cp()` was:
```bash
bash gate_preflight.sh g1 || true
gate_g1_recall.py ... &
G1=$!
bash gate_preflight.sh g3 || true   # G1 already running!
gate_g3_levenshtein.py ... &
G3=$!
wait $G1 $G3
```

Preflight correctly refused G3 (G1 already in flight), but `|| true` swallowed the refusal and `&` launched G3 anyway. **Hard Rule #10 violated by the script itself.** CP-2 ran G1 + G3 concurrently against the shared bge-reranker → exactly the contamination scenario codified after CP-1 incident on 2026-04-26.

### Permanent fix

Rewrote `reingest_spec/scripts/run_batch_loop.sh` (synced to `D:/_gpu_rig_ai/reingest_spec/scripts/run_batch_loop.sh`):

- New `run_serial_gate()` helper: preflight refusal → `exit 6` (HALT, not warn). Each gate runs to completion before the next. No `&`, no `wait`.
- New `run_lint()` helper: honors lint exit code. P0 (exit 2) → `exit 5`. P1 (exit 1) → `exit 5`. Clean (0) → continue.

Both helpers respect Hard Rules #1 and #10 by construction; cannot be bypassed via `|| true`.

**Standing rule:** any wrapper script that calls a mandatory preflight or lint MUST treat refusal/non-zero as a hard halt, not a soft warning. `|| true` after such calls is a code smell and a Hard Rule violation.

## Part 5: Operational trivia codified once and for all

These keep getting re-discovered every session. Pinning them here:

- **Qdrant runs in Docker container `qdrant-cbic`** on host ports **6343→6333** and **6344→6334**. NOT a systemd unit. Find: `docker ps --filter name=qdrant`.
- **cbic-rag-api on port 9500**, NOT 8087. Health: `/health`.
- **`/retrieve` body field is `question`, NOT `q`.** Sample: `{"question":"...","k":5}`.
- **`post_batch_lint.py` exit code is meaningful** (P0=2, P1=1, clean=0) but gets masked when piped (e.g. `... | tail`). When testing manually, capture rc directly: `python3 post_batch_lint.py N CSV; echo rc=$?`.

## Part 6: state at end of session

- cbic_v2 = 26,026+ pts (batch 6 done, batch 7 in flight at JOURNAL write time).
- Loop `run_batch_loop.sh 6 10` running with all fixes.
- Lint CLEAN on batches 3, 4, 5 (re-validated post-fix).
- CP-2 was never properly run (killed during concurrent gate violation). The fixed loop runs CP-2 only at N=5 (already past). **Need separate manual CP-2 run** OR rely on CP-3 after batch 10 as the authoritative gate panel for the full re-ingest.
- All 5 codified fixes deployed and persisted (rig + Windows tree + MEMORY + JOURNAL + DECISIONS).

## Permanent artifacts produced today

| Artifact | Location |
|---|---|
| Low-density tier in chunker | `reingest_spec/ingest_v2.py:589-606` |
| Carve-out file (831 doc_ids) | `/opt/indian-legal-ai/data/eval/known_d_defect_carveouts.json` |
| Carve-out generator (regenerable) | `reingest_spec/scripts/generate_carveouts.py` |
| Fixed batch loop | `reingest_spec/scripts/run_batch_loop.sh` |
| Lint with carveout integration | `reingest_spec/scripts/post_batch_lint.py:56-71` |
| Power-state systemd drop-ins | `reingest_spec/systemd/{qwen3-14b,bge-reranker}.service.d/power_dpm.conf` |
| This evolution doc | `reingest_spec/EVOLUTION_2026-05-07.md` |
| MEMORY.md top entries (5 new) | `~/.claude/projects/D---gpu-rig-ai/memory/MEMORY.md` |
| JOURNAL.md entry | `reingest_spec/JOURNAL.md` (separate append) |

---

## Part 7: build_batch.py manifest path mismatch (discovered while batches 8-10 were running)

**Symptom:** batch 8 ingest ran 210s, exited 0, but produced **+0 pts** (expected ~3500-5000). Loop's safety check (`DELTA < 1000`) correctly HALTed.

**Root cause:** two scripts pointing at different manifest sqlite files for the same purpose.
- `ingest_v2.py` writes to `/opt/indian-legal-ai/data/ingest_manifest_v2.sqlite` (line 78).
- `build_batch.py` reads from `/opt/indian-legal-ai/data/ingest_manifest_v2_full.sqlite` (line 24, `_full` suffix).

The `_full.sqlite` file never existed at runtime. So `get_already_ingested()` returned an empty set every call. Each batch picked from the FULL corpus minus nothing. Overlap accumulated:

| Batch | Overlap with priors | Cumulative union |
|---|---|---|
| 1 | 0 | 1500 |
| 2 | 142 | 2858 |
| 3 | 284 | 4074 |
| 4 | 420 | 5154 |
| 5 | 488 | 6166 |
| 6 | 594 | 7072 |
| 7 | 673 | 7899 |
| 8 | 742 | 8657 |

By batch 8, 1500 of the picked doc_ids were already phase2_done in the actual manifest. ingest_v2 correctly skipped them all → 0 chunks emitted → safety check HALTed.

**Permanent fix at `reingest_spec/build_batch.py:24`** (synced rig + Windows tree):
- Changed `MANIFEST_V2 = ".../ingest_manifest_v2_full.sqlite"` → `".../ingest_manifest_v2.sqlite"`.

**Verification post-fix:**
- `[batch8] full corpus: 15776 docs`
- `[batch8] already ingested into cbic_v2: 9346 docs`
- `[batch8] available for batching: 6430 docs`
- `[batch8] FINAL: 1500 docs picked` — of which 1,162 truly fresh.

**Standing rule:** any two scripts that share state via a sqlite path MUST resolve the path through the same constant/env var. Never hardcode divergent paths in sibling scripts.

**Cumulative state implication for tomorrow:** batches 1-7 actually ingested 9,346 unique doc_ids (the overlapping CSVs were deduped by ingest_v2 doing the right thing). Remaining unprocessed = 14,945 (carve-out-adjusted corpus) - 9,346 = ~5,500 docs. Batches 8-10 with the fixed build_batch should cover ~4,500-5,000. May need 1-2 more batches after CP-3 to reach 100% of the ingestable corpus.

---

## Part 8: Lint runtime-classifies unknown zero-chunk docs (self-healing)

**Symptom:** twice in this session lint HALTed on docs that should have been carve-outs but weren't. Static carve-out file generated by `generate_carveouts.py` had a size-based pre-filter that decided which PDFs to verify with pdftotext:
- Batch 7: 5 docs in the 9-25KB range, dropped because original threshold was <5KB.
- Batch 8: 1 doc at 459KB (`cbic-instruction-msts:1000539`, scanned customs instruction with 221 chars eOffice header), dropped because raised threshold of <50KB still missed it.

The fundamental problem: **any static size threshold for "what to verify" leaves a long tail of false negatives**, and rescrapes / new docs can inject more.

**Permanent fix at `post_batch_lint.py:103-141`** — runtime classification of any zero-chunk doc not in the static carveout. For each missing doc:
1. Query manifest for `path_en` + `sha256_en` + cluster size.
2. If path missing → D-2a (no PDF) → carved out at runtime.
3. If sha shared with > 1 doc → D-1 (cluster member) → carved out at runtime.
4. Run `pdftotext`. If text < 500 chars → D-2b (junk content) → carved out at runtime.
5. Otherwise → real defect → HALT.

**Verified 2026-05-07:** batch 8 re-lint after fix:
- 1 zero-chunk doc auto-classified as D-2b (`runtime-classified 1 zero-chunk docs as known carve-out: 0 D-2a, 1 D-2b, 0 D-1`)
- `[lint] CLEAN — no drift detected, exit 0`

**Architectural improvement:** static carveout file remains as a fast pre-filter (subtracts before scrolling Qdrant), but is no longer the source of truth. Lint is self-healing against:
- Rescraped junk PDFs (will classify as D-2b at runtime)
- Newly merged shared-PDF clusters (D-1 at runtime)
- Manifest-only docs where the file was deleted (D-2a at runtime)

**Standing rule:** any "static exclusion list" should have a runtime-classification fallback for the same condition. Static lists go stale, runtime checks don't.


## Part 7: build_batch.py manifest path mismatch (discovered while batches 8-10 were running)

**Symptom:** batch 8 ingest ran 210s, exited 0, but produced **+0 pts** (expected ~3500-5000). Loop's safety check () correctly HALTed.

**Root cause:** two scripts pointing at different manifest sqlite files for the same purpose.
- ingest_v2.py: writes to  (line 78).
- build_batch.py: reads from  (line 24, with  suffix).

The  file never existed at runtime. So  returned an empty set every call. Each batch picked from the FULL corpus minus nothing. Overlap accumulated:

| Batch | Overlap with priors | Cumulative union |
|---|---|---|
| 1 | 0 | 1500 |
| 2 | 142 | 2858 |
| 3 | 284 | 4074 |
| 4 | 420 | 5154 |
| 5 | 488 | 6166 |
| 6 | 594 | 7072 |
| 7 | 673 | 7899 |
| 8 | 742 | 8657 |

By batch 8, 1500 of the picked doc_ids were already phase2_done in the actual manifest. ingest_v2 correctly skipped them all → 0 chunks emitted.

**Permanent fix at ** (synced rig + Windows tree):


**Verification post-fix:**
- 
- 
- 
-  — of which 1162 truly fresh (vs all 1500 already-done before fix)

**Standing rule:** any two scripts that share state via a sqlite path MUST resolve the path through the same constant/env var. Never hardcode divergent paths in sibling scripts.

**Cumulative state implication for tomorrow:** batches 1-7 actually ingested 9346 unique doc_ids (overlapping batch CSVs were correctly deduplicated by ingest_v2). Remaining unprocessed = 14,945 carve-out-adjusted - 9,346 = ~5,500 docs. Batches 8-10 with the fixed build_batch should cover ~4,500 of those (if 1500/batch new + small overlap). May need 1-2 more batches after CP-3 to reach 100% of the ingestable corpus.
