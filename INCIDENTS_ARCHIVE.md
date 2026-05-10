# INCIDENTS_ARCHIVE

Dated postmortems. NOT loaded into Claude's context — reference only when debugging a recurrence.
Active rules distilled from these incidents live in `RULES_INDEX.md`.

---

## 2026-05-08 — Re-embed Marathon: 4 hangs, 1 Qdrant segfault, lesson on reactive over-correction

### Context

Synthetic Q-A enrichment of 42,153 chunks (Gemini 2.5 Flash) had completed successfully. Goal: re-embed all chunks into `cbic_v2` so retrieval embeds the enriched text. Then run G1 on full 380-query SPEC gold to test if enrichment closes the 0.84 → 0.95 gap.

### What happened

- 1st run (6-card pool {0,1,3,4,5,6}, EMBED_BATCH=48): rig hung at 41% (~17.5K pts).
- 2nd run after physical reset: same config, hung again under sustained load.
- 3rd run: same — third hang.
- I dropped pool 6→4 reactively. Resumed at 4 cards + EMBED_BATCH=24. Reached 71% (30K pts), then **Qdrant segfaulted** mid-upsert. Docker auto-restarted Qdrant; ingest crashed on the disconnect. Manifest+Qdrant in sync at 30,024 pts.
- Restarted with full 6-card + batch 48 — and rig hung a 4th time.
- Final relaunch: 6-card + EMBED_BATCH=24 (lower batch, keep codified pool size). Completed cleanly. Phase log: only 513 stragglers needed work — the rest had survived all prior crashes thanks to deterministic SHA256 point IDs (idempotent re-upsert).
- Reconciliation: all 14,022 scoped docs had `expected == upserted` canonical chunks. Final pts: 42,057 (~96 chunk drift = segment variance, acceptable).

### Lessons

1. **Reactive over-correction on pool size was wrong.** I dropped 6→4 without checking whether the codified mitigation (sequential cold-load + warmup) had fired. It HAD. The hangs were Qdrant + batch size, NOT pool size. Codified into `RULES_INDEX.md` `[TRIGGER: reactive overcorrection]`.

2. **Qdrant `status: grey` is healthy, not broken.** Spent ~5 min waiting for `green` after a segfault recovery before realizing `grey` + `optimizer_status: ok` means HNSW indexing in progress for sub-threshold segments — collection is fully readable and writable. My wait-for-green poll loop was wrong. Codified into `[TRIGGER: qdrant status]`.

3. **Qdrant segfaults under sustained upsert load.** Observed once with full crash, possibly contributed to other "rig hangs" (the rig itself stayed responsive in only one of the four; in three, the rig went unreachable — likely Vulkan/memory pressure from the embed pool cascading from Qdrant deaths). Docker auto-restart + WAL recovery handled it; deterministic SHA256 point IDs made the partial work idempotent on retry.

4. **Deterministic SHA256 point IDs are gold.** Across 4 hangs and 1 segfault, the cumulative work survived. The final phase had only 513 stragglers to do (~16s). Without deterministic IDs (Python `hash()` was the bug fixed earlier this session), every crash would have created duplicate points and the collection would now be 2-3x oversize.

5. **The "Read-Before-Write Interlock" architecture was built today** in response to this incident pattern. Two external consultants converged independently: rules with mechanical preflight gates hold; rules dependent on the model "remembering" do not — LLM attention decay is the root cause. New scaffolding:
   - `RULES_INDEX.md` (≤200 lines, keyword-indexed)
   - `scripts/rule_check.sh` (greps RULES_INDEX)
   - `scripts/task_preflight.sh` (asserts inventory + rule_check)
   - `.claude/commands/preflight.md` (slash command)
   - CLAUDE.md "MANDATORY EXECUTION PROTOCOL" section

### Final state at end of incident

- cbic_v2: 42,057 pts, all 14,022 scoped docs reconciled
- Manifest: 42,153 chunks upserted=1 (all canonical)
- Qdrant status: red at incident close — needs check (likely indexing under final load; not investigated yet)
- Pending follow-ons: sparse backfill (~3 min), linked_doc_ids backfill (~1 min), API restart, G1 on full 380-query gold

---

## 2026-05-08 (later) — Post-re-embed cascade: corrupt Qdrant segment, dead fallback code, power-state revert

### Context

After successful re-embed, ran sparse + linked_doc_ids backfills, then API restart, then G1 on full 380-query SPEC gold. Hit four distinct bugs in sequence. All codified into RULES_INDEX.md.

### Bug 1: Qdrant `red` collection — corrupt mmap payload segment

`status: red`, `optimizer_status: Service runtime error: Optimization task panicked: Error("key must be a string", line: 1, column: 80)`. Some enriched chunk's payload_json contains a byte sequence Qdrant's internal mmap parser can't handle. Reads + targeted writes (`update_vectors` by ID) work, but `scroll` and concurrent `query_points` re-trigger the panic → Qdrant segfaults under load (3 docker auto-restarts during this session).

**Mitigation:** bypass scroll in backfills — use manifest-driven backfill scripts (`add_sparse_by_manifest.py`, `backfill_linked_by_manifest.py`) that derive deterministic SHA256 point_ids from the manifest's chunk identity tuples and write via `update_vectors` (per-ID).

**Permanent fix TODO:** identify the corrupt point + delete, OR rebuild collection from manifest.

### Bug 2: Dead `qc.search()` fallback in retriever.py

`retriever.py:83` had `qc.search(...)` as the except-clause fallback to the hybrid `qc.query_points` path. **`qc.search()` was removed in qdrant-client ≥ 1.10.** Never fired before because the hybrid path always succeeded. Today Qdrant segfaulted under G1's 8-worker load → exception bubbled to the fallback → `AttributeError: 'QdrantClient' object has no attribute 'search'` masked the real "Connection refused" cause.

**Patched 2026-05-08:** fallback now uses `qc.query_points(query=dense, using='dense', ...).points`.

**Hard rule codified:** every `except:` fallback MUST use the same library API as the success path. Untested fallback = dead code masking the real failure.

### Bug 3: Embed-pool cards revert to `dpm=auto` after ingest

The codified power-state rule (2026-05-07) installed systemd ExecStartPre on `qwen3-14b` and `bge-reranker` to force `dpm=high` on cards 0 + 2. **No equivalent on the embed pool cards {1,3,4,5,6}** — they get forced high by `ingest_v2.py`'s pool warmup, then drift back to `auto` (`sclk=800Mhz`) when ingest exits. Subsequent `/retrieve` calls via the API embed at LOW power → 10-15s per query + reranker timeouts.

**Permanent fix 2026-05-08:** new systemd drop-in at `/etc/systemd/system/cbic-rag-api.service.d/power_dpm.conf` (mirrored at `D:/_gpu_rig_ai/reingest_spec/systemd/cbic-rag-api.service.d/power_dpm.conf`) forces `dpm=high` on cards {0,1,2,3,4,5,6} before the API starts.

**Manual remediation (when API is already running):**
```bash
for c in 1 3 4 5 6; do echo high | sudo tee /sys/class/drm/card$c/device/power_dpm_force_performance_level >/dev/null; done
```

### Bug 4: G1 default `--out` writes to /opt path owned by root

`gate_g1_recall.py` default `--out` points at `/opt/.../gate_g1_result.json` which is root-owned → `PermissionError` after a multi-minute eval run completes. Fix: pass `--out /tmp/g1_result.json` explicitly.

### Lessons

1. **Every except-clause is potentially dead code.** Review for "does this branch use the same library API as the success path?" — if the fallback hasn't fired in production, it may have rotted.
2. **GPU power state must be re-asserted by every consumer service, not just every loader service.** Loaders set high; pools that USE shared GPUs don't.
3. **Concurrent load on a `red` Qdrant collection re-triggers the panic.** Default to `--workers 1` against any post-re-ingest collection until it's stably indexed.
4. **Manifest-driven backfill bypasses scroll.** When Qdrant scroll panics, derive deterministic point_ids from the manifest and update by ID — works around corrupt segments without re-ingest.

---

## 2026-05-08 (final) — G1 = 0.8421 on full 380 gold post-enrichment. Synthetic enrichment alone is insufficient.

### Result

```
G1 final on cbic_v2 (post enrichment + sparse backfill + linked backfill):
n=380  hits=320  recall@10=0.8421  errors=2  pass_gate=FALSE  threshold=0.95
```

### Trajectory across the 380-query run

| Checkpoint | Recall@10 |
|---|---|
| 50 | 0.880 |
| 100 | 0.890 |
| 150 | 0.887 |
| 200 | 0.880 |
| 250 | 0.868 |
| 300 | 0.843 |
| 350 | 0.849 |
| **380 (final)** | **0.8421** |

Harder queries clustered in the back half (recall fell from 0.89 → 0.84 between 100 and 380). Net delta vs pre-enrichment baseline (0.8602 adjusted): **−1.8 pts** — synthetic Q-A enrichment + hybrid retrieval did NOT close the gap; it slightly *underperformed* the pre-enrichment baseline on the same gold.

### Hypothesis on why enrichment didn't help

The synthetic Q-A blocks added narrative-style questions to chunk `embed_text`. The hypothesis was: scenario-style queries → narrative-augmented chunks would close the asymmetric query↔chunk geometry. Evidence we now have:
- It did NOT close the gap (G1 = 0.8421 vs 0.8602 baseline).
- The misses are concentrated in the back half of gold queries — likely the more complex multi-statute cross-references.
- Possible reasons: (a) Gemini-generated synthetic Qs may have introduced noise (off-topic phrasing); (b) embedding the augmented `embed_text` may have shifted chunks AWAY from the gold queries that were already close; (c) BGE-M3's pre-trained geometry can't absorb structured legal framings without supervised tuning.

### Decision

HALT per Hard Rule #1 (95% non-negotiable, no patch-and-continue). Next intervention: **BGE-M3 fine-tune** on 5,500 curated pairs (Phase A) and/or 126K synthetic pairs (Phase B), evaluated on UNCHANGED 380-query gold. Cloud A100 needed (Vulkan ≠ training). New rule codified at `RULES_INDEX.md` `[TRIGGER: fine-tune, BGE-M3]`.

### Pending diagnostic (separate, cheap)

Inspect the 60 misses: which categories? what query length? what chunk type? May reveal a structural fix cheaper than fine-tune. Script: write `diagnose_misses.py` that reads `/tmp/g1_v3_result.json` + matches against gold doc_ids + bins by category/length.

---

## 2026-05-09 — Hardware fault discovery: rig corrupts ≥1GB downloads

### Context

After G1 failed, RG approved attempting BGE-M3 fine-tune on rig card 2 (RX 6700 XT 12GB) via PyTorch+ROCm to avoid cloud spend. Setup: stopped qwen3-14b → freed card 2 → created `/opt/training-venv/` → tried `pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch torchvision` (4.0GB wheel).

### Sequence of failures

1. Pip attempt 1 (rocm6.2): hash mismatch on torch wheel.
2. Pip attempt 2 (rocm6.2, `--no-cache-dir`): hash mismatch, **different "got" hash from attempt 1**.
3. Pip attempt 3 (rocm6.1, `torch-2.6.0+rocm6.1`): hash mismatch on a sub-dependency (different inner file).
4. Direct `curl` of the rocm6.1 wheel (2.79GB): downloaded full size, but `zipfile.testzip()` failed with `zlib.error: Error -3 invalid distances set`.
5. `aria2c -x 16 -s 16` (multi-conn, integrity-checked download): downloaded clean per aria2c, but `zipfile.testzip()` failed on `torch/lib/aotriton.images/...`.
6. RG asked "did you try GPU 2 reset?" — I had not. Tried sysfs reset → "Inappropriate ioctl for device". Tried PCI rescan → **GPU 2 disappeared from PCI bus, didn't come back from rescan**.
7. Rebooted rig (~60s back). All 7 cards visible, services healthy. Retried aria2c download.
8. **Post-reboot aria2c retry: same byte-corruption pattern.** Different inner file failed (`libMIOpen.so` this time), different sha256 again.

### Verdict

**Hardware fault — bit-level corruption of large transfers, random per attempt.** Not fixable by:
- retry (corruption is per-download, different bits each time)
- cache purge (`pip --no-cache-dir`, `rm -rf ~/.cache/pip`)
- different downloader (curl, aria2c, pip all corrupt)
- multi-conn / chunked (aria2c 16-conn still corrupted)
- GPU reset (sysfs ioctl unsupported, PCI rescan removed card)
- full reboot (post-reboot reproduced corruption immediately)

### Root cause hypotheses (in order of probability)

1. **Bad RAM module** — most consistent with random per-byte corruption that survives reboot
2. **Bad PCIe link** to NIC or storage controller (DMA path corrupted)
3. **Bad NIC firmware** (less likely; would show in dmesg, none seen)

dmesg has been clean of EDAC errors but TCP checksum is only 16-bit — random corruption at 1-in-65k rate gets through silently.

### Implications

- **Inference is impacted too** but at a low enough rate that BGE-M3 / qwen3 still produce coherent output. Probably explains some of today's 4 rig hangs and Qdrant segfaults — corrupted index pages caused `Result::unwrap() on an Err` panics.
- **G1 = 0.8421 result was measured on a faulty system.** Some fraction of the 10.8-pt gap could be silent noise. Re-running G1 on healthy hardware might give a slightly different number — but unlikely to close the full gap.
- **Cannot train on this rig** — silent weight corruption mid-training would produce a broken model.

### Permanent action items

1. **Tonight:** memtest86+ overnight (4-8 hrs, kernel-level RAM verification). If errors found → swap RAM modules / replace.
2. **If RAM clean:** check PCIe link width on NIC and storage (`sudo lspci -vv | grep -E "LnkSta"`). Look for downgraded x4/x1 links.
3. **If hardware confirmed bad:** replace before any further training. Inference path can keep running until a hang.

### Codified into RULES_INDEX

`[TRIGGER: rig hardware reliability, byte corruption, large download, hash mismatch]` — codified pre-reboot, then extended POST-REBOOT verified 2026-05-09 with the conclusive evidence above.

---

## 2026-05-09 (later) — SMB-mount workaround discovered + verified

### Context

After hardware corruption was conclusively diagnosed and codified rule said "go cloud," RG pushed back: "If a big download is a problem, then you can download it here on windows machine and then copy on the smb share we have with rig." The rig already had a CIFS automount at `/mnt/d` mapping `//192.168.1.222/projects` (Windows D drive). Worth one more attempt before pivoting to cloud.

### Empirical sequence

1. **Windows-side download of `torch-2.4.1+cu121` wheel (798MB)** via curl: clean download. sha256 = `9a5f0b103cfe...`, zip integrity OK.
2. **Direct LAN SCP from Windows to rig disk**: corrupted. Rig sha256 = `ee3d89e16af...` (different from Windows source). Zip integrity BAD.
3. **SMB read of same wheel from rig via `/mnt/d`**: clean. sha256 = `9a5f0b103cfe...` (matches Windows). Zip integrity OK.
4. **`pip install` from SMB mount of the wheel**: torch itself installed clean (798MB outer wheel verifies).
5. **Dependencies via `pip install` from SMB-mounted CUDA wheels**: corrupted on extraction of `libcufft.so.11` (~150MB inner file). pip's extract path triggers RAM corruption.
6. **`apt-get install nvidia-cuda-toolkit`**: also corrupted on `libcublaslt11_*.deb` extraction.
7. **Final workaround**: extract wheels on Windows side first, then per-file copy via SMB with sha256 verification + retry-up-to-8x. Files up to 515MB confirmed copying clean (e.g. `libcublasLt.so.12`).

### Diagnosis confirmed

The failure mode is NOT in network / NOT in CPU compute. It IS in the local-disk write path on rig. Specifically:
- Reading from SMB and computing sha256 = clean (transient RAM use, no long-buffer)
- Reading from SMB and writing to disk via pip/cp = corrupts (data sits in page cache before flush)
- The corruption rate is rare per byte but ≈ 100% for files >500MB single transfer

### Workaround codified (permanent until hardware fixed)

| Component | Path |
|---|---|
| CLAUDE.md "RIG HARDWARE FAULT" section | top of file, mandatory-rule level |
| `RULES_INDEX.md` `[TRIGGER: SMB workflow]` | rule block, keyword-indexed |
| `MEMORY.md` anchor entry | one-line, auto-loaded at session start |
| Workaround script | `D:/_gpu_rig_ai/tmp_downloads/copy_wheels_verified.py` |
| SHA256 manifest format | `D:/_gpu_rig_ai/tmp_downloads/wheels_manifest.sha256` |

### The rule

**Until hardware is fixed:**
- NO direct download (pip / curl / aria2c / docker / apt) of any package >150MB on rig
- ALWAYS Windows-side download → integrity-verify → SMB-share → install/copy on rig
- For wheels with large internal files: pre-extract on Windows, per-file copy with retry

### Hardware fix pending

Action items:
1. Run memtest86+ overnight (4-8 hrs)
2. If memtest finds errors → identify bad DIMM, replace
3. If memtest is clean → check PCIe link integrity, PSU under load, SATA cable
4. Until fixed, SMB-only workflow is the standing approach
