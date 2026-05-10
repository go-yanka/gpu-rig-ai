# RULES_INDEX — Keyword-Indexed Hard Rules

This file is the single source of truth for the **active** rules I (Claude) must follow.
Format: `[TRIGGER: <keywords>]` blocks. `rule_check.sh <keyword>` greps this file.

Historical incidents that produced these rules live in `INCIDENTS_ARCHIVE.md`.
This file stays under 200 lines so the model can hold all of it in working attention.

---

## How to use this file

Before ANY architectural change / pool size adjustment / chunker edit / threshold change / model swap / fine-tune launch:

1. Run `bash D:/_gpu_rig_ai/scripts/rule_check.sh "<keyword>"` for at least one keyword from the situation.
2. Quote the matching block verbatim in a `> [CODIFIED RULE]` block in your response.
3. Fill the Self-Audit template (see bottom).
4. Only then act.

If you cannot quote a block, you have not consulted the rules. Stop.

---

## Rules

[TRIGGER: pool size, embed pool, EMBED_GPUS, GPU pool, hang under load, rig hang, dropped pool]
DEFAULT: 6-card pool {0,1,3,4,5,6}, EMBED_BATCH=48. GPU 2 reserved for qwen3-14b.
MITIGATION FOR HANGS: sequential cold-load (`sequential_cold_load=true` in `embed_pool_profiles.json`) + concurrent pool warmup before measurement. Both fire from `embedder.py` / `embedder_direct.py`. Verify by grepping log for `cold-load mode: SEQUENTIAL` and per-GPU `warmup_p50=`.
DO NOT drop pool size as first response to a hang — verify the codified mitigation fired first. If it did and the rig still hung, the issue is hardware/Vulkan/Qdrant-segfault under sustained upsert load — mitigate via EMBED_BATCH reduction or inter-batch pacing, NOT pool size.
REFERENCE: MEMORY.md "EMBED POOL = 6 CARDS" 2026-04-26; "SEQUENTIAL COLD-LOAD + CONCURRENT WARMUP IS MANDATORY" 2026-04-26.

[TRIGGER: inventory, new task, new plan, gold queries, eval, draft, generate, create new]
DEFAULT: MANDATORY FIRST STEP. Run `bash D:/_gpu_rig_ai/inventory.sh`. Grep output for keywords related to the task. Paste relevant lines into the conversation BEFORE proposing a plan.
WHY: 2026-04-23 — Claude planned to draft 180–230 gold queries from scratch when 5,781 already existed in `eval/training_pairs/`. Would have wasted 6+ hours.
NO EXCEPTIONS. If you skipped it, you are violating Hard Rule #5.

[TRIGGER: embedder, ingest.py, chunker, reuse code, adjacent code, existing module, refactor]
DEFAULT: Before reusing/extending any component (`embedder.py`, `ingest.py`, `chunker_v2.py`, `retriever.py`, etc.), open and cite the proven playbook for the SCALE you are operating at.
PLAYBOOK FILES: `~/.claude/projects/D---gpu-rig-ai/memory/ingest_playbook_cbic.md` (TL;DR + "what we tried vs what failed" table); `known_good_configs.md` (RADV_DEBUG=nodcc mandatory); `project_cbic_reingest_v2.md` ("Session continuity essentials").
WHY: 2026-04-23 — Claude trusted `embedder.py`'s Ollama wiring without checking; spent hours when llama-cpp-python Vulkan was the proven recipe.
HARD RULE #8: Query-time behaviour ≠ ingest-scale behaviour. Verify coverage before reuse.

[TRIGGER: trust gate, G1, G2, G3, G4, G5, recall, refusal, gate launch, eval gate]
DEFAULT: Trust gates run SERIAL — never concurrent. Every launch MUST be prefixed by `/opt/indian-legal-ai/reingest_spec/scripts/gate_preflight.sh <gate_name>`.
WHY: G1–G5 share infrastructure (reranker:9085, qwen3:9082 single-slot). Concurrent gates → reranker timeouts → contaminated results → ~45 min wasted reruns. Codified 2026-04-26 CP-1 incident.
DO NOT rationalize ("but G3 is retrieval-only" / "but G4 uses different model") — the rule is the rule. Hard Rule #10.

[TRIGGER: 95%, threshold, accuracy, trust criterion, adjusted recall, fake pass]
DEFAULT: 95% on all four gates G1/G2/G3/G4 on FULL gold. No "adjusted recall" carve-outs. No "patch-and-continue" on a <95% gate.
WHY: Independent audit will rerun unadjusted gates. Selection bias = fake pass. Hard Rule #1.
ESCALATION: <95% → HALT, fix spec, re-run. Do NOT proceed with known defects. Hard Rule #2.

[TRIGGER: parallelize, parallel, serial loop, throughput, GPU idle, resource utilization]
DEFAULT: Default to parallelism. Before launching any `for q in queries` serial loop, justify why parallel is unsafe.
HARD EXCEPTIONS: cold-load ≤2 cards concurrent (now sequential per pool config); G2 judges must not contend with /retrieve; trust gates serial.
PRE-LAUNCH CHECKLIST (apply BEFORE launching, not after RG asks): (1) Can N steps run as N parallel? (2) Any GPU idle while job runs? (3) Recompute vs cache? (4) Multi-category serial → parallel?
WHY: Hard Rule #9. If RG asks "are we maximizing resources?" you missed something — fix it without being asked twice.

[TRIGGER: proxy, LiteLLM, gateway, 4444]
DEFAULT: All app→LLM calls go DIRECT to llama-server (`http://127.0.0.1:9082` for qwen3, `9085` for reranker). Never through LiteLLM gateway (`:4444`).
WHY: 2026-04-26 — RG explicitly forbade proxies; codified rule. `rag/cbic_rag/api.py` defaults `LLM_DIRECT_URL=http://127.0.0.1:9082`.
TO BYPASS FOR TESTING ONLY: set `LLM_DIRECT_URL=$LITELLM_URL`.

[TRIGGER: deletion, delete, rm, drop collection, cleanup]
DEFAULT: Never delete during project. Accumulate to `cleanup_backlog.md` for batch review at project end. Hard Rule #3.
EXCEPTION: explicit RG instruction in the chat for a specific path/collection.

[TRIGGER: power_dpm, GPU power, low power, slow inference, qwen3 slow, performance level, dpm=auto, embed pool slow, retrieve slow]
DEFAULT: GPU power state must be `high` BEFORE any model loads. Enforced via systemd ExecStartPre drop-ins at `/etc/systemd/system/{qwen3-14b,bge-reranker}.service.d/power_dpm.conf` (cards 0 + 2 only).
VERIFY: `for c in 0 1 2 3 4 5 6; do printf "card%s: " $c; cat /sys/class/drm/card$c/device/power_dpm_force_performance_level; done` → all `high`.
WHY: 2026-05-07 — 300× qwen3 slowdown root-caused to GPU loading in low-power state.
GAP CODIFIED 2026-05-08: embed pool cards (1,3,4,5,6) revert to `dpm=auto` AFTER ingest_v2 exits — the embed pool init forces them high during pool warmup, but they drift back to auto when pool releases. Subsequent /retrieve queries via API run at low power → 10-15s/query latency, reranker timeouts at 30s under any concurrency.
FIX (manual until codified to systemd): after any ingest_v2 run, before any G1/G2/G3 launch, run:
  `for c in 1 3 4 5 6; do echo high | sudo tee /sys/class/drm/card$c/device/power_dpm_force_performance_level >/dev/null; done`
PERMANENT FIX (TODO): add ExecStartPre on `cbic-rag-api.service` that forces all embed pool cards to high.
RULE: any new GPU-using systemd service MUST have equivalent ExecStartPre. Any service that USES (not loads) GPUs via a pool ALSO needs power-state assertion at start.

[TRIGGER: point_id, hash, deterministic, duplicate points, re-embed, re-ingest]
DEFAULT: point_id MUST be SHA256-derived from `(doc_id, page, char_start)`. Python `hash()` is per-process random — produces duplicates on re-embed.
LOCATION: `rag/cbic_rag/ingest.py` `upsert_chunks()` lines ~163–164.
WHY: 2026-05-08 — Python hash() created 3.4% duplicate points after 8 min of re-embed; SHA256 fix is idempotent on re-touch.

[TRIGGER: wait=False, upsert, ghost docs, missing chunks]
DEFAULT: Qdrant upsert MUST use `wait=True`. With `wait=False`, manifest commit can outrun Qdrant write → ghost docs (manifest says upserted=1 but point not in Qdrant).
LOCATION: `rag/cbic_rag/ingest.py` `qc.upsert(QCOLL, points=points, wait=True)`.
WHY: 2026-05-08 — 2,393 doc_ids reported upserted=1 but missing from cbic_v2.

[TRIGGER: DENSE_ONLY, sparse, BM25, hybrid retrieval]
DEFAULT: DENSE_ONLY=1 must be set at INGEST AND at API service env. Setting only at ingest leaves sparse vectors empty in payload but API still queries them → hybrid scoring drops dense rank.
SPARSE BACKFILL: `add_sparse_v2.py` (Qdrant scroll + update_vectors, ~100s for 38K chunks; idempotent).
WHY: 2026-05-08 — sparse vectors empty in cbic_v2 because DENSE_ONLY scoped only to ingest, not API.

[TRIGGER: G2 judge, dual judge, Claude API, Anthropic key, claude CLI, paid analysis, paid external, anthropic billing, gemini api cost, external LLM cost]
DEFAULT: ALL paid external-LLM analysis MUST go through `claude` CLI on the rig (`/usr/bin/claude`, RG's subscription, no per-call cost). NOT Anthropic API key, NOT a paid Gemini key when the same job can run via Claude CLI.
SCOPE (extended 2026-05-09 from G2-only to ALL paid analysis):
  - G2 dual-judge (was the original case): `evaluators/gate_g2_dual_judge.py` `judge_claude_cli()`, `CLAUDE_USE_CLI=1`
  - Synthetic Q-A generation (replace Gemini API path): use Claude CLI subprocess instead
  - Adversarial pair generation: use Claude CLI
  - Hard-negative labeling: use Claude CLI
  - Any consultant-style "rewrite this query / rate this answer" call: use Claude CLI
  - EXCEPTION: free/local models (qwen3-14b on the rig, gemini-2.5-flash via the bundled free quota) are still allowed when faster or higher-quality. The rule is "never spend money on Anthropic API key when CLI is available."
USAGE: `echo "<prompt>" | claude -p --output-format text` (single-shot, no session). For JSON outputs ask the prompt to emit JSON; CLI honours that.
AUTH NOTE: Claude CLI auth on the rig may expire — verify with a smoke test before any batch job. Re-auth requires interactive `claude` then `/login` (RG action). Codified 2026-05-09 after smoke test on rig returned 401.
WHY: 2026-05-08 — earlier evaluator runs used Anthropic API key wrongly. 2026-05-09 — RG generalized: any paid-analysis path must default to CLI, not paid API.

[TRIGGER: SMB workflow, smb mount, /mnt/d, large file install, package download, wheel install, byte corruption workaround]
DEFAULT (codified 2026-05-09): the rig has demonstrated silent bit-flip corruption on any large transfer landing on local disk. Workaround: ALL large package/file operations MUST originate from the SMB share at `/mnt/d` (mounted from Windows `//192.168.1.222/projects` = `D:/`).
WORKFLOW for installing any non-trivial package on rig:
  1. Download package to Windows (clean network, clean RAM): `D:/<some-path>/<file>`
  2. Verify integrity on Windows side (sha256 + zip integrity)
  3. On rig: read via SMB mount `/mnt/d/<some-path>/<file>` (CIFS chunked transport is verified clean)
  4. For pip wheels: `pip install /mnt/d/path/wheel.whl` (works for files <150MB internal members)
  5. For wheels with large internal files (CUDA libs, triton, BGE-M3): EXTRACT on Windows side, then per-file copy with sha256 retry via `copy_wheels_verified.py`
WHY: 2026-05-09 — direct internet download, SCP, Docker pull, apt install, and pip install all corrupt files >150MB on this rig. SMB read of a verified-clean file produces correct bytes (CIFS retransmits corrupt chunks at protocol level). Per-file sha256 retry workaround is the only reliable install path until hardware is fixed.
HARD RULE: do NOT attempt direct download (curl/wget/aria2c/pip/docker/apt) of any package >150MB on this rig. Use Windows-side download → SMB-share → verified-copy workflow.

[TRIGGER: rig hardware reliability, byte corruption, large download, hash mismatch, pip install ROCm, pytorch install fails, network corruption]
DEFAULT: This rig has DEMONSTRATED bit-level corruption on large file transfers (≥1GB) — verified 2026-05-09 across 3 separate downloads of PyTorch ROCm wheels. Same size, different hashes, zip-integrity test fails on different inner files each time.
SYMPTOMS: pip "hashes don't match" errors; aria2c integrity-checked download still produces structurally-broken zip; dmesg shows NO errors (TCP checksum is 16-bit, random low-grade corruption passes through).
ROOT CAUSE (hypothesis): bad RAM, bad PCIe link, or DMA corruption between NIC and disk. NOT software-fixable.
RULE: do NOT attempt large package installs on this rig (PyTorch+ROCm is the canonical case). Train on cloud (codified default) — `[TRIGGER: fine-tune]`.
COMBINED EVIDENCE: this rig hung 4× during 2026-05-08 re-embed AND corrupts ≥1GB downloads — production training on this hardware would silently corrupt model weights mid-training. The rig is fine for INFERENCE (Vulkan path is working with all 7 GPUs at high power) but NOT for training.
PERMANENT FIX TODO: memtest86+ overnight; PCIe link width check; consider replacing the motherboard/RAM if memtest fails.
POST-REBOOT VERIFIED 2026-05-09: corruption persists across full reboot. Same 4GB wheel downloaded with aria2c (16-conn, 20 MiB/s, fresh kernel state) → different sha256 each attempt → zip integrity fails on different inner file each time. Confirmed hardware fault, not state issue. NOT fixable by retry/reboot/sysfs reset/PCI rescan.

[TRIGGER: training resume, checkpoint, save_steps, reboot, hang during train, training survives, persistent checkpoint]
DEFAULT: ANY long training job on this rig MUST be resumable across reboots. Rig has hung 4× in a single session (2026-05-08) — multi-hour training without checkpointing = total work loss.
REQUIRED MECHANISMS:
  - HF Trainer `save_strategy="steps"`, `save_steps=200` (~3-5 min on rig card 2)
  - `save_total_limit=3` (keeps last 3 checkpoint dirs, ~12GB)
  - `output_dir` MUST be on persistent disk (`/opt/indian-legal-ai/models/...`), never /tmp
  - At launcher start: `trainer.train(resume_from_checkpoint=True)` (HF auto-finds latest)
  - systemd service with `Restart=on-failure` + `RestartSec=30` so reboot auto-resumes
  - Logs append to `/opt/indian-legal-ai/logs/<job>.log` (not journalctl-only)
  - Pre-launch disk-space assertion: >=30GB free in models dir
  - Idempotency: if final model exists, service exits without re-training
WHY: 2026-05-08 — RG flagged "make sure model doesn't lose training if rig reboots" before launching first fine-tune. Codified as standing requirement for ANY future training job.

[TRIGGER: fine-tune, BGE-M3, embedder fine-tune, training pairs, sentence-transformers, MultipleNegativesRankingLoss, hard negatives]
DEFAULT: BGE-M3 dense head can be fine-tuned via sentence-transformers + MultipleNegativesRankingLoss + mined hard negatives. Vulkan/AMD does NOT support training — only inference. Train on cloud (A100/L40, ~$1-2/hr), ship weights back to rig for Vulkan inference.
DATA INVENTORY (codified 2026-05-08): `~/eval/training_pairs/` has ~5,500 curated Q-chunk pairs (qa_gemini=2559, pairs_2000=1909, qa_sonnet_high=426, pairs_opus_highcomplex=213, qa_claude_opus=152, pairs_claude_opus=76, plus smaller files). Plus 126K synthetic pairs in `cbic_v2.embed_text` from 2026-05-08 Gemini enrichment (42,153 × 3 Qs).
HARD NEG MINING: `D:/_gpu_rig_ai/scripts/mine_hard_negatives.py` is the proven script.
EVAL PROTOCOL: train, embed into NEW collection `cbic_v3_ft` (do NOT touch cbic_v2 — that's the rollback), re-run G1 on UNCHANGED 380-query SPEC gold (`v2_gold.json`). Gold MUST NOT leak into training.
RISK: domain-specialized fine-tune may hurt cross-domain queries (FEMA, NDPS allied acts). Mitigation: low LR, few epochs; measure on held-out adversarials before committing.
WHY: 2026-05-08 — G1 stuck at 0.8421 / 380 queries after enrichment + hybrid + linked backfill. Standard retrieval levers exhausted. Fine-tune is the next unexhausted lever per consultant brief.

[TRIGGER: G1 push order, fine-tune order, clean re-embed first, mandatory first step, hardware-test before train]
DEFAULT (locked 2026-05-09 by 3-way consultant convergence): execution order to close G1 0.84 → 0.95:
  STEP 1 (MANDATORY FIRST): clean cloud re-embed (~$1, ~2 hrs). Until hardware contamination of existing embeddings is ruled out, no diagnosis is trustworthy. If G1 jumps to 0.88+, that's the gap and we save engineering days.
  STEP 2: fine-tune Phase A (BGE-M3 dense head, MNRL + hard negatives from cbic_v2 top-K, 5,500 curated pairs, ~$5 cloud, ~1 day end-to-end).
  STEP 3 (HOLD): multi-granularity only if miss diagnosis proves >50% structural-boundary failures.
CONDITIONAL MATRIX (apply once miss diagnosis lands):
  - Query-length correlated misses → Fine-tune Phase A
  - Hierarchical Acts miss, Notifications/Circulars OK → Multi-granularity
  - Sibling notifications fighting top-5 → Hard-negative fine-tune
  - Cross-statute confusion → Query routing
  - Procedural phrase / amendment → Adjust sparse RRF weight
DO NOT REORDER without new evidence. RG asked all 3 consultants whether to redirect from fine-tune to multi-granularity; all 3 said no.

[TRIGGER: multi-granularity, multi-scale RAG, parent-child retrieval, RAPTOR, hierarchical chunking, multi-index, granularity ensemble]
DEFAULT: For the CBIC G1=0.84 block specifically, multi-granularity is SECONDARY POLISH (+1-4 pts max), NOT the primary lever. Three independent consultants on 2026-05-09 converged: dominant miss mode is long-narrative→formal-statute semantic mismatch; granularity doesn't reshape embedding geometry.
PRIMARY LEVER: fine-tune BGE-M3 dense head (MNRL + hard negatives mined from current Qdrant top-K, 5,500 curated Q-chunk pairs).
ORDERING: (1) diagnose misses first — let pattern dictate; (2) clean cloud re-embed (~$1) to rule out hardware contamination of existing embeddings; (3) fine-tune Phase A (~$5); (4) ONLY THEN consider multi-granularity if specific patterns warrant (e.g., heavy sub-clause/hierarchy misses in long Acts).
DO NOT REDIRECT FROM FINE-TUNE TO MULTI-GRANULARITY without evidence in the miss diagnosis. Synthetic Q-A enrichment already failed via a related zero-shot mechanism — supervised adaptation is required.
WHY: 2026-05-09 — RG asked all three external consultants whether multi-granularity could solve the G1 block specifically. All three said no. Codified to prevent re-debate.

[TRIGGER: HyDE, query rewrite, hypothetical, query expansion]
DEFAULT: HyDE alone HURTS recall on long-scenario queries (verified 2026-05-08: 0.7585 → 0.7542). Use sparse+dense hybrid first (gained 10pts: 0.7585 → 0.8602). HyDE+hybrid was a wash.
LOCATION: `rag/cbic_rag/hyde.py` (extraction-style, with `/no_think`).
DECISION: prefer hybrid before HyDE; consider synthetic Q-A enrichment of chunks (per 2026-05-08 consultant brief).

[TRIGGER: lint, post_batch_lint, D-DEFECT, carveout, zero-chunk]
DEFAULT: `post_batch_lint.py` runtime-classifies zero-chunk docs into D-2a (no PDF), D-2b (junk content <500 chars), D-1 (shared-PDF cluster). Static carveout file is fast pre-filter; runtime fallback handles new edge cases.
EXIT CODES: P0=2 (HALT), P1=1 (HALT historically; now run_batch_loop.sh halts only on P0), 0=clean.
LOCATION: `reingest_spec/scripts/post_batch_lint.py` lines 103–141.

[TRIGGER: build_batch, manifest path, ingest_manifest_v2]
DEFAULT: Manifest path is `/opt/indian-legal-ai/data/ingest_manifest_v2.sqlite` (no `_full` suffix). All sibling scripts MUST resolve through the same constant.
WHY: 2026-05-07 — `build_batch.py` had stale `_full.sqlite` path → empty already-ingested set → CSV overlap accumulated 142→742 over 8 batches.

[TRIGGER: qdrant client, qc.search, query_points, fallback, dead code, AttributeError]
DEFAULT: `qc.search()` was REMOVED in qdrant-client >= 1.10. Use `qc.query_points(query=vec, using='dense', limit=k, query_filter=qfilter, with_payload=True).points` for dense-only queries.
WHY: 2026-05-08 — `retriever.py` had `qc.search()` as except-clause fallback. Never fired before because hybrid path always succeeded. Today Qdrant segfaulted under concurrent G1 load → exception bubbled to fallback → AttributeError → 500 error masked the real "Qdrant down" cause. Patched 2026-05-08.
HARD RULE: every `except:` fallback must use the SAME library API as the success path. Untested fallback = dead code masking the real failure.

[TRIGGER: G1 launch, gate workers, parallel workers, concurrent retrieve, reranker timeout]
DEFAULT: For G1/any gate that hits /retrieve at scale, START with `--workers 1` if Qdrant has just been re-ingested or shows status=red/grey. Concurrent /retrieve fires N parallel hybrid searches → reranker queue saturates → timeouts → AND can re-segfault Qdrant if a corrupt segment exists.
ESCALATE TO `--workers 4-8` ONLY after a full serial pass succeeds.
WHY: 2026-05-08 — G1 launched with default workers=8 against status=red cbic_v2 → Qdrant segfaulted (3rd restart) → 380/380 errors. Codified RERANK_TIMEOUT=30s helps but doesn't fix the segfault re-trigger.

[TRIGGER: gate result, output path, PermissionError, /opt write]
DEFAULT: Write gate evaluator outputs to `/tmp/<gate>_result.json` or pass `--out /tmp/...` explicitly. Files in `/opt/indian-legal-ai/reingest_spec/evaluators/` may have been created by sudo — non-root can't overwrite.
WHY: 2026-05-08 — gate_g1_recall.py default `--out` path was `/opt/.../gate_g1_result.json`, owned by root → PermissionError after a 5-min eval run.

[TRIGGER: SSH, rig connection, control master, mux]
DEFAULT: Use `ssh -o ControlMaster=no -o ControlPath=none rig` when prior mux sockets may be stale (after rig reset). Clean stale sockets: `rm -f ~/.ssh/cm-*`.

[TRIGGER: qdrant status, grey, red, yellow, segfault, optimizer_status, indexing_threshold]
DEFAULT: Qdrant collection `status` field has 4 states. DO NOT block on `green`:
  - `green` = all segments ≥ indexing_threshold (10K) are HNSW-indexed
  - `grey` = some segments below threshold use brute-force search; HEALTHY, reads/writes work
  - `yellow` = optimizing in progress
  - `red` = error state — check `optimizer_status` field; investigate
HEALTH CHECK: use `optimizer_status: ok`, NOT `status: green`. A `grey` collection is fully usable for upsert + query.
SEGFAULT RECOVERY: Qdrant can segfault under sustained upsert load (observed 2026-05-08, after ~30K cumulative upserts via 6-card pool). Docker auto-restarts the container; collection recovers from WAL. Deterministic SHA256 point_ids (see `[TRIGGER: point_id]`) make re-upsert idempotent — no duplicates.
MITIGATION FOR REPEATED SEGFAULTS: lower EMBED_BATCH (48 → 24), keep pool at codified 6 cards.

[TRIGGER: reactive overcorrection, dropped pool, drop concurrency, hang reaction]
DEFAULT: When a hang/crash happens, the FIRST response is NOT to lower concurrency. The first response is to verify the codified mitigation (sequential cold-load + warmup) actually fired — grep log for `cold-load mode: SEQUENTIAL` and per-GPU `warmup_p50=`. If those fired and rig still hung, it's downstream (Qdrant segfault, EMBED_BATCH too aggressive, hardware), NOT pool size.
WHY: 2026-05-08 — after 3 rig hangs, I dropped 6→4 cards reactively without checking whether the codified mitigation had fired. It HAD. The hangs were caused by Qdrant segfaults under upsert load + EMBED_BATCH=48; pool size was the wrong lever. Cost: half a day of debug + thrashing.
HARD RULE: cite log evidence of mitigation firing BEFORE proposing any lever change. Self-Audit (c) must include the grep output.

[TRIGGER: scheduling, cron, scheduled task]
DEFAULT: User-requested scheduled tasks go in CronCreate or scheduled-tasks MCP. Do not invent autonomous loops without RG approval.

---

## Self-Audit Template (MANDATORY before architectural decisions)

Fill verbatim in your response before pool/chunker/threshold/model changes:

```
## Self-Audit for [DECISION]
(a) Codified default (quote RULES_INDEX block verbatim):
> [paste]

(b) Codified mitigation for the failure mode I'm seeing (quote verbatim):
> [paste]

(c) Why the codified mitigation does NOT fit this case (evidence required, with log lines / metrics — or "N/A — following codified default"):
[reason]

(d) Proposed change + expected impact:
[change + metric you expect to move]
```

If you cannot fill (a) and (b) by quoting from RULES_INDEX.md, you have not consulted the rules. Run `rule_check.sh` first.

---

## Maintenance

- This file MUST stay under 200 lines.
- New rules: add a `[TRIGGER: ...]` block. Keep blocks ≤ 6 lines.
- Incident postmortems: do NOT add here — go to `INCIDENTS_ARCHIVE.md`.
- Periodic prune: every ~10 incidents, audit for stale or superseded rules.
