# BACKLOG — Parked Improvements (pick up after CBIC v2 re-ingest)

This is the deferred-work registry. Every item here is something we've thought about, scoped, or partially designed but consciously parked because it would digress from the current critical path. **Do not start any item here until the CBIC v2 full re-ingest is complete and at 95% on every gate.**

When picking up an item: read its full block here (not just the title), update CLAUDE.md if it changes the workflow, then move to in-progress in `JOURNAL.md`.

---

## B1 — Generic Rig LLM/GPU Loader (parked 2026-04-25)

### Why we want it
Every project (CBIC RAG, OpenClaw, LiteLLM gateway, Ritu's job agent, future ones) keeps re-deriving the same per-GPU recipe. Each has its own:
- Path to the model file
- llama-server flags
- GPU pinning via `GGML_VK_VISIBLE_DEVICES`
- Mandatory env (`RADV_DEBUG=nodcc`, `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1`)
- systemd unit
- Health check

When a new model arrives we re-derive it. When a card moves we re-derive it. When a flag-bug is discovered (e.g. `--flash-attn` broken on RADV) the fix lives in one project's memory and the others rediscover the burn.

User directive 2026-04-25 (verbatim): *"any project and any llm, the script should be a generic script that loads the model from the other place we maintain the table of llm gpu and how to load it. so that we can work with these gpu with no pain in future"*

### Proposed design

**1. Single canonical manifest:** `D:/_gpu_rig_ai/rig_models.yaml`

```yaml
version: 1
host: rig (192.168.1.107)

# GPU table — physical truth (mirrors known_good_configs.md)
gpus:
  0: {name: "RX 5700 XT (ASUS)",     pci: "03:00.0", vram_mb: 8176,  bar: full,  arch: RDNA1}
  1: {name: "RX 5700 XT (Dell)",     pci: "06:00.0", vram_mb: 8176,  bar: full,  arch: RDNA1}
  2: {name: "RX 6700 XT (ASRock)",   pci: "09:00.0", vram_mb: 12272, bar: full,  arch: RDNA2}
  3: {name: "RX 5700 XT (Sapphire)", pci: "0f:00.0", vram_mb: 8192,  bar: 256MB, arch: RDNA1}
  4: {name: "RX 5700 XT (PwrColor)", pci: "14:00.0", vram_mb: 8192,  bar: 256MB, arch: RDNA1}
  5: {name: "RX 5700 XT (ASRock)",   pci: "17:00.0", vram_mb: 8192,  bar: 256MB, arch: RDNA1}
  6: {name: "RX 5700 XT (ASRock)",   pci: "1a:00.0", vram_mb: 8192,  bar: 256MB, arch: RDNA1}

mandatory_env:
  RADV_DEBUG: nodcc
  GGML_VK_DISABLE_INTEGER_DOT_PRODUCT: "1"

# Hard rules — codified, never violated
rules:
  - never_per_card_pcie_reset_more_than_one
  - never_concurrent_cold_load_more_than_two
  - sequential_cold_load_default_true
  - never_set_flash_attn_on_radv
  - mlock_forbidden
  - cache_type_k_v_q8_forbidden_radv

# Models — every LLM that runs on this rig
models:
  qwen3-14b:
    path: /opt/ai-models/qwen3-14b-q4_k_m.gguf
    role: chat-completion
    runtime: llama-server
    binary: /opt/llama-server-b8840/llama-server
    default_gpu: 2
    port: 9082
    vram_required_mb: 9000
    args: [-ngl, 99, -c, 16384, --parallel, 1, --mmap, -b, 2048, -ub, 512, -t, 4,
           --jinja, --chat-template-file, /opt/chat-templates/qwen3-unsloth.jinja]
    systemd_unit: qwen3-14b.service
    health_check: "curl -sf http://127.0.0.1:9082/health"

  bge-m3:
    path: /usr/share/ollama/.ollama/models/blobs/sha256-daec91ffb5dd0c27411bd71f29932917c49cf529a641d0168496c3a501e3062c
    role: embedder-dense-1024
    runtime: llama-cpp-python-pool
    default_gpus: [0, 1, 3, 4, 5, 6]
    profiles_file: /opt/indian-legal-ai/embed_pool_profiles.json
    sequential_cold_load: true

  bge-reranker-v2-m3:
    path: /opt/indian-legal-ai/models/bge-reranker-v2-m3-Q4_K_M.gguf
    role: reranker
    runtime: llama-cpp-python
    # default_gpu: TBD post-reranker integration

  # also: gemma-4-e4b, mistral-nemo-12b, qwen2.5-coder-7b, qwen3-8b,
  #       qwen3.5-4b/9b, llama-3.1-8b, qwen2.5-1.5b, bge-small-en-v1.5
  # populate from MEMORY.md "MODEL PATHS ON RIG" block
```

**2. Generic CLI:** `D:/_gpu_rig_ai/rig.py` deployed as `/usr/local/bin/rig` on the rig

```
rig list                                 # show models + status (running/stopped)
rig load qwen3-14b                       # start on default GPU
rig load qwen3-14b --gpu 2 --port 9082   # explicit override
rig unload qwen3-14b                     # stop systemd or kill process
rig pool start bge-m3 --gpus 0,1,3,4,5,6 # honors sequential_cold_load
rig pool stop bge-m3
rig pool stats bge-m3                    # hits /admin/embed_pool
rig status                               # what's running where, per-GPU VRAM
rig swap qwen3-14b bge-m3 --gpu 2        # graceful swap (phase 3-4-5 use case)
rig validate                             # rules check: cold-load count, flag bans
```

**3. Project integration:** Each project imports the loader instead of rolling its own. `cbic-rag-api.service` would be `ExecStart=rig pool start bge-m3 ... && exec uvicorn ...`. New projects don't reinvent.

**4. Memory canonization:** Update `CLAUDE.md` to add: *"Before launching any LLM/embedder, consult `rig_models.yaml`. Never inline model paths or flags in project code."*

**5. Refactor:** Move `embed_pool_profiles.json` content INTO the manifest as `models.bge-m3.profiles_per_gpu` so there's one file, not two.

### Estimated scope
- `rig_models.yaml`: ~250 lines
- `rig.py`: ~400 lines Python (subprocess + systemd + curl health)
- Refactor to point at manifest: small

### Source material to copy from when starting
- `~/.claude/projects/D---gpu-rig-ai/memory/known_good_configs.md` — qwen3-14b flags, RADV rules, flag bans
- `~/.claude/projects/D---gpu-rig-ai/memory/MEMORY.md` "MODEL PATHS ON RIG" block — all 9 model paths
- `D:/_gpu_rig_ai/embed_pool_profiles.json` — current per-GPU embed config (move to manifest)
- `D:/_gpu_rig_ai/embedder_direct.py` `_Pool` — sequential_cold_load implementation reference
- `/etc/systemd/system/qwen3-14b.service` — proven flag set

### Pre-conditions before pickup
- CBIC v2 full re-ingest complete (all 5 gates ≥95%)
- No active production load on the rig (so we can break the embed pool restart-loop while developing)

### Why parked
Building a generic loader now would digress from the critical path: **finishing the CBIC v2 re-ingest blockers (reranker, Defect D, phase 6 patches, gate concurrency).** The current per-project loaders work; the generic loader is a quality-of-life refactor, not a blocker. Pick up after the v2 re-ingest is at 95% on every gate.

---

## B2 — Weighted-Deficit Scheduler for Embed Pool (parked 2026-04-25)

### Why parked
Current `embedder_direct.py` uses equal-weight round-robin. That works perfectly when all GPUs are uniform-throughput (proven 2026-04-25: 166-167 calls each across 6 GPUs in 1000-call burst, p50 40-44ms ±4ms).

The reason `rebalance_after_warmup: false` is currently codified: warmup_p50 reflects **cold-vs-warm shader cache** state, NOT steady-state throughput. When we tried to use warmup latency as weights, GPU 3 got weight 2.04 vs others 0.07, the `top_tier` filter `abs(s - top_score) < 0.01` left only GPU 3 in the pool, and 600 calls all went to GPU 3.

### What needs building
- Track running latency p50 over last N=200 calls per GPU
- Replace equal-RR `_pick_for_retrieve` with **deficit round-robin** weighted by `1/p50`
- Weights recomputed every 60s based on rolling steady-state, not warmup
- Re-enable `rebalance_after_warmup: true` only after this lands

### Source material
- `embedder_direct.py` `_pick_for_retrieve` (current equal-RR)
- `embedder_direct.py` `_rebalance_weights_from_warmup` (broken — feeds warmup data)

### When to revisit
After CBIC v2 re-ingest. Today's 6-card uniform-RX-5700-XT pool means equal-RR is empirically optimal. If/when we mix GPU 2 (RX 6700 XT, ~1.5× faster) into the embed pool, weighted-deficit becomes useful.

---

## B3 — Investigation: Why does GPU 6 take 2.7s solo but caused 300s timeout in 6-card concurrent? (parked 2026-04-25)

### Status
Empirical answer is sufficient (concurrent Vulkan init contention). Sequential cold-load eliminates the problem. No further investigation needed for production. **Park indefinitely** unless we see a different concurrent-init failure mode in another workload.

---
