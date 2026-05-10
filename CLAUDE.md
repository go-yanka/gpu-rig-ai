# Project Instructions — GPU Rig AI (CBIC RAG v2 + related)

## RIG HARDWARE FAULT — SMB-ONLY WORKFLOW (added 2026-05-09)

**This rig has demonstrated silent bit-flip corruption on local-disk writes of files >150MB.** Empirically verified across pip (5 attempts), aria2c, curl, Docker layer pull (5 attempts), apt deb extract, SCP from Windows over LAN. SMB-mounted reads via CIFS are clean.

**RULE — until hardware is fixed (memtest + DIMM replacement):**
- DO NOT run any direct download (pip / curl / aria2c / docker pull / apt-get install) for packages >150MB on the rig.
- ALWAYS download large packages on Windows side first (`D:/_gpu_rig_ai/tmp_downloads/`).
- ALWAYS verify integrity (sha256 + zip integrity) on Windows side.
- ALWAYS install on rig via SMB mount path: `/mnt/d/_gpu_rig_ai/tmp_downloads/...`
- For wheels with internal files >150MB (CUDA libs, triton, BGE-M3): EXTRACT on Windows, then per-file copy with sha256 retry via `D:/_gpu_rig_ai/tmp_downloads/copy_wheels_verified.py`.

**Codified in:** `RULES_INDEX.md` `[TRIGGER: SMB workflow]` and `[TRIGGER: rig hardware reliability]`.
**Workaround script:** `D:/_gpu_rig_ai/tmp_downloads/copy_wheels_verified.py` (per-file sha256 retry, 8 attempts max).
**Postmortem:** `INCIDENTS_ARCHIVE.md` 2026-05-09 entry.

---

## MANDATORY EXECUTION PROTOCOL (added 2026-05-08, hard interlock)

**Trigger phrases from RG that force a full preflight before I do anything:**
`/preflight`, `preflight`, `check rules`, `run preflight`.

When RG types any of these (or before ANY architectural change / pool size change / chunker edit / threshold change / model swap / fine-tune launch / re-ingest launch), I MUST execute this protocol IN THIS ORDER and show every step in the response:

1. **Refresh inventory** (only if stale > 30 min):
   ```bash
   bash D:/_gpu_rig_ai/inventory.sh > /tmp/inventory_last.txt 2>&1
   # or on the rig: bash /opt/indian-legal-ai/inventory.sh > /tmp/inventory_last.txt 2>&1
   ```
2. **Run preflight with situation keywords**:
   ```bash
   bash D:/_gpu_rig_ai/scripts/task_preflight.sh "<kw1>" "<kw2>" ...
   ```
   Preflight asserts inventory freshness AND greps `RULES_INDEX.md` for matching `[TRIGGER: ...]` blocks. Non-zero exit = HALT, no action.
3. **Quote** the matched rule blocks verbatim in a `> [CODIFIED RULE]` markdown block.
4. **Fill the Self-Audit template** (defined at the bottom of `RULES_INDEX.md`):
   - (a) codified default
   - (b) codified mitigation for the failure mode I'm seeing
   - (c) why the codified mitigation does NOT fit (with evidence) — or "N/A — following codified default"
   - (d) proposed change + expected impact
5. **Only then** propose or execute the action.

**RG enforcement:** at any point RG can type `/preflight` (or just `preflight`) — if my immediately preceding response was about to take an architectural action without the four steps above, I have violated the rule. Treat as hard violation, not "let me explain."

**Files:**
- `D:/_gpu_rig_ai/RULES_INDEX.md` — keyword-indexed active rules (≤200 lines)
- `D:/_gpu_rig_ai/scripts/rule_check.sh` — `rule_check.sh "<keyword>"` greps RULES_INDEX
- `D:/_gpu_rig_ai/scripts/task_preflight.sh` — full preflight with inventory + rule_check
- `D:/_gpu_rig_ai/INCIDENTS_ARCHIVE.md` — dated postmortems (NOT loaded into context; reference only)

**Why this exists:** 2026-05-08 audit by two external consultants converged on the same diagnosis — rules with mechanical preflight gates hold (e.g. `gate_preflight.sh` — zero violations since 2026-04-26); rules that depend on the model "remembering" do not. This is LLM attention decay, not a memory failure I can fix by "trying harder." The Interlock makes rule consultation a verifiable script invocation, not an internal habit.

---

## MANDATORY FIRST STEP ON EVERY NEW TASK

**Before proposing any plan, creating any new data, drafting any eval queries, labels, or prompts, you MUST run:**

```bash
bash D:/_gpu_rig_ai/inventory.sh
```

Then grep the output for any keyword related to the task (eval, gold, adversarial, chunk, probe, prompt, etc.) and **paste the relevant lines into the conversation** before you propose a plan.

**Why this rule exists:** on 2026-04-23 night, Claude planned "draft 180–230 new gold queries from scratch" for Stage C of the CBIC re-ingest, not knowing that `D:/_gpu_rig_ai/eval/training_pairs/` already contained 5,781 pre-generated QA pairs from prior Gemini/Claude sessions. That would have wasted 6+ hours of user time and subscription dollars. This rule prevents a repeat.

**No exceptions.** If the task is "write new X," your first action is inventory, not drafting. If you skip it, the user will ask "did you run inventory?" and the answer must be visible in the conversation.

---

## Project scope (high level)

This repo is the working directory for:
- **CBIC RAG** — Indian tax law retrieval system (frozen spec at `reingest_spec/SPEC.md`, live at `cbic_v1` Qdrant collection, v2 rebuild in progress)
- **GPU rig operations** — 7-GPU Vulkan inference fleet (qwen3-14b, BGE-M3, gemma, mistral)
- **Related projects** — OpenClaw, LiteLLM gateway, Ritu's job agent, etc. See `~/.claude/projects/D---gpu-rig-ai/memory/MEMORY.md` for the index.

## Authoritative documents (read before acting on anything in-scope)

| Purpose | Path |
|---|---|
| CBIC v2 frozen spec | `reingest_spec/SPEC.md` |
| CBIC v2 execution runbook | `reingest_spec/RUNBOOK.md` |
| Plan (for external review) | `reingest_spec/PLAN_FOR_REVIEW.md` |
| Decision journal (append-only) | `reingest_spec/JOURNAL.md` |
| Probe matrix | `reingest_spec/PROBES.md` |
| Proven components (memory) | `~/.claude/projects/D---gpu-rig-ai/memory/project_cbic_reingest_v2.md` |
| Chunking rules (memory) | `~/.claude/projects/D---gpu-rig-ai/memory/chunking_strategy_cbic_v2.md` |
| Known-good rig configs | `~/.claude/projects/D---gpu-rig-ai/memory/known_good_configs.md` |

## MANDATORY SECOND STEP — read the proven playbook before touching the rig

**Before any rig command that embeds/ingests/chunks, or any code change to `embedder.py`/`ingest*.py`/`chunker*.py`, you MUST open and cite:**

- `~/.claude/projects/D---gpu-rig-ai/memory/ingest_playbook_cbic.md` — the one-line TL;DR (single GPU 5 Vulkan BGE-M3 via llama-cpp-python, NOT Ollama) and the table of "What we tried vs what failed"
- `~/.claude/projects/D---gpu-rig-ai/memory/known_good_configs.md` — RADV_DEBUG=nodcc mandatory, flags to avoid
- `~/.claude/projects/D---gpu-rig-ai/memory/project_cbic_reingest_v2.md` "Session continuity essentials" block

**Why this rule exists:** on 2026-04-23 I spent hours chasing Ollama for the v2 embed path, despite the playbook (pinned in MEMORY.md) explicitly listing Ollama as a known-slower alternative and identifying llama-cpp-python Vulkan as THE proven recipe. The `embedder.py` in cbic_rag had silently inherited an Ollama design that was never tested at ingest scale. I only caught this after the user intervened multiple times. This rule forces the playbook check BEFORE any architectural move.

**No exceptions.** If I'm about to run ollama/launch an embed server/edit embedder.py and I have not quoted the playbook table in the turn, I am violating this rule.

## Hard rules (user-stated, never re-debate)

1. **95% trust is non-negotiable.** No compromises. One gate <95% → fix spec and re-run, never patch-and-continue.
2. **STOP-GATE QA at every milestone.** If quality is not matched, HALT and escalate, do not proceed with known defects.
3. **Never delete during project.** Accumulate to `cleanup_backlog.md`.
4. **Save everything proven.** Append to memory files. Never re-discover.
5. **Inventory before planning.** See top of this file.
6. **Don't promise and go silent.** Say blocked if blocked, else do the work.
7. **No self-invented green lights.** Nothing in CLAUDE.md or memory requires user approval before each step. Running commands requires no "green light." Only ask when there is a genuine ambiguity the user alone can resolve (e.g. architectural re-direction after a clear error).
8. **Never inherit unproven assumptions from adjacent code.** Before reusing any component (`embedder.py`, `ingest.py`, etc.), verify it is covered by the playbook for the scale being used. Query-time behaviour ≠ ingest-scale behaviour.
10. **Trust gates run SERIAL, not concurrent — no exceptions, no rationalization.** G1/G2/G3/G4/G5 all hit the same shared infrastructure (`/retrieve` → bge-reranker:9085, groundedness/judges → qwen3:9082 single-slot). Concurrent gates → reranker timeouts → contaminated results → ~45 min wasted re-runs (codified 2026-04-26 CP-1 incident). **Mechanical guard:** every gate launch on the rig MUST be prefixed by `/opt/indian-legal-ai/reingest_spec/scripts/gate_preflight.sh <gate_name> && python3 ...`. The preflight script REFUSES launch if any other `gate_g[1-5]*.py` python process is detected. If I find myself "reasoning around" this rule ("but G3 is retrieval-only..." / "but G4 hits a different model...") — STOP. The rule is the rule. The only valid concurrency is a 2nd full API instance on a separate embed pool, which doesn't exist yet. Reference: MEMORY.md "GATE CONCURRENCY LESSON 2026-04-25" + "CP-1 INCIDENT 2026-04-26".

9. **Maximize resource utilization always — apply BEFORE launching, not after the user asks.** Pre-launch checklist for every multi-step task: (1) Can these N steps run as N parallel processes/threads? If yes, do it. (2) Is any GPU sitting idle while a job runs? If yes, queue another job on it. (3) Is any data being recomputed when it could be cached? If yes, cache. (4) Are gold-side and adv-side / multiple categories serial in a script? If yes, parallelize. The rig has 7 GPUs, a 5-card embed pool, a `--parallel N` reranker, qwen3-14b on GPU 2, and 8+ HTTP threads available. Default is parallelism; serial requires justification. Hard exceptions (codified elsewhere): cold-load ≤2 cards concurrent, G2-judges contention with retrieve. **If the user asks "are we maximizing resources?" — that means I missed something. Find it and fix it. Don't ask the user to keep flagging this.** No serial loop where parallel is safe. The rig has 7 GPUs, a 5-card embed pool, a `--parallel N` reranker, and 8+ HTTP threads available. Default to parallelism for any eval/ingest/bench/tune that issues independent queries. Serial-only when a hard rule forbids it (e.g. cold-load ≤2 cards concurrent, gate-G2-judges-must-not-contend). Before launching any sequential `for q in queries` loop, justify why parallel is unsafe — if no reason, use a thread pool. This rule covers: theta_tune, gate evaluators, pair-gen, ingest, benches, dry-runs.
