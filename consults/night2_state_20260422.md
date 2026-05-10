# Night-2 State Snapshot (2026-04-22, resume after orchestrator crash)

## Timestamp
- Resume start: 2026-04-22 ~01:05 EDT (~05:05 UTC)
- Previous orchestrator crashed ~04:45 UTC (Anthropic 529)

## Rig state (192.168.1.107)
- `cbic-rag-api` service up.
- Feature flags at resume:
  - `TWO_PASS_ENABLED=0` (from `/etc/systemd/system/cbic-rag-api.service.d/two_pass.conf`)
  - `TWO_PASS_CHUNK_REFEED` — not yet present (P1.3 not deployed)
- `eval_set.json` on rig = dict with 170 queries (version=2). Was already swapped. OK.
- P1.2 (validator simplify) deployed: backup `api.py.bak.p1_2.1776829554` exists. No two_pass sentinel changes since.
- No eval process running.

## Local state (D:\_gpu_rig_ai)
- `eval/gold_set.yaml` = 170 items (version=2).
- Latest run: `eval/runs/p1_2_postdeploy_20260422_003323/` completed.
  - **50 items, 32.29% (103/319)** — used `--limit 50`, NOT full 170.
  - **LLM judge returned None for every item** (keyword/section fallback only).
- Consults available: `p1_2_validator_simplify_patch.md`, `p1_3_a3_chunk_refeed_patch_v2.md`, `p1_3_a3_chunk_refeed_v2.patch`.
- `p1_1` retrieval patch does NOT exist → SKIP per plan.

## Root cause of judge=None (diagnosed this session)
The judge endpoint (qwen3-14b on :9082) is a `<think>`-mode model. `run_eval.py` sent `max_tokens=8`, which is fully consumed by the think tokens before any digit is emitted. Direct curl reproduces empty content. Adding `/no_think` (qwen3 directive) causes the model to answer with a single digit immediately.

**Mitigation applied:** patched `D:\_gpu_rig_ai\eval\run_eval.py` locally:
- Inject `/no_think\n` prefix in judge user message.
- Raise `max_tokens` from 8 → 16.
- Verified: `curl .../v1/chat/completions` with `/no_think` + max_tokens=8 returns `"3"` cleanly.

## Baseline / prior pass rates
- Claimed pre-P1.2 baseline: 33.23% (keyword-only; judge was non-functional same as now).
- P1.2 postdeploy: 32.29% / 50 items (keyword-only). No statistically meaningful delta.

## Implication for tonight's plan
The user's overnight plan assumes judge-meaningful numbers. With the judge now fixed:
- A fresh 170-item baseline (P1.2 state, flags OFF) is needed to re-anchor.
- Then P1.3 Eval-A (flags 1/0) and Eval-B (flags 1/1).
- Each 170-item eval ≈ 60–90 min at median 26s/item single-worker. With judge now producing real scores, we add ~1–3s judge latency per item — still within budget.

## Decisions taken this session
1. Applied local-only judge fix to run_eval.py (minor, reversible).
2. Will NOT attempt a second P1.2 170-item re-eval before P1.3 — the 50-item run already validates P1.2 deployed cleanly (no errors). Instead, will run a single 170-item baseline with `TWO_PASS_ENABLED=0` as the "true" anchor, then deploy P1.3 and eval A & B. This costs 3 evals instead of 4, keeping us in budget.
3. Cloudflare auth NOT wired (per user).

## Budget
~5h wall-clock remaining. 3 × ~75min evals = ~3.75h + deploy/verification overhead ~20min. Feasible.
