# Morning Status — 2026-04-22 (Overnight Driver Run #4)

Agent: attempt #4. Completed all 3 evals and landed final decision.

## TL;DR

- **Final flag state on rig**: `TWO_PASS_ENABLED=0`, `TWO_PASS_CHUNK_REFEED=0` (two-pass A3 fully OFF; patched code is behind flags but dormant).
- **Decision**: Two-pass A3 caused a large regression. Reverted to single-shot legacy behavior. P1.3 chunk-refeed adds no benefit over A3 alone (within ±3pp).
- **cbic-rag-api**: active, /health = 200 OK.

## Pass rates (all 170-item, judge-functional)

| Run | Flags | Score | Pass rate | Median latency | p95 |
|---|---|---:|---:|---:|---:|
| Prior claim (pre-P1.2, keyword-only) | — | — | 33.23% | — | — |
| **P1.2 baseline170** (anchor) | ENABLED=0 | 582.0 / 1439 | **40.44%** | 26.3s | 33.1s |
| **P1.3 Eval-A** | ENABLED=1, REFEED=0 | 389.0 / 1439 | **27.03%** | 31.3s | 64.1s |
| **P1.3 Eval-B** | ENABLED=1, REFEED=1 | 383.0 / 1439 | **26.62%** | 33.1s | 61.1s |

Delta vs P1.2 baseline: Eval-A **-13.41pp**, Eval-B **-13.82pp**. Both are catastrophic regressions.

Delta B-A: **-0.41pp** (within ±3pp band → per decision matrix, keep REFEED=0; code stays behind flag).

## Per-category (% correct)

| Category | N | P1.2 | Eval-A | Eval-B | A-P1.2 | B-P1.2 |
|---|---:|---:|---:|---:|---:|---:|
| central_excise | 10 | 47.73 | 31.82 | 26.14 | -15.9 | -21.6 |
| customs | 45 | 34.64 | 19.53 | 18.49 | -15.1 | -16.2 |
| gst | 81 | 44.35 | 31.42 | 31.57 | -12.9 | -12.8 |
| others | 14 | 28.10 | 21.49 | 21.49 | -6.6 | -6.6 |
| service_tax | 20 | 43.03 | 27.88 | 29.09 | -15.2 | -13.9 |

Every category regressed under two-pass. customs worst hit.

Latency also much worse: p95 roughly doubled (33s → 61–64s), median +5–7s.

## Decision + rationale

**Keep P1.2 validator-simplify deployed. Disable two-pass entirely.**

Rationale:
1. The two-pass extraction→validate→synthesis pipeline loses signal regardless of whether chunks are re-fed at pass 2.
2. Chunk re-feed (Eval-B) did not rescue pass 2; within noise of Eval-A.
3. Latency cost is unacceptable (p95 ~60s+).
4. Per the decision matrix the B−A delta (-0.41pp) falls in ±3pp → REFEED=0 default; I kept the P1.3 code in place (behind a disabled flag) so it is reversible and auditable.
5. Because P1.3_A already regresses vs baseline by >3pp, I further reverted `TWO_PASS_ENABLED` to 0 on the live drop-in. Both flags are now off. This returns the service to the P1.2-baseline configuration that scored 40.44%.

## Files deployed / changed tonight

- Rig: `/opt/indian-legal-ai/rag/cbic_rag/api.py` — P1.3 chunk-refeed patch APPLIED.
  - New env flag `TWO_PASS_CHUNK_REFEED` parsed at module load (line 79).
  - `_synthesize_pass2()` gained optional `chunks` arg + gated re-feed block (line 391ff).
  - `two_pass_generate()` now forwards `chunks` to pass-2 (line 457).
  - Flags are OFF so code path is dormant.
- Rig: `/etc/systemd/system/cbic-rag-api.service.d/two_pass.conf` — final state `ENABLED=0 REFEED=0`.
- Local: `D:\_gpu_rig_ai\eval\runs\p1_2_baseline170_20260422_050751\` — 170-item baseline (judge fixed).
- Local: `D:\_gpu_rig_ai\eval\runs\p1_3_evalA_20260422_023102\` — Eval-A.
- Local: `D:\_gpu_rig_ai\eval\runs\p1_3_evalB_20260422_041318\` — Eval-B.

Backups on rig (restore points):
- `/opt/indian-legal-ai/rag/cbic_rag/api.py.bak.chunk_refeed.20260422_022926` (pre-P1.3, identical to current P1.2 source minus patched fn signature).
- `/opt/indian-legal-ai/rag/cbic_rag/api.py.bak.p1_2.1776829554` (pre-P1.2 validator-simplify).
- `/opt/indian-legal-ai/rag/cbic_rag/storyformat.py.bak.chunk_refeed.20260422_022926` (unchanged but snapshotted).

## Rollback commands

Revert P1.3 (chunk_refeed) patch only:
```
ssh rig
sudo cp /opt/indian-legal-ai/rag/cbic_rag/api.py.bak.chunk_refeed.20260422_022926 \
        /opt/indian-legal-ai/rag/cbic_rag/api.py
sudo python3 -m py_compile /opt/indian-legal-ai/rag/cbic_rag/api.py
sudo systemctl restart cbic-rag-api
```

Revert P1.2 (validator-simplify) as well, back to pre-two-pass state:
```
ssh rig
sudo cp /opt/indian-legal-ai/rag/cbic_rag/api.py.bak.p1_2.1776829554 \
        /opt/indian-legal-ai/rag/cbic_rag/api.py
sudo python3 -m py_compile /opt/indian-legal-ai/rag/cbic_rag/api.py
sudo rm -f /etc/systemd/system/cbic-rag-api.service.d/two_pass.conf
sudo systemctl daemon-reload
sudo systemctl restart cbic-rag-api
```

Re-enable P1.3 Eval-A config (if we want to investigate further):
```
sudo tee /etc/systemd/system/cbic-rag-api.service.d/two_pass.conf <<'EOF'
[Service]
Environment=TWO_PASS_ENABLED=1
Environment=TWO_PASS_CHUNK_REFEED=0
EOF
sudo systemctl daemon-reload && sudo systemctl restart cbic-rag-api
```

## Skips / failures

- **P1.1 retrieval patch**: SKIPPED per plan — no `p1_1_*.md` consult existed on disk.
- **Patch file format**: `p1_3_a3_chunk_refeed_v2.patch` had malformed hunk headers (wrong line counts). Instead of `patch`, the edits were applied by pulling `api.py` to the laptop, editing via Read/Edit, byte-compiling locally, then scp'ing back. Result identical to the intended diff; verified with grep for sentinel lines.
- **Cloudflare auth**: DEFERRED per user. Tunnel still at https://mileage-demographic-duplicate-oven.trycloudflare.com (no auth).
- No agent/orchestrator crashes this run. No 529s. Three consecutive evals completed cleanly.

## Recommended next steps

1. **Do NOT ship two-pass in any flavor.** The design as currently specified loses >13pp across every category; chunk re-feed does not help. Suggest parking P1.3 and revisiting the extraction/validation prompts — suspect pass-1 extraction is under-recalling verbatim spans, which then makes pass-2 synthesis unable to compose.
2. **Investigate zero-point items.** Many Eval-A/B items scored `pts=0.0 judge=0`, including factual basics (gst_rate_*, gst_cess_*, gst_itc_004). Pull 5–10 such responses from `results.jsonl` and inspect the answer text vs expected_sections. Likely pass-1 is returning empty or validator is rejecting everything.
3. **Re-anchor future experiments against 40.44% (P1.2_170).** The 33.23% figure was keyword-only; it under-states true baseline. Use P1.2_170 from tonight as the canonical anchor.
4. **Judge stability**: judge scores now populate (fix from state doc). Monitor max_tokens=16 is sufficient; no truncation observed.
5. **Budget for next experiment**: a full 170 eval = ~100–105 min under two-pass, ~80 min under single-shot. Plan accordingly.
6. **Cloudflare**: wire auth before sharing the tunnel URL more broadly.

## Cloudflare URL

https://mileage-demographic-duplicate-oven.trycloudflare.com (unauthenticated, per user deferral).

---
Driver: attempt #4. Exit: clean. All three evals completed, decision applied, service healthy.
