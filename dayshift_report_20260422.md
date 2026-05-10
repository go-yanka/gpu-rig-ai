# Day-Shift Report — 2026-04-22

Driver: day-shift #1. Budget: ~4 h wall-clock actually used.

## TL;DR

- **P1.3 triage**: Root cause = two-pass "validator-refusal" gate. 58/170 items (34%) emitted the canned "no defensible verbatim span" message; 55 of those had scored positive under P1.2 baseline. Accounts for the full 13.4 pp regression. Needs redesign (soft-fallback), not a retry. P1.3 stays PARKED.
- **P2.1 tariff-DB**: wired in behind `TARIFF_ROUTER_ENABLED`. Full 170-item eval → 39.54 % (−0.90 pp vs 40.44 % baseline). Within ±3 pp band → **code kept, flag OFF** per decision matrix. Router fired on 7 items and lost a net 16 points on those 7 (bare SQL rows lacked the narrative colour the LLM needed to score 4-6 pts).
- **P2.2**: out of scope.
- **P2.4 NER hard-filter**: **BLOCKED**. Design requires payload arrays (`sections`, `rules`, `notifications`, `hsn`, …). Live `cbic_v1` payloads only carry scalar `section_ref` + `doc_number`, and `section_ref` represents the chunk's own internal clause number, not cited sections. Implementing P2.4 requires P2.3 corpus enrichment first. See `consults/p2_4_blocker_20260422.md`.
- **cbic-rag-api** healthy after every restart; /health = 200. Nothing reverted due to crash.

## Score table

| Run | Flags | Pass rate |
|---|---|---:|
| Prior (pre-P1.2, keyword judge) | — | 33.23 % |
| P1.2 baseline (anchor) | all OFF | **40.44 %** |
| P1.3 Eval-A (overnight) | TWO_PASS=1 | 27.03 % |
| P1.3 Eval-B (overnight) | TWO_PASS=1, REFEED=1 | 26.62 % |
| **P2.1 tariff-router ON** (today) | TARIFF_ROUTER=1 | **39.54 %** |
| **Final deployed state** | all OFF | identical to P1.2 |

## Final flag state on rig

```
TWO_PASS_ENABLED=0
TWO_PASS_CHUNK_REFEED=0
TARIFF_ROUTER_ENABLED=0
# NER_FILTER_ENABLED not set (P2.4 never deployed)
```

Drop-in files:
- `/etc/systemd/system/cbic-rag-api.service.d/two_pass.conf` → both two-pass flags OFF
- `/etc/systemd/system/cbic-rag-api.service.d/tariff_router.conf` → TARIFF_ROUTER_ENABLED=0

## P1.3 triage findings (detail)

Source: `consults/p1_3_triage_20260422.md`. Top failure modes among the 57 zero-point items in Eval-A:

| # | Mode | Count |
|---|---|---:|
| 1 | Validator-refusal canned message | 54 |
| 2 | Wrong-content (off-topic/hallucinated) | 3 |
| 3 | Error/empty | 0 |

Dominant mode is a hard refusal emitted when pass-1 extraction yields no validator-accepted verbatim spans. The pipeline should soft-fallback to single-shot RAG in that case; it currently force-refuses. Chunk-refeed (Eval-B) does not help because pass-2 still requires pass-1 spans.

## P2.1 tariff-DB findings (detail)

Wire-in: `api.py` step-0 pre-retrieval router. On accepted reasons (`rate-lookup:*`, `exemption-lookup:*`, `notif-sno-lookup:*`, `notif-meta:*`, code-bound list-membership) it renders SQL rows + notif_id citations and returns immediately. Otherwise falls through to the full RAG pipeline unchanged. Excludes `list-membership:*:all` (keyword over-matching producing unrelated dumps).

Observed on eval:
- 7 router hits (latency 24-51 ms each, vs 25-30 s RAG). That's huge on latency.
- But pts_router = 9, pts_baseline_same_items = 25 → net −16 pts. The canned row rendering strips narrative. Judge gave 1/max on most hits because the answer is technically correct but lacks the context the rubric rewards.
- Net effect 170-item: −0.9 pp. Within ±3 pp band, so code stays but flag OFF per matrix.

Future improvement (not done today): have the router return rows PLUS fall through to RAG, then merge — SQL row as authoritative rate citation, RAG prose as context. Or: feed the rows into the LLM as pre-chunk context rather than emitting them directly.

## Rollback commands

### P2.1 (tariff router) — remove entirely
```
ssh rig
sudo cp /opt/indian-legal-ai/rag/cbic_rag/api.py.bak.phase_p2_1.1776860023 \
        /opt/indian-legal-ai/rag/cbic_rag/api.py
sudo rm -f /etc/systemd/system/cbic-rag-api.service.d/tariff_router.conf
sudo rm -f /opt/indian-legal-ai/rag/cbic_rag/tariff.db \
           /opt/indian-legal-ai/rag/cbic_rag/tariff_router.py
sudo python3 -m py_compile /opt/indian-legal-ai/rag/cbic_rag/api.py
sudo systemctl daemon-reload && sudo systemctl restart cbic-rag-api
```

### Re-enable P2.1 (for further experimentation)
```
sudo tee /etc/systemd/system/cbic-rag-api.service.d/tariff_router.conf <<'EOF'
[Service]
Environment=TARIFF_ROUTER_ENABLED=1
EOF
sudo systemctl daemon-reload && sudo systemctl restart cbic-rag-api
```

### P1.3 / P1.2 rollbacks — unchanged, see `morning_status_20260422.md`.

## Files changed today

Rig:
- `/opt/indian-legal-ai/rag/cbic_rag/api.py` — +60 LOC for TARIFF_ROUTER flag, import, DB connection, `_format_tariff_rows`, and step-0 call in `_run_pipeline`. Behind flag.
- `/opt/indian-legal-ai/rag/cbic_rag/api.py.bak.phase_p2_1.1776860023` — pre-change backup.
- `/opt/indian-legal-ai/rag/cbic_rag/tariff.db` — new (73 KB, seed data).
- `/opt/indian-legal-ai/rag/cbic_rag/tariff_router.py` — new (copy of local `tariff_db/router.py`).
- `/etc/systemd/system/cbic-rag-api.service.d/tariff_router.conf` — new drop-in, flag OFF.

Laptop (git status-tracked):
- `D:\_gpu_rig_ai\rig_mirror\api.py.current` — updated mirror for reference.
- `D:\_gpu_rig_ai\consults\p1_3_triage_20260422.md` — triage.
- `D:\_gpu_rig_ai\consults\p2_4_blocker_20260422.md` — P2.4 design-vs-reality gap.
- `D:\_gpu_rig_ai\eval\runs\p2_1_evalA_20260422_081545\` — 170-item run, tariff router ON.
- `D:\_gpu_rig_ai\dayshift_report_20260422.md` — this file.

## Recommended next moves

1. **P1.3 redesign.** Change two-pass pass-1-empty behaviour from hard-refusal to soft-fallback to single-shot. Small code change; eval alone would tell us if this preserves the +38pp claim of A3 on the items where two-pass actually helps, without the 58-item cliff. Worth a single Eval-A run.
2. **P2.3 corpus enrichment (payload arrays for citations).** Blocking prerequisite for P2.4, and would help P2.1 too — router could fall-back to RAG with a *boosted* filter rather than replace it. Batch job, ~2–4 h of re-ingest.
3. **P2.1 v2: merge don't replace.** Feed SQL rows into LLM context as authoritative grounding rather than emitting them directly. Requires touching `build_prompt` + the payload plumbing; ~1 h; should eliminate the −16 pt penalty seen today.
4. **Do not revisit P2.4 until P2.3 is done.**
5. **Anchor future experiments on 40.44 %** (P1.2 baseline, 170-item, judge-functional). Next shift's deltas should be quoted against that.

## Abort / incident log

- No 529s. No crashes. No reverts on restart.
- One slow eval (~85 min for 170 items, ~30 s/item under single-shot + tariff router-miss path; 7 items <60 ms on router hit). Matches budget.
- P2.4 aborted before code change (design blocker); no rig touch.

---
Exit: clean. All flags returned to P1.2 baseline state. Service healthy.
