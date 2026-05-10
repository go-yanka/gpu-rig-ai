
### 2026-04-25 (later 3) — Set 5 G1–G4 results + 7 readiness blockers for full re-ingest

**Set 5 cohort (`cbic_v2_set5`, 100 docs, 653 gold queries) — gate panel ran in parallel against single API instance:**

| Gate | Result | Pass? | Root cause if FAIL |
|---|---|---|---|
| G1 recall@10 | raw 0.84, adj 0.96 (after dropping 96 shared-PDF form misses) | ❌ raw / ✅ adj | Defect D — shared `CGST-Rules-2017-Part-B-Forms.pdf` referenced by ~50 form doc_ids; chunker re-reads whole PDF per doc_id, dedup zeros all but the first. |
| G2 dual-judge | 10/200 errors=10 → killed | ❌ broken | Concurrent gate load on `/query` (LLM endpoint) — same root cause as the 4-gate concurrency lesson codified earlier today. Re-run alone after G1/G3. |
| G3 levenshtein | g1=193/225, g3_saves=0 | ❌ broken | Set 5 hand-authored gold has no `expected_text` field; G3 scoring path can't fire. Decision pending: accept G3≡G1 vs enrich gold. |
| G4 adversarial | 12/50 = 24% refusal vs 90% target | ❌ separability | Gold band [0.51-0.82] overlaps adversarial [0.42-0.74]. Dense-only retrieve cannot distinguish. **Reranker required.** |

**Code patches deployed (not in spec, retroactively documented here):**
- `evaluators/gate_g3_levenshtein.py` — accept singular gold keys (`expected_chunk_id`/`expected_doc_id`/`expected_section`) in addition to plural. Set 5 gold uses singular.
- `evaluators/gate_g4_adversarial.py` — `API` flipped from `/query` to `/retrieve` because /query was overloaded under concurrent gate load and G4 only needs the top retrieve score vs theta (no LLM answer needed).
- Both patches mirrored to `D:/_gpu_rig_ai/reingest_spec/evaluators/` 2026-04-25.

**Rig observation during gate run — heterogeneous embed pool not load-balanced:**
- Codified `EMBED_GPUS=4,5,6` pool at full gate load showed GPU 4 = 99% busy, GPU 5/6 = 0%.
- Suspect: embedder_direct.py worker fan-out single-worker by default; pool only exercised under multi-request burst.
- Consequence for full re-ingest: paper projection of 22 ch/s on {4,5,6} may not hold in production.

**Seven readiness blockers — full 14,925-doc re-ingest is NOT READY:**

1. G4 separability — integrate `bge-reranker-v2-m3-Q4_K_M.gguf` into `/retrieve`, re-tune θ. Without this, 95% gates are mathematically unreachable. (1-2 days)
2. G2 — re-run dual-judge alone (no concurrency) with `set -a; source /root/.cbic_env; set +a`. (30 min)
3. G3 — decide accept G3≡G1 vs enrich gold with `expected_text`. (30 min decision; 2h enrichment if chosen)
4. Defect D — count shared-PDF docs in 14,925 corpus; patch chunker to scope per doc_id offsets. (0.5-1 day)
5. phase6_pairs — patch `summary.json` writer (didn't fire on last 200-doc test) + add hard cap of 12 q/chunk on qwen3 emission (outliers `cbic-allied-act-dtls:1000201` emitted 48; `cbic-others-document-msts:1000038` emitted 49). (2-4 hours)
6. EMBED_GPUS — bench solo + 4-card pools `{0,4,5,6}` `{1,4,5,6}` `{3,4,5,6}` per the existing MEMORY note. Right now only GPU 4 saturates; pool concurrency itself is suspect. (1-2 hours bench)
7. Gate concurrency — codified lesson must be operationalized: either serialize gate panel, or stand up a second API instance with separate embed pool, before any production gate panel.

**Critical path before kicking 14,925-doc re-ingest:**
- Day 1: blockers 1 + 4 + 5 in parallel agents
- Day 2: blocker 6 → freeze pool → re-tune θ → serialized re-run all 4 gates on Set 5
- Day 3: 200-doc dry run end-to-end (Set 6)
- Day 4+: 14,925-doc kick-off (~8.5 days at qwen3-bound Phase 6 limit, single GPU 2 12GB)

