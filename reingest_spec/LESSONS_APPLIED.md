# LESSONS_APPLIED.md — v1 pain → v2 fix ledger

**This file is mandatory.** RUNBOOK Stage-B/C/D/E/F exit gate must diff this file
against the code it claims to cover. If a row says FIX=applied but `grep` can't
find the fix in the referenced file+line, the gate fails.

Regen: `python3 reingest_spec/check_lessons.py` (writes STATUS column).
Re-run at every stage exit, not just once.

Last audited: 2026-04-23 (ingest_v2.py revision: hash pending commit).

| # | v1 pain (source)                               | v2 fix location                                                  | STATUS | Notes |
|---|------------------------------------------------|------------------------------------------------------------------|--------|-------|
| 1 | Size-only chunking mangled sections            | `chunker_v2.py:303 chunk_document` two-pass                      | APPLIED | T5 test green |
| 2 | Orphan provisos/explanations (lost context)    | `chunker_v2.py find_critical_unit_spans` + `is_unusable_cut`     | APPLIED | T2,T3,T4 green |
| 3 | Tables split mid-row → hallucinated tax rates  | `chunker_v2.py:339 _extract_table_chunks` (R1 atomic)            | APPLIED | T1,T1b green |
| 4 | ~30% duplicate boilerplate re-embedded         | `ingest_v2.py:162,198 ChunkDeduper` (V21 proven 31.7% savings)   | APPLIED | canonical SHA after NFKC+lc+ws-collapse |
| 5 | No topic tags → router blind                   | `ingest_v2.py:193 topic_tagger.tag_chunk` (V20)                  | APPLIED | multi-label, stored in payload |
| 6 | Bilingual twins lost (EN and HI unlinked)      | `ingest_v2.py _find_twin` + `twin_doc_id` self-link on bilingual docs | APPLIED | 2026-04-23. Self-twin when both path_en+path_hi present; cross-doc twin via (category,subcategory,title) index when only one path. Skipped for scoped `--doc-ids` smoke runs (closed set). |
| 7 | Hindi orphan-verb cuts (dev. script)           | `chunker_v2.py is_unusable_cut` 12 devanagari tokens             | APPLIED | T4 Hindi leg green |
| 8 | No amendment/version metadata                  | `chunker_v2.py notification_id` (+ as_of_date, superseded_by)    | APPLIED | G2 fixed — T10.a/b/c/f green. |
| 9 | OCR text silently mixed with born-digital      | `ingest_v2.py detect_text_source` (density<200 chars/pg → ocr)   | APPLIED | G3 fixed — T12.a/b green. OCR docs skipped until thaw. |
| 10| Sparse BM25 format drift between v1/v2         | `ingest_v2.py:244 from ingest import embed_batch` (reuse)        | APPLIED | format identity guaranteed by reuse |
| 11| `upsert_chunks` keys `c['page']`, `c['char_start']` | `chunker_v2.py to_payload`                                       | APPLIED | G1 fixed — emits page/char_start/char_end. T9.a green. |
| 12| Collection name hardcoded                      | `ingest_v2.py QDRANT_COLL_V2` (set before `from ingest import`)  | APPLIED | G4 fixed — T11.a green. |
| 13| No resume granularity per phase                | `ingest_v2.py init_manifest_v2` (phase1_done, phase2_done, embedded, upserted) | APPLIED | per-chunk flags |
| 14| No Pass-1 plan persisted → unreproducible      | `ingest_v2.py:209 plan_json` stored per doc                      | APPLIED | plus `plan_confidence` for QA |
| 15| Router picks wrong collection under shadow     | `api_v2_shadow.py _jaccard_divergence`                           | APPLIED | T14/T15/T16/T17 green. Kill-switch auto-trips on >2% avg over 200 req. |
| 16| No θ calibration → random refuse threshold     | `theta_tune.py pick_theta`                                       | APPLIED | T18 green. Grid search, fails hard if infeasible. |
| 17| No snapshot rotation / corruption detection    | `snapshot_v2.sh MIN_BYTES`                                       | APPLIED | T19 green. 50%-shrink corruption check + 14d rotation. |
| 18| No v1 archive → rollback impossible post-cutover | `archive_v1.sh` + `rollback_v1.sh`                             | APPLIED | tar.gz + sha256 + points_count verify. test_archive.sh 17/17 static-checks. |
| 19| No cutover visibility → divergence goes unnoticed | `static/shadow.html` + `/shadow/recent` endpoint               | APPLIED | Sparkline + 2s polling, kill-switch banner, mirrors quality.html style. |
| 20| Shadow dual-writer mutated os.environ (race)   | `api_v2_shadow.py _call_collection` + `retriever.py retrieve(collection=)` | APPLIED | H1. T20.a-d green. Collection threaded through kwargs, no env mutation. |
| 21| Evaluators used fabricated `_collection` filter Qdrant ignored | `api.py QueryReq` adds `collection: Optional[str]` field | APPLIED | H2. T21.a-e green. 5 evaluators + theta_tune all switched. |
| 22| Gold singular schema broke `_is_hit` lookups + v1 chunk_id never matches v2 SHA256 | `reingest_spec/evaluators/gate_g1_recall.py _norm_gold` | APPLIED | H3. T22.a-f green. Drops chunk_id match, adds text-substring fallback. |
| 23| Rollback wrote config but never restarted service → change invisible | `rollback_v1.sh` systemctl + SERVICE_CMD fallback + /v1/stats health-check | APPLIED | H4. T23.a-e green. |
| 24| Ollama silently replaced llama-cpp Vulkan facade (2026-04-23, burned 12h) — CPU fallback returned zero vectors, no error signal | `embedder.py` `_FACADE_VERSION = "direct-v1"` sentinel + preflight grep + `ingest_v2.py run_preflight()._preflight_embedder_facade` rejects any embedder lacking the sentinel or containing "ollama"/"11434" in source | APPLIED | 2026-04-23. `grep _FACADE_VERSION /opt/indian-legal-ai/rag/cbic_rag/embedder.py` must return 1 hit. |
| 25| Preflight was component-only (swap/Qdrant/VRAM/model-file) — missed the Ollama regression because no check exercised real embed on real hardware | `preflight.sh` step 12 runs a full 1-doc end-to-end dry-run (ingest_v2.py --doc-ids <smallest-doc>) to a throwaway collection, verifies points_count>0, drops collection. `ingest_v2.py run_preflight()` also runs a hello-world embed with variance check (rejects zero-vectors) | APPLIED | 2026-04-23. Set `SKIP_E2E=1` to bypass (component-smoke only — never production). |
| 26| `embed_dense_batch` swallowed exceptions and returned zero vectors on failure → hid the Ollama CPU fallback AND any future embed failure | `embedder.py.direct.ref` — removed the `except Exception: return [[0.0]*DENSE_DIM ...]` fallback; now raises | APPLIED | 2026-04-23. Any embed failure now propagates to phase3_4_5._flush_batch and kills the run visibly. |
| 27| `phase3_4_5` could print "DONE" with 0 real upserts ("silent success") | `ingest_v2.py phase3_4_5` final assertion: `if done > 0 and pts < done: raise RuntimeError("SILENT-SUCCESS: submitted N but Qdrant has only M")` | APPLIED | 2026-04-23. Catches both zero-point-collection and partial-drop cases. |
| 28| `EMBED_GPUS` could be set to include GPU 2 (qwen3 host) or GPU 3 (SMU-faulted) → ingest fights qwen3 for VRAM or hangs | `ingest_v2.py EMBED_GPUS_FORBIDDEN` + default `EMBED_GPUS_DEFAULT` + `_preflight_embed_gpus` rejects forbidden GPUs | APPLIED | 2026-04-23. |
| 29| `LLM_URL` (qwen3) could be dead at Phase-2 start → 14,925 × HTTP timeout before failing | `ingest_v2.py _preflight_qwen3` pings LLM_URL with /health or /v1/models in <10s and aborts if unreachable | APPLIED | 2026-04-23. |
| 30| Launch via `screen -dmS` silent-failed on this rig (no socket, no error) → launched "dry" ingest that did nothing | `RUNBOOK.md` Stage 0.5.5 codifies `nohup` + log-to-/tmp + `tail -f` as the canonical launch pattern. `screen` deprecated for long-running ingest. | APPLIED | 2026-04-23. |
| 31| 5,781 pre-generated QA pairs in `eval/training_pairs/` lacked `doc_id` → could not scope G2 to smoke doc subset → would have drafted new gold from scratch (the 2026-04-23 near-miss) | `reingest_spec/eval_smoke/filter_training_pairs.py` filters the 7 jsonl pair files by 5-gram overlap between `why_this_chunk` and chunks of a given doc-id set; emits G2-compatible schema with `expected_doc_ids` | APPLIED | 2026-04-23. Usage in RUNBOOK Stage J / Stage 0.5. |
| 32| Wrong Python binary / missing PYTHONPATH gave cryptic `ModuleNotFoundError: llama_cpp` deep in pool init — hid real cause for minutes | `ingest_v2.py _preflight_python_stack` catches ImportError up-front with an actionable message listing the correct PYTHONPATH + python binary + Vulkan rebuild command | APPLIED | 2026-04-23. First check in `run_preflight()`. |
| 33| Forbidden Vulkan flags (`--flash-attn`, `--mlock`, `--cache-type-k/v q*`) known to crash the pool on Navi 10 but only documented in memory `known_good_configs.md` — no code enforcement | `preflight.sh` step 11b greps all llama-server launch scripts + systemd unit files for the 4 forbidden flags; aborts preflight if any found | APPLIED | 2026-04-23. FORBIDDEN_FLAGS array + SCAN_PATHS array drive the scan. |
| 34| Chunker T1–T8 self-tests existed in `test_chunker.py` but were never wired into preflight → a chunker regression (e.g. proviso splitting) would only be caught by post-G1 audit after 25h of ingest | `preflight.sh` step 11c runs `pytest -x -q test_chunker.py` (with script-mode fallback); aborts if any T1–T8 case fails | APPLIED | 2026-04-23. `SKIP_CHUNKER_TESTS=1` escape hatch for component-only smokes. |
| 35| G3 Levenshtein `--sim-threshold` default was 0.85 but SPEC.md §1 G3 row says ≥0.95 → threshold drift between spec and evaluator. Would have passed queries that SPEC calls fails. | `reingest_spec/evaluators/gate_g3_levenshtein.py` default `--sim-threshold=0.95` (matches SPEC verbatim) + docstring note that lowering requires SPEC amendment + JOURNAL entry | APPLIED | 2026-04-23. Caught by codification audit. |
| 36| Operational learnings (Ollama trap, preflight gaps, launch pattern) were only written into memory MDs, not fed into operational artifacts → every new Claude session repeats the same mistakes | LESSONS_APPLIED rows 24–36 + `reingest_spec/check_lessons.py` greps code locations on every stage-exit + `CLAUDE.md` MANDATORY SECOND STEP requires playbook citation before touching embed/ingest code | APPLIED | 2026-04-23. This is the meta-rule: fixes-in-code, not fixes-in-memory-only. |

## Real gaps caught by this audit (must fix before Stage-B exit)

### G1 — BROKEN: phase3_4_5 will crash on first batch
`cbic_rag/ingest.py:138` computes `pid = abs(hash((c['doc_id'], c['page'], c['char_start'])))` but `chunker_v2.Chunk.to_payload()` emits `page_range: [start,end]` — no `page`, no `char_start` keys. First upsert raises KeyError.
**Fix:** extend `Chunk.to_payload` to also emit `page = page_range[0]` and `char_start = <span offset>`. One-line addition. MUST do before any phase3 smoke test.

### G2 — MISSING: no amendment/version metadata in chunks
v1 had no `as_of_date` / `notification_id` / `superseded_by` fields. v2 spec §4 requires them (D8). `Chunk.to_payload` currently has none.
**Fix:** Pass-1 prompt already asks Claude for `doc_type` and `effective_date` — wire those into `meta` and propagate into every chunk's payload.

### G3 — INCOMPLETE: OCR text_source hardcoded "born"
471 image-only PDFs (per ocr_research_cbic.md). v2 silently tags them as born-digital. Retrieval will surface garbage OCR alongside clean text with no way to filter.
**Fix:** before `classify_and_chunk`, call a cheap text-density probe on the extracted text (if chars/page < 200, tag `text_source="ocr"` and route to OCR pipeline per frozen plan — or skip until OCR thawed).

### G4 — UNVERIFIED: QDRANT_COLL env override
`ingest.py` line ~80 reads `QCOLL = os.environ.get("QDRANT_COLL", …)` at module import. Our `os.environ["QDRANT_COLL"] = QDRANT_COLL_V2` happens AFTER import. Need to check whether `ensure_collection(qc, dim)` uses module-level QCOLL or recomputes.
**Fix:** either pass collection name as explicit arg or set env before `from ingest import …`.

## Mechanical enforcement

1. `reingest_spec/check_lessons.py` (next file) greps each `v2 fix location` cell and writes STATUS. Fails CI if any row regresses from APPLIED → MISSING.
2. `RUNBOOK.md` Stage-B/C/D/E/F exit clauses now reference this file: "no stage exits while any row is BROKEN/MISSING/PARTIAL without an explicit waiver JOURNAL entry."
3. Every new script added to Stage B (B5, B6, B7) MUST append rows to this table BEFORE the script lands.
