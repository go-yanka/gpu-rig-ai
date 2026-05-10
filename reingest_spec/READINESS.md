# CBIC Re-Ingestion — Readiness & Execution Playbook

**Purpose.** When we press "go" on cbic_v2 ingestion, there must be **zero** "I have this issue and that issue" moments. This file is the pre-flight: everything prepared, every probe status with evidence, every known dependency, every planned command. If something isn't on this list, it's not ready.

Source of truth for decisions: `SPEC.md` v2 (frozen 2026-04-23).
Decision log: `JOURNAL.md`.
Probe definitions: `PROBES.md`.
Proven components registry: `~/.claude/projects/D---gpu-rig-ai/memory/project_cbic_reingest_v2.md`.
Known-good hardware/model configs: `~/.claude/projects/D---gpu-rig-ai/memory/known_good_configs.md`.

---

## 1. What's PREPARED (ready on disk, tested, committed)

### Code components
| Component | File | Status | Proof |
|---|---|---|---|
| Topic tagger (60 topics, 5 cats, multi-label) | `reingest_spec/topic_tagger.py` | ✅ READY | V20 PASS (57/57 gold topics ≥10 chunks, 49% corpus tagged) |
| Chunk deduplicator (SHA256+NFKC) | `reingest_spec/dedupe_chunks.py` | ✅ READY | V21 PASS (114,626→78,291 canonical, 0 residual dups, 31.7% saved) |
| Probe harness (V1–V24) | `benchmarks/probes/probe_v*.py` | ✅ READY | All scripts present, run recipes in PROBES.md |
| V17 throttle/retry (Gemini judge) | `benchmarks/probes/probe_v17_judge.py` | ✅ READY | 16s throttle + exp backoff (5→10→20→40→60s) + all_samples_complete gate |
| Gold eval set patches | `/opt/indian-legal-ai/rag/cbic_rag/eval_set.json` | ✅ READY | 9 must_cite_verbatim flips applied; backup `.bak-20260423-083446`; 4 OOC refusals exempted via probe's REFUSAL_SUBS |

### Infrastructure
| Asset | State | Proof |
|---|---|---|
| Qdrant `cbic_v1` (114,626 pts, live) | ✅ running, port 6343 | `/collections/cbic_v1` |
| cbic-rag-api.service | ✅ running, port 9500 | `/health` OK |
| qwen3-14b.service (GPU 2, RX 6700 XT) | ✅ post-reboot verified 2026-04-23 | VRAM 11,045 MB, gen 32.9 tok/s |
| SSH key access (`id_ed25519_rig`) | ✅ hardened | `ssh.service` + `ttyd.service` Restart=always |
| Cloudflared tunnels (api + ttyd) | ✅ running, URLs rotate on restart | journalctl check |
| BGE-M3 embedder pool (GPUs 0,1,4,5,6) | ✅ known-good (last session 5.2–9.6 ch/s per GPU) | spawned on demand by api.py |

### Decisions locked (D1–D15)
All 15 decisions resolved. See `project_cbic_reingest_v2.md` § "Locked decisions". Highlights:
- **D1 = Claude CLI primary** for regex-miss extraction. qwen3-14b secondary.
- **D5 = Gemini paid tier ~$6.56 worst-case**.
- D4/D14 = English queries, cite Hindi twin when available.
- D10 = Never delete v1 during project.
- D13 = Shadow cutover with dual-write log.

---

## 2. Probe results — FULL matrix

Legend: 🟢 PASS · 🟡 manual/info · 🔴 fail (none) · 🔵 running · ⚪ pending-downstream

| # | Probe | Status | Evidence |
|---|---|---|---|
| V1 | qwen3 strict JSON | 🔵 running post-reboot | task `bnm3z8y2h` on rig |
| V2 | qwen3 extraction latency | 🔵 queued after V1 | same task |
| V3 | langdetect on samples | 🟢 PASS | `v3_result.json` |
| V4 | BM25 Hindi (no silent drops) | 🟢 PASS | `v4_result.json` — 0 drops both languages |
| V5 | Pool 1-hr soak | ⚪ pending | schedule pre-ingest |
| V6 | Qdrant snapshot+restore | 🟢 PASS | `v6_result.json` |
| V7 | Disk dual-collection | 🟢 info (annotated) | `v7_result.json` |
| V8 | Table-aware OCR prompt | ⚪ pending | manual review of 10 tables |
| V9 | Gemini daily budget | 🟢 PASS | $6.56 est, paid tier OK |
| V10 | Router LLM latency | 🔵 queued after V2 | same task |
| V11 | Chunker parent_hierarchy | ⚪ needs chunker v2 | do during chunker dev |
| V12 | 7 empty PDFs | 🟡 manual re-OCR queued | `v12_result.json` |
| V13 | Chunker runtime | ⚪ needs chunker v2 | do during chunker dev |
| V14 | must_cite intent | 🟢 PASS | 0 missing |
| V15 | Payload update perf | 🟢 PASS | 1092/s |
| V16 | θ_retrieve threshold | ⚪ pending | schedule after retrieve layer ready |
| V17 | Gemini judge stability | 🔵 re-running | task `bh0csma82` on laptop |
| V18 | Claude CLI extraction | 🟢 PASS | 50/50 parse, p50 3.38s, p95 4.58s → D1 |
| V19 | api.py during ingest | ⚪ pending | run during Phase 5 stage of v2 |
| V20 | Taxonomy coverage | 🟢 PASS | 57/57 gold topics ≥10 chunks |
| V21 | sha256 dedup at scale | 🟢 PASS | 0 residual, 31.7% saved |
| V22 | OCR-tolerant regex variants | 🟢 info | 5 variants; best picked during chunker v2 |
| V23 | api.py refactor | ⚪ pending | after V19 |
| V24 | Validator dry-run | ⚪ pending | after chunker v2 |

**Hard blockers remaining: ZERO.** The ⚪ items are downstream of components we haven't built yet (chunker v2, validator, /query_v2 route). They will be exercised as those components get built.

---

## 3. What remains to be BUILT before Phase 1 can start

Concrete, ordered, scoped. Each item has acceptance criteria.

### B-1. Chunker v2 (TWO-PASS, structure-aware)
**Location:** `reingest_spec/chunker_v2.py` (does not yet exist)
**Authority:** non-negotiable rules in `~/.claude/projects/D---gpu-rig-ai/memory/chunking_strategy_cbic_v2.md` (R1–R7, failure modes, self-tests T1–T8). Any deviation requires a dated JOURNAL entry explaining why.

**Pass 1 — doc understanding (LLM-assisted):**
- Per doc: **qwen3-14b on rig GPU 2 port 9082 (D1 primary as of 2026-04-23)** — call with metadata + first 2000 chars + last 1500 chars + TOC → emits `chunking_plan` JSON (doc_type, structure, primary_splitter, critical_units, hard_boundaries, table_regions, language, confidence). Claude CLI is the legacy fallback, retired from default after 40% JSON-fail on production smoke.
- Ambiguous (confidence <0.6) → Gemini second-opinion; disagreement on doc_type → manual review queue
- Budget: **~24.9 h wall for 14,925 born-digital docs** (6s/doc × 14,925). Corpus is 15,776 total minus 851 image-only OCR-queued (deferred).

**Pass 2 — rule-driven chunking:**
- Hard size rules: target 3500 / cap 5500 / ceiling 8000 / floor 500
- R1 tables atomic, R2 critical-units whole (proviso/explanation/definition/footnote/form-field-block), R3 unusable-cut validator, R4 hierarchy-aware splits, R5 overlap policy (700 mid-section, 0 at section-start / table transitions)
- Dual PDF extract: PyMuPDF primary, pdfplumber fallback for tables flagged by Pass 1
- `section_ref` extraction: best V22 regex variant → on miss Claude CLI → on miss leave null + `section_ref_extraction_failed=true`
- `topic_tags` via `reingest_spec/topic_tagger.py.tag_chunk(text, category)`
- `chunk_type` classifier (heuristic from text + parent hierarchy)

**Emits per chunk:** `chunk_id, doc_id, sha256, source, category, subcategory, lang, text, embed_text, section_ref, parent_hierarchy_text, chunk_type, is_table, table_part, page_range, effective_date, text_source, hindi_twin_chunk_ids, topic_tags, chunking_plan_used, chunking_rule_triggered`

**Acceptance:**
- Self-tests T1–T8 pass (`reingest_spec/test_chunker.py`)
- V11 (5/5 docs breadcrumb correct)
- V13 (≤15 min per 100 docs → ≤4h wall for 115k incl. Pass 1)
- V24 (≤2% validator reject)
- Post-run audit: <1% R3 rejections, 0 table splits mid-row, 0 proviso orphans, ≥90% Hindi twin link coverage

### B-2. Ingest pipeline wrapper
**Location:** `reingest_spec/ingest_v2.py`
**Responsibility:**
1. Walk manifest → for each doc, load text → chunker v2 → emit chunks
2. Run chunks through `ChunkDeduper` → skip duplicates but link them to canonical via `also_appears_in`
3. Dense embed canonical chunks via BGE-M3 pool
4. Sparse embed via fastembed BM25
5. Upsert to `cbic_v2`
**Acceptance:** Dry-run on 100 docs produces valid Qdrant upsert payloads; validator rejects ≤2%.

### B-3. Payload validator
**Location:** `reingest_spec/validator.py`
**Contract:** Pydantic (or equivalent) schema enforcing required fields + types + enum membership (chunk_type, lang, text_source).
**Acceptance:** V24 — on 100-doc sample, ≤2% reject rate.

### B-4. `/query_v2` endpoint in api.py
**Location:** `/opt/indian-legal-ai/rag/cbic_rag/api.py` (extend, don't rewrite)
**Contract:** Same request schema as `/query`, but retrieves from `cbic_v2`. Response format identical.
**Acceptance:** V19 (no 5xx on `/query` while v2 ingests), V23 (12/12 existing endpoints unchanged behaviour).

### B-5. Dual-writer middleware (shadow mode, D13)
**Location:** `api.py` — async tap on every `/query` inbound that mirrors to `/query_v2` and logs diff to `shadow_log.sqlite`
**Acceptance:** Shadow log fills during live traffic with no visible impact on `/query` latency.

### B-6. θ_retrieve refusal threshold
**Determined by:** V16 — run 350–400 gold + 50 OOC through retrieval, compute score distributions, pick θ with clean separation (OOC max < gold min × 0.9). Per-category θ per D8.
**Stored in:** `/opt/indian-legal-ai/rag/cbic_rag/retrieval_config.json`.

### B-7. Snapshot cbic_v1 (Phase 0)
`curl -X POST http://192.168.1.107:6343/collections/cbic_v1/snapshots` → verified by V6.
Store the snapshot filename + date + size somewhere permanent (append to JOURNAL).

---

## 4. Execution plan — exact commands

### Phase 0 — Pre-flight (≤30 min)
```bash
# On rig
curl -X POST http://127.0.0.1:6343/collections/cbic_v1/snapshots
curl http://127.0.0.1:6343/collections/cbic_v1/snapshots | tee /opt/indian-legal-ai/data/snapshot_v1_pre_v2ingest.json
# Confirm:
df -h /opt/indian-legal-ai  # need ≥1.5× current collection
systemctl is-active qwen3-14b.service cbic-rag-api.service
curl -s http://127.0.0.1:9082/health   # expect {"status":"ok"}
awk '{printf "%.0f MB\n", $1/1024/1024}' /sys/bus/pci/devices/0000:09:00.0/mem_info_vram_used  # expect ≥8000

# Git freeze
cd /opt/indian-legal-ai && git tag reingest-v2-start && git log -1
```
**If any fail:** stop. Resolve before proceeding.

### Phase 1 — Manifest prep
```bash
python3 reingest_spec/build_manifest_v2.py  # TO BE WRITTEN — derives from existing v1 scrape + OCR cache
# Validates: all 15,776 manifest rows have text source (851 tagged ocr, ~14,925 born); Hindi twins linked; no orphan rows
```

### Phase 2 — Chunk
```bash
python3 reingest_spec/chunker_v2.py --manifest ingest_manifest_v2.sqlite --out chunks_v2.jsonl
# V13 budget: ≤3h wall time. Emits per-doc chunk count to manifest.
```

### Phase 2b — Dedup + tag (in-process via ingest_v2.py wrapper)
Handled by `ingest_v2.py` between chunk emit and upsert. Stats logged to `dedup_stats.json`.

### Phase 3 — Dense embed
```bash
# Start BGE-M3 pool (on-demand, already configured in embedder_direct.py)
python3 reingest_spec/ingest_v2.py --phase dense --chunks chunks_v2.jsonl
```

### Phase 4 — Sparse embed
Handled inline in Phase 3 via fastembed; no separate command.

### Phase 5 — Upsert
```bash
python3 reingest_spec/ingest_v2.py --phase upsert --collection cbic_v2
# V19 must pass concurrently: one-shell hammers /query at 1 QPS
```

### Phase 6 — Shadow mode
```bash
# Restart api.py with /query_v2 + dual-write enabled
systemctl restart cbic-rag-api.service
# Verify shadow log fills
sqlite3 /opt/indian-legal-ai/data/shadow_log.sqlite "select count(*) from shadow_queries;"
```

### Phase 7 — 4-gate validation
```bash
python3 benchmarks/eval_4gate.py --collection cbic_v2 --gold /opt/indian-legal-ai/rag/cbic_rag/eval_set.json
# Writes 4gate_result.json: G1 recall@10, G2 judge mean, G3 evidence hit%, G4 refusal%
```
**Pass:** all four ≥95% (G2 ≥4.5 avg with ≥95% ≥4). **Fail on any:** stop, amend SPEC, re-run failed phase.

### Phase 8 — Promotion
```bash
# If all gates pass:
# Flip /query to read from cbic_v2 in api.py config (single line: COLLECTION = "cbic_v2")
systemctl restart cbic-rag-api.service
# Keep cbic_v1 online (D10)
```

---

## 5. Failure response matrix

| Symptom | Likely cause | Recovery |
|---|---|---|
| qwen3-14b `/health` 503 lasts >5 min after start | mmap tensor load (normal 45s-90s) | `journalctl -u qwen3-14b` — if steady progress, wait. If frozen, `systemctl restart`. |
| qwen3-14b VRAM <1 GB after load completes | RADV degradation | `systemctl reboot` (known-good recovery per `known_good_configs.md`) |
| Chunker Phase 2 exceeds 3h budget | Dual-pass overhead | Run `--pymupdf-only` pass first for scale; pdfplumber pass only for `is_table` candidates |
| Embedder pool one GPU stops producing | VRAM orphan | Kill + respawn (embedder_direct.py auto-recovers; if not, restart cbic-rag-api.service) |
| Upsert rate drops to near zero | Qdrant memory pressure | Pause, check `du -sh /opt/.../qdrant_cbic_storage`, add disk or reduce batch |
| G1 recall@10 <95% | Retrieval config wrong OR chunker dropped content | Inspect false negatives; usually fixes: raise k, fix section_ref extraction, check topic_tag keyword index |
| G2 judge variance high | Judge instability | V17 re-verify throttle; if needed swap judge model to 2.0-pro |
| G3 evidence <95% | Chunker split evidence across chunks | Lower chunk target, raise overlap to 1000 |
| G4 refusal leaks | θ_retrieve too low | V16 re-fit with larger OOC pool |

---

## 6. Rollback

cbic_v1 stays online throughout (D10). Rollback is single-command:
```python
# In api.py config
COLLECTION = "cbic_v1"  # was "cbic_v2"
```
```bash
systemctl restart cbic-rag-api.service
```
cbic_v2 is not deleted — it stays for diagnosis.

---

## 7. Definition of DONE for this readiness doc

- [x] SPEC v2 frozen
- [x] All 5 previously-failing probes fixed + demonstrated working
- [x] qwen3-14b recovery recipe proven end-to-end
- [x] D1 (extraction LLM) resolved → Claude CLI
- [x] D5 (Gemini budget) resolved → paid ~$6.56
- [x] Topic tagger + ChunkDeduper code on disk, self-tested
- [x] Gold set patched
- [x] Known-good configs persisted in memory
- [ ] V1, V2, V10 final verification on recovered qwen3 (**in progress**)
- [ ] V17 final stability reading on throttled run (**in progress**)
- [ ] B-1..B-7 implemented (NOT yet done — these are the to-build list for next sessions)
- [ ] Chunker v2 probes V11/V13/V22 (during B-1)
- [ ] Validator V24 (during B-3)
- [ ] Shadow-mode V19/V23 (during B-4/B-5)
- [ ] θ_retrieve V16 (during B-6)

**We do NOT start Phase 1 until every box above is ticked.** When they are, this file becomes the execution runbook — open it alongside a terminal and work down §4.

---

## 8. Open questions for the user (before ingestion)

Only one, and it's schedulable not blocking:
- **Re-OCR the 7 empty CE PDFs now or skip them?** V12 flagged 7 tiny OCR outputs (16 bytes each). They may be genuinely blank scans. Need 15 min to re-OCR at 300 DPI with the table-aware prompt. Recommendation: do it in the Phase 1 manifest stage.

No other decisions pending.
