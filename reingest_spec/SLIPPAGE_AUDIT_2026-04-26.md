# Slippage audit — 2026-04-26 ~21:00 IST

**Trigger:** RG flagged that `phase6_pairs` was codified but not running. Demanded full audit + permanent fix to prevent recurrence.

## Root cause of phase6 slip

**Codification ≠ implementation.** The decision was logged in MEMORY.md, PAIR_GEN_SPEC.md, DECISIONS.yaml on 2026-04-25. Nobody wrote the actual `phase6` Python code in `ingest_v2.py`. Tonight I launched batch 1 with `--phase all` assuming "all" meant all codified phases. Hard Rule "Inventory before planning" was not applied tonight — it would have caught this.

## Audit table

| # | Decision (codified) | Reality | Slip? | Impact |
|---|---|---|---|---|
| 1 | phase6_pairs inline per batch | Not in `ingest_v2.py` | **CAUGHT** | Deferred to single post-batch-10 pass per RG decision (JOURNAL "later 6") |
| 2 | Defect D chunker — shared mega-PDFs emit per-doc chunks | `cbic-form-msts:1000360` produced 0 chunks in batch 1 | **UNFIXED** | Will degrade CP-2/CP-3 G3 same as CP-1 |
| 3 | Defect F1 mega-chunk (`_SINGLE_SECTION_LIMIT = int(TARGET*1.5)`) | Present at line 1130 | ✅ fixed | – |
| 4 | classify_doc_qwen `max_tokens 4096→512` | Present at line 556 | ✅ fixed | – |
| 5 | Reranker `-c 8192 -b 4096 -ub 4096 --parallel 8` | Service runs `-c 32768 ...` | drift (bigger ctx, safer; not a slip) | – |
| 6 | Embed pool expand to `{1,3,4,5,6}` after CP-1 | Still `5,6,4` (3 GPUs) | **DEFERRED** by spec; not slipped | Slower batch wall-clock |
| 7 | section_ref propagation into chunk payload (post-G3 root-cause) | Bug exists (3/6 batch-1 misses had `None`) | **UNFIXED** | Will degrade CP-2/CP-3 G3 |
| 8 | API drop-in `set5_collection.conf` disabled post-CP-1 | `.disabled` extension active | ✅ done by AI agent earlier today | – |
| 9 | `sequential_cold_load: true` in embed pool | confirmed | ✅ | – |
| 10 | `_preflight_classify_latency_slo` mandatory in ingest | line 227 def, line 376 call | ✅ | – |

## Items needing action BEFORE CP-2 (after batch 5)

- **Slip #2 — Defect D shared mega-PDF chunker.** Pre-CP-2 risk: G3 will repeat the same forced miss pattern. Mitigation: surface in JOURNAL + decision deferred to post-CP-3 since impact is small (~1 doc per batch).
- **Slip #7 — section_ref None.** Pre-CP-2 risk: same gold queries will miss. Mitigation: ingest_v2 already populates `section_ref` for most chunks; the bug is on a minority path. Defer to post-CP-3.

If CP-2 G3 < 0.92 (CP-2 threshold), the run halts per Hard Rule #1, and we re-fix #2 + #7 before continuing.

## Items deferred with explicit RG awareness

- phase6 pair generation → single post-batch-10 pass (RG decision B 2026-04-26 21:00 IST)
- Embed pool expansion → after full re-ingest done
- 481 orphan docs from attempt-1 partial flushes → audit post-CP-3

## Permanent fix (rule strengthened in CLAUDE.md)

When a decision is codified, the inventory check at the START of any session must include:

```bash
bash inventory.sh | grep -i <new_decision_keyword>
```

If the keyword has files but no implementation grep matches, **the codification is dead text** — surface as a slip and either implement or explicitly defer.

## Process change going forward

For every multi-step plan I propose:
1. Run `inventory.sh` and grep for plan keywords (existing rule).
2. **NEW: For each codified decision involved, verify the implementation exists in code** (not just the spec doc).
3. Surface gaps in the conversation BEFORE launching.
