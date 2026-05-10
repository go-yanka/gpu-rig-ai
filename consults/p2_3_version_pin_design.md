# P2.3 — Version-Pinning Payload Schema (Design Only)

Status: DESIGN. No rig deployment. Reference: `golden_plan_v2.md` §6 (P2.3).

## 1. Current payload (verified on rig, 2026-04-21)

Brief note: the collection named in the task (`cbic_v1`) does **not** exist. Live collections on `192.168.1.107:6333` are `indian_legal_t1_v2` and `indian_legal_full`. Scroll sample from `indian_legal_t1_v2` shows actual fields:

```
text, source, file, pdf_sha256, dataset, tier,
act_name, act_year, act_number, section_no, chapter_no, anchor,
type (statute_chunk | concordance_chunk | ...),
status (current | legacy),
effective_until (ISO date, present on legacy chunks only)
```

So a **partial** version-pinning scheme already exists (`status` + `effective_until`). It is coarse (chunk-level only, no `effective_from`, no amendment graph) and not applied uniformly. P2.3 formalises and extends it.

Fields listed in the task brief (`doc_id`, `title`, `subcategory`, `page`, `text_full`, `char_start`, `is_table`, `source_url`) are **not** present in the live collection — the brief appears to describe an older or planned schema. Design below treats the live schema as ground truth.

## 2. Proposed payload (additions only)

| Field | Type | Null? | Meaning |
|---|---|---|---|
| `effective_from` | ISO date string `YYYY-MM-DD` | no | Date this chunk's text started being law. |
| `effective_to` | ISO date string or `null` | yes | Date it ceased; `null` = still in force. Replaces `effective_until` (keep as alias for back-compat). |
| `amendment_of` | list[str] | yes (`[]`) | Notif/Act IDs this chunk amends, e.g. `["05/2020-CT(R)"]`. |
| `amended_by` | list[str] | yes (`[]`) | Later notifs that amend THIS chunk. Populated by post-pass graph build. |
| `version_id` | str | no | Stable key: `{doc_id}#{section_no}@{effective_from}`. Lets UI show "v1 (2017-07-01) / v2 (2024-11-01)". |
| `pin_confidence` | float 0..1 | no | Heuristic extractor confidence. <0.5 = show "date uncertain" in UI. |

Keep existing `status` as a derived convenience (`current` iff `effective_to IS NULL`).

## 3. Extraction heuristics

Run during chunking, before embedding. All regex case-insensitive, multiline.

**(a) Notifications / circulars**
- `effective_from` ← first match of `come[s]? into force (?:on|with effect from|w\.e\.f\.?)\s+(\d{1,2}(?:st|nd|rd|th)?\s+\w+[, ]+\d{4})`. Normalise via `dateutil`.
- Fallback: notif issue date from filename (`05_2020-CT-R.pdf` → gazette date lookup table).
- `amendment_of` ← regex `in supersession of (?:the )?notification(?:s)? No\.?\s*([\d/\-A-Z(),\s]+)` → split on comma.

**(b) Amendment Acts (Finance Act, CGST Amendment Act)**
- Per-section `effective_from` ← `This section shall .*? come into force (?:on|w\.e\.f\.?)\s+(<date>)` **or** the Act's appointed-day notification if the clause says "on such date as Central Government may notify".
- Emit one chunk per amended section; set `amendment_of` = [parent Act section ID].

**(c) Original Act sections (CGST/IGST/UTGST/Compensation Cess)**
- Default `effective_from = "2017-07-01"` (GST appointed day). For IT Act 1961, default `1962-04-01`.
- `effective_to` = min(`effective_from` of any later chunk with same `act_name`+`section_no`+`sub_section`), else `null`. Computed by post-pass (§5).

**(d) Tariff / rate tables (bucket 3 item 025 class)**
- Parse "w.e.f." column if present. Otherwise treat whole notification's effective date as row effective date.

`pin_confidence`: 1.0 if explicit regex hit; 0.7 if filename-derived; 0.4 if defaulted to appointed day; 0.2 if no signal (flag for manual review).

## 4. Qdrant filter pattern ("as of date D")

```json
{
  "must": [
    { "key": "effective_from", "range": { "lte": "2024-10-15" } }
  ],
  "must_not": [
    { "key": "effective_to",   "range": { "lt":  "2024-10-15" } }
  ]
}
```

Note: Qdrant treats missing key in `must_not` range as non-match, so `effective_to = null` (still in force) correctly passes. Create payload indexes on both fields: `PUT /collections/{c}/index` with `field_name=effective_from, field_schema=datetime`, same for `effective_to`. Without indexes, filter cost on 108k points is ~80 ms; with, <5 ms.

Convenience shortcut: query param `as_of=today` → server substitutes `date.today().isoformat()`; `as_of=null` → no filter (show all eras, rank later).

## 5. Migration options

| Option | Cost | Quality | Recommendation |
|---|---|---|---|
| **A. Backfill heuristic** — iterate 108k points, run §3 regex on `text`+`source`+`file`, `set_payload` in batches of 1k. | ~3 h compute, 1 day dev. | ~85% auto; ~15% defaulted low-confidence. | **Chosen** for P2.3 Phase 1. |
| B. Full re-ingest | 2–3 days wall clock, re-OCR risk | Cleanest; chunker emits fields natively | Defer to next corpus refresh (Income Tax / MCA ingest). |
| C. New chunks only | ~0 h | Mixed-era corpus; "as of" filter silently drops un-pinned chunks | **Reject** — breaks recall. |

Phase 1 = A + make chunker emit fields going forward (so future ingests are native). Phase 2 = amendment-graph post-pass (populate `amended_by` by matching `amendment_of` edges across collection).

## 6. Test queries where pinning changes the answer

1. **GSTAT pre-deposit (bucket-3 item 025)** — "Pre-deposit for appeal to GST Appellate Tribunal under Sec 112(8)?" `as_of=2024-06-01` → 20% / Rs 50 cr. `as_of=2025-01-01` → 10% / Rs 20 cr (Finance (No.2) Act 2024, notified 01-Nov-2024).
2. **ITC on CSR expenditure** — "Can ITC be claimed on goods/services used for CSR?" `as_of=2023-09-30` → allowed (no bar). `as_of=2023-10-01` → blocked by Sec 17(5)(fa) inserted by Finance Act 2023.
3. **E-invoicing threshold** — "Turnover threshold for mandatory e-invoicing?" pre-2022-10-01 Rs 20 cr, 2022-10-01→2023-07-31 Rs 10 cr, from 2023-08-01 Rs 5 cr. Pinning lets RAG give the correct rung.

## 7. Effort estimate

| Task | Hrs |
|---|---|
| Finalise schema + ADR + update chunker contract | 2 |
| Heuristic extractor module + unit tests (20 fixtures) | 6 |
| Backfill script (scroll + batched `set_payload`) + dry-run on 1k sample | 4 |
| Create Qdrant payload indexes, benchmark filter latency | 1 |
| RAG API: accept `as_of` query param, inject filter, surface `version_id` + `pin_confidence` in response | 3 |
| UI: date picker + "showing law as of …" banner + version badge on each citation | 4 |
| Eval pack: 15-query pinning regression set (inc. 3 above), wire into nightly | 3 |
| Docs + memory update | 1 |
| **Total** | **~24 h** (~3 focused days) |

## 8. Risks

- **R1 Date parser drift**: Indian tax notifs use "1st day of April, 2024" and "w.e.f. 01.04.2024" and "with effect from 1-4-2024". Mitigation: `dateutil.parser` + handwritten fallback; unit fixtures from 30 real notifs.
- **R2 Retrospective amendments**: some amendments declare `deemed to have come into force` on a past date. Heuristic must prefer the deemed date for `effective_from`, not the notif issue date. Add explicit regex branch.
- **R3 Sub-section granularity**: chunker currently keys on section, not sub-section/proviso. An amendment touching only `Sec 17(5)(fa)` will mark the entire Sec 17 chunk as amended, harming recall for untouched sub-sections. Document as known limitation for Phase 1; Phase 2 re-chunks to proviso level.
- **R4 Amendment graph cycles**: notifs sometimes cross-amend. Build `amended_by` as a DAG with cycle detection; on cycle, log and keep both edges, do not auto-resolve.
- **R5 User confusion**: "as of today" may hide chunks users expect to see (old rates for historic disputes). UI must offer an easy "show all versions" toggle, default to today for forward-looking queries, default to "date of cause of action" field when provided.
- **R6 Low-confidence defaults contaminate filter**: 15% of chunks with `pin_confidence=0.4` defaulted to 2017-07-01 will pass every "as_of >= 2017" filter. Mitigation: expose confidence in ranker; down-weight low-confidence when a high-confidence competitor exists for same section.
- **R7 Schema drift vs live `effective_until`**: back-compat alias required; migration script must read `effective_until` → write `effective_to` then drop alias after one release.

— end —
