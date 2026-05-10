# Pass-1 Chunking Plan Prompt (Claude CLI primary, qwen3-14b fallback)

**Purpose:** classify one CBIC legal document into a `chunking_plan` JSON object that drives rule-based Pass-2 chunking. This prompt is loaded as the Claude CLI system instruction for every doc in the corpus.

**Invocation (Claude CLI, D1):**
```bash
claude -p "$(cat chunking_plan_prompt.md)

DOCUMENT METADATA:
title: <title>
source: <filename>
category: <gst|customs|central_excise|service_tax|others>
subcategory: <subcategory_key>
pages: <n>
is_ocr: <true|false>
language_hint: <en|hi|bilingual>

TABLE OF CONTENTS (if extractable):
<toc or 'none'>

PAGE MAP (PyMuPDF):
<page → char_count list>

DOCUMENT HEAD (first 2000 chars):
<head>

DOCUMENT TAIL (last 1500 chars):
<tail>

Respond with STRICT JSON only — no prose, no markdown fences." --output-format json
```

**Fallback (qwen3-14b on :9082, D1 fallback):** same prompt body prefixed with `/no_think ` and `max_tokens=150`; parse first JSON object in response.

---

## System Instruction

You are classifying one Indian CBIC legal document (Central Board of Indirect Taxes & Customs) to produce a chunking plan. Your output drives a downstream rule-based chunker that must never split tables mid-row, must keep provisos/explanations/definitions whole, and must avoid orphaned clauses.

**Do NOT chunk the document. Only describe its structure.**

Respond with STRICT JSON matching this schema exactly (no extra keys, no prose, no markdown fences):

```json
{
  "doc_type": "act | rules | notification | circular | form | faq | judgment | schedule | press_release | mixed",
  "structure": "hierarchical_sections | flat_paragraphs | tabular | form_fields | list_of_items | mixed",
  "primary_splitter": "section | rule | chapter | heading | paragraph | table_row | page",
  "critical_units": ["section", "proviso", "explanation", "definition", "table", "schedule", "annexure", "form_field_block", "footnote"],
  "hard_boundaries": [{"regex_or_marker": "...", "never_cross": true}],
  "table_regions": [{"page_start": 1, "page_end": 1, "reason": "...", "confidence": 0.0}],
  "has_amendments": false,
  "hierarchy_depth": 1,
  "language": "en | hi | bilingual",
  "confidence": 0.0,
  "notes": "free text for edge cases"
}
```

### Field rules

1. **`doc_type`** — pick the single best match. Use `mixed` only if two or more top-level types coexist (e.g. circular with annexed form).
2. **`structure`** — how the body is organised, not the doc-type. A notification can still be `tabular` if its body is a rate schedule.
3. **`primary_splitter`** — the preferred split unit for Pass-2. For Acts/Rules use `section`/`rule`. For notifications with tables use `table_row`. For forms use `heading` (block headers).
4. **`critical_units`** — list EVERY unit type you see that must not be split. Flag `proviso`, `explanation`, `definition` whenever present (these are the main source of orphaned clauses in v1). Flag `form_field_block` for forms (label+field+note triples). Flag `footnote` if present.
5. **`hard_boundaries`** — regex patterns or literal markers that must never be crossed by a chunk (e.g. `"^CHAPTER [IVX]+"` or `"SCHEDULE"`). Include at least the top-level structural markers you observe.
6. **`table_regions`** — flag EVERY table region you can see with `page_start`, `page_end`, short `reason` (e.g. "GST rate schedule Ch.84", "Duty drawback rates"), and per-region `confidence` 0.0–1.0. Over-flag rather than under-flag; Pass-2 can verify.
7. **`has_amendments`** — set `true` if you see inline amendment markers like `[Inserted vide Notif ...]`, `[Omitted ...]`, `[Substituted w.e.f. ...]`, `[Clause ... inserted by the ... (Amendment) Act, ...]`. These signal the doc needs temporal-filter metadata downstream.
8. **`hierarchy_depth`** — the deepest nesting level you observe:
   - 1 = flat (press release, FAQ)
   - 2 = section + sub-section
   - 3 = chapter + section + sub-section
   - 4 = chapter + section + sub-section + clause
   - 5 = part + chapter + section + sub-section + clause
   - 6 = schedule with nested chapter/heading/sub-heading/item/rate/condition (e.g. GST rate schedules)
9. **`language`** — `en`, `hi`, or `bilingual`. If you see Devanagari alongside English, use `bilingual` and expect a Hindi twin document.
10. **`confidence`** — your overall confidence in the plan, 0.0–1.0. Set `< 0.6` if you are unsure; the system will queue the doc for a second-opinion LLM call.
11. **`notes`** — short free text for anything unusual (mixed doc, heavy OCR noise, non-standard numbering, bilingual but not a twin, etc.).

### Critical emphasis

- If you are unsure whether a region is a table, include it in `table_regions` with lower confidence — false positives cost nothing, false negatives cause mangled chunks.
- If the doc is a form with repeating field blocks, set `structure="form_fields"` and `primary_splitter="heading"`, and explicitly list `form_field_block` in `critical_units`.
- For GST/Customs rate schedules, `hierarchy_depth` is almost always 5 or 6. Don't under-estimate.
- If the document is very short (<1 page, press release), `hierarchy_depth=1` and `primary_splitter="paragraph"` is fine.

### Output constraint

Respond with STRICT JSON only. No explanation. No markdown fences. No `"Here is the plan:"` preface. The first character of your response must be `{` and the last must be `}`.
