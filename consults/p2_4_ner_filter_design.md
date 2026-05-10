# P2.4 — NER Hard-Filter Design

**Goal.** Extract structured entities (sections, rules, notifications, HSN/SAC, forms, Acts) from the user query via regex and build Qdrant `must` filters so dense+BM25 retrieval is hard-scoped to chunks citing the exact entity. Eliminates semantic drift on rare-term lookups (e.g. "Notif 12/2017-CT(R)") where embeddings collapse numeric tokens.

## 1. Entity inventory

| Type | Regex | Notes |
|---|---|---|
| `section` | `\bSection\s+(\d+[A-Z]?)(?:\((\d+)\))?(?:\(([a-z]+)\))?` | CGST/IGST/UTGST act-agnostic; capture base + sub + clause |
| `rule` | `\bRule\s+(\d+[A-Z]?)(?:\((\d+)\))?` | CGST Rules 2017 |
| `notification` | `\b(\d{1,3})/(\d{4})-(CT\(R\)|IT\(R\)|Cus|Cus\(ADD\)|UTT|Comp-Cess|ST|CE|CT)\b` | Rate/tariff/customs |
| `hsn` | `\b(?:HSN\s*)?(\d{4}|\d{6}|\d{8})\b` + `\bheading\s+(\d{4})\b` + `\bchapter\s+(\d{1,2})\b` | Prefix-match 4/6/8 |
| `sac` | `\bSAC\s*(\d{4,6})\b` | Services codes |
| `form` | `\bGSTR-(\d+[A-Z]?)\b|\bForm\s+GST\s+([A-Z]+)-(\d+)\b` | Returns + RFD/REG/etc. |
| `act` | `\b(Finance Act|Finance \(No\.\s*\d+\)\s*Act)\s+(\d{4})\b` | Amending Acts |

Regex runs CPU-only, <1ms per query. No ML model.

## 2. Qdrant payload fields (requires P2.3 enrichment)

Each chunk payload should carry arrays (a chunk can cite several):

- `sections: ["16", "16(2)", "16(2)(c)"]` (normalized: base + each sub-level as own token)
- `rules: ["36", "36(4)"]`
- `notifications: ["12/2017-CT(R)"]` (canonical form: `<num>/<year>-<type>`)
- `hsn: ["9983", "998311"]` (all prefixes 4/6/8 indexed)
- `sac`, `forms`, `acts` — same shape.

Index each as Qdrant keyword payload index for O(1) filter.

## 3. Filter construction

- Within a single entity type with multiple hits in query → OR (any match).
- Across entity types → AND (all types must be satisfied).
- Hierarchical match for sections/rules/HSN: query `Section 16(2)(c)` matches chunks tagged `16`, `16(2)`, or `16(2)(c)` (widen, don't narrow, at index time).

```python
def build_qdrant_filter(entities: dict) -> dict:
    must = []
    for field, vals in entities.items():
        if vals:
            must.append({"key": field, "match": {"any": vals}})
    return {"must": must} if must else None
```

## 4. Fallback chain

1. Full filter → if hit count ≥ 8 after fusion, proceed.
2. If 0 hits: drop lowest-precision entity type (HSN 8-digit → 6 → 4, or drop `act`), retry.
3. If still 0: downgrade to **soft boost** — run unfiltered retrieval, add +0.15 score to chunks matching any entity.
4. If still nothing useful: current behavior (no filter).

Log the fallback level per query for eval.

## 5. Implementation sketch

```python
def extract_entities(q: str) -> dict:
    return {
      "sections": _expand(SECTION_RE.findall(q)),
      "rules":    _expand(RULE_RE.findall(q)),
      "notifications": [f"{n}/{y}-{t}" for n,y,t in NOTIF_RE.findall(q)],
      "hsn":      _hsn_prefixes(HSN_RE.findall(q)),
      "sac":      SAC_RE.findall(q),
      "forms":    [f"GSTR-{m}" for m in GSTR_RE.findall(q)],
      "acts":     [f"{a} {y}" for a,y in ACT_RE.findall(q)],
    }
```

Hook: in `/v1/query` before dense+BM25 call, compute filter, pass to both Qdrant search and BM25 (BM25 filters by `chunk_id in allowed_set`).

## 6. Test queries (from gold_set.yaml)

| # | Query fragment | Expected entities |
|---|---|---|
| 1 | "ITC under Section 16(2)(c)" | sections=["16","16(2)","16(2)(c)"] |
| 2 | "Rule 42 reversal" | rules=["42"] |
| 3 | "Notification 12/2017-CT(R) exempt services" | notifications=["12/2017-CT(R)"] |
| 4 | "HSN 9983 rate" | hsn=["9983"] |
| 5 | "GSTR-3B late fee" | forms=["GSTR-3B"] |
| 6 | "SAC 998314 GST rate" | sac=["998314"], hsn prefixes |
| 7 | "Finance Act 2023 amendment to Section 17" | acts=["Finance Act 2023"], sections=["17"] |
| 8 | "Chapter 84 customs duty" | hsn=["84"] (chapter-level) |
| 9 | "Rule 36(4) ITC restriction" | rules=["36","36(4)"] |
| 10 | "Form GST RFD-01 refund" | forms=["RFD-01"] |

## 7. Risks

- **Over-restriction.** User cites non-existent section → 0 hits. Mitigated by fallback chain step 3.
- **Corpus gaps.** P2.3 extractor must be lossless; miss one citation and that chunk is invisible under hard filter. Run coverage audit: % of known-citation chunks with non-empty payload array, target ≥95%.
- **Ambiguity.** "Section 16" alone matches CGST, IGST, UTGST — keep act-agnostic; rank by dense score.
- **False positives from regex.** `Rule 42` vs "rule of 42" — require capitalized `Section`/`Rule` token, reject if preceded by lowercase word.
- **Payload index cost.** Keyword indexes on 6 fields over 108k chunks is negligible (<100MB).
