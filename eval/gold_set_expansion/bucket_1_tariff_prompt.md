# External-LLM Prompt — Bucket 1: Tariff / Rate Queries (25 items)

**Purpose:** generate 25 YAML test items for the CBIC RAG eval gold set, focused on rate / HSN / SAC / notification questions — the highest-value bucket because our pipeline's A4 tariff-routing feature has no eval coverage today.

**Where to paste this:** Claude 3.5 Sonnet, Claude Opus 4, GPT-5, Gemini 2.5 Pro, or any frontier model with current knowledge of Indian GST/Customs rates. Use whichever you trust most on Indian tax specifics.

**After it replies:** save the YAML block as `D:\_gpu_rig_ai\eval\gold_set_expansion\bucket_1_tariff.yaml`. I'll verify each item against real CBIC notifications before merging into `gold_set.yaml`.

---

## PROMPT BEGINS — copy everything below this line

You are an Indian indirect-tax expert building an evaluation set for a CBIC RAG system. Generate **exactly 25 test questions** about **GST rates, Customs tariff (BCD/IGST), HSN codes, SAC codes, and rate-notification lookups**. These are the questions real practitioners (CAs, CFOs, consultants, importers) ask every day.

### Scope

Cover these sub-topics roughly evenly:

- **GST rate by HSN** (e.g. "What is the GST rate for HSN 8471?" — computers)
- **GST rate by SAC** (e.g. "What SAC covers restaurant services and what is the rate?")
- **Rate-change notifications** (e.g. "What did notification 03/2022-CT(R) change?")
- **Exemption notifications** (e.g. "Is HSN 0401 — fresh milk — exempt under GST?")
- **Compensation cess** (e.g. "Is cess levied on HSN 2402 — cigarettes?")
- **Customs BCD + IGST combo** (e.g. "What is the total customs duty on HSN 8517 — smartphones imported from China?")
- **Concessional rates via notification** (e.g. "What is the IGST rate on medical equipment under notification 50/2017-Cus?")
- **Inverted duty rate** scenarios (supply HSN vs input HSN rate difference)
- **Nil-rated vs exempt vs zero-rated** distinction questions
- **Composite / mixed supply** with different rates on components

### Difficulty mix (mandatory)

- 12 × **basic** — single-concept lookup, e.g. "GST rate for HSN 8471"
- 9 × **intermediate** — 2-step reasoning, e.g. "Rate change history of HSN X between 2017 and 2023"
- 4 × **complex** — multi-part scenario, e.g. importing a composite machine containing parts under different HSN chapters

### Output format — strict YAML, one item per block

```yaml
- id: gst_rate_001              # format: <cat>_<subcat>_<NNN>. cat ∈ {gst, customs}. subcat ∈ {rate, hsn, sac, notif, cess, exempt, ccy_supply, inverted}
  category: gst                  # or customs
  subcategory: rate              # short tag
  difficulty: basic              # basic | intermediate | complex
  question: "What is the GST rate for computers under HSN 8471 for supply within India?"
  expected_sections: []          # only if a CGST/IGST Act section is directly on point
  expected_rules: []             # only if a CGST Rule is on point
  expected_notifications: ["01/2017-CT(R) Schedule III", "Notification 1/2017 Central Tax (Rate)"]
  expected_conclusion_keywords: ["18%", "HSN 8471", "computer"]
  must_not_say: ["12%", "28%", "exempt"]
  must_cite_verbatim: true       # true for most rate questions; false for interpretive questions
  notes: "HSN 8471 is in Schedule III of notification 01/2017 — 18% (9% CGST + 9% SGST for intra-state, 18% IGST inter-state)."
```

### Rules (must follow all)

1. **Real references only.** Every `expected_notifications` entry must be a real CBIC notification you are confident exists (e.g. `01/2017-CT(R)`, `50/2017-Cus`, `03/2022-CT(R)`). If you are not certain about a specific notification number, use a category placeholder like `"GST rate schedule notification"` and mark the item `difficulty: intermediate` with a note flagging "notification number to be verified".
2. **Current rates preferred.** Use rates effective as of early 2026 wherever you know them. If a rate has changed during the GST era, pick one version (current) and flag the change in `notes`.
3. **Keywords must be literal.** `expected_conclusion_keywords` must be terms the LLM would need to produce to earn credit — not concepts. `"18%"` is good; `"higher tax rate"` is bad.
4. **must_not_say must be concrete.** Forbidden phrases should be the specific wrong answers a sloppy model might produce. `"exempt"` on a rated item, `"28%"` when the right answer is `"18%"`, etc.
5. **HSN / SAC codes must be real** — don't invent codes. If uncertain, use commonly-cited examples (8471 computers, 8517 phones, 3004 pharmaceuticals, SAC 9963 restaurant services, SAC 9972 real estate, etc.).
6. **Unique IDs** — number sequentially `001` to `025` within this bucket.
7. **One question per item** — don't bundle multiple questions.
8. **Questions must be realistic** — phrase them the way a practitioner would type into a search bar. No leading hints, no "hint: see section X" framing.

### Spread HSN across chapters

Don't cluster all questions in Chapter 84 (machinery) or 85 (electronics). Touch:
- Food & agri (Ch 1-24)
- Textiles (Ch 50-63)
- Metals (Ch 72-83)
- Machinery & electronics (Ch 84-85)
- Vehicles (Ch 87)
- Pharma (Ch 30)
- Services (SAC 99xx)
- Petroleum / excluded goods (Ch 27) — to test "GST not applicable, under State VAT/Central Excise" edge case

### Examples of the spirit we want (DO generate questions of this shape, not copies)

```
question: "What is the IGST rate on import of smartphones (HSN 8517.12) from Vietnam into India?"
question: "Distinguish the GST rate treatment of raw cashew nuts (HSN 0801) vs roasted cashew nuts (HSN 2008)."
question: "An exporter buys inputs taxed at 18% and exports finished goods under LUT at 0% IGST — what provision allows ITC refund?"
question: "What changed in GST rate on textiles when notification 14/2019-CT(R) was issued?"
question: "A hotel supplies both lodging (SAC 9963) and restaurant service — can it charge different GST rates on a single bill?"
```

### Output

Return **only** a single YAML block containing all 25 items (no prose, no markdown wrapping, no comments before or after the YAML). Start directly with `- id: gst_rate_001` and end at the 25th item.

## PROMPT ENDS

---

## Post-generation checklist (for you, the human, before merging)

- [ ] Spot-check 5 random items: open the cited notification PDF on cbic-gst.gov.in and confirm the rate + HSN match
- [ ] Run `python -c "import yaml; yaml.safe_load(open('bucket_1_tariff.yaml'))"` to confirm valid YAML
- [ ] Confirm count = 25 items, difficulty spread = 12/9/4
- [ ] Confirm HSN chapter spread covers at least 6 different chapters
- [ ] Rename IDs if any collide with existing `gold_set.yaml` (probably won't since existing set has no `gst_rate_*` items)

Once verified, I'll merge into `gold_set.yaml` and re-run the eval to measure A4's lift on tariff queries specifically.
