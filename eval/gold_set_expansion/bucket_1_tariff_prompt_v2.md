# External-LLM Prompt — Bucket 1: Tariff / Rate Queries (25 items) — v2 ENHANCED

**Purpose:** generate 25 YAML test items for the CBIC RAG eval gold set, focused on rate / HSN / SAC / notification questions. This v2 prompt adds guardrails learned from v1 output quality issues (factual traps, ID-naming drift, HSN duplicates, composite/mixed confusion).

**Where to paste this:** Claude 3.5 Sonnet, Claude Opus 4, GPT-5, Gemini 2.5 Pro, or any frontier model with current knowledge of Indian GST/Customs rates.

**After it replies:** save the YAML block as `D:\_gpu_rig_ai\eval\gold_set_expansion\bucket_1_tariff.yaml`. Each item will be verified against real CBIC notifications before merging.

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
- **Inverted duty rate** scenarios (supply HSN vs input HSN rate difference — Rule 89(5))
- **Nil-rated vs exempt vs zero-rated** distinction questions
- **Composite / mixed supply** with different rates on components

### Difficulty mix (mandatory)

- 12 × **basic** — single-concept lookup
- 9 × **intermediate** — 2-step reasoning
- 4 × **complex** — multi-part scenario

### Output format — strict YAML, one item per block

```yaml
- id: gst_rate_001              # format: <cat>_<subcat>_<NNN>. cat ∈ {gst, customs}. subcat ∈ {rate, hsn, sac, notif, cess, exempt, ccy_supply, inverted}
  category: gst                  # or customs
  subcategory: rate              # short tag matching the subcat in the id
  difficulty: basic              # basic | intermediate | complex
  question: "What is the GST rate for computers under HSN 8471 for supply within India?"
  expected_sections: []
  expected_rules: []
  expected_notifications: ["01/2017-CT(R) Schedule III"]
  expected_conclusion_keywords: ["18%", "HSN 8471", "computer"]
  must_not_say: ["12%", "28%", "exempt"]
  must_cite_verbatim: true
  notes: "HSN 8471 Schedule III of 01/2017-CT(R) — 18% (9+9 intra / 18 IGST inter)."
```

### CRITICAL QUALITY RULES — read before generating

#### R1. ID naming must vary by subcategory
The id **must** reflect the actual subcategory. Do NOT label every item `gst_rate_NNN`.
- ✅ `gst_rate_001`, `gst_sac_002`, `gst_notif_003`, `gst_exempt_004`, `gst_cess_005`, `gst_inverted_006`, `gst_ccy_supply_007`, `customs_rate_008`
- ❌ `gst_rate_001` through `gst_rate_025` — this is wrong and will be rejected.

Numbering resets are fine, but **subcat in id must equal the `subcategory` field**.

#### R2. No duplicate primary HSN / SAC across items
Each HSN code (e.g. 8517) or SAC code (e.g. 9963) may appear as the **primary subject** of at most ONE item. You may reference a code in passing in `notes` or `must_not_say`, but two items both asking "what is the rate on HSN 8517" is forbidden. Spread across chapters per the list below.

#### R3. Factual traps — DO NOT fall into these
The following are common wrong answers. Do not generate items that assert these:
- ❌ "Air conditioners are inverted duty" — FALSE. AC output is 28%, typical inputs (compressors, copper) are 18% — that is **normal** (output > input), not inverted. Real inverted-duty examples: fertilizer (5% out / 18% in), railway locomotives, footwear below threshold, solar modules pre-2021, some textiles.
- ❌ "Packaged drinking water is nil-rated" — FALSE. Only **20-litre jars** are nil (entry 99 of 02/2017-CT(R)). Retail bottles (1L, 500ml) are **18%** under HSN 2201.
- ❌ "Mobile phones are 12%" — FALSE since April 2020. Phones (HSN 8517.13 / 8517.14) are **18%** via 03/2020-CT(R).
- ❌ "Composite supply = mixed supply" — they are **different**. Composite: naturally bundled, one principal supply, rate of principal applies (Sec 8(a)). Mixed: not naturally bundled, priced together, **highest rate** applies (Sec 8(b)). Example of composite: hotel room + breakfast (principal = accommodation). Example of mixed: gift hamper with chocolates + dry fruits + juice.
- ❌ "Petrol / diesel / ATF / crude / natural gas are taxed under GST" — FALSE. They remain under VAT + Central Excise per **Section 9(2) CGST Act** (deferred from GST). Use this as a nuance question, don't state they have a GST rate.
- ❌ Inventing notification numbers. If unsure, write `"GST rate schedule (01/2017-CT(R))"` as a generic placeholder rather than a fake number like `"45/2023-CT(R)"`.

#### R4. Composite vs mixed — if you include these, use canonical examples
- **Composite (principal rate)**: hotel room + breakfast; laptop + charger + bag sold together; AMC that includes parts; transportation + insurance in a CIF supply.
- **Mixed (highest rate)**: gift hamper (chocolates 18% + dry fruits 12% + aerated drink 28% → all at 28%); Diwali gift box; combo pack of unrelated items priced as one.
- **Not a composite/mixed question**: "AC + installation service" — installation is typically a separate works-contract / service, not always bundled as composite. Avoid this as an example.

#### R5. Inverted duty — use real examples
If you write an inverted-duty question, the output HSN rate must actually be **lower** than input HSN rate. Canonical examples:
- Fertilizer (HSN 3102, 5%) with inputs at 18% — classic inverted.
- Footwear below ₹1000 (HSN 6401-6405, 5%) vs inputs (rubber, leather at 12-18%) — was inverted, rate rationalisation history worth testing.
- Railway locomotives & parts (Ch 86, 5%) with inputs at 18%.
- EV batteries / solar cells — check current rate before using.
Rule **89(5)** governs refund of accumulated ITC on inverted-duty; **input services and capital goods are excluded** from the formula — worth a nuance item.

#### R6. HSN chapter spread — touch at least 7 of these
| Chapter | Example HSN | Typical item |
|---|---|---|
| 01-24 food/agri | 0401, 0801, 1701, 2008, 2106, 2201, 2202, 2402 | milk, cashew, sugar, roasted nuts, namkeen, water, aerated drinks, cigarettes |
| 27 petroleum | 2709, 2710, 2711 | crude, petrol/diesel, natural gas — **Sec 9(2) non-GST** |
| 30 pharma | 3003, 3004, 3006 | medicines, life-saving drugs |
| 50-63 textiles | 5208, 6109, 6203, 6403 | cotton fabric, t-shirts, trousers, footwear |
| 71 gems | 7108, 7113 | gold, jewellery (3% special rate) |
| 72-83 metals | 7204, 7208, 7210 | steel scrap, HR coil |
| 84 machinery | 8471, 8413, 8418 | computer, pumps, refrigerators |
| 85 electronics | 8517, 8528, 8415 | phones, TV, AC |
| 87 vehicles | 8703, 8711, 8712 | cars/SUV (+ cess), motorcycles, bicycles |
| SAC 99xx services | 9954, 9963, 9964, 9965, 9971, 9972, 9984, 9988 | works contract, hotel/restaurant, passenger/goods transport, financial, real estate, telecom, job work |
| Ch 9801 | Project Imports | concessional BCD under Project Imports Regulations |

#### R7. Real notifications only
Every `expected_notifications` entry must be a real CBIC notification. Canonical ones you may safely use:
- `01/2017-CT(R)` — GST rate schedule (goods), Schedules I (5%) / II (12%) / III (18%) / IV (28%)
- `02/2017-CT(R)` — nil-rated goods
- `11/2017-CT(R)` — services rate schedule
- `12/2017-CT(R)` — services exemption
- `50/2017-Cus` — concessional BCD/IGST schedule (customs)
- `03/2017-CT(R)` — concessional rate for specified supplies (petroleum ops etc.)
- `03/2022-CT(R)`, `14/2019-CT(R)`, `03/2020-CT(R)` — rate-change notifications
- If you're not confident of a specific number, write `"GST rate schedule notification (01/2017-CT(R) as amended)"` and add `"notification number to verify"` to notes.

#### R8. Keywords / must_not_say must be literal
- `expected_conclusion_keywords` = exact strings a correct answer will contain. `"18%"` ✅, `"higher rate"` ❌.
- `must_not_say` = specific wrong answers. `"28%"` when correct is `"18%"` ✅, `"incorrect"` ❌.

#### R9. Realistic phrasing
Questions must read like a practitioner typing into a search bar. No hints, no "per Section X". Don't bundle two questions into one item.

#### R10. Unique IDs, sequential within subcat
Number sequentially per subcat (e.g. `gst_rate_001..007`, `gst_sac_001..003`, `gst_notif_001..004`, `customs_rate_001..004`). Total = 25.

### Self-check before emitting YAML
Before you output, verify:
- [ ] Exactly 25 items
- [ ] Difficulty split = 12 basic / 9 intermediate / 4 complex
- [ ] ID prefixes span at least 6 different subcats (not all `_rate_`)
- [ ] No HSN code appears as primary subject in two items
- [ ] HSN spread covers at least 7 chapters from R6 table
- [ ] No item asserts any of the R3 factual traps
- [ ] All notification numbers are from R7 list OR flagged as "to verify"
- [ ] Every `subcategory` field matches the subcat in its `id`

### Output

Return **only** a single YAML block containing all 25 items (no prose, no markdown wrapping, no comments before or after the YAML). Start directly with the first `- id:` line and end at the 25th item.

## PROMPT ENDS
