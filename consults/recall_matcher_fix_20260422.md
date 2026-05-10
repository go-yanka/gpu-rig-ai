# Recall Matcher Fix — 2026-04-22

Fixes the `chunk_contains` notation-mismatch false negatives by parsing gold entities into `(section_code | rule_code | notif_num, act_name)` and matching each field against the correct chunk metadata field individually.

> **Data limitation (option b):** we only have `top_k_meta` (section_ref, doc_number, title, doc_id) — no full chunk `text`. The OLD matcher also searched `chunk.text[:2000]`, so any old hit that relied on the text body is invisible to this offline re-scoring and shows up as an apparent regression. Regressions below are mostly this; the true fixed-recall lower-bound is still ≥ the number shown. Net @5 delta here: **+13**.

- Source: `D:\_gpu_rig_ai\eval\recall_audit_20260422.jsonl`
- Output: `D:\_gpu_rig_ai\eval\recall_audit_fixed_matcher_20260422.jsonl`
- Items: **170**

## Headline — old vs fixed matcher

| Metric | Old | Fixed | Delta |
|---|---:|---:|---:|
| recall@1 | 6/170 (3.5%) | 14/170 (8.2%) | +8 |
| recall@5 | 21/170 (12.4%) | 34/170 (20.0%) | +13 |

**Formerly-miss items recovered at k=5: 27**  (out of 149 original misses)
**Apparent regressions at k=5: 14**  (almost certainly chunks where old match relied on `text` body, invisible in `top_k_meta`)

## Per-category recall@5

| category | n | old hit@5 | fixed hit@5 | delta |
|---|---:|---:|---:|---:|
| central_excise | 10 | 2 (20.0%) | 5 (50.0%) | +3 |
| customs | 45 | 7 (15.6%) | 17 (37.8%) | +10 |
| gst | 81 | 6 (7.4%) | 9 (11.1%) | +3 |
| others | 14 | 3 (21.4%) | 0 (0.0%) | +-3 |
| service_tax | 20 | 3 (15.0%) | 3 (15.0%) | +0 |

## Per-category recall@1

| category | n | old hit@1 | fixed hit@1 | delta |
|---|---:|---:|---:|---:|
| central_excise | 10 | 1 (10.0%) | 2 (20.0%) | +1 |
| customs | 45 | 2 (4.4%) | 7 (15.6%) | +5 |
| gst | 81 | 1 (1.2%) | 3 (3.7%) | +2 |
| others | 14 | 1 (7.1%) | 0 (0.0%) | +-1 |
| service_tax | 20 | 1 (5.0%) | 2 (10.0%) | +1 |

## 5 recovered-hit examples  (old matcher missed, fixed matcher finds)

### cus_val_001  (customs)
- Question: What is the primary method of valuation for imported goods under the Customs Act?
- Expected: `['14 Customs Act', 'Rule 3 CVR 2007']`
- Matched entity ranks (fixed): `{'14 Customs Act': [], 'Rule 3 CVR 2007': [1]}`
- Top-5 chunk meta:
  - rank 1: section_ref=`3(b)` doc_number=`94/2007-Custom` title=`This relates to Customs valuation (Determination of Price of`
  - rank 2: section_ref=`` doc_number=`37/2007-Custom` title=`Customs valuation (Determination of Value of Export Goods) R`
  - rank 3: section_ref=`` doc_number=`38/2007-Custom` title=`Customs Valuation (Determination of Value of Imported Goods)`
  - rank 4: section_ref=`` doc_number=`94/2007-Custom` title=`This relates to Customs valuation (Determination of Price of`
  - rank 5: section_ref=`` doc_number=`94/2007-Custom` title=`This relates to Customs valuation (Determination of Price of`

### cus_val_002  (customs)
- Question: An Indian importer buys from a foreign parent company (related party). What valuation scrutiny applies?
- Expected: `['14 Customs Act', 'Rule 3(3) CVR 2007', 'Rule 2(2) CVR 2007']`
- Matched entity ranks (fixed): `{'14 Customs Act': [], 'Rule 3(3) CVR 2007': [], 'Rule 2(2) CVR 2007': [4]}`
- Top-5 chunk meta:
  - rank 1: section_ref=`` doc_number=`97/2004-Custom` title=`Export Promotion Capital Goods Scheme - Customs Duty Exempti`
  - rank 2: section_ref=`` doc_number=`94/2007-Custom` title=`This relates to Customs valuation (Determination of Price of`
  - rank 3: section_ref=`` doc_number=`` title=`cs21-2k2-cond.pdf`
  - rank 4: section_ref=`2(2)(i)` doc_number=`91/2017-Custom` title=`Customs Valuation (Determination of value of Imported goods)`
  - rank 5: section_ref=`` doc_number=`94/2007-Custom` title=`Makes Customs Valuation (Determination of Price of imported `

### cus_svb_001  (customs)
- Question: What is the role of the Special Valuation Branch (SVB) in customs?
- Expected: `['Circular 5/2016-Customs']`
- Matched entity ranks (fixed): `{'Circular 5/2016-Customs': [2]}`
- Top-5 chunk meta:
  - rank 1: section_ref=`14(23)(a)` doc_number=`65/2017-Custom` title=`Bill of entry(Forms)(Amendment) Regulations, 2017`
  - rank 2: section_ref=`` doc_number=`5/2016-Custom` title=`Procedure for investigation of related party import cases an`
  - rank 3: section_ref=`14(24)` doc_number=`90/2020-Custom` title=`Amendment to Bill of Entry (Forms) Regulations, 1976`
  - rank 4: section_ref=`` doc_number=`1/98-Cus` title=`Valuation (Customs) - Cases handled by Special Valuation Bra`
  - rank 5: section_ref=`` doc_number=`1/98-Cus` title=`Valuation (Customs) - Cases handled by Special Valuation Bra`

### cus_ig_001  (customs)
- Question: On import of goods into India, what are the components of customs duty levied?
- Expected: `['3 CTA', '12 Customs Act', '5(1) IGST']`
- Matched entity ranks (fixed): `{'3 CTA': [5], '12 Customs Act': [], '5(1) IGST': []}`
- Top-5 chunk meta:
  - rank 1: section_ref=`` doc_number=`` title=`The Finance Act, 1998`
  - rank 2: section_ref=`11N` doc_number=`` title=`Customs Act, 1962`
  - rank 3: section_ref=`` doc_number=`6/2012-Custom` title=`Amend Custom Tariff (Identification, Assessment and Collecti`
  - rank 4: section_ref=`2(6)` doc_number=`` title=`Customs Tariff Act, 1975`
  - rank 5: section_ref=`3` doc_number=`1/2016-Custom` title=`Seeks to levy safeguard duty on imports of Hot-rolled flat p`

### exc_val_002  (central_excise)
- Question: For certain notified goods, valuation is based on MRP rather than transaction value. Which provision governs this?
- Expected: `['4A Central Excise Act']`
- Matched entity ranks (fixed): `{'4A Central Excise Act': [1, 4]}`
- Top-5 chunk meta:
  - rank 1: section_ref=`` doc_number=`625/16` title=`Valuation of goods under Section 4A of the Central Excise Ac`
  - rank 2: section_ref=`` doc_number=`336/52` title=`Introduction of MRP based levy to tooth paste, tooth powder `
  - rank 3: section_ref=`` doc_number=`737/53` title=`Levy of excise duty on readymade garments on the basis of Re`
  - rank 4: section_ref=`` doc_number=`625/16` title=`Valuation of goods under Section 4A of the Central Excise Ac`
  - rank 5: section_ref=`` doc_number=`5/2001-C` title=`This notification specifies the commodities to which MRP bas`

## 3 remaining-miss examples  (confirm real retrieval failure)

### gst_pos_001  (gst)
- Question: A registered supplier in Maharashtra sells goods to a registered buyer in Karnataka. Goods are shipped directly from Maharashtra to Karnataka. What is the place
- Expected: `['10(1)(a) IGST', 'Section 10 IGST']`
- Top-5 chunk meta:
  - rank 1: section_ref=`` doc_number=`` title=`Seeks to supersede 31 customs exemption notifications and pr`
  - rank 2: section_ref=`` doc_number=`40/2017-Centra` title=`Seeks to prescribe Central Tax rate of 0.05% on intra-State `
  - rank 3: section_ref=`` doc_number=`01/2025-Compen` title=`Seeks to prescribe Compensation cess  rate of 0.1% on supply`
  - rank 4: section_ref=`` doc_number=`` title=`Seeks to supersede 31 customs exemption notifications and pr`
  - rank 5: section_ref=`` doc_number=`` title=`Seeks to supersede 31 customs exemption notifications and pr`
- Verdict: none of top-5 contain the expected section/rule/notification in any metadata field. Real retrieval miss, not matcher bug.

### gst_pos_002  (gst)
- Question: A buyer in Delhi instructs a Mumbai supplier to deliver goods directly to the buyer's customer in Chennai (bill-to-ship-to). What is the place of supply for eac
- Expected: `['10(1)(b) IGST']`
- Top-5 chunk meta:
  - rank 1: section_ref=`2(11)(ii)` doc_number=`` title=`Integrated Goods And Services Tax Act, 2017`
  - rank 2: section_ref=`` doc_number=`01/2025-Compen` title=`Seeks to prescribe Compensation cess  rate of 0.1% on supply`
  - rank 3: section_ref=`9(b)` doc_number=`` title=`Integrated Goods And Services Tax Act, 2017`
  - rank 4: section_ref=`` doc_number=`40/2017-Centra` title=`Seeks to prescribe Central Tax rate of 0.05% on intra-State `
  - rank 5: section_ref=`` doc_number=`` title=`Regarding procedure for direct supply by intermediate suppli`
- Verdict: none of top-5 contain the expected section/rule/notification in any metadata field. Real retrieval miss, not matcher bug.

### gst_pos_003  (gst)
- Question: Services of a chartered accountant in Pune are rendered to a registered client in Gujarat. What is the place of supply?
- Expected: `['12(2) IGST']`
- Top-5 chunk meta:
  - rank 1: section_ref=`` doc_number=`3/2017-Centra` title=`Continuation of Recovery Proceedings`
  - rank 2: section_ref=`` doc_number=`` title=`Central Goods and Services Tax Act, 2017`
  - rank 3: section_ref=`` doc_number=`10/2017–Centra` title=`Seeks to amend CGST Rules notification no 3/2017-Central Tax`
  - rank 4: section_ref=`` doc_number=`3/2017-Centra` title=`Order for rejection / allowance of compounding of offence`
  - rank 5: section_ref=`2(11)(ii)` doc_number=`` title=`Integrated Goods And Services Tax Act, 2017`
- Verdict: none of top-5 contain the expected section/rule/notification in any metadata field. Real retrieval miss, not matcher bug.
