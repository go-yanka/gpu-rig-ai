# Failure-Mode Taxonomy — Recall Audit 2026-04-22

Source: `D:\_gpu_rig_ai\eval\recall_audit_20260422.jsonl`  
Misses analyzed: **149** (of 170 items)

## Taxonomy table

| Failure mode | Count | % of misses |
|---|---:|---:|
| Right_act_wrong_section | 88 | 59.1% |
| Notation_mismatch | 43 | 28.9% |
| Wrong_domain | 10 | 6.7% |
| Procedural_vs_substantive_confusion | 5 | 3.4% |
| Rate_notification_miss | 3 | 2.0% |

## Per-category breakdown

| category | total | Right_act_wrong_section | Notation_mismatch | Wrong_domain | Procedural_vs_substantive_confusion | Rate_notification_miss |
|---|---|---|---|---|---|---|
| central_excise | 8 | 3 | 5 | 0 | 0 | 0 |
| customs | 38 | 19 | 18 | 0 | 0 | 1 |
| gst | 75 | 44 | 17 | 8 | 4 | 2 |
| others | 11 | 7 | 1 | 2 | 1 | 0 |
| service_tax | 17 | 15 | 2 | 0 | 0 | 0 |

## Examples (up to 5 per mode)

### Right_act_wrong_section  (88)

- **gst_pos_002** (gst): A buyer in Delhi instructs a Mumbai supplier to deliver goods directly to the buyer's customer in Chennai (bill-to-ship-to). What is the place of supply for eac…
  - expected: `['10(1)(b) IGST']`
  - top-3 titles:
    - Integrated Goods And Services Tax Act, 2017
    - Seeks to prescribe Compensation cess  rate of 0.1% on supply of taxable goods by
    - Integrated Goods And Services Tax Act, 2017
  - evidence: act overlap ['igst']; expected_sec=['10'] top_sec=['2', '9']
- **gst_pos_003** (gst): Services of a chartered accountant in Pune are rendered to a registered client in Gujarat. What is the place of supply?
  - expected: `['12(2) IGST']`
  - top-3 titles:
    - Continuation of Recovery Proceedings
    - Central Goods and Services Tax Act, 2017
    - Seeks to amend CGST Rules notification no 3/2017-Central Tax dt 19.06.2017
  - evidence: act overlap ['igst']; expected_sec=['12'] top_sec=['2']
- **gst_pos_004** (gst): An event-management company organizes a conference in Bengaluru for a client registered in Delhi. Where is the place of supply?
  - expected: `['12(7) IGST']`
  - top-3 titles:
    - Integrated Goods and Services Tax Rules, 2017
    - Order for rejection / allowance of compounding of offence
    - Order for acceptance/rejection of application for deferred payment / payment in 
  - evidence: act overlap ['igst']; expected_sec=['12'] top_sec=['2', '3']
- **gst_cs_002** (gst): A gift hamper contains chocolates, a fruit juice bottle, and a toy, sold at one price for Diwali. Composite or mixed supply?
  - expected: `['2(74) CGST', '8(b) CGST']`
  - top-3 titles:
    - Central Goods and Services Tax Act, 2017
    - Central Goods and Services Tax Act, 2017
    - Circular clarifying various doubts related to treatment of sales promotion schem
  - evidence: act overlap ['cgst']; expected_sec=['2', '8'] top_sec=[]
- **gst_tos_001** (gst): A supplier of goods receives an advance of Rs 5 lakhs on 10 April 2025 against a future supply. Is GST payable on the advance?
  - expected: `['12(2) CGST', '66/2017-Central Tax']`
  - top-3 titles:
    - Authorisation / withdrawal of authorisation for Goods and Services Tax Practitio
    - All Forms - Hindi
    - Monthly return
  - evidence: no rule matched; default bucket

### Notation_mismatch  (43)

- **gst_pos_005** (gst): Quantum Tech Pvt Ltd (Karnataka GSTIN) receives a purchase order from a UK buyer. The UK buyer asks Quantum Tech to deliver the goods directly to a Mumbai-based…
  - expected: `['10(1)(b) IGST', '2(5) IGST', '13(8) CGST', 'Section 7 IGST']`
  - top-3 titles:
    - Integrated Goods And Services Tax Act, 2017
    - Seeks to supercede notification number 21/2002-customs dated 01.03.2002
    - Regarding detailed manual scrutiny of Service Tax Returns
  - evidence: expected section(s) ['10', '13', '2', '7'] present in top-k section_ref/title
- **gst_cs_001** (gst): A hotel provides a package of room + breakfast + airport transfer for a single price. How is this taxed — as composite supply or mixed supply?
  - expected: `['2(30) CGST', '8(a) CGST']`
  - top-3 titles:
    - Central Goods and Services Tax Act, 2017
    - Clarifications regarding levy of GST on accommodation services, betting and gamb
    - Clarifications regarding levy of GST on accommodation services, betting and gamb
  - evidence: expected section(s) ['2', '8'] present in top-k section_ref/title
- **gst_itc_001** (gst): Can a registered taxpayer claim ITC on GST paid for the purchase of a motor car (seating capacity 5) used for business?
  - expected: `['17(5)(a) CGST']`
  - top-3 titles:
    - Clarification on availability of input tax credit in respect of demo vehicles.
    - Clarification on various issue pertaining to GST
    - Clarification on refund related issues. Rescinded vide Circular No. 125/44/2019 
  - evidence: expected section(s) ['17'] present in top-k section_ref/title
- **gst_itc_002** (gst): A manufacturer buys capital goods in April 2024 and claims full ITC. In March 2026 the capital goods are sold as scrap. What is the reversal liability?
  - expected: `['18(6) CGST', 'Rule 40(2)', 'Rule 44(6)']`
  - top-3 titles:
    - Clarification in respect of GST liability and input tax credit (ITC) availabilit
    - Extension of Deferred Payment of Customs Duty benefits to ‘Eligible Manufacturer
    - Orders of Supreme Court, High Courts and CESTAT accepted by the Department and o
  - evidence: expected section(s) ['18', '40', '44'] present in top-k section_ref/title
- **cus_val_001** (customs): What is the primary method of valuation for imported goods under the Customs Act?
  - expected: `['14 Customs Act', 'Rule 3 CVR 2007']`
  - top-3 titles:
    - This relates to Customs valuation (Determination of Price of Imported Goods) Rul
    - Customs valuation (Determination of Value of Export Goods) Rules, 2007- Instruct
    - Customs Valuation (Determination of Value of Imported Goods) Rules, 2007 - Instr
  - evidence: expected section(s) ['14', '200', '3'] present in top-k section_ref/title

### Wrong_domain  (10)

- **gst_pos_001** (gst): A registered supplier in Maharashtra sells goods to a registered buyer in Karnataka. Goods are shipped directly from Maharashtra to Karnataka. What is the place…
  - expected: `['10(1)(a) IGST', 'Section 10 IGST']`
  - top-3 titles:
    - Seeks to supersede 31 customs exemption notifications and prescribes effective r
    - Seeks to prescribe Central Tax rate of 0.05% on intra-State supply of taxable go
    - Seeks to prescribe Compensation cess  rate of 0.1% on supply of taxable goods by
  - evidence: expected_acts=['igst'] topk_acts=['cess', 'customs'] disjoint
- **gst_itc_004** (gst): Is ITC available on GST paid for office renovation of a rented building (capitalized in books)?
  - expected: `['17(5)(d) CGST', '17(5)(c) CGST']`
  - top-3 titles:
    - Clarification on refund related issues. Rescinded vide Circular No. 125/44/2019 
    - All Forms - Hindi
    - Issuance of SCNS in Time Bound Manner– Regarding.
  - evidence: expected_acts=['cgst'] topk_acts=['cess'] disjoint
- **gst_rcm_002** (gst): A GTA provides transport services to a registered manufacturer and charges 5% GST on forward-charge basis having filed Annexure V. Is RCM applicable to the reci…
  - expected: `['9(3) CGST', '13/2017-Central Tax (Rate)', '11/2017-Central Tax (Rate)']`
  - top-3 titles:
    - Seeks to amend notification no. 22/2022-Customs to notify the fifth tranche of t
    - Seeks to amend Notification No 8/2017- Integrated Tax (Rate) dated 28.06.2017
    - Seeks to amend notification no. 22/2022-Customs to notify the fifth tranche of t
  - evidence: expected_acts=['cgst'] topk_acts=['customs'] disjoint
- **oth_gaar_001** (others): Does GAAR (General Anti-Avoidance Rule) apply under GST law, or is it only under the Income Tax Act?
  - expected: `['Chapter X-A Income Tax Act']`
  - top-3 titles:
    - Seeks to clarify verification for grant of new registration.
    - Notification No. 1/2018 dated 14.11.2018 which notifies the list of Acts of Cent
    - Notification No. 1/2018 dated 14.11.2018 which notifies the list of Acts of Cent
  - evidence: expected_acts=['income'] topk_acts=['cess', 'cgst'] disjoint
- **gst_exempt_002** (gst): An exporter ships cane sugar (HSN 1701) to the UK. While sugar is taxable domestically at 5%, what is the GST treatment on this export, and can they claim ITC o…
  - expected: `['Section 16 IGST Act']`
  - top-3 titles:
    - Seeks to supersede 31 customs exemption notifications and prescribes effective r
    - Circular No. 52/26/2018-GST dated 09.08.2018 i.r.o. clarification regarding appl
    - Fourteenth amendment to the CGST Rules, 2017.
  - evidence: expected_acts=['igst'] topk_acts=['cgst', 'customs'] disjoint

### Procedural_vs_substantive_confusion  (5)

- **oth_ap_002** (others): What is the time limit for filing an appeal to the First Appellate Authority under the CGST Act?
  - expected: `['107 CGST Act']`
  - top-3 titles:
    - Guidelines for recovery of outstanding dues, in cases wherein first appeal has b
    - Central Goods and Services Tax Act, 2017
    - Central Goods and Services Tax Act, 2017
  - evidence: procedural question; only 20% of top-k look procedural
- **oth_penalty_001** (gst): A truck carrying taxable goods without an e-way bill is intercepted. The owner of the goods voluntarily comes forward to pay the applicable tax and penalty. Und…
  - expected: `['Section 129(1)(a) CGST Act']`
  - top-3 titles:
    - clarifying the procedure for interception of conveyances for inspection of goods
    - Seeks to amend notification no. 22/2022-Customs to notify the fifth tranche of t
    - Central Goods and Services Tax Act, 2017
  - evidence: procedural question; only 20% of top-k look procedural
- **oth_advance_ruling_002** (gst): A taxpayer files an application before the Authority for Advance Ruling (AAR) asking whether a notice issued to them for a delay in filing GSTR-3B is legally va…
  - expected: `['Section 97(2) CGST Act']`
  - top-3 titles:
    - Order of Authority for Advance Rulings (Customs & Central Excise) in respect of 
    - Continuation of Recovery Proceedings
    - Application for refund by Canteen Stores Department
  - evidence: procedural question; only 20% of top-k look procedural
- **oth_advance_ruling_004** (gst): Company Z receives an adverse Advance Ruling from the state AAR. They want to appeal the decision to the Appellate Authority for Advance Ruling (AAAR). Under Se…
  - expected: `['Section 100(2) CGST Act']`
  - top-3 titles:
    - Central Goods and Services Tax Act, 2017
    - Central Goods and Services Tax Rules, 2017
    - Central Goods and Services Tax Rules, 2017
  - evidence: procedural question; only 0% of top-k look procedural
- **oth_interest_002** (gst): A taxpayer files a valid refund application under Section 54. The proper officer sanctions and disburses the refund 85 days after the receipt of the application…
  - expected: `['Section 56 CGST Act']`
  - top-3 titles:
    - Seeks to clarify the fully electronic refund process through FORM GST RFD-01 and
    - The Finance Act, 2001
    - Central Goods and Services Tax Act, 2017
  - evidence: procedural question; only 20% of top-k look procedural

### Rate_notification_miss  (3)

- **gst_inverted_001** (gst): A manufacturer of electric railway locomotives (HSN 8601) procures electrical components and parts taxed at 18% GST, while their output is taxed at a lower rate…
  - expected: `['Section 54(3) CGST Act', '05/2017-CT(R)']`
  - top-3 titles:
    - Notice to a third person under section 79(1) (c)
    - Seeks to clarify the fully electronic refund process through FORM GST RFD-01 and
    - Intimation to Liquidator for recovery of amount
  - evidence: rate/notif question; expected notification; top-k dominated by Act/Rule chunks
- **gst_complex_012** (gst): A supplier receives an advance payment for services on 20th June. The services are completed on 15th July, and the invoice is issued on 10th August. When is the…
  - expected: `['Section 13(2) CGST Act', 'Section 12(2) CGST Act', 'Rule 47 CGST Rules', '66/2017-CT']`
  - top-3 titles:
    - Order for rejection / allowance of compounding of offence
    - Central Goods and Services Tax Act, 2017
    - Central Goods and Services Tax Act, 2017
  - evidence: rate/notif question; expected notification; top-k dominated by Act/Rule chunks
- **customs_complex_017** (customs): An exporter imports raw materials duty-free under an Advance Authorisation scheme. They manufacture finished goods and export them. Later, they realize they exp…
  - expected: `['Section 28 Customs Act', 'Advance Authorisation Notification (e.g., 18/2015-Cus)']`
  - top-3 titles:
    - Annexure-B
    - To notify AIR of Duty Drawback w.e.f. 21.9.2013Drawback Schedule-2013-14
    - Regarding implementation of Advance Authorisation Scheme for deemed export under
  - evidence: rate/notif question; expected notification; top-k dominated by Act/Rule chunks

## Intervention fix-rate estimate

| Intervention | Wrong_domain | Right_act_wrong_section | Notation_mismatch | Rate_notification_miss | Procedural_vs_substantive | No_authoritative_chunk |
|---|---|---|---|---|---|---|
| **meta_filter** (parent_act/doc_type) | FIX (high) | partial | partial | FIX (filter to notification) | partial | no |
| **HyDE** (hypothetical doc expansion) | partial | FIX (high) | partial | partial | FIX (clarifies intent) | no |
| **Fine-tune** embedder on CBIC pairs | FIX | FIX | small | partial | FIX | no |
| **Hard negatives** (same-act wrong-section) | small | FIX (high) | no | no | partial | no |
| **Matcher fix** (multi-field substring) | no | no | FIX (complete) | no | no | no |
| **Corpus ingest** (missing docs) | no | no | no | partial | no | FIX |

## Recommendation — target these first

1. **Right_act_wrong_section** (88 misses, 59%) — see fix-rate row above for best intervention.
2. **Notation_mismatch** (43 misses, 29%) — second priority.

- Right_act_wrong_section: Hard-negative mining within the same parent_act + HyDE give the largest lift.
- Notation_mismatch: Matcher fix is near-zero-cost and fully recovers these; do it before any model work.