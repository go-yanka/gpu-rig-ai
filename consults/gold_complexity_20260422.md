# Gold Set Complexity Audit — 2026-04-22

Input: `D:\_gpu_rig_ai\eval\gold_set.yaml` (170 items)
Recall data: `D:\_gpu_rig_ai\eval\recall_audit_20260422.jsonl`

## Total distribution

| Bucket | N | % |
|---|---:|---:|
| LOW | 80 | 47.1% |
| MEDIUM | 79 | 46.5% |
| HIGH | 11 | 6.5% |
| **Total** | **170** | **100.0%** |

## Per-category distribution

| Category | LOW | MEDIUM | HIGH | Total |
|---|---:|---:|---:|---:|
| central_excise | 6 | 4 | 0 | 10 |
| customs | 20 | 23 | 2 | 45 |
| gst | 33 | 41 | 7 | 81 |
| others | 11 | 2 | 1 | 14 |
| service_tax | 10 | 9 | 1 | 20 |

## Examples per bucket

### LOW
- **gst_cs_002** (cat=gst, n_ent=2, q_len=22, score=0): A gift hamper contains chocolates, a fruit juice bottle, and a toy, sold at one price for Diwali. Composite or mixed supply?
- **gst_itc_001** (cat=gst, n_ent=1, q_len=22, score=-1): Can a registered taxpayer claim ITC on GST paid for the purchase of a motor car (seating capacity 5) used for business?
- **gst_itc_004** (cat=gst, n_ent=2, q_len=16, score=0): Is ITC available on GST paid for office renovation of a rented building (capitalized in books)?

### MEDIUM
- **gst_pos_001** (cat=gst, n_ent=2, q_len=31, score=2): A registered supplier in Maharashtra sells goods to a registered buyer in Karnataka. Goods are shipped directly from Maharashtra to Karnataka. What is the place of supply and which tax applies?
- **gst_pos_003** (cat=gst, n_ent=1, q_len=21, score=1): Services of a chartered accountant in Pune are rendered to a registered client in Gujarat. What is the place of supply?
- **gst_pos_004** (cat=gst, n_ent=1, q_len=20, score=2): An event-management company organizes a conference in Bengaluru for a client registered in Delhi. Where is the place of supply?

### HIGH
- **gst_pos_002** (cat=gst, n_ent=1, q_len=28, score=4): A buyer in Delhi instructs a Mumbai supplier to deliver goods directly to the buyer's customer in Chennai (bill-to-ship-to). What is the place of supply for each leg?
- **gst_pos_005** (cat=gst, n_ent=4, q_len=74, score=8): Quantum Tech Pvt Ltd (Karnataka GSTIN) receives a purchase order from a UK buyer. The UK buyer asks Quantum Tech to deliver the goods directly to a Mumbai-based third party (the UK buyer's Indian agent). Quantum Tech raises the invoice on the UK buyer in USD and receives an advance of USD 50,000. Determine: (a) place of supply for the goods, (b) whether this is export of goods, (c) tax treatment of the advance.
- **oth_pen_001** (cat=others, n_ent=4, q_len=58, score=5): A taxpayer fails to furnish a tax invoice for a supply of Rs 10 lakh (tax Rs 1.8 lakh) and the officer finds this during an inspection. The taxpayer claims there was no intent to evade. Determine: (a) applicable penalty provision, (b) quantum, (c) whether Sec 73 or Sec 74 applies, (d) remedy to avoid penalty before SCN.

## Recall@5 by complexity bucket

| Bucket | N evaluated | recall@5 | recall@1 | missing from audit |
|---|---:|---:|---:|---:|
| LOW | 80 | 12.5% | 2.5% | 0 |
| MEDIUM | 79 | 11.4% | 5.1% | 0 |
| HIGH | 11 | 18.2% | 0.0% | 0 |

## Interpretation

- Overall recall@5 across audited items: 12.4% (21/170)
- LOW vs HIGH delta: -5.7 pp (LOW 12.5% vs HIGH 18.2%)
- **Failure concentrates on LOW complexity** — unexpected; likely a term-mismatch / entity-resolution issue, not a reasoning issue.
