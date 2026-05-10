# CBIC Chunk Audit — cbic_v1

Date: 2026-04-22  |  Points: 108,802  |  Scroll time: 36.9s

## Red Flags

- 19.34% of chunks are over 2400 chars (imprecise retrieval, may exceed effective BGE-M3 window).
- Boilerplate contamination: top duplicate prefix repeats 273x. Sample: '| 1. | GSTIN |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | -'
- 83.17% of chunks missing section_ref (weak citation grounding).

## Length distribution

Median: 1298 chars (~324 tokens)  |  Mean: 1495.9 chars

| Bucket (chars) | Count | % |
|---|---:|---:|
| 0-100 | 25 | 0.02% |
| 100-300 | 11,510 | 10.58% |
| 300-600 | 13,762 | 12.65% |
| 600-1200 | 25,014 | 22.99% |
| 1200-2400 | 37,452 | 34.42% |
| 2400+ | 21,039 | 19.34% |

## Categories

| Category | Count | Median len | Mean len |
|---|---:|---:|---:|
| customs | 53,044 | 1501 | 1640.5 |
| gst | 41,356 | 932 | 1158.0 |
| central_excise | 8,940 | 1658 | 1792.4 |
| others | 2,952 | 2872 | 2556.2 |
| service_tax | 2,510 | 1483 | 1704.2 |

## Orphan headings (<80 chars + title + section_ref)

Count: 0 (0.0%)


## Tables

Count: 43,654  |  Median len: 636  |  Mean len: 1018.2

## Empty / near-empty

- Zero-length text: 0
- Under 20 chars: 2

## Near-duplicate prefixes (len 100-500, hash of text[:200])

| Count | Sample text (first 160 chars) | Sample doc_numbers |
|---:|---|---|
| 273 | \| 1. \| GSTIN \|  \|  \|  \|  \|  \|  \|  \|  \|  \|  \|  \|  \|  \|  \|  \|  \| \| --- \| --- \| --- \| --- \| --- \| --- \| --- \| --- \| --- \| --- \| --- \| --- \| --- \| --- \| --- \| --- \| | 3/2017-Centra, 3/2017-Centra, 3/2017-Centra |
| 188 | \| Tariff Item \| Description of goods \| Unit \| Drawback Rate \| Drawback cap per unit in Rs. (₹) \| \| --- \| --- \| --- \| --- \| --- \| \| (1) \| (2) \| (3) \| (4) \| (5) \| | 07/2020-CUSTOM, 07/2020-CUSTOM, 07/2020-CUSTOM |
| 170 | \| Description \| Central tax \| State/ UT tax \| Integrated tax \| Cess \| \| --- \| --- \| --- \| --- \| --- \| \| a) Tax/ Cess \|  \|  \|  \|  \| \| b) Interest \|  \|  \|  \|  \| \| | 3/2017-Centra, 3/2017-Centra, 3/2017-Centra |
| 139 | \| Turnover of zero rated supply of goods and services \| Net input tax credit \| Adjusted total turnover \| Refund amount (1×2÷3) \| \| --- \| --- \| --- \| --- \| \| 1 \| | 3/2017-Centra, 3/2017-Centra, 3/2017-Centra |
| 115 | \| 12119019- \| \| --- \| \| Seeds, Kernel, \| \| Aril, Fruit, \| \| Pericarp, \| \| Fruit rind, \| \| Endosperm, \| \| Mesocarp, \| \| Endocarp of \| \| column(1) \| | 15/2023-Custom, 15/2023-Custom, 15/2023-Custom |
| 104 | \| Description \| Tax \| Interest \| Penalty \| Fee \| Other \| Debit Entry Nos. \| \| --- \| --- \| --- \| --- \| --- \| --- \| --- \| \| 1 \| 2 \| 3 \| 4 \| 5 \| 6 \| 7 \| \| (a) Inte | 3/2017-Centra, 3/2017-Centra, 10/2017–Centra |
| 102 | \| Account Number \|  \|  \|  \|  \|  \|  \|  \|  \|  \|  \|  \|  \|  \|  \|  \| \| --- \| --- \| --- \| --- \| --- \| --- \| --- \| --- \| --- \| --- \| --- \| --- \| --- \| --- \| --- \| ---  | 3/2017-Centra, 3/2017-Centra, 3/2017-Centra |
| 95 | \| Sl. No. \| Foreign Currency \| Rate of exchange of 100 units of foreign currency equivalent to Indian rupees \|  \| \| --- \| --- \| --- \| --- \| \| (1) \| (2) \| (3) \|  | 40/2024-Custom, 26/2023-Custom, 66/2022-Custom |
| 82 | \| Sl.No. \| Foreign Currency \| Rate of exchange of 100 units of foreign currency equivalent to Indian rupees \|  \| \| --- \| --- \| --- \| --- \| \| (1) \| (2) \| (3) \|   | 97/2015-Custom, 81/2017-Custom, 37/2019-Custom |
| 73 | \| HS Code 2022 \| Description of product \| Working or processing, carried out on non-originating materials, which confers originating status \| \| --- \| --- \| ---  | 59/2025-Custom, 59/2025-Custom, 59/2025-Custom |

## Missing metadata

- section_ref missing: 90,491 (83.17%)
- parent_act missing:  3,753 (3.45%)
- category missing:    0 (0.0%)

## Top 20 parent_acts

| parent_act | count |
|---|---:|
| GST | 34,065 |
| Customs Act 1962 | 18,884 |
| CUSTOMS | 14,120 |
| Customs Tariff Act 1975 | 13,955 |
| Finance Act | 5,568 |
| Service Tax | 4,523 |
| Central Excise Act 1944 | 4,244 |
| CGST Act 2017 | 3,845 |
| CENTRAL_EXCISE | 2,434 |
| IGST Act 2017 | 1,278 |
| CGST Rules 2017 | 991 |
| UTGST Act 2017 | 833 |
| OTHERS | 250 |
| SERVICE_TAX | 59 |
