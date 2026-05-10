# P2.1 tariff.db — Schema & Ingestion Plan

**Date:** 2026-04-21
**Status:** Draft design (agent output). To be wired after P1 eval ≥40%.
**Purpose:** SQLite sidecar for structured tariff/rate/HSN/notification lookups, bypassing RAG for near-100% accuracy on bucket-1-style queries.

---

## Schema

```sql
-- 1. Master code list (HSN goods + SAC services, all granularities)
CREATE TABLE codes (
  code           TEXT PRIMARY KEY,          -- '1006', '10063020', '996331'
  code_type      TEXT NOT NULL,             -- 'HSN' | 'SAC'
  level          INTEGER NOT NULL,          -- 2/4/6/8 for HSN, 4/6 for SAC
  parent_code    TEXT REFERENCES codes(code),
  description    TEXT NOT NULL,
  chapter        INTEGER                    -- 1..99 HSN chapter
);

-- 2. Notifications (provenance spine)
CREATE TABLE notifications (
  notif_id       TEXT PRIMARY KEY,          -- '01/2017-CT(R)', '50/2017-Cus'
  series         TEXT NOT NULL,             -- 'CT(R)','IT(R)','Cus','Cus(ADD)','Comp-Cess'
  number         INTEGER NOT NULL,
  year           INTEGER NOT NULL,
  issued_on      DATE NOT NULL,
  effective_from DATE NOT NULL,
  superseded_by  TEXT REFERENCES notifications(notif_id),
  title          TEXT,
  source_doc_id  TEXT                       -- link to RAG chunk / PDF path
);

-- 3. Rate history (bi-temporal: one row per (code, levy, period))
CREATE TABLE rates (
  rate_id        INTEGER PRIMARY KEY,
  code           TEXT NOT NULL REFERENCES codes(code),
  levy_type      TEXT NOT NULL,             -- 'CGST','SGST','IGST','BCD','SWS','COMP_CESS','AIDC'
  rate_pct       REAL,                      -- NULL if specific/ad-valorem+specific
  rate_specific  TEXT,                      -- 'Rs 400/tonne', free text for complex
  condition_no   TEXT,
  schedule       TEXT,                      -- Sch I/II/III/IV/V/VI for 01/2017-CT(R)
  sno            INTEGER,
  effective_from DATE NOT NULL,
  effective_to   DATE,                      -- NULL = currently in force
  notif_id       TEXT NOT NULL REFERENCES notifications(notif_id),
  amended_by     TEXT REFERENCES notifications(notif_id)
);

-- 4. List memberships (RCM, inverted-duty, exempt, nil-rated)
CREATE TABLE list_membership (
  id             INTEGER PRIMARY KEY,
  code           TEXT REFERENCES codes(code),
  list_type      TEXT NOT NULL,             -- 'RCM','INVERTED_DUTY','EXEMPT','NIL','COMP_CESS_LEVIABLE'
  sno            INTEGER,
  description    TEXT,                      -- when code is NULL (service descriptions)
  notif_id       TEXT NOT NULL REFERENCES notifications(notif_id),
  effective_from DATE NOT NULL,
  effective_to   DATE
);

-- 5. Exemption entries (50/2017-Cus style S.No. table)
CREATE TABLE exemptions (
  id             INTEGER PRIMARY KEY,
  notif_id       TEXT NOT NULL REFERENCES notifications(notif_id),
  sno            INTEGER NOT NULL,
  code           TEXT REFERENCES codes(code),
  description    TEXT NOT NULL,
  std_rate       TEXT,
  igst_rate      TEXT,
  condition_no   TEXT,
  effective_from DATE NOT NULL,
  effective_to   DATE,
  UNIQUE(notif_id, sno, code)
);
```

## Indices

```sql
CREATE INDEX idx_rates_code_date ON rates(code, effective_from, effective_to);
CREATE INDEX idx_rates_notif ON rates(notif_id);
CREATE INDEX idx_list_code_type ON list_membership(list_type, code);
CREATE INDEX idx_exempt_notif_sno ON exemptions(notif_id, sno);
CREATE INDEX idx_codes_chapter ON codes(chapter, level);
```

## Ingestion Sources

| Table | Source |
|---|---|
| `codes` | Customs Tariff Act First Schedule (HSN), CBIC SAC scheme list |
| `notifications` | Metadata scrape of all CBIC `*-CT(R)`, `*-IT(R)`, `*-Cus`, Comp-Cess notifs |
| `rates` | 01/2017-CT(R) Sch I-VI, 02/2017-CT(R) nil, Customs Tariff BCD, Comp-Cess, every amending notif |
| `list_membership` | 13/2017-CT(R) (RCM), 05/2017-CT(R) (inverted duty), 02/2017-CT(R) (exempt) |
| `exemptions` | 50/2017-Cus + amendments, service exemption 12/2017-CT(R) |

## Sample Queries (against bucket-1 gold set)

```sql
-- GST rate on HSN 1006 as of 2023-01-01
SELECT levy_type, rate_pct, notif_id FROM rates
WHERE code='1006' AND levy_type IN ('CGST','SGST','IGST')
  AND effective_from<='2023-01-01' AND (effective_to IS NULL OR effective_to>'2023-01-01');

-- Is HSN 8703 on RCM currently?
SELECT notif_id, sno FROM list_membership
WHERE code='8703' AND list_type='RCM' AND effective_to IS NULL;

-- Latest rate change for HSN 2202
SELECT notif_id, effective_from, rate_pct FROM rates
WHERE code='2202' ORDER BY effective_from DESC LIMIT 1;

-- 50/2017-Cus S.No. 404 detail
SELECT code, description, std_rate, condition_no FROM exemptions
WHERE notif_id='50/2017-Cus' AND sno=404;

-- All inverted-duty goods in chapter 54
SELECT lm.code, c.description FROM list_membership lm JOIN codes c USING(code)
WHERE lm.list_type='INVERTED_DUTY' AND c.chapter=54 AND lm.effective_to IS NULL;
```

## Gaps (keep RAG)

- Advance Rulings / case-law interpretation of HSN classification disputes
- Procedural questions ("how to file refund under inverted duty") — narrative
- Circulars clarifying scope — reasoning text
- Cross-notification interactions (GST + cess + BCD combined narrative)
- Anti-dumping / safeguard duties (separate regime — add later if needed)
- Place-of-supply / composition-scheme rate questions

## Router Rule

Route to SQL if query matches any:
```
\b(HSN|SAC|chapter heading)\s*\d{2,8}\b
\b(rate|duty|BCD|IGST|CGST|compensation cess)\s+(on|of|for)\b
\b(notification|notif)\s*\d{1,3}/\d{4}\b
\bS\.?\s*No\.?\s*\d+\b.*\b(50/2017|01/2017|12/2017)\b
\b(RCM|reverse charge|inverted duty|negative list|nil rated)\b
```
Fallback: small prompt "is this a lookup on structured tariff data? y/n". On SQL miss or empty → fall through to RAG (never fail closed).

## Size Estimate

| Table | Rows |
|---|---|
| `codes` | ~14k |
| `notifications` | ~1.5k |
| `rates` | ~80k |
| `list_membership` | ~3k |
| `exemptions` | ~5k |

**Total DB size: ~40-60 MB**, fits entirely in page cache. Sub-ms lookups trivially achievable.

---

## Next steps

1. After P1 eval ≥40%, write ingestion scripts per-notification
2. Build router shim in `api.py` (regex match → SQL attempt → fall back to RAG on miss)
3. Back-test against bucket-1 gold set: each of 25 items should resolve via SQL or get a clean fall-through
4. Measure: SQL-hit rate, SQL-accuracy, fall-through latency impact
