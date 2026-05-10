# tariff_db — Structured sidecar for CBIC RAG (P2.1)

Local-only SQLite that answers bucket-1 tariff/rate/HSN/notification-SNo lookups directly, bypassing RAG for near-100% accuracy and sub-ms latency.

Status: **prototype, not deployed.** Seed data only. Nothing has been touched on the rig.

## Layout

```
tariff_db/
  schema.sql                          CREATE TABLE + INDEX (5 tables per p2_1 design)
  ingest.py                           init + loaders + verify driver
  router.py                           maybe_route(query) -> (rows | None, reason)
  test_router.py                      11 gold items, currently 11/11 pass
  tariff.db                           built artefact (~80 KB with seed, target ~40-60 MB at scale)
  seed/
    codes.csv                         110 HSN/SAC entries spanning chapters 1-99
    notifications.csv                 5 notifs: 01, 02, 05, 13 / 2017-CT(R) + 50/2017-Cus
    rates_01_2017_CTR.csv             108 rate rows across Sch I-VI (CGST+SGST+IGST triples)
    list_02_2017_CTR_exempt.csv       13 exempt-supply rows
    list_05_2017_CTR_inverted.csv     2 inverted-duty rows (ch. 54)
    list_13_2017_CTR_rcm.csv          7 RCM rows (goods + services)
    exemptions_50_2017_Cus.csv        10 customs BCD exemption S.No. rows
```

## Build & test

```bash
cd D:/_gpu_rig_ai/tariff_db
python ingest.py build          # init + load seeds + run verify queries
python test_router.py           # 11/11 should PASS
```

## Adding more notifications

### Rate notifications (01/2017-CT(R) amendments, etc.)

1. Append a row to `seed/notifications.csv` for the new `notif_id`.
2. Drop a CSV into `seed/` with the rates-table header:

   ```
   code,levy_type,rate_pct,rate_specific,condition_no,schedule,sno,effective_from,effective_to
   ```

3. Add a line to `ingest.py::build_all`:

   ```python
   load_notification("18/2021-CT(R)", SEED / "rates_18_2021_CTR.csv", conn)
   ```

4. For supersessions (rate changes), set `effective_to` on the outgoing rows in the existing CSV, then insert the new rows with `effective_from` = amendment date and `effective_to` = NULL.

### List-membership notifications (RCM, exempt, inverted-duty)

Same process, CSV header:

```
code,list_type,sno,description,effective_from,effective_to
```

`list_type` ∈ `RCM | INVERTED_DUTY | EXEMPT | NIL | COMP_CESS_LEVIABLE`.

`code` can be NULL when the list item is a service description (13/2017 pure-services rows).

### Customs exemption S.No. tables (50/2017-Cus, 12/2017-CT(R))

Use `load_exemption_table(notif_id, csv_path, conn)`. Header:

```
sno,code,description,std_rate,igst_rate,condition_no,effective_from,effective_to
```

### PDF ingestion

Not implemented in this stub. Hook a `pdfplumber` or `tabula-py` parser into `ingest.py::load_notification` when the source is `.pdf`. The raw CBIC PDFs live on the rig at `/opt/indian-legal-ai/data/scraped/cbic/{gst,customs}/notifications/<year>/`.

## Router contract

```python
from tariff_db.router import maybe_route

rows, reason = maybe_route("GST rate on HSN 1006")
# rows = [{'levy_type': 'CGST', 'rate_pct': 2.5, ...}, ...]
# reason = 'rate-lookup:code=1006:date=9999-12-31'
```

Return values:

| `rows`       | `reason`                   | Caller action                                  |
|--------------|----------------------------|-----------------------------------------------|
| `None`       | `'no-match'`               | Fall through to RAG                           |
| `[]`         | `'<rule>-empty'`           | Fall through to RAG; optionally log the miss  |
| non-empty    | `'<rule>:…'`               | Format rows + cite `notif_id` in the response |

**Never fail closed** — router always allows a RAG fallback.

## Wiring into `api.py` (not yet done)

Sketch:

```python
from tariff_db.router import maybe_route
import sqlite3

TARIFF_CONN = sqlite3.connect("tariff_db/tariff.db", check_same_thread=False)

@app.post("/query")
async def query(req: QueryReq):
    rows, reason = maybe_route(req.q, TARIFF_CONN)
    if rows:
        return format_sql_answer(rows, reason)   # authoritative, cite notif_id
    # rows is None or [] -> fall through to existing RAG
    return await rag_answer(req)
```

Latency budget: SQL path < 2 ms on indexed lookup; RAG path unchanged. Measure SQL-hit rate, SQL-accuracy vs bucket-1 gold, and fall-through latency before/after.

## Gaps (deliberate — route to RAG)

- Advance Rulings / HSN classification disputes
- Procedural narrative ("how to file a refund")
- Circulars clarifying scope
- Cross-notification narratives (GST + cess + BCD combined)
- Anti-dumping / safeguards (add later)
- Place-of-supply / composition scheme

## Next steps when user resumes

1. Parse real CBIC 01/2017-CT(R) Schedules I-VI into full rates CSV (~80 k rows) — requires robust PDF-table parser. Rig has `pdfplumber`; can run as an offline batch.
2. Back-test router against the full 25-item bucket-1 gold set once that set is finalised.
3. Wire into `api.py` behind a feature flag; compare before/after eval scores.
4. Daily cron to pick up new CBIC notifications from `/opt/indian-legal-ai/data/scraped/cbic/*/notifications/` and queue them for ingestion.
