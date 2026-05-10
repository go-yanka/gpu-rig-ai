# tariff_v1 — A4 patch (table extraction pipeline)

**Sentinel:** `tariff_v1`
**Generated:** 2026-04-21 21:32:09 (laptop local)
**Patch dir:** `D:\_gpu_rig_ai\patches\tariff_v1_20260421_213209\`
**Status:** SCAFFOLD ONLY — NOT DEPLOYED. Ship after A3 and A1 land.

## What this is

A NEW parallel pipeline for rate / HSN / SAC queries. Today, asking "GST rate on HSN 8471?" goes through semantic RAG and returns paraphrased guesses. This patch routes those queries to a SQLite tariff DB for exact-match lookups, and leaves RAG only for the interpretive paragraph around the rate.

Does NOT touch existing RAG code paths. Live backend `b11b12a8_v1` is unaffected.

## Files in this patch

| File | Purpose |
|---|---|
| `tariff_schema.sql` | SQLite DDL: `tariff_rate` table, indexes, FTS5 + triggers, `tariff_meta` sentinel. |
| `tariff_ingest.py`  | PDF-to-DB ingester. `init_db()` works. `extract_tables_from_pdf()` is TODO (Docling). |
| `tariff_query.py`   | Working query module. `is_rate_query()` + `lookup()`. No TODOs. |
| `tariff_endpoint.py`| FastAPI router: `POST /v1/rate-query`, `GET /v1/rate-query/health`. |
| `gold_set_tariff.yaml` | 10 rate-lookup gold items (separate from the 50-Q `gold_set.yaml`). |
| `apply.py`          | Deploy script (dry-run by default; `--yes` to execute). |
| `README.md`         | This file. |

## Target layout on rig

- DB file: `/opt/indian-legal-ai/tariff.db`
- Python modules: `/opt/indian-legal-ai/rag/cbic_rag/tariff_*.py`
- Gold file: `/opt/indian-legal-ai/rag/cbic_rag/gold_set_tariff.yaml`
- Endpoint: `POST http://192.168.1.107:9500/v1/rate-query`
- Health:   `GET  http://192.168.1.107:9500/v1/rate-query/health`

## Smoke tests

After deploy (run on laptop or rig):

```bash
# 1. DB sentinel (on rig)
sqlite3 /opt/indian-legal-ai/tariff.db "SELECT value FROM tariff_meta WHERE key='sentinel';"
# expect: tariff_v1

# 2. init-only dry run (on rig; safe, idempotent)
python /opt/indian-legal-ai/rag/cbic_rag/tariff_ingest.py --init-only

# 3. Router health
curl -s http://192.168.1.107:9500/v1/rate-query/health
# expect: {"sentinel":"tariff_v1","status":"ok"}

# 4. Rate lookup (will return empty all_matches until real PDFs ingested)
curl -s -X POST http://192.168.1.107:9500/v1/rate-query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the GST rate on HSN 8471?"}'
# expect: {"sentinel":"tariff_v1","rate_table_hit":null,...,"note":"No matching tariff rows..."}

# 5. Non-rate query should 400
curl -s -X POST http://192.168.1.107:9500/v1/rate-query \
  -H 'Content-Type: application/json' \
  -d '{"question":"what is input tax credit?"}'
# expect: HTTP 400
```

## Rollback

```bash
# 1. Remove tariff_v1 block from api.py (sentinel-guarded)
ssh user@192.168.1.107 "sed -i '/# BEGIN tariff_v1/,/# END tariff_v1/d' /opt/indian-legal-ai/rag/cbic_rag/api.py"

# 2. Restart service
ssh user@192.168.1.107 'sudo systemctl restart cbic-rag.service'

# 3. (Optional) drop the DB — only if you want to fully uninstall
ssh user@192.168.1.107 'mv /opt/indian-legal-ai/tariff.db /opt/indian-legal-ai/tariff.db.rollback.$(date +%s)'

# 4. (Optional) remove module files
ssh user@192.168.1.107 'cd /opt/indian-legal-ai/rag/cbic_rag && rm -f tariff_schema.sql tariff_ingest.py tariff_query.py tariff_endpoint.py gold_set_tariff.yaml'
```

The DB file removal is reversible up to the rename — it is not deleted.

## Follow-up TODOs (not in this patch)

1. `pip install docling` on rig (ROCm wheel if possible).
2. Implement `extract_tables_from_pdf()` in `tariff_ingest.py` against 2–3 real CBIC rate notification PDFs (start with `1/2017-CT(Rate)` and its amendments).
3. Forward-fill logic for merged cells / chapter headers.
4. Build an `index.json` mapping PDF → (notification_id, effective_from, effective_to).
5. Wire the interpretive-answer step in `tariff_endpoint.py`: call existing RAG with `filter={notification_id: best.notification_id}`.
6. Fill exact `expected_rate_igst_any_of` / `expected_notification_any_of` in `gold_set_tariff.yaml` from ingested data.
7. Add eval harness run mode that scores rate-lookup hits (exact-match, not BERTScore).
8. Dashboard surface: show `rate_table_hit` as a table, not a paragraph.

## Design notes / divergences from spec

- Added `tariff_meta` table with `sentinel='tariff_v1'` — gives a cheap SQL check that the right schema is applied, parallel to how router exposes `{sentinel: tariff_v1}`.
- Added `idx_tariff_notif` index on `notification_id` — needed once the follow-up RAG scoping step filters by notification.
- `HSN_RE` / `SAC_RE` accept optional `:` or `-` separators (`HSN: 8471`, `HSN-8471`) — common in user questions.
- `RATE_RE` expanded to include `tariff`, `duty`, and `applicable to`.
- Added `GET /v1/rate-query/health` — gives apply.py a one-shot mount verification without a POST body.
- `apply.py` is dry-run by default; requires explicit `--yes` to execute. api.py patch is sentinel-guarded (`# BEGIN tariff_v1`) so re-running is idempotent.
- Service restart in apply.py is intentionally left as a printed manual step, not auto-executed.
