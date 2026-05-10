# Probes V1–V24 — Runbook

Each probe is standalone. All write to `/opt/indian-legal-ai/data/probes/v{N}_result.json` (rig-side) or `D:/_gpu_rig_ai/reingest_spec/v{N}_result.json` (laptop-runnable).

## Run order (once rig is back up)

### Can run from laptop (Qdrant HTTP only):
```
python probe_v6_snapshot.py
python probe_v7_disk.py         # coarse without rig shell
python probe_v9_gemini_quota.py
python probe_v15_payload_update.py
python probe_v17_judge.py
python probe_v21_dedup.py
python probe_v22_regex.py
```

### Must run on rig (SSH/ttyd needed):
```
# Wave B — LLM evaluation (resolves D1)
python3 probe_v1_qwen_json.py
python3 probe_v2_qwen_latency.py
python3 probe_v10_route_latency.py
python3 probe_v18_claude_cli.py

# Wave C — Chunker (requires chunker v2 implementation)
python3 probe_v3_langdetect.py
python3 probe_v4_bm25_hindi.py
python3 probe_v11_chunker_hierarchy.py
python3 probe_v13_chunker_timing.py
python3 probe_v20_taxonomy.py
python3 probe_v24_validator.py

# Wave D — Infra
python3 probe_v5_pool_soak.py       # 1-hour run
V8_SAMPLES="/path1.pdf:3,/path2.pdf:7,..." python3 probe_v8_table_ocr.py
python3 probe_v12_empty_pdfs.py
python3 probe_v19_api_stability.py  # needs concurrent mini-ingest

# Wave E — Gate validity
python3 probe_v14_mustcite.py
python3 probe_v23_api_refactor.py baseline   # before any api.py edit
# ... edit api.py to add /query_v2 ...
python3 probe_v23_api_refactor.py verify
```

## Deferred / TBD in scripts

- **V11, V24** need chunker v2 implementation (or adapter)
- **V17** `SAMPLES` list is 3 stubs — extend to 20 real triples from logs
- **V8** needs `V8_SAMPLES` env var with table-rich PDF paths
- **V19** needs a concurrent ingest session against `cbic_v2_test`
- **V23** two-phase: baseline then verify around api.py refactor

## Rollup after all probes

Collect every `v*_result.json`, tabulate pass/fail, update `PROBES.md` status column.
