# A3 Two-Pass Structured Extraction — Patch Bundle

- **Sentinel:** `two_pass_v1`
- **Timestamp:** `20260421_213026`
- **Feature flag:** `TWO_PASS_ENABLED` (env var). `0` (default) = legacy single-shot byte-identical. `1` = new two-pass path.
- **Target:** CBIC RAG on `192.168.1.107` at `/opt/indian-legal-ai/rag/cbic_rag/`.

## Files in this bundle

| File | Purpose |
|---|---|
| `apply.py` | Idempotent ssh/scp deploy script. Does NOT restart the service. |
| `api.patched.py` | Full rewrite of rig `api.py`: adds `two_pass_generate`, gates at step 5 on `TWO_PASS_ENABLED`, extends `_call_llm` with optional `response_format`. |
| `storyformat.patched.py` | Full rewrite of rig `storyformat.py`: adds `EXTRACTION_SYS`, `SYNTHESIS_SYS`, `DECOMP_VERIFY_SYS`, `DECOMP_SYS`, JSON schemas, `build_extraction_user`, `build_synthesis_user`, `build_decomp_user`, `build_citations`. All legacy exports (`SYS_PROMPT`, `build_prompt`, `build_response_payload`, `verify_quotes`) preserved verbatim. |
| `validator.py` | New file on rig. `validate_span(span, chunk_text, is_table) -> (ok, reason)`. Ladder: length + clause terminator → canon substring → table bypass → 6-gram Jaccard ≥ 0.80 → BGE-M3 cosine ≥ 0.92. |
| `README.md` | This file. |

## Files touched on rig (after manual deploy)

- **Modified:** `/opt/indian-legal-ai/rag/cbic_rag/api.py`
- **Modified:** `/opt/indian-legal-ai/rag/cbic_rag/storyformat.py`
- **New:**      `/opt/indian-legal-ai/rag/cbic_rag/validator.py`
- **Backups:** `api.py.bak.two_pass_v1.20260421_213026`, `storyformat.py.bak.two_pass_v1.20260421_213026` (created by `apply.py`, only if not already present)
- **Sidecars:** `*.patched.two_pass_v1.20260421_213026` (exact copy of what was uploaded, for diffing)

## Deploy (manual, after you're ready)

```bash
cd "D:\_gpu_rig_ai\patches\two_pass_v1_20260421_213026"
python apply.py
# … review output; nothing is restarted.

# Phase 1: restart with flag OFF — must be byte-identical to current live
ssh user@192.168.1.107 'sudo systemctl restart cbic-rag'

# Phase 2: flip flag ON
ssh user@192.168.1.107 'sudo systemctl edit cbic-rag'
#   add: Environment=TWO_PASS_ENABLED=1
ssh user@192.168.1.107 'sudo systemctl daemon-reload && sudo systemctl restart cbic-rag'
```

## Rollback (any time, any phase)

```bash
ssh user@192.168.1.107 \
  'cp /opt/indian-legal-ai/rag/cbic_rag/api.py.bak.two_pass_v1.20260421_213026 /opt/indian-legal-ai/rag/cbic_rag/api.py \
   && cp /opt/indian-legal-ai/rag/cbic_rag/storyformat.py.bak.two_pass_v1.20260421_213026 /opt/indian-legal-ai/rag/cbic_rag/storyformat.py \
   && sudo systemctl restart cbic-rag'
```

(The stray `validator.py` does no harm if it's left on disk — nothing imports it once `api.py` is reverted.)

## Smoke test

After **Phase 1** restart (flag OFF) — should match pre-deploy byte-for-byte:

```bash
curl -sS -X POST http://192.168.1.107:9500/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What are the conditions under Section 16(2) of the CGST Act for claiming input tax credit?","k":6}' \
  | python -m json.tool | head -80

# Expect: `timings.two_pass_enabled` = false, no `two_pass_*` timing keys,
# quality_warnings should be identical in shape to current live responses.
```

After **Phase 2** restart (flag ON):

```bash
curl -sS -X POST http://192.168.1.107:9500/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What are the conditions under Section 16(2) of the CGST Act for claiming input tax credit, and how does Rule 36(4) restrict provisional ITC?","k":8}' \
  | python -m json.tool | head -120

# Expect in payload:
#   - "two_pass": true
#   - timings.two_pass_enabled = true
#   - timings.two_pass_decomp_ms / two_pass_extract_ms / two_pass_validate_ms / two_pass_synth_ms all populated
#   - verified_quotes: list with {sid, quote, chunk_id, match: "exact"|"jaccard_0.xx"|"cos_0.xx"}
#   - suspicious_quotes: may be non-empty; each entry has {sub_question, quote, reason}
#   - answer_markdown quotes verbatim_spans EXACTLY (no paraphrase)
```

Flag-flip verification (single command):

```bash
curl -sS http://192.168.1.107:9500/v1/meta | python -c "import json,sys; d=json.load(sys.stdin); print('two_pass_enabled=', d.get('two_pass_enabled'))"
```

## Env var to flip

```
TWO_PASS_ENABLED=1       # enable two-pass
TWO_PASS_ENABLED=0       # disable (default, legacy path)
```

(Set via `systemctl edit cbic-rag` → `[Service]` → `Environment=TWO_PASS_ENABLED=1`.)

## Design choices / deviations from spec

1. **`cited_chunk_id` = `"S<n>"` string.** The spec says the validator must map `cited_chunk_id` to a chunk. Since the extraction prompt tags each source with `[S<n>]`, the cleanest stable ID is just the S-index as a string (e.g. `"S3"`). `_sid_to_index()` tolerates `"S3"`, `"s3"`, `"3"`, or `"[S3]"`. This avoids the LLM inventing hashed IDs it can't see.
2. **Cosine fallback reuses `retriever._cached_embed_query`.** The spec named `_embed_single`; that function does not exist in `retriever.py`. `_cached_embed_query(q_norm)` returns `{"dense": [...], "sparse": {...}}`, so we pull `["dense"]`. Falls back to `embedder.embed_query` if the retriever import path fails.
3. **`build_citations()` is a new helper** extracted from `build_response_payload()` so the two-pass path can emit the exact same `citations` shape without going through the legacy fuzzy verifier. Legacy `build_response_payload()` is untouched — single-shot fallback is byte-identical.
4. **`_call_llm` signature extended, not replaced.** Added optional `response_format=None` and `max_tokens=6144` kwargs. All pre-existing call sites pass neither, so behavior is unchanged for them.
5. **JSON-mode graceful degradation.** `_call_llm_json()` tries `json_schema` first, falls back to `json_object`, then to plain text + regex brace-extract. Any LiteLLM/llama.cpp variant that supports at least one of these will work.
6. **Decomposition planner prompt added.** The spec provided `DECOMP_VERIFY_SYS` but no decomposition prompt itself. Added a minimal `DECOMP_SYS` that emits `{"sub_questions": [...]}`. On any failure the pipeline falls back to `[question]` (single sub-question) so pass-1 still runs.
7. **Cap of 6 sub-questions per Pass-1 call** implemented via `math.ceil(N/6)` batches; results are merged.
8. **Table detection** uses `chunk.get("is_table")` or `chunk.get("type") == "table"`. If neither flag is populated in current payloads, `is_table` will always be False and the 6-gram/cosine ladder applies uniformly. This is safe: the stricter "exact only" path only engages if your ingestor starts flagging tables.
9. **Synthesis prose — fallback path.** If Pass 2 HTTP call fails, we emit a minimal deterministic markdown from `verified` so the user still gets verbatim quotes + citations rather than a 502.
10. **Did NOT touch `retriever.py`** per spec.
11. **Did NOT touch `/query_stream`** — it still uses the single-shot path regardless of flag. Two-pass is inherently multi-LLM-call and doesn't map cleanly onto token streaming; streaming remains legacy until a future `A3b`.
