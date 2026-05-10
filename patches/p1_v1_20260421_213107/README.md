# p1_v1 — A1 retrieval-side BM25 boost for statute queries

**Sentinel:** `p1_v1`
**Timestamp:** `20260421_213107`
**Target:** `user@192.168.1.107:/opt/indian-legal-ai/rag/cbic_rag/`
**Status:** DO NOT DEPLOY yet. Hold until A3 ships.

## Files touched (remote)

| Remote file | Change |
| --- | --- |
| `retriever.py` | Added `classify_query()`, rewritten `retrieve()` with `qclass` kwarg + client-side RRF fusion + BM25 rank-shift boost when `tier in ('hard','soft')`. Added `enforce_min_non_act()` helper. `rerank()` unchanged. |
| `api.py` | Imports `classify_query`, `enforce_min_non_act`. In `_run_pipeline()` only: classify, pass `qclass` to `retrieve()`, log `retrieval_tier` / `statute_refs_matched` / `fusion_mode` / `non_act_count` to `timings`. Backfill runs after rerank. Generation block (lines ~248-285) untouched. |
| `storyformat.py` | **NOT MODIFIED.** |

## Design summary

The live code defers RRF to the Qdrant server via `qm.FusionQuery(fusion=qm.Fusion.RRF)`, so there is no single client-side RRF site to inject a multiplier into. The patch preserves the original fast path for `tier == 'none'` (zero regression on ordinary queries) and takes a separate path for `tier in ('hard','soft')`:

1. Run dense and sparse `query_points` as **two independent prefetches** (no fusion).
2. On the sparse (BM25) list only, for any chunk whose `payload.doc_type in ('act','rules')`, shift its rank to `max(1, floor(rank / mult))` with `mult = 3.5` (hard) or `1.8` (soft). Re-sort & re-number ranks.
3. Do client-side RRF `1/(60 + rank)` across the two lists. Dense side is unchanged.
4. After rerank/MMR, `enforce_min_non_act(top, hits, min_non_act=2)` backfills top-k by swapping the lowest-scoring act/rules chunk(s) for the best non-act candidates that were retrieved but dropped.

No Qdrant hard payload filter is ever applied, so non-act circulars remain visible to the LLM as the spec requires.

## Deploy (dry-run first)

```bash
cd D:/_gpu_rig_ai/patches/p1_v1_20260421_213107
python apply.py                 # prints all ssh/scp commands, does nothing
python apply.py --apply         # actually deploy (HOLD until A3 ships)
ssh -o ControlMaster=no -o ControlPath=none -i ~/.ssh/id_ed25519 \
    user@192.168.1.107 'sudo systemctl restart cbic-rag'
```

## Rollback

```bash
python apply.py --rollback --apply
ssh -o ControlMaster=no -o ControlPath=none -i ~/.ssh/id_ed25519 \
    user@192.168.1.107 'sudo systemctl restart cbic-rag'
```

This restores `retriever.py.bak.p1_v1.20260421_213107` and `api.py.bak.p1_v1.20260421_213107` over the live files.

## Smoke test

After deploy + restart, hit `/query` with a hard statute query and confirm the new fields appear in `timings`:

```bash
curl -s http://192.168.1.107:9500/query \
    -H 'content-type: application/json' \
    -d '{"question":"What does Section 16(2)(b) of the CGST Act say about ITC?","k":6}' \
  | python -c 'import json,sys; j=json.load(sys.stdin); t=j.get("timings",{}); \
               print("tier        =", t.get("retrieval_tier")); \
               print("refs        =", t.get("statute_refs_matched")); \
               print("fusion_mode =", t.get("fusion_mode")); \
               print("fusion_mult =", t.get("fusion_mult")); \
               print("non_act_cnt =", t.get("non_act_count"));'
```

Expected:
```
tier        = hard
refs        = ['16', '2', 'b']
fusion_mode = client_rrf_boost_hard
fusion_mult = 3.5
non_act_cnt = 2        # >=2 (backfill honoured)
```

Control query (should take fast path, no regression):
```bash
curl -s http://192.168.1.107:9500/query \
    -H 'content-type: application/json' \
    -d '{"question":"How do I file a refund for exports?","k":6}' \
  | python -c 'import json,sys; t=json.load(sys.stdin).get("timings",{}); \
               print(t.get("retrieval_tier"), t.get("fusion_mode"))'
# expected: none qdrant_rrf
```

## Files in this patch dir

| File | Purpose |
| --- | --- |
| `retriever.patched.py` | Full patched retriever (drop-in replacement). |
| `api.patched.py` | Full patched api.py (drop-in replacement). |
| `apply.py` | Idempotent deploy script with backup + rollback. |
| `README.md` | This file. |
