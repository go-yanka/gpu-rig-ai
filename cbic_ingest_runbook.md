# CBIC RAG Ingest Runbook — Direct Multi-GPU (Vulkan)

Last known-good: 2026-04-20. Survives reboots, power cuts, and resumes cleanly.

## Topology

- **Collection:** `cbic_v1` on Qdrant at `http://127.0.0.1:6343`
- **Source manifest:** `/opt/indian-legal-ai/data/scraped/cbic/_manifest.sqlite`
  (~15.5k PDFs: customs 9,883 / central_excise 3,083 / gst 1,847 / service_tax 918 / others 45)
- **Embedder:** BGE-M3 (1024d dense + BM25 sparse), GGUF F16
  reused from Ollama blob: `/usr/share/ollama/.ollama/models/blobs/sha256-daec91ffb5dd0c27411bd71f29932917c49cf529a641d0168496c3a501e3062c`
- **Compute:** 6× RX 5700 XT via Vulkan, in-process `llama-cpp-python` pool
  GPUs used: `0,1,3,4,5,6` (GPU 2 = RX 6700 XT excluded — vk DeviceLostError)
- **Throughput:** ~295 items/s raw embed, ~55 items/s end-to-end after chunking+upsert

## Files (all persistent, nothing in /tmp)

| Path | Purpose |
|---|---|
| `/opt/indian-legal-ai/rag/cbic_rag/ingest.py` | Main ingest script |
| `/opt/indian-legal-ai/rag/cbic_rag/embedder.py` | Facade → direct pool |
| `/opt/indian-legal-ai/rag/cbic_rag/embedder_direct.py` | Multi-GPU Vulkan pool |
| `/opt/indian-legal-ai/rag/cbic_rag/ingest_monitor.py` | Rich TUI monitor |
| `/opt/indian-legal-ai/scripts/cbic_ingest/run-ingest-direct.sh` | Launcher with env |
| `/opt/indian-legal-ai/scripts/cbic_ingest/build-llamacpp-vulkan.sh` | Rebuild recipe |
| `/opt/indian-legal-ai/scripts/cbic_ingest/patch_ingest.py` | Re-apply fixes if ingest.py ever gets clobbered |
| `/opt/indian-legal-ai/state/cbic-files.log` | Per-file ingest log (persistent resume checkpoint) |
| `/etc/systemd/system/cbic-ingest.service` | Systemd unit |

## Pre-flight checklist (before starting a fresh ingest)

1. **Ollama bge-m3 services must be STOPPED** — they hold VRAM and block our Vulkan load:
   ```
   sudo systemctl stop ollama-embed@{0,1,3,4,5,6}.service
   sudo systemctl disable ollama-embed@{0,1,3,4,5,6}.service   # optional
   ```
2. **Qdrant must be healthy:**
   ```
   curl -s http://localhost:6343/healthz
   curl -s http://localhost:6343/collections
   ```
3. **bge-m3 GGUF blob present:**
   ```
   ls -la /usr/share/ollama/.ollama/models/blobs/sha256-daec91ffb5dd0c27411bd71f29932917c49cf529a641d0168496c3a501e3062c
   ```
4. **llama-cpp-python has Vulkan:**
   ```
   python3 -c "import llama_cpp; print(llama_cpp.__version__)"
   python3 -c "from llama_cpp import llama_supports_gpu_offload; print(llama_supports_gpu_offload())"   # must be True
   ```
   If not: run `/opt/indian-legal-ai/scripts/cbic_ingest/build-llamacpp-vulkan.sh`

## Four critical bugs this setup fixes

| Bug | Symptom | Fix in ingest.py |
|---|---|---|
| Random-seeded `hash()` for point_id | Each restart creates DIFFERENT ids for same chunks → duplicates stack. Saw 88k gst points for 235 docs (377×). | Deterministic `blake2b(doc_id|page|char_start)` |
| FILES_LOG in `/tmp` | Reboot wipes resume checkpoint → re-ingests from scratch | `/opt/indian-legal-ai/state/cbic-files.log` |
| `existing_doc_ids()` silently swallowing errors | Resume fails → thinks nothing is done → re-ingests | Raise on scroll error |
| Ollama bge-m3 services running | VRAM conflict → direct llama_cpp hangs loading | Stop them in pre-flight |

## Starting a fresh (clean-slate) ingest

```
# 1. Stop any running ingest
sudo systemctl stop cbic-ingest.service

# 2. Drop collection (only if you want to start from zero)
curl -X DELETE http://localhost:6343/collections/cbic_v1

# 3. Stop Ollama embed workers
sudo systemctl stop ollama-embed@{0,1,3,4,5,6}.service

# 4. Re-apply patches if ingest.py was reverted
python3 /opt/indian-legal-ai/scripts/cbic_ingest/patch_ingest.py

# 5. Launch
sudo systemctl start cbic-ingest.service

# 6. Watch
sudo journalctl -u cbic-ingest.service -f -o cat | grep -v 'embeddings required'
# OR the rich TUI:
python3 /opt/indian-legal-ai/rag/cbic_rag/ingest_monitor.py --refresh 2
```

## Resuming after crash / reboot

Same as fresh, but **skip step 2** (keep the collection). With deterministic IDs,
already-ingested chunks get overwritten harmlessly; untouched docs get added.
`--resume` in the launcher script still scans Qdrant for done doc_ids.

## Verification (collection sanity)

```
# Total + per-category point counts
curl -s http://localhost:6343/collections/cbic_v1 | jq '.result | {points_count, indexed_vectors_count, status}'
for c in customs central_excise gst service_tax others; do
  curl -s -X POST http://localhost:6343/collections/cbic_v1/points/count \
    -H 'Content-Type: application/json' \
    -d "{\"filter\":{\"must\":[{\"key\":\"category\",\"match\":{\"value\":\"$c\"}}]},\"exact\":true}" \
    | jq -r --arg c "$c" '"\($c): \(.result.count)"'
done

# Sanity: unique doc_ids vs points per category. Expect ~10-50 chunks/doc.
# If ratio > 200, point_id collision bug is back — re-check patch_ingest.py applied.
python3 -c "
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
qc=QdrantClient('http://127.0.0.1:6343', timeout=60)
for cat in ['gst','customs','central_excise','service_tax','others']:
    flt=qm.Filter(must=[qm.FieldCondition(key='category', match=qm.MatchValue(value=cat))])
    seen=set(); off=None; n=0
    while True:
        pts,off=qc.scroll('cbic_v1', limit=5000, with_payload=['doc_id'], with_vectors=False, offset=off, scroll_filter=flt)
        for p in pts: seen.add(p.payload.get('doc_id'))
        n+=len(pts)
        if off is None: break
    print(f'{cat}: {n} points / {len(seen)} docs ({n/max(1,len(seen)):.1f} chunks/doc)')
"
```

## Environment vars (launcher sets these)

| Var | Default | Meaning |
|---|---|---|
| `EMBED_GPUS` | `0,1,3,4,5,6` | Comma list of GPU ids for embed pool |
| `EMBED_BATCH` | `128` | Chunks per embed call |
| `EMBED_THREADS` | `2` | Concurrent embed+upsert workers |
| `QUEUE_MAXSIZE` | `64` | Chunk queue backpressure |
| `EMBED_CTX` | `8192` | Llama context size |
| `EMBED_MODEL_PATH` | (Ollama blob) | Override GGUF path |
| `CBIC_FILES_LOG` | `/opt/indian-legal-ai/state/cbic-files.log` | Per-file log |
| `QDRANT_URL` | `http://127.0.0.1:6343` | |
| `QDRANT_COLL` | `cbic_v1` | |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` | `1` | Keep CPU out of Vulkan's way |

## When things go wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| Hang at "loading model" | Ollama holding bge-m3 VRAM | `systemctl stop ollama-embed@*` |
| `vk::DeviceLostError` | GPU 2 (RX 6700 XT) in pool | Remove from `EMBED_GPUS` |
| `n_seq_max` / seq_id error | llama-cpp batch embed limit | Per-item embed already in worker |
| Very slow (< 10 items/s) | CPU fallback — Vulkan not loaded | Check `llama_supports_gpu_offload()` |
| Coverage > 100% in monitor | Point ID collision bug back | Re-run `patch_ingest.py` |
| Resume skips nothing after reboot | `/tmp` resume log | Verify `FILES_LOG` points to `/opt/...state/` |
| apt broken | Known | Use miniforge conda-forge gcc-13, not system apt |

## Numbers to expect (15.5k docs, 6 GPUs)

- Per-doc chunks: ~10-50 (avg ~12, GST PDFs can hit 50)
- Expected final points: ~180,000-250,000
- Raw embed throughput: ~295 items/s across 6 GPUs
- End-to-end throughput: ~55 items/s (I/O + chunking + upsert)
- Full ingest time: ~60-90 minutes from empty collection
- Per-GPU VRAM: ~620 MB (bge-m3 F16)
- Per-GPU power under load: 120-180 W
