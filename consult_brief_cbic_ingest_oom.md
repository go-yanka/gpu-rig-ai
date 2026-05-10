# Consult Brief: CBIC RAG Ingest OOM — Review Remediation Plan

**Audience:** Senior LLM/ML-ops engineer. I need a sanity check on my proposed fixes to a Python batch ingest pipeline that OOM'd mid-run. Please flag anything wrong, missing, or better done differently.

## Context

- **Hardware:** 6× AMD RX 5700 XT + 1× RX 6700 XT, Intel i5-6400T (4 cores), 31 GiB RAM, 16 GB swap, Ubuntu 22.04, kernel 6.8, all storage on USB (1 TB Sabrent enclosure for rootfs + data).
- **Workload:** Ingest 15,559 tax PDFs (CBIC GST/customs/central-excise/service-tax) into Qdrant collection `cbic_v1` with hybrid retrieval: 1024d dense (BGE-M3) + BM25 sparse.
- **Embedding backend:** `llama-cpp-python` compiled from source with Vulkan (`GGML_VULKAN=ON`, gcc-13 via conda-forge), in-process multi-GPU pool — one `multiprocessing.Process` per GPU, each sets `GGML_VK_VISIBLE_DEVICES=N` before `import llama_cpp`. Using BGE-M3 GGUF F16 (1.1 GB), 6 GPUs × ~620 MB VRAM each, `pooling_type=2` (CLS), per-item embed (`n_seq_max=1` can't be exposed via `Llama.__init__`). Achieved ~295 items/s raw across 6 GPUs.
- **Chunker:** `ProcessPoolExecutor` (default `max_workers=None` → one per CPU). Each worker runs `chunk_doc(doc, filepath)` and `return doc_id, [asdict(c) for c in chunks]` — **returns the whole chunk list for a doc as a single result**.
- **Consumer:** Main thread pulls `as_completed()` results, dumps chunks into a `queue.Queue(maxsize=64)`, three embed+upsert worker threads drain it in batches of 128.

## What happened

- Ran ingest with `--resume`. 15,559 docs total.
- After ~15 min: file-log (one line per processed doc) had all 15,559 entries → **chunker finished everything** (15,559 docs → 167,106 chunks).
- Qdrant only had **42,752 points** — embedder/upserter finished only 26% of what the chunker produced.
- Box became SSH-unresponsive (sshd session auth OK but command exec hung → classic memory thrashing / fork failing).
- I hard-power-cut. On reboot, ext4 journal replay left some blocks returning transient `I/O error` on `/usr/bin/*`. Box effectively unbootable until I booted a recovery USB and ran `fsck -y` on the data partition (which found only minor free-counter drift — FS was fine).
- Currently recovered. State preserved: Qdrant has 42,752 points, file-log has full 15,559 entries, all on-disk scripts intact.

## Root cause hypothesis

- A 4 MB customs PDF produces 457 chunks. That whole 457-item `list[dict]` (each dict has `text` up to ~2KB + metadata) sits in the ProcessPool result until main thread pulls it.
- With ProcessPoolExecutor submitting all 15,559 docs up front (default eager submission), many `Future` results are buffered inside the pool's internal result queue even before main thread asks for them.
- Embedder is the bottleneck (~55 items/s end-to-end). Chunker keeps outrunning it.
- Result: internal buffers + the 16-entry `queue.Queue` + per-doc lists → RSS climbs past 25-30 GiB, swap fills, OOM killer picks sshd (higher `oom_score` than our Python process).

## Proposed fixes (what I want critiqued)

### 1. Stream chunks from chunker instead of per-doc lists
Replace `return doc_id, [asdict(c) for c in chunks]` with a chunker worker that writes each chunk into a `multiprocessing.Queue` as it produces them. Drop ProcessPoolExecutor; use `multiprocessing.Process` per chunker-worker (N=4), share one queue with main thread, backpressure via `maxsize`.

### 2. Cap concurrency explicitly
- Chunker workers: 4 (was implicit = nproc = 4 already, but rely on it)
- Embed+upsert threads: unchanged (3)
- `queue.Queue(maxsize=16)` (was 64)
- Batch size: 128 → 64

### 3. Per-worker RSS limit
`resource.setrlimit(RLIMIT_AS, (4*1024**3, 4*1024**3))` at chunker-worker startup. One OOM kills the worker, not the box.

### 4. Systemd kernel-level limits
```ini
[Service]
MemoryHigh=22G
MemoryMax=26G
OOMScoreAdjust=500
TasksMax=200
```
- `OOMScoreAdjust=500` makes our process the OOM-killer target, protecting sshd.
- `MemoryHigh=22G` triggers kernel page reclaim before we hit hard limit.
- `MemoryMax=26G` is hard cap — kernel kills us before total system OOM.

### 5. Verify swap is actually on
Seemed obvious but might not be — HiveOS had 0 B swap, maybe Sabrent install missed enabling it. `swapon --show` on boot, added to preflight.

### 6. Preflight script
Run before `systemctl start`, fails hard if: Qdrant unreachable, `ollama-embed@*` still running (they'd hold VRAM), swap not active, any GPU has >500 MB VRAM used, `llama_supports_gpu_offload()` false, state dir not writable.

### 7. Deterministic point IDs (already done)
Was using `hash()` which is randomized per-Python-process → restarts created new point IDs for same chunks → duplicates stacked. Replaced with `blake2b(doc_id|page|char_start)`. Verified: now re-ingest overwrites cleanly.

### 8. Persistent state path (already done)
Moved per-file log from `/tmp/cbic-files.log` (wiped on reboot) to `/opt/indian-legal-ai/state/cbic-files.log`.

### 9. Monitor fix
Current `ingest_monitor.py` estimates category coverage as `(sampled_category_points / 12) / source_docs` — ignores the fact that sampled count is not scaled by (sampled / total). Showed 175% coverage. Replace with **exact** per-category counts via `/collections/{name}/points/count` + filter (O(ms) in Qdrant).

## Specific questions

1. **Is streaming via `multiprocessing.Queue` actually the right pattern here**, or is there something Pythonic I'm missing (e.g., `concurrent.futures.Executor.map` with a generator and tight backpressure, or `asyncio` + subprocess chunkers)? Specifically worried about queue serialization cost for millions of small chunk dicts.

2. **Is the RLIMIT_AS cap safe for a chunker that uses PDF libraries (`pypdf`, `fitz`/PyMuPDF)?** Those can have large mmap'd regions on big PDFs. RLIMIT_AS counts mmap against virtual address space — might kill legit PDFs unfairly.

3. **Does `OOMScoreAdjust=500` make sense** or should it be higher (up to 1000) to guarantee sshd/dbus/systemd are spared? What is the conventional tradeoff?

4. **16 GB swap on a 31 GB RAM box running a pipeline that can hit 20 GB+ RSS** — is that adequate or should it be bumped to 32+ GB? Swap on USB-attached ext4 — how bad is the IOPS hit? (Sabrent USB 3.0 enclosure, RTL9210 bridge.)

5. **ProcessPoolExecutor submit pattern** — if I submit 15,559 futures up front (even with `map(..., chunksize=N)`), does Python buffer all 15,559 work-items in the internal queue? If yes, that's another source of memory growth I hadn't accounted for. Should I use a throttled submit pattern (submit next future only after one completes)?

6. **Any known pitfalls with llama-cpp-python Vulkan multi-GPU in spawned multiprocessing workers** that could cause memory leaks over time? Each worker does tens of thousands of `create_embedding()` calls over the ingest's 1-2 hour duration.

7. **Is there a simpler "just use Ray / Dask / Prefect"-style answer** that I should consider instead of hand-rolling the backpressure? (I'd prefer not to add dependencies, but worth hearing.)

8. **On watchdogs**: `/proc/pressure/memory` avg10 threshold — what's a sensible value for "memory pressure too high, pause ingest" signal? I was going to use `> 80` for 2 min but I'm just guessing.

Please critique the full list. If any item is wrong or unnecessary, say so and why. If I'm missing something important, flag it. Short answers welcome; depth where it matters.

## Key code fragments for reference

**Current chunker (the problem):**
```python
def chunk_worker(doc):
    chunks = list(chunk_doc(doc, doc['file_path']))   # whole list materialized
    return doc['doc_id'], [asdict(c) for c in chunks]  # returned as single result

with ProcessPoolExecutor() as ex:   # default max_workers = nproc
    futures = [ex.submit(chunk_worker, d) for d in docs]   # 15559 submissions up front
    for fut in as_completed(futures):
        doc_id, clist = fut.result()
        for c in clist:
            q.put(c)   # q is queue.Queue(maxsize=64)
```

**Embedder pool (likely OK, but verify):**
```python
# One mp.Process per GPU, spawns before import llama_cpp
def _gpu_worker(gpu_id, req_q, resp_q):
    os.environ['GGML_VK_VISIBLE_DEVICES'] = str(gpu_id)
    from llama_cpp import Llama
    m = Llama(model_path=..., embedding=True, n_gpu_layers=-1,
              n_ctx=8192, n_batch=512, pooling_type=2)
    while True:
        item = req_q.get()
        if item is None: return
        req_id, texts = item
        out = [list(m.create_embedding([t])['data'][0]['embedding']) for t in texts]
        resp_q.put((req_id, out))
```

**Current point ID (fixed):**
```python
# was: pid = abs(hash((c['doc_id'], c['page'], c['char_start']))) % (10**15)
_pk = f"{c['doc_id']}|{c['page']}|{c['char_start']}".encode()
pid = int.from_bytes(hashlib.blake2b(_pk, digest_size=8).digest(), 'big') % (10**15)
```

Thanks.
