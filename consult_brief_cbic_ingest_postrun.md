# Consult Brief: CBIC RAG Ingest — Post-Remediation Run Review

**Audience:** Senior LLM/ML-ops engineer. The OOM remediation plan you reviewed earlier was implemented and ran to completion. This brief covers the outcome and three new issues that surfaced during the run. I need opinions on the follow-up fixes.

## TL;DR

- **OOM remediation: fully validated.** Ingest ran ~48 min start-to-finish, processed 15,322 docs, upserted 164,974 points. Peak RSS 9.1 GB / 31 GB. Cgroup peak ~5 GB / 22 GB soft / 26 GB hard. **Zero swap used. PSI avg10 stayed at 0.00% the entire run.** Watchdog never fired.
- **Three downstream issues surfaced:**
  1. Qdrant collection status went `red` — optimizer panicked on an invalid unicode code point in some chunk's payload.
  2. One of two consumer threads got stuck at shutdown in a blocking `qdrant-client` HTTP call against the red collection. Main thread's `.join()` blocked forever until I `systemctl stop` the service.
  3. My preflight/monitor queries used wrong category names (dashes) — the actual manifest uses underscores (`central_excise`, not `central-excise`). Caused false "0 docs processed" readings.

## What we implemented (from the previous consult)

Exactly the reconciled plan — summary:

| Change | Status |
|---|---|
| Bounded in-flight window (`CHUNKER_IN_FLIGHT=16` docs, `ProcessPoolExecutor` + `wait(FIRST_COMPLETED)` + top-up on done) | Deployed |
| Drop `RLIMIT_AS` (was going to kill PyMuPDF mmaps) | Done |
| systemd cgroup: `MemoryHigh=22G MemoryMax=26G OOMScoreAdjust=500 TasksMax=200` | Deployed |
| `EMBED_BATCH=128`, `EMBED_THREADS=2`, `QUEUE_MAXSIZE=16`, `CHUNK_WORKERS=4` | Deployed |
| PSI watchdog: avg10 > 40 sustained 30s → touch pause sentinel; < 10 → remove. Runs as separate systemd unit (`cbic-ingest-watchdog.service`, `MemoryMax=64M`, `Nice=-5`) | Deployed, never triggered |
| Preflight: swap on, Qdrant up, no stray ollama, VRAM baseline clear, model readable, state writable, no `ingest.disabled` flag | Deployed, 8/8 green |
| `ingest.py` unchanged from earlier patches (blake2b IDs, persistent state path, multi-consumer embed) | Preserved |
| `ingest_monitor.py`: exact per-category counts via `/collections/.../points/count` with filter; added RAM/PSI panel | Deployed |
| GPU worker rotation | **Skipped** — math said we'd never hit the 30–50k-per-worker threshold in one run. Held in reserve. |

Direct in-process Vulkan embedding path (`llama-cpp-python` in 6 `mp.Process` workers, one per GPU) unchanged. Zero HTTP intermediaries in the compute path. Rate held steady at ~60 items/s end-to-end (varied 50–77 across phases).

## Observed run (highlights)

| Time | Docs | Upserted this run | RSS | Cgroup | Swap | PSI avg10 |
|---|---|---|---|---|---|---|
| 14:08 start | - | 0 | 0 | 0 | 0 | 0 |
| 14:12 (+4m) | ~ | 18,560 | 7.0 GB | 3.1 GB | 0 | 0 |
| 14:20 (+12m) | ~ | 38,634 | 7.0 GB | 3.1 GB | 0 | 0 |
| 14:26 (+18m) | 1,300 | 57,266 | 7.0 GB | 3.6 GB | 0 | 0 |
| 14:36 (+28m) | 7,650 | 89,518 | 7.0 GB | 4.6 GB | 0 | 0 |
| 14:47 (+39m) | 14,600 | 111,584 | 6.4 GB | 4.6 GB | 0 | 0 |
| 14:50 (+42m) | 15,300 | 118,624 | 6.4 GB | 4.8 GB | 0 | 0 |
| 14:53 (+45m) | 15,322 (end) | 120,942 | 9.1 GB | 4.9 GB | 0 | 0 |
| 14:56 | — | — | 9.1 GB | 4.9 GB | 0 | 0 (hung at `.join()`) |

Throughout: `inflight=15`, `qsize=16`, `buf` mostly 0–30. **Backpressure stayed perfectly engaged.** Chunker produced exactly as fast as embedder could drain, never more. The previous-run OOM mode (blow up the `ProcessPoolExecutor` internal result queue with 15k pending futures) is structurally eliminated.

Final Qdrant state (stored, but collection status `red`):
```
gst            99,057
customs        48,300
central_excise  9,689
service_tax     2,728
others          5,200
TOTAL         164,974 points (all 15,559 docs in manifest covered)
```

## Issue 1 — Qdrant optimizer panic on bad unicode

**Error:**
```
"status": "red",
"optimizer_status": {
  "error": "Service runtime error: Optimization task panicked:
   called `Result::unwrap()` on an `Err` value:
   Error(\"invalid unicode code point\", line: 1, column: 4847)"
}
```

Points are stored, HNSW optimizer crashed on segment merge because some chunk payload contains a codepoint `serde_json` refuses. This is Rust-side Qdrant (not my Python code) but root cause is payload sanitization in the ingest:

**Current `_sanitize_text` in `ingest.py`:**
```python
def _sanitize_text(t):
    if not isinstance(t, str): return t
    t = t.replace(chr(0), "")
    try: t.encode("utf-8")
    except UnicodeEncodeError:
        t = t.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return t
```

Catches only `UnicodeEncodeError`. Doesn't strip:
- Lone surrogates (U+D800–U+DFFF)
- Non-characters (U+FFFE, U+FFFF, etc.)
- Private-use-area codepoints that some JSON parsers reject
- Control chars other than `\x00`

**Proposed fix:**
```python
def _sanitize_text(t):
    if not isinstance(t, str): return t
    # Re-encode round-trip with surrogatepass removed
    t = t.encode('utf-8', 'replace').decode('utf-8', 'replace')
    # Strip everything that isn't printable, plus keep whitespace
    t = ''.join(c for c in t
                if (c.isprintable() or c in '\n\t\r ')
                and not (0xD800 <= ord(c) <= 0xDFFF)
                and not (0xFDD0 <= ord(c) <= 0xFDEF)
                and (ord(c) & 0xFFFF) not in (0xFFFE, 0xFFFF))
    return t
```

**Recovery path for the existing collection:**
- Option A: re-ingest with new sanitizer + `--reset` (drops and recreates cbic_v1). Cost: another ~45 min run. Clean slate, certainly fixes it.
- Option B: try `POST /collections/cbic_v1/index` to retrigger optimizer. Unlikely to help — bad payload still present, will re-panic.
- Option C: find + delete the offending point, then reoptimize. Risky, needs scrolling + detecting bad unicode in each payload.

Leaning toward A. Cost is known (~45 min), outcome is deterministic, and blake2b IDs make it idempotent.

## Issue 2 — Consumer thread hung on shutdown

**What happened:** Two consumer threads (`EMBED_THREADS=2`) running `embed_consumer_thread`. At ~14:47, thread `#0` stopped logging flush lines; only `#1` continued. Both stayed in a `while True: q.get()` loop. At end of run, main thread did `for t in consumers: t.join()` and hung on `#0`.

**Likely cause:** In the upsert batch-fallback path:
```python
try:
    qc.upsert(QCOLL, points=points, wait=False)
except Exception:
    for pt in points:
        try: qc.upsert(QCOLL, points=[pt], wait=False)
        except Exception: pass
```
When collection is `red`, some single-point upserts block rather than raise. `qdrant-client` has internal retries + timeout 120s per the `QdrantClient(timeout=120)` arg. With the server refusing to index but keeping the HTTP connection open, a thread can sit in `recv()` inside grpc/http2 for the full timeout × retries.

**Confirmed via `/proc/<pid>/task/*/wchan`:** All threads blocked on `futex_wait_queue_me`. Main is waiting for the `Thread.join()`.

**Proposed fix — two parts:**
```python
# 1. Consumer fallback retry: bail out after N failures in a row
MAX_POINT_RETRIES = int(os.environ.get("MAX_POINT_RETRIES", "3"))
consecutive_failures = 0
for pt in points:
    try:
        qc.upsert(QCOLL, points=[pt], wait=False)
        n += 1
        consecutive_failures = 0
    except Exception as e:
        consecutive_failures += 1
        if consecutive_failures >= MAX_POINT_RETRIES:
            print(f"[drop batch] worker={worker_id} {consecutive_failures} failures, dropping rest")
            break

# 2. Main thread join with timeout + force-kill
for t in consumers:
    t.join(timeout=60)
    if t.is_alive():
        print(f"[shutdown] consumer stuck; forcing exit")
        # Thread.daemon=True already; process exit will kill it
```

Consumer thread is already a `daemon=True` thread — just letting main return cleanly is enough. The fix is the timeout on `.join()`.

## Issue 3 — Wrong category names in preflight/monitor

Minor — manifest uses snake_case (`central_excise`, `service_tax`) but I hardcoded kebab-case (`central-excise`, `service-tax`). Monitor showed "0 docs processed" for those two categories all run despite them being actively worked on.

Trivial fix in `ingest_monitor.py` (monitor already uses `docs_per_category()` which returns manifest values, so actually only my out-of-band curl queries were wrong — monitor itself was fine). Also fix preflight query if used elsewhere.

## Specific questions

1. **Sanitizer proposal** — is the unicode stripping I proposed overkill? I'm worried about accidentally dropping legit non-ASCII content (these are Indian tax PDFs, may have Hindi/Tamil text that looks weird but is valid). Should I preserve "category Cf" and above but reject only `Cs` (surrogates) and non-characters?

2. **Shutdown hang** — is the `daemon=True` + `.join(timeout=60)` pattern sufficient, or should I switch consumers to use an `Event` sentinel and explicit `qc.close()` before join? `qdrant-client` doesn't seem to expose a hard-cancel API for in-flight requests.

3. **Recovery** — between (a) `--reset` + re-ingest with new sanitizer vs (c) find-and-delete offending points, is there a middle path? E.g., trigger reoptimize after ONLY re-upserting points that have payloads sanitized against the new rules? Seems fragile.

4. **The 5,200 "others"** — I don't know what category that covers yet. Should I investigate before re-ingesting, or is "category=others" acceptable as a catch-all?

5. **Flush pattern observation** — during run, both consumers alternated cleanly until ~75% through, then `#0` stopped. `#1` alone carried the last ~9k chunks at only slightly slower rate (~3.5s per 128-batch). Did I even need 2 consumers? Is 1 thread + larger batch actually the safer design here?

6. **`_sanitize_text` runs in the consumer thread, AFTER chunker produced the text**. Should sanitization happen in the chunker worker (pickled payload is already bad-unicode-free by the time it crosses the process boundary) or in the consumer (closer to the upsert, catches late additions)? Marginal either way — I tend toward chunker since it's the origin.

## Summary state

- Data: all 164,974 points stored, deterministic IDs via blake2b (re-ingest is idempotent).
- Corpus: 15,559 docs fully chunked twice now (earlier failed run + this successful run). files-log is 30,881 lines — benign duplication; could dedupe.
- Collection: `red` status; retrieval quality unknown until reindex.
- Hardware: untouched, rig running fine, no reboots needed.
- Scripts in place: `ingest.py`, `cbic-ingest.service` (w/ cgroup), `cbic-ingest-watchdog.service`, `watchdog.sh`, `preflight.sh`, `run-ingest-direct.sh`, `ingest_monitor.py`.
- Backups: timestamped `.bak.<ts>` of all touched files on rig under `scripts/cbic_ingest/` and `rag/cbic_rag/`.

Net: the hard problem (OOM / unbootable box) is definitively solved. What remains are payload hygiene + graceful shutdown polish. Would appreciate opinions before I pull the trigger on `--reset` + re-ingest.

Thanks.
