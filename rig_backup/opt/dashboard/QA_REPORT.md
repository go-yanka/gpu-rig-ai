# Dashboard QA Report
**Date:** 2026-04-09  
**Files reviewed:** `dashboard/app.py`, `dashboard/static/index.html`, `dashboard/gpu_profiles.json`, `dashboard/start.sh`, `dashboard/install.sh`, `dashboard/deploy.sh`

---

## Summary

| Severity | Count | Fixed |
|----------|-------|-------|
| CRITICAL | 1     | YES   |
| MAJOR    | 8     | YES   |
| MINOR    | 6     | No (low-risk, noted for awareness) |

---

## CRITICAL Issues

### C1 — Path Traversal / Command Injection in model_path
**File:** `app.py:308`  
**Status:** FIXED  

**Problem:** `POST /api/gpu/{id}/load` accepted an arbitrary `model_path` string from the request body with no validation. A malicious caller could supply `../../../../etc/passwd` or a path outside MODEL_DIR. If `start_gpu` in the shell module constructs a subprocess command with this path (likely), this is exploitable as a command injection vector.

**Fix applied (`app.py:301-318`):**
- Resolves the path with `Path.resolve()` to eliminate traversal sequences (`../`)
- Asserts the resolved path starts with `MODEL_DIR.resolve()`
- Checks the file exists on disk
- Restricts extensions to `.gguf` and `.bin`
- Passes the sanitized string (not the raw user input) to `start_gpu`

---

## MAJOR Issues

### M1 — `api_unload` and `api_kill_zombie` missing try/except
**File:** `app.py:314-331` (original)  
**Status:** FIXED  

**Problem:** Both endpoints called `_call("kill_port", ...)` and `_call("save_state")` with no exception handling. Any failure would cause FastAPI to return an unhandled 500 with a non-JSON traceback body, breaking the frontend's `apiPost()` JSON parser.

**Fix:** Wrapped both bodies in `try/except Exception as e: raise HTTPException(500, str(e))`.

---

### M2 — No concurrent benchmark protection (backend)
**File:** `app.py:472` (original)  
**Status:** FIXED  

**Problem:** Two simultaneous GET requests to `/api/bench/{id}/stream` on the same GPU would both launch `bench_event_generator`, firing duplicate HTTP requests to llama-server and doubling load without warning.

**Fix:** Added `_bench_locks: dict[int, bool]` (one flag per GPU). The route returns HTTP 409 if a benchmark is already running on that GPU. The generator sets the lock on entry and clears it in a `finally` block to guarantee cleanup on disconnect or error.

---

### M3 — `asyncio.get_event_loop()` deprecated (Python 3.10+)
**File:** `app.py:444` (original)  
**Status:** FIXED  

**Problem:** `asyncio.get_event_loop()` is deprecated inside async functions in Python 3.10+ and raises a DeprecationWarning. In Python 3.12 it will emit a RuntimeWarning and in future versions may error.

**Fix:** Changed to `asyncio.get_running_loop()` which is the correct call inside a running async context.

---

### M4 — Double `get_gpu_data()` call in SSE generator
**File:** `app.py:395-400` (original)  
**Status:** FIXED  

**Problem:** `gpu_event_generator` called `get_gpu_data()` twice per tick — once for the `gpus` key and once inside the `state` dict to count active GPUs. In live mode this means two full rocm-smi polls per 2-second cycle, doubling GPU probe overhead.

**Fix:** Call `get_gpu_data()` once, assign to `gpus`, reuse for both keys.

---

### M5 — `Math.max()` crash on empty perf arrays
**File:** `static/index.html` — `updateChart()` function  
**Status:** FIXED  

**Problem:** When no perf log entries exist yet (fresh rig, or perf log cleared), all `byGpu[i]` arrays are empty. `Math.max(...[0,0,0,0,0,0,0])` = `0` is fine, but `Math.max()` with spread of all-zero-length arrays correctly returns `0` — however, if the spread produces no arguments at all (edge case where `Object.values` returns empty), `Math.max()` returns `-Infinity`. `Array.from({length: -Infinity})` throws a `RangeError: Invalid array length`, crashing the chart entirely.

**Fix:** Guard with `isFinite(maxRaw)` check; return early if `maxLen === 0` (no data, leave chart untouched).

---

### M6 — Chart type mutation causes Chart.js rendering artifacts
**File:** `static/index.html` — `updateChart()` function  
**Status:** FIXED  

**Problem:** Switching tabs (speed=line → vram=bar, bar → temp=bar, back to line) mutated `perfChart.config.type` directly on a live Chart.js instance. Chart.js does not support changing chart type after initialization — this causes rendering artifacts, legend corruption, and in some versions silent failures where the chart stops updating.

**Fix:** Before switching type, check `perfChart.config.type !== targetType`. If the type changed, call `perfChart.destroy()` then `initChart()` to recreate cleanly, then set the new type.

---

### M7 — `apiPost()` doesn't check `res.ok`
**File:** `static/index.html` — `apiPost()` function  
**Status:** FIXED  

**Problem:** `apiPost()` called `return res.json()` unconditionally. When the server returns an HTTP 4xx/5xx with an HTML error body (e.g., uvicorn default 422 validation error), `res.json()` throws a `SyntaxError: Unexpected token`, swallowing the actual error message. The caller's `catch(e)` would see a confusing JSON parse error instead of the real problem.

**Fix:** Check `if (!res.ok)` first. Try to parse the JSON detail/message for a clean error string, then `throw new Error(detail)` so callers get the real server error.

---

### M8 — Mobile layout overflow in bottom row
**File:** `static/index.html` — bottom row grid div  
**Status:** FIXED  

**Problem:** The Charts + Controls bottom row used `grid-template-columns: 1fr 340px` inline style with no responsive override. On screens narrower than ~400px this causes the 340px panel to overflow the viewport. The existing `@media (max-width: 640px)` block only targeted `.gpu-grid`.

**Fix:** Added `bottom-row` class to the grid div and added `.bottom-row { grid-template-columns: 1fr !important; }` to the 640px media query, stacking charts above controls on small screens.

---

## MINOR Issues (not fixed — low risk, noted for awareness)

### m1 — CORS fully open
**File:** `app.py:224`  
`allow_origins=["*"]` with `allow_methods=["*"]` and `allow_headers=["*"]`. Acceptable for a private LAN dashboard with no authentication, but means any page on any domain a user visits could make API calls to the rig if the user is on the same network. Not a concern if the rig is on a private network.

### m2 — No upper bound on `/api/perf/recent?n=`
**File:** `app.py:255`  
A caller can request `?n=9999999`. The function reads the entire JSONL file into memory and returns the last N lines. If the log is large, this could spike memory. Recommend capping at `min(n, 2000)`.

### m3 — `esc()` missing `>` escape
**File:** `static/index.html` — `esc()` function  
Only escapes `&`, `"`, `<`. The `>` character is not escaped. In practice this doesn't create XSS since all esc() output is placed inside already-opened HTML tags or as text content, but a fully correct HTML escaper should include `>` and `'`.

### m4 — Empty GPU grid shows no message
**File:** `static/index.html` — `renderGpuGrid()`  
If `gpuState` is empty (zero GPUs reported), the grid renders blank with no user-facing explanation. Should show a "No GPU data" placeholder.

### m5 — SSE `onmessage` parse errors silently swallowed
**File:** `static/index.html` — `sseConn.onmessage`  
The `catch {}` block swallows JSON parse errors and malformed events with no console output, making debugging harder. Should at minimum do `catch(e) { console.warn('SSE parse error', e); }`.

### m6 — Benchmark per-prompt timeout is too long (was 120s)
**File:** `app.py:447` (original, now fixed to 60s)  
The original `urlopen(..., timeout=120)` × 5 prompts = up to 10 minutes of blocking per benchmark stream. Reduced to 60s per prompt as part of Fix M3 (still conservative enough for slow models). Consider further reduction to 30s for interactive use.

---

## Scripts Review (start.sh, install.sh, deploy.sh)

**No security issues found.** Observations:
- `start.sh` — clean, exports env vars, no shell injection vectors (port comes from `$1` used only in `export`, not in eval/exec strings)
- `install.sh` — uses `sudo cp -r` of the entire dashboard dir. Fine. `--break-system-packages` pip flag is appropriate for the rig's Ubuntu setup.
- `deploy.sh` — hardcodes `user@192.168.1.107` and `/opt/dashboard`. The SSH heredoc uses `${RIG_DIR}` which is a static string — no injection vector. The only concern is that `scp -r` copies everything including any `.env` or secret files that might be co-located; ensure no secrets land in the dashboard directory.

---

## Files Modified

| File | Changes |
|------|---------|
| `dashboard/app.py` | C1, M1, M2, M3, M4 |
| `dashboard/static/index.html` | M5, M6, M7, M8 |
