#!/usr/bin/env python3
"""
AI Rig Dashboard — FastAPI backend
Runs on the rig (192.168.1.107), serves the web UI and proxies shell commands.

Usage:
  python3 /opt/dashboard/app.py
  # or via start.sh

Endpoints:
  GET  /              → serves index.html
  GET  /api/gpus      → live GPU status
  GET  /api/models    → available GGUF models
  GET  /api/perf      → performance summary per GPU
  GET  /api/perf/recent?n=200 → recent JSONL entries (for charts)
  GET  /api/logs/{gpu_id} → stderr log tail
  GET  /api/state     → system state file + uptime
  GET  /api/profiles  → GPU hardware profiles (from gpu_profiles.json)
  PUT  /api/profiles/{gpu_id} → update a GPU profile
  GET  /stream        → SSE: live GPU stats every 2s
  POST /api/gpu/{id}/load   → body: {"model_path": "..."}
  POST /api/gpu/{id}/unload → stop GPU
  POST /api/gpu/{id}/kill   → kill zombie
  POST /api/nginx/rebuild   → rebuild nginx LB config
  POST /api/bench/{id}      → run benchmark (streams back results)
  GET  /api/bench/{id}/stream → SSE benchmark stream
"""

import os, sys, json, time, asyncio, threading, subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, AsyncGenerator

# ── FastAPI / Starlette ────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ── Import shell functions ─────────────────────────────────────────────────
SHELL_PATHS = [
    "/opt/ai-rig-shell.py",
    str(Path(__file__).parent.parent / "ai_rig_shell.py"),
]

shell = None
for path in SHELL_PATHS:
    if os.path.exists(path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("ai_rig_shell", path)
        shell = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(shell)
        break

if shell is None:
    # Stub mode: dashboard runs but all calls return mock data.
    # Useful for UI development without the rig.
    print("WARNING: ai_rig_shell.py not found — running in STUB mode")

# ── Constants (mirror shell, overrideable via env) ─────────────────────────
MODEL_DIR  = Path(os.environ.get("AI_MODEL_DIR",  "/opt/ai-models"))
STATE_FILE = Path(os.environ.get("AI_STATE_FILE", "/opt/ai-rig-state.json"))
PERF_LOG   = Path(os.environ.get("AI_PERF_LOG",   "/opt/ai-rig-perf.jsonl"))
BASE_PORT  = int(os.environ.get("AI_BASE_PORT",   "9080"))
NUM_GPUS   = 7
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "8080"))

GPU_MAP = {
    0: {"name": "RX 5700 XT (ASUS)",    "vram_mb": 8176,  "bar": "8GB",   "pci": "03:00.0"},
    1: {"name": "RX 5700 XT (Dell)",     "vram_mb": 8176,  "bar": "8GB",   "pci": "06:00.0"},
    2: {"name": "RX 6700 XT (ASRock)",   "vram_mb": 12272, "bar": "16GB",  "pci": "09:00.0"},
    3: {"name": "RX 5700 XT (Sapphire)", "vram_mb": 8192,  "bar": "256MB", "pci": "0f:00.0"},
    4: {"name": "RX 5700 XT (PwrColor)", "vram_mb": 8192,  "bar": "256MB", "pci": "14:00.0"},
    5: {"name": "RX 5700 XT (ASRock)",   "vram_mb": 8192,  "bar": "256MB", "pci": "17:00.0"},
    6: {"name": "RX 5700 XT (ASRock)",   "vram_mb": 8192,  "bar": "256MB", "pci": "1a:00.0"},
}

# ── Helpers ────────────────────────────────────────────────────────────────

DB_PATH = "/dev/shm/ai-rig.db"  # Runtime — same as shell (fast, no USB wear)

def _load_defaults():
    """Read defaults from SQLite (single source of truth)."""
    try:
        conn = _db()
        gpus = {}
        sleep_watts = {}
        awake_watts = {}
        for r in conn.execute("SELECT gpu_id, default_model, default_mode, default_temp, sleep_watts_cal, awake_watts_cal FROM gpu_state"):
            gid = str(r["gpu_id"])
            if r["default_model"]:
                gpus[gid] = {"model": r["default_model"], "mode": r["default_mode"], "temp": r["default_temp"]}
            if r["sleep_watts_cal"] is not None:
                sleep_watts[gid] = r["sleep_watts_cal"]
            if r["awake_watts_cal"] is not None:
                awake_watts[gid] = r["awake_watts_cal"]
        sys = conn.execute("SELECT auto_sleep_minutes, auto_load FROM system_state WHERE id=1").fetchone()
        conn.close()
        return {
            "auto_load": bool(sys["auto_load"]) if sys else False,
            "auto_sleep_minutes": sys["auto_sleep_minutes"] if sys else 0,
            "gpus": gpus,
            "gpu_sleep_watts": sleep_watts,
            "gpu_awake_watts": awake_watts,
        }
    except:
        return {}

def _get_sleep_watts(gpu_id):
    """Get calibrated or estimated sleep watts for a GPU."""
    try:
        conn = _db()
        r = conn.execute("SELECT sleep_watts_cal FROM gpu_state WHERE gpu_id=?", (gpu_id,)).fetchone()
        conn.close()
        if r and r["sleep_watts_cal"] is not None:
            return round(r["sleep_watts_cal"], 1), True
    except: pass
    return (6.0 if gpu_id == 2 else 5.0), False

def _get_awake_watts(gpu_id):
    """Get calibrated awake watts for savings calculation."""
    try:
        conn = _db()
        r = conn.execute("SELECT awake_watts_cal FROM gpu_state WHERE gpu_id=?", (gpu_id,)).fetchone()
        conn.close()
        if r and r["awake_watts_cal"] is not None:
            return round(r["awake_watts_cal"], 1)
    except: pass
    return 10.0

def _db():
    """Get a SQLite connection. Dashboard ONLY reads from this DB."""
    import sqlite3
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn

def _call(fn_name, *args, **kwargs):
    """Call a shell function for ACTIONS only (load, unload, power, etc).
    NEVER use for data reads — read from SQLite instead."""
    if shell is None:
        raise RuntimeError("Shell not loaded — running in stub mode")
    fn = getattr(shell, fn_name, None)
    if fn is None:
        raise RuntimeError(f"Function '{fn_name}' not found in shell")
    return fn(*args, **kwargs)

def get_gpu_data() -> list[dict]:
    """Read GPU status from SQLite — the single source of truth.
    Shell writes to this DB every time get_instances() runs."""
    try:
        conn = _db()
        rows = conn.execute("SELECT * FROM gpu_state ORDER BY gpu_id").fetchall()
        conn.close()
        result = []
        for r in rows:
            temp = r["gpu_temp"]
            temp_level = "ok"
            if temp:
                if temp >= 85: temp_level = "hot"
                elif temp >= 70: temp_level = "warm"
            card_level = "ok"
            status = r["status"] or "stopped"
            if status in ("crashed", "zombie"): card_level = "error"
            elif temp_level == "hot": card_level = "hot"
            elif temp_level == "warm": card_level = "warm"

            vt = r["vram_total"] or r["vram_mb"] or 8192
            vu = r["vram_used"] or 0
            vram_pct = round(vu / vt * 100, 1) if vt else 0

            result.append({
                "id":          r["gpu_id"],
                "name":        r["card_name"],
                "pci":         r["pci_addr"] or GPU_MAP.get(r["gpu_id"], {}).get("pci", ""),
                "bar":         r["bar_size"],
                "vram_total":  vt,
                "vram_used":   vu,
                "vram_pct":    vram_pct,
                "temp":        temp,
                "temp_level":  temp_level,
                "card_level":  card_level,
                "model":       r["model_file"],
                "model_name":  r["model_name"],
                "mode":        r["mode"],
                "port":        r["port"],
                "pid":         r["pid"],
                "status":      status,
                "power_watts": r["power_watts"] or 0,
                "perf_level":  r["perf_level"] or "on",
                "sleep_watts": _get_sleep_watts(r["gpu_id"])[0],
                "sleep_calibrated": _get_sleep_watts(r["gpu_id"])[1],
                "awake_watts": _get_awake_watts(r["gpu_id"]),
                "default_model": r["default_model"] if "default_model" in r.keys() else None,
                "default_mode": r["default_mode"] if "default_mode" in r.keys() else None,
            })
        return result
    except Exception as e:
        print(f"DB read error: {e}")
        return []

def get_system_state_from_db() -> dict:
    """Read system state from SQLite."""
    try:
        conn = _db()
        r = conn.execute("SELECT * FROM system_state WHERE id = 1").fetchone()
        conn.close()
        if not r: return {}
        uptime_sec = r["uptime_sec"] or 0
        h, rem = divmod(uptime_sec, 3600)
        m, _ = divmod(rem, 60)
        return {
            "ram": {"total": r["ram_total"] or 0, "used": r["ram_used"] or 0, "avail": r["ram_avail"] or 0},
            "uptime": f"{h}h {m}m",
            "nginx_active": r["nginx_active"] or 0,
            "updated_at": r["updated_at"],
        }
    except:
        return {}

def get_perf_recent(n: int = 200) -> list[dict]:
    """Read last N entries from JSONL perf log."""
    entries = []
    if PERF_LOG.exists():
        try:
            lines = PERF_LOG.read_text().strip().split("\n")
            for line in lines[-n:]:
                if line.strip():
                    entries.append(json.loads(line))
        except Exception as e:
            print(f"PERF read error: {e}")
    return entries

def get_models_list() -> list[dict]:
    """List GGUF model files on disk."""
    result = []
    if MODEL_DIR.exists():
        for f in sorted(MODEL_DIR.glob("*.gguf")):
            result.append({"name": f.name, "path": str(f), "size_gb": round(f.stat().st_size / 1e9, 1)})
    return result

# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(title="AI Rig Dashboard", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Initialize SQLite DB on startup (DB lives on /dev/shm — wiped on reboot)
if shell:
    try:
        _call("init_db")
        _call("get_instances")  # Populate DB with fresh data
    except Exception as e:
        print(f"DB init: {e}")

STATIC_DIR   = Path(__file__).parent / "static"
PROFILES_FILE = Path(__file__).parent / "gpu_profiles.json"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Routes — serve UI ─────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return HTMLResponse(html_file.read_text())
    return HTMLResponse("<h1>Dashboard starting...</h1><p>index.html not found</p>")

# ── Routes — read data ────────────────────────────────────────────────────

@app.get("/api/gpus")
async def api_gpus():
    return get_gpu_data()

@app.get("/api/models")
async def api_models():
    return get_models_list()

@app.get("/api/perf/recent")
async def api_perf_recent(n: int = 200):
    return get_perf_recent(n)

@app.get("/api/state")
async def api_state():
    """All data from SQLite — single source of truth."""
    gpus = get_gpu_data()
    sys = get_system_state_from_db()
    active = sum(1 for g in gpus if g["status"] in ("ready", "loading"))
    total_vram = sum(g["vram_total"] for g in gpus)
    used_vram  = sum(g["vram_used"] for g in gpus)
    return {
        "uptime":        sys.get("uptime", "?"),
        "active_gpus":   active,
        "total_gpus":    NUM_GPUS,
        "total_vram_mb": total_vram,
        "used_vram_mb":  used_vram,
        "ram":           sys.get("ram", {"total":0,"used":0,"avail":0}),
        "total_tok_s":   0,
        "stub_mode":     shell is None,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }

@app.get("/api/logs/{gpu_id}")
async def api_logs(gpu_id: int):
    if gpu_id < 0 or gpu_id >= NUM_GPUS:
        raise HTTPException(404, "GPU not found")
    err_log = Path(f"/tmp/gpu{gpu_id}_err.log")
    out_log = Path(f"/tmp/gpu{gpu_id}.log")
    def read_tail(path, n=50):
        if not path.exists(): return []
        lines = path.read_text(errors="replace").splitlines()
        return lines[-n:]
    return {
        "gpu_id": gpu_id,
        "stderr": read_tail(err_log),
        "stdout": read_tail(out_log),
    }

# ── Routes — control ──────────────────────────────────────────────────────

class LoadRequest(BaseModel):
    model_path: str

@app.post("/api/gpu/{gpu_id}/load")
async def api_load(gpu_id: int, req: LoadRequest):
    if gpu_id < 0 or gpu_id >= NUM_GPUS:
        raise HTTPException(404, "GPU not found")
    if shell is None:
        return {"ok": False, "message": "Stub mode — cannot load models"}
    # Validate model_path: must be under MODEL_DIR, no traversal
    try:
        model_path = Path(req.model_path).resolve()
        model_dir_resolved = MODEL_DIR.resolve()
        if not str(model_path).startswith(str(model_dir_resolved)):
            raise HTTPException(400, f"model_path must be inside {MODEL_DIR}")
        if not model_path.exists():
            raise HTTPException(400, "model_path does not exist")
        if model_path.suffix.lower() not in (".gguf", ".bin"):
            raise HTTPException(400, "model_path must be a .gguf or .bin file")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Invalid model_path: {e}")
    try:
        pid = _call("start_gpu", gpu_id, str(model_path))
        _call("save_state")
        return {"ok": True, "pid": pid, "port": BASE_PORT + gpu_id}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/gpu/{gpu_id}/unload")
async def api_unload(gpu_id: int):
    if gpu_id < 0 or gpu_id >= NUM_GPUS:
        raise HTTPException(404, "GPU not found")
    if shell is None:
        return {"ok": False, "message": "Stub mode"}
    try:
        _call("kill_port", BASE_PORT + gpu_id)
        _call("save_state")
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/gpu/{gpu_id}/kill")
async def api_kill_zombie(gpu_id: int):
    if gpu_id < 0 or gpu_id >= NUM_GPUS:
        raise HTTPException(404, "GPU not found")
    if shell is None:
        return {"ok": False, "message": "Stub mode"}
    try:
        _call("kill_port", BASE_PORT + gpu_id)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, str(e))

class PowerRequest(BaseModel):
    level: str   # "low" or "on"

@app.post("/api/gpu/{gpu_id}/power")
async def api_gpu_power(gpu_id: int, req: PowerRequest):
    """Set GPU power: 'sleep' (D3hot BACO ~0W) or 'wake' (D0 full power)."""
    if gpu_id < 0 or gpu_id >= NUM_GPUS:
        raise HTTPException(404, "GPU not found")
    if shell is None:
        return {"ok": False, "message": "Stub mode"}
    level = req.level
    if level in ("wake", "on"): level = "auto"
    elif level in ("sleep", "low", "off"): level = "sleep"
    else:
        raise HTTPException(400, "level must be 'sleep' or 'wake'")
    try:
        loop = asyncio.get_running_loop()
        ok = await loop.run_in_executor(None, lambda: _call("set_gpu_power", gpu_id, level))
        return {"ok": ok, "gpu_id": gpu_id, "level": req.level}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/power/all")
async def api_power_all(req: PowerRequest):
    """Set power on all GPUs."""
    if shell is None:
        return {"ok": False, "message": "Stub mode"}
    level = req.level
    if level in ("wake", "on"): level = "auto"
    elif level in ("sleep", "low", "off"): level = "sleep"
    else:
        raise HTTPException(400, "level must be 'sleep' or 'wake'")
    try:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, lambda: _call("set_all_gpu_power", level))
        return {"ok": True, "results": results}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/nginx/rebuild")
async def api_nginx():
    if shell is None:
        return {"ok": False, "message": "Stub mode"}
    try:
        ok = _call("rebuild_nginx")
        return {"ok": ok}
    except Exception as e:
        raise HTTPException(500, str(e))

# ── Defaults API ──────────────────────────────────────────────────────────

@app.get("/api/defaults")
async def api_defaults():
    """Return GPU defaults from persistent file."""
    return _load_defaults()

class DefaultRequest(BaseModel):
    model: Optional[str] = None
    mode: Optional[str] = None
    temp: Optional[float] = None

@app.post("/api/gpu/{gpu_id}/default")
async def api_set_default(gpu_id: int, req: DefaultRequest):
    if shell is None: raise HTTPException(503, "Shell not loaded")
    if gpu_id < 0 or gpu_id >= NUM_GPUS: raise HTTPException(404, "GPU not found")
    loop = asyncio.get_running_loop()
    if req.model:
        await loop.run_in_executor(None, lambda: _call("set_default", gpu_id, req.model, req.mode or "", req.temp))
    else:
        await loop.run_in_executor(None, lambda: _call("clear_default", gpu_id))
    return {"ok": True}

@app.post("/api/defaults/load")
async def api_defaults_load():
    if shell is None: raise HTTPException(503, "Shell not loaded")
    loop = asyncio.get_running_loop()
    loaded = await loop.run_in_executor(None, _call, "load_all_defaults")
    return {"ok": True, "loaded": len(loaded)}

@app.post("/api/defaults/save-running")
async def api_defaults_save():
    if shell is None: raise HTTPException(503, "Shell not loaded")
    loop = asyncio.get_running_loop()
    saved = await loop.run_in_executor(None, _call, "save_running_as_defaults")
    return {"ok": True, "saved": saved}

class AutoSleepRequest(BaseModel):
    minutes: int

@app.post("/api/power/autosleep")
async def api_autosleep(req: AutoSleepRequest):
    if shell is None: raise HTTPException(503, "Shell not loaded")
    defaults = _load_defaults()
    defaults["auto_sleep_minutes"] = max(0, req.minutes)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: _call("save_defaults_file", defaults))
    return {"ok": True, "minutes": defaults["auto_sleep_minutes"]}

# ── Rig control ───────────────────────────────────────────────────────────

@app.post("/api/rig/reboot")
async def api_rig_reboot():
    """Graceful reboot. Saves state, stops GPUs."""
    loop = asyncio.get_running_loop()
    if shell:
        await loop.run_in_executor(None, lambda: (_call("save_running_as_defaults"), _call("save_state")))
    await loop.run_in_executor(None, lambda: subprocess.run(["sudo", "reboot"], timeout=5))
    return {"ok": True}

@app.post("/api/rig/suspend")
async def api_rig_suspend():
    """Suspend to RAM (S3). Risky with 7 GPUs on risers."""
    loop = asyncio.get_running_loop()
    # Save state before suspend
    if shell:
        await loop.run_in_executor(None, lambda: (_call("save_running_as_defaults"), _call("save_state")))
    await loop.run_in_executor(None, lambda: subprocess.run(["sudo", "systemctl", "suspend"], timeout=5))
    return {"ok": True}

@app.post("/api/rig/shutdown")
async def api_rig_shutdown():
    """Graceful shutdown (S5). Saves state, stops GPUs, powers off."""
    loop = asyncio.get_running_loop()
    if shell:
        await loop.run_in_executor(None, lambda: (_call("save_running_as_defaults"), _call("save_state")))
        for gpu in range(NUM_GPUS):
            try: await loop.run_in_executor(None, _call, "kill_port", BASE_PORT + gpu)
            except: pass
    await loop.run_in_executor(None, lambda: subprocess.run(["sudo", "systemctl", "poweroff"], timeout=5))
    return {"ok": True}

@app.get("/api/profiles")
async def api_profiles():
    """Return GPU hardware profiles from gpu_profiles.json."""
    if PROFILES_FILE.exists():
        try:
            return json.loads(PROFILES_FILE.read_text())
        except Exception as e:
            raise HTTPException(500, f"Could not read profiles: {e}")
    return {}

class ProfileUpdate(BaseModel):
    strengths:    list[str] | None = None
    limitations:  list[str] | None = None
    workloads:    list[dict] | None = None
    max_context:  str | None = None
    notes:        str | None = None

@app.put("/api/profiles/{gpu_id}")
async def api_update_profile(gpu_id: int, update: ProfileUpdate):
    """Persist edits to a single GPU's profile."""
    if gpu_id < 0 or gpu_id >= NUM_GPUS:
        raise HTTPException(404, "GPU not found")
    profiles = {}
    if PROFILES_FILE.exists():
        try:
            profiles = json.loads(PROFILES_FILE.read_text())
        except:
            pass
    key = str(gpu_id)
    current = profiles.get(key, {})
    payload = update.model_dump(exclude_none=True)
    current.update(payload)
    profiles[key] = current
    tmp = str(PROFILES_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(profiles, f, indent=2)
    os.rename(tmp, str(PROFILES_FILE))
    return {"ok": True, "profile": current}

@app.post("/api/state/save")
async def api_save_state():
    if shell is None:
        return {"ok": False, "message": "Stub mode"}
    _call("save_state")
    return {"ok": True}

# ── SSE — live GPU stream ──────────────────────────────────────────────────

async def gpu_event_generator() -> AsyncGenerator[str, None]:
    """Yield SSE events with GPU stats every 2 seconds.
    Flow: trigger shell to refresh DB → read from SQLite → send to browser."""
    loop = asyncio.get_running_loop()
    while True:
        try:
            # ACTION: tell shell to probe GPUs and write fresh data to SQLite
            if shell:
                await loop.run_in_executor(None, _call, "get_instances")
            gpus = get_gpu_data()        # READ: from SQLite
            sys = get_system_state_from_db()  # READ: from SQLite
            data = {
                "gpus":  gpus,
                "state": {
                    "uptime":      sys.get("uptime", "?"),
                    "active_gpus": sum(1 for g in gpus if g["status"] in ("ready", "loading")),
                    "total_gpus":  NUM_GPUS,
                    "ram":         sys.get("ram", {"total":0,"used":0,"avail":0}),
                    "timestamp":   datetime.now(timezone.utc).isoformat(),
                }
            }
            yield f"data: {json.dumps(data)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        await asyncio.sleep(2)

@app.get("/stream")
async def stream():
    return StreamingResponse(
        gpu_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )

# ── SSE — benchmark stream ─────────────────────────────────────────────────

import urllib.request, urllib.error

# Per-GPU benchmark lock — prevents two concurrent bench runs on the same GPU
_bench_locks: dict[int, bool] = {i: False for i in range(NUM_GPUS)}

async def bench_event_generator(gpu_id: int) -> AsyncGenerator[str, None]:
    _bench_locks[gpu_id] = True
    port = BASE_PORT + gpu_id
    prompts = [
        {"messages": [{"role":"user","content":"What is 2+2? Answer in one word."}], "max_tokens": 50},
        {"messages": [{"role":"user","content":"Write a Python function to reverse a string."}], "max_tokens": 400},
        {"messages": [{"role":"system","content":"Extract JSON: {name, age}"},{"role":"user","content":"John Smith, 35 years old."}], "max_tokens": 100},
        {"messages": [{"role":"user","content":"Explain TCP vs UDP in 3 sentences."}], "max_tokens": 400},
        {"messages": [{"role":"user","content":"A store has 20% off. Item costs $80. Final price?"}], "max_tokens": 300},
    ]
    speeds = []
    try:
        yield f"data: {json.dumps({'type':'start','gpu_id':gpu_id,'total':len(prompts)})}\n\n"
        for i, prompt in enumerate(prompts):
            start = time.time()
            try:
                data = json.dumps(prompt).encode()
                req  = urllib.request.Request(
                    f"http://localhost:{port}/v1/chat/completions", data=data,
                    headers={"Content-Type": "application/json"}
                )
                # Run blocking request in thread pool
                loop = asyncio.get_running_loop()
                def do_req():
                    resp = urllib.request.urlopen(req, timeout=60)
                    return json.loads(resp.read())
                r = await loop.run_in_executor(None, do_req)
                elapsed = time.time() - start
                speed  = r["timings"]["predicted_per_second"]
                tokens = r["usage"]["completion_tokens"]
                speeds.append(speed)
                # Log it
                if shell:
                    model = None
                    try:
                        instances = _call("get_instances")
                        model = instances[gpu_id].get("model")
                    except:
                        pass
                    try:
                        _call("log_perf", gpu_id, tokens, speed, elapsed, model or "?", "bench")
                    except:
                        pass
                yield f"data: {json.dumps({'type':'result','step':i+1,'tokens':tokens,'speed':round(speed,1),'elapsed':round(elapsed,2),'ok':True})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type':'result','step':i+1,'ok':False,'error':str(e)[:80]})}\n\n"
            await asyncio.sleep(0.1)
        avg = round(sum(speeds)/len(speeds), 1) if speeds else 0
        yield f"data: {json.dumps({'type':'done','avg_speed':avg,'passed':len(speeds),'total':len(prompts)})}\n\n"
    finally:
        _bench_locks[gpu_id] = False

@app.get("/api/bench/{gpu_id}/stream")
async def api_bench_stream(gpu_id: int):
    if gpu_id < 0 or gpu_id >= NUM_GPUS:
        raise HTTPException(404, "GPU not found")
    if _bench_locks.get(gpu_id):
        raise HTTPException(409, f"Benchmark already running on GPU {gpu_id}")
    return StreamingResponse(
        bench_event_generator(gpu_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

# ── Route: Testing ────────────────────────────────────────────────────────

_test_running = False

# ── Test prompts: same 5-level structure as shell verify ──────────────────

long_text = "The speed of light in a vacuum is approximately 299,792,458 meters per second. " * 30 + "Water boils at 100 degrees Celsius at standard atmospheric pressure. The chemical formula for water is H2O. The Earth orbits the Sun at an average distance of about 149.6 million kilometers. " * 10

VERIFY_TESTS = [
    {
        "name": "Basic", "desc": "Arithmetic",
        "prompt": {"messages": [{"role":"user","content":"What is 37 * 43? Show only the final number."}], "max_tokens":500},
        "expect_all": ["1591"], "timeout": 30,
    },
    {
        "name": "JSON", "desc": "Structured extraction",
        "prompt": {"messages": [
            {"role":"system","content":'Extract these facts as JSON only, no explanation: {"element", "symbol", "atomic_number", "state_at_room_temp", "group"}'},
            {"role":"user","content":"Gold is a chemical element with the symbol Au and atomic number 79. It is a solid at room temperature and belongs to group 11 of the periodic table."}
        ], "max_tokens":2000},
        "expect_all": ["Au", "79", "solid", "11", "Gold"], "timeout": 60,
    },
    {
        "name": "Code", "desc": "Function generation",
        "prompt": {"messages": [{"role":"user","content":"Write a Python function called celsius_to_fahrenheit that converts Celsius to Fahrenheit using the formula F = C * 9/5 + 32. Include a docstring. Show a test: celsius_to_fahrenheit(100) should return 212."}], "max_tokens":2000},
        "expect_all": ["def ", "celsius_to_fahrenheit", "return", "212", "32"], "timeout": 60,
    },
    {
        "name": "Reasoning", "desc": "Multi-step math",
        "prompt": {"messages": [{"role":"user","content":"A rectangle has length 15 cm and width 8 cm. Calculate: 1) The area. 2) The perimeter. 3) The length of the diagonal (round to 2 decimal places). Show each step."}], "max_tokens":2000},
        "expect_all": ["120", "46", "17"], "timeout": 60,
    },
    {
        "name": "Stress", "desc": "Large prompt (~3000 tokens)",
        "prompt": {"messages": [
            {"role":"system","content":"Answer only: What is the boiling point of water in Celsius? And what is the chemical formula for water?"},
            {"role":"user","content": long_text + " Based on the text above, answer the system's questions."}
        ], "max_tokens":500},
        "expect_all": ["100", "H2O"], "timeout": 120,
    },
]

# ── Model configs for all 7 test combos ──────────────────────────────────

TEST_CONFIGS = [
    ("gemma4",       "gemma-4-E4B-it-Q4_K_M.gguf",           0, 9080, "-ngl 99 -c 8192 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 --mmap -b 512 -ub 256 -t 4",                    "fast",        "Gemma4 fast (GPU0 8GB)"),
    ("gemma4",       "gemma-4-E4B-it-Q4_K_M.gguf",           0, 9080, "-ngl 99 -c 16384 --parallel 1 --cache-ram 0 --mmap -b 2048 -ub 512 -t 4 --flash-attn off",                                          "16k",         "Gemma4 16k (GPU0 8GB)"),
    ("qwen35-4b",    "qwen3.5-4b-q4_k_m.gguf",               0, 9080, "-ngl 99 -c 16384 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 --mmap -b 2048 -ub 512 -t 4",                    "default",     "Qwen3.5 4B (GPU0 8GB)"),
    ("qwen35-9b",    "qwen3.5-9b-q4_k_m.gguf",               0, 9080, "-ngl 99 -c 4096 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 --mmap -b 2048 -ub 512 -t 4",                     "default-8gb", "Qwen3.5 9B (GPU0 8GB 4K ctx)"),
    ("qwen35-9b",    "qwen3.5-9b-q4_k_m.gguf",               2, 9082, "-ngl 99 -c 16384 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 --mmap -b 2048 -ub 512 -t 4",                    "default-12gb","Qwen3.5 9B (GPU2 12GB 16K ctx)"),
    ("mistral-nemo", "mistral-nemo-12b-instruct-q4_k_m.gguf", 2, 9082, "-ngl 99 -c 8192 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 --mmap -b 2048 -ub 512 -t 4",                    "default",     "Mistral Nemo 12B (GPU2 12GB)"),
    ("qwen25-coder", "qwen2.5-coder-7b-instruct-q4_k_m.gguf", 0, 9080, "-ngl 99 -c 16384 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 --mmap -b 2048 -ub 512 -t 4",                   "default",     "Qwen2.5 Coder 7B (GPU0 8GB)"),
]

# GPU fitness: default model per GPU for GPU test mode
GPU_FITNESS_MODEL = {
    # 8GB GPUs get gemma4 16k (proven stable, fast load)
    0: ("gemma4", "16k"), 1: ("gemma4", "16k"),
    3: ("gemma4", "16k"), 4: ("gemma4", "16k"),
    5: ("gemma4", "16k"), 6: ("gemma4", "16k"),
    # 12GB GPU gets mistral-nemo
    2: ("mistral-nemo", "default"),
}

# ── Model configs API ────────────────────────────────────────────────────

@app.get("/api/model-configs")
async def api_model_configs():
    """Return MODEL_CONFIGS from shell for frontend dropdowns."""
    if shell is None:
        return {}
    configs = {}
    for name, cfg in shell.MODEL_CONFIGS.items():
        modes = {}
        for mname, mcfg in cfg["modes"].items():
            modes[mname] = {"desc": mcfg["desc"], "speed": mcfg["speed"], "context": mcfg["context"]}
        configs[name] = {
            "file": cfg["file"],
            "size_gb": cfg["size_gb"],
            "fits_8gb": cfg["fits_8gb"],
            "fits_12gb": cfg.get("fits_12gb", True),
            "default_mode": cfg["default_mode"],
            "thinking": cfg.get("thinking", False),
            "modes": modes,
        }
    return configs


# ── Core test runner (shared by all 3 modes) ─────────────────────────────

# ── SSE generators for each test mode ────────────────────────────────────

async def _test_stream(configs_iter, total):
    """Generic SSE generator. Streams sub-test results live as each verify prompt completes."""
    global _test_running
    _test_running = True
    try:
        yield f"data: {json.dumps({'type':'start','total':total,'timestamp':datetime.now().isoformat()})}\n\n"
        all_results = []
        for i, (name, file, gpu, port, flags, mode, label) in enumerate(configs_iter):
            yield f"data: {json.dumps({'type':'loading','step':i+1,'total':total,'label':label,'model':name,'mode':mode,'gpu':gpu})}\n\n"

            # Run model test, yielding sub-test events as they complete
            result = {"name": name, "mode": mode, "label": label, "gpu": gpu, "port": port,
                      "tests": [], "load_time": 0, "speed": 0, "status": "starting"}
            loop = asyncio.get_running_loop()

            # Kill + launch
            try:
                await loop.run_in_executor(None, shell.kill_port, port)
                await asyncio.sleep(2)
            except: pass

            model_path = str(MODEL_DIR / shell.MODEL_CONFIGS.get(name, {}).get("file", file))
            if not Path(model_path).exists():
                result["status"] = "missing"; result["error"] = f"File not found: {file}"
                all_results.append(result)
                yield f"data: {json.dumps({'type':'result','step':i+1,'total':total,**result})}\n\n"
                continue

            try:
                await loop.run_in_executor(None, lambda: shell.start_gpu(gpu, model_path, flags))
            except Exception as e:
                result["status"] = "launch_fail"; result["error"] = str(e)[:80]
                all_results.append(result)
                yield f"data: {json.dumps({'type':'result','step':i+1,'total':total,**result})}\n\n"
                continue

            # Wait for ready
            t0 = time.time()
            ready = False
            for _ in range(48):
                await asyncio.sleep(5)
                try:
                    def _ck(): return urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2).status
                    st = await loop.run_in_executor(None, _ck)
                    if st == 200: ready = True; break
                except urllib.error.HTTPError as e:
                    if e.code != 503: break
                except: pass

            result["load_time"] = round(time.time() - t0, 1)
            if not ready:
                result["status"] = "timeout"; result["error"] = f"Not ready after {result['load_time']}s"
                try: await loop.run_in_executor(None, shell.kill_port, port)
                except: pass
                all_results.append(result)
                yield f"data: {json.dumps({'type':'result','step':i+1,'total':total,**result})}\n\n"
                continue

            # Stream: send 'ready' event so frontend knows model is loaded
            yield f"data: {json.dumps({'type':'ready','step':i+1,'total':total,'label':label,'load_time':result['load_time']})}\n\n"

            # Run each verify test and stream result immediately
            speeds = []
            for ti, test in enumerate(VERIFY_TESTS):
                tprompt = test["prompt"]
                user_msg = ""; sys_msg = ""
                for m in tprompt.get("messages", []):
                    if m.get("role") == "user": user_msg = m.get("content", "")
                    elif m.get("role") == "system": sys_msg = m.get("content", "")

                t1 = time.time()
                try:
                    def _inf(p=tprompt, tout=test.get("timeout",60)):
                        data = json.dumps(p).encode()
                        req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                            data=data, headers={"Content-Type":"application/json"})
                        resp = urllib.request.urlopen(req, timeout=tout)
                        return json.loads(resp.read())
                    r = await loop.run_in_executor(None, _inf)
                    elapsed = round(time.time() - t1, 2)
                    content = r["choices"][0]["message"].get("content", "")
                    reasoning = r["choices"][0]["message"].get("reasoning_content", "")
                    full = (content + " " + reasoning).lower()
                    speed = round(r["timings"]["predicted_per_second"], 1)
                    tokens = r["usage"]["completion_tokens"]
                    prompt_tokens = r["usage"].get("prompt_tokens", 0)
                    expect_all = test.get("expect_all", [])
                    missing = [kw for kw in expect_all if kw.lower() not in full]
                    passed = len(missing) == 0
                    speeds.append(speed)
                    sub = {"name": test["name"], "desc": test.get("desc",""), "passed": passed,
                           "speed": speed, "tokens": tokens, "prompt_tokens": prompt_tokens, "elapsed": elapsed,
                           "prompt_text": user_msg[:300], "system_text": sys_msg[:200],
                           "expected": expect_all, "response": content[:500],
                           "reasoning": reasoning[:300] if reasoning else "", "missing": missing}
                except Exception as e:
                    sub = {"name": test["name"], "desc": test.get("desc",""), "passed": False,
                           "speed": 0, "tokens": 0, "prompt_tokens": 0, "elapsed": round(time.time()-t1,2),
                           "prompt_text": user_msg[:300], "system_text": sys_msg[:200],
                           "expected": test.get("expect_all",[]), "response": "", "reasoning": "", "missing": [],
                           "error": str(e)[:80]}

                result["tests"].append(sub)
                # Stream each sub-test live
                yield f"data: {json.dumps({'type':'subtest','step':i+1,'total':total,'test_idx':ti,'test_total':len(VERIFY_TESTS),'label':label,'subtest':sub})}\n\n"

            result["speed"] = round(sum(speeds)/len(speeds), 1) if speeds else 0
            result["status"] = "pass" if all(t["passed"] for t in result["tests"]) else "partial"

            # Unload
            try:
                await loop.run_in_executor(None, shell.kill_port, port)
                await asyncio.sleep(2)
            except: pass

            all_results.append(result)
            yield f"data: {json.dumps({'type':'result','step':i+1,'total':total,**result})}\n\n"

        passed = sum(1 for r in all_results if r["status"] == "pass")
        partial = sum(1 for r in all_results if r["status"] == "partial")
        failed = total - passed - partial
        yield f"data: {json.dumps({'type':'done','passed':passed,'partial':partial,'failed':failed,'total':total,'results':all_results})}\n\n"
    finally:
        _test_running = False


def _sse_response(gen):
    return StreamingResponse(gen, media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Mode 1: Model test (all model configs) ───────────────────────────────

@app.get("/api/test/run")
async def api_test_run():
    if shell is None: raise HTTPException(503, "Shell not loaded")
    if _test_running: raise HTTPException(409, "Test already running")
    return _sse_response(_test_stream(TEST_CONFIGS, len(TEST_CONFIGS)))


# ── Mode 2: GPU fitness test ─────────────────────────────────────────────

@app.get("/api/test/gpu/run")
async def api_test_gpu_run():
    """Test each GPU with a standard model to check fitness."""
    if shell is None: raise HTTPException(503, "Shell not loaded")
    if _test_running: raise HTTPException(409, "Test already running")

    configs = []
    for gpu_id in range(NUM_GPUS):
        model_name, mode_name = GPU_FITNESS_MODEL.get(gpu_id, ("gemma4", "16k"))
        cfg = shell.MODEL_CONFIGS.get(model_name, {})
        mode_cfg = cfg.get("modes", {}).get(mode_name, {})
        flags = mode_cfg.get("flags_12gb", mode_cfg.get("flags")) if gpu_id == 2 else mode_cfg.get("flags")
        if not flags:
            continue
        info = GPU_MAP.get(gpu_id, {})
        label = f"GPU {gpu_id} {info.get('name','')} [{info.get('bar','')}]"
        configs.append((model_name, cfg["file"], gpu_id, BASE_PORT + gpu_id, flags, mode_name, label))

    return _sse_response(_test_stream(configs, len(configs)))


# ── Mode 3: Custom test (user picks GPU + model + mode + temp) ───────────

@app.get("/api/test/custom/run")
async def api_test_custom_run(gpu: str = "0", model: str = "gemma4", mode: str = "", temp: Optional[float] = None):
    """Run test on specific GPU(s) with specific model/mode/temperature."""
    if shell is None: raise HTTPException(503, "Shell not loaded")
    if _test_running: raise HTTPException(409, "Test already running")

    cfg = shell.MODEL_CONFIGS.get(model)
    if not cfg:
        raise HTTPException(400, f"Unknown model: {model}. Available: {list(shell.MODEL_CONFIGS.keys())}")

    if not mode:
        mode = cfg["default_mode"]
    mode_cfg = cfg.get("modes", {}).get(mode)
    if not mode_cfg:
        raise HTTPException(400, f"Unknown mode '{mode}' for {model}. Available: {list(cfg['modes'].keys())}")

    # Parse GPU list
    if gpu.lower() == "all":
        gpu_ids = [g for g in range(NUM_GPUS) if cfg["fits_8gb"] or g == 2]
    else:
        try:
            gpu_ids = [int(x) for x in gpu.split(",")]
        except:
            raise HTTPException(400, f"Invalid gpu: {gpu}")

    configs = []
    for gpu_id in gpu_ids:
        if gpu_id < 0 or gpu_id >= NUM_GPUS:
            continue
        if not cfg["fits_8gb"] and gpu_id != 2:
            continue  # Model too big for 8GB
        flags = mode_cfg.get("flags_12gb", mode_cfg.get("flags")) if gpu_id == 2 else mode_cfg.get("flags")
        if temp is not None:
            flags += f" --temp {temp}"
        info = GPU_MAP.get(gpu_id, {})
        temp_str = f" temp={temp}" if temp is not None else ""
        label = f"{model} {mode}{temp_str} (GPU{gpu_id} {info.get('bar','')})"
        configs.append((model, cfg["file"], gpu_id, BASE_PORT + gpu_id, flags, mode, label))

    if not configs:
        raise HTTPException(400, "No valid GPU/model combinations")

    return _sse_response(_test_stream(configs, len(configs)))


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    try:
        ip = socket.gethostbyname(hostname)
    except:
        ip = "127.0.0.1"
    print(f"\n  AI Rig Dashboard")
    print(f"  ─────────────────────────────────────────")
    print(f"  Local:   http://127.0.0.1:{DASHBOARD_PORT}")
    print(f"  Network: http://{ip}:{DASHBOARD_PORT}")
    if shell is None:
        print(f"  Mode:    STUB (no rig shell found)")
    else:
        print(f"  Mode:    LIVE (shell loaded)")
    print(f"  ─────────────────────────────────────────\n")
    uvicorn.run(app, host="0.0.0.0", port=DASHBOARD_PORT, log_level="warning")
