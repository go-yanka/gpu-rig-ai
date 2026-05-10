#!/usr/bin/env python3
"""
AI Rig Management Shell v1.0
Run on rig: python3 /opt/ai-rig-shell.py
"""

import os, sys, re, time, json, signal, subprocess, shutil, sqlite3
import urllib.request, urllib.error, urllib.parse, curses
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════

MODEL_DIR = Path("/opt/ai-models")
LLAMA_BIN = "/opt/llama-server/llama-server-rocm"
LLAMA_LIB = "/opt/llama-server"
BASE_PORT = 9080
NUM_GPUS = 7
LB_PORT = 4000
STATE_FILE = "/opt/ai-rig-state.json"
DEFAULTS_FILE = "/opt/ai-rig-defaults.json"
PERF_LOG = "/opt/ai-rig-perf.jsonl"
PERF_LOG_MAX_MB = 10
PROFILES_FILE = "/opt/gpu_profiles.json"

GPU_MAP = {
    0: {"name": "RX 5700 XT (ASUS)",     "vram_mb": 8176, "bar": "8GB",   "pci": "03:00.0", "arch": "RDNA1", "grade": "A"},
    1: {"name": "RX 5700 XT (Dell)",      "vram_mb": 8176, "bar": "8GB",   "pci": "06:00.0", "arch": "RDNA1", "grade": "A"},
    2: {"name": "RX 6700 XT (ASRock)",    "vram_mb": 12272,"bar": "16GB",  "pci": "09:00.0", "arch": "RDNA2", "grade": "A+"},
    3: {"name": "RX 5700 XT (Sapphire)",  "vram_mb": 8192, "bar": "256MB", "pci": "0f:00.0", "arch": "RDNA1", "grade": "B"},
    4: {"name": "RX 5700 XT (PwrColor)",  "vram_mb": 8192, "bar": "256MB", "pci": "14:00.0", "arch": "RDNA1", "grade": "B"},
    5: {"name": "RX 5700 XT (ASRock)",    "vram_mb": 8192, "bar": "256MB", "pci": "17:00.0", "arch": "RDNA1", "grade": "B"},
    6: {"name": "RX 5700 XT (ASRock)",    "vram_mb": 8192, "bar": "256MB", "pci": "1a:00.0", "arch": "RDNA1", "grade": "B"},
    # Grade: A+ = full BAR + 12GB RDNA2, A = full BAR + 8GB, B = 256MB BAR (slow load, ~25% slower inference)
}

# ── MODEL + MODE CONFIGS (proven stable, do NOT change without testing) ──
# Each model has one or more modes with exact flags that work.
# The "why" is documented so nobody removes a flag without understanding.

MODEL_CONFIGS = {
    "gemma4": {
        "file": "gemma-4-E4B-it-Q4_K_M.gguf",
        "size_gb": 4.7,
        "default_mode": "16k",
        "thinking": True,
        "modes": {
            "fast": {
                "desc": "Thinking ON, 8K ctx, small batch",
                "speed": "~47 tok/s",
                "context": 8192,
                "flags": "-ngl 99 -c 8192 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 --mmap -b 512 -ub 256 -t 4 --reasoning-format deepseek",
                "why": "-b 512 -ub 256 avoids llama.cpp issue #21336. Thinking ON by default (reasoning_content populated).",
            },
            "16k": {
                "desc": "Thinking ON, 16K ctx, flash-attn off",
                "speed": "~48 tok/s",
                "context": 16384,
                "flags": "-ngl 99 -c 16384 --parallel 1 --cache-ram 0 --mmap -b 2048 -ub 512 -t 4 --flash-attn off --reasoning-format deepseek",
                "why": "--flash-attn off fixes issue #21336. Thinking ON. No KV quant with flash-attn off.",
            },
            "nothink": {
                "desc": "Thinking OFF, 16K ctx, direct answers",
                "speed": "~48 tok/s",
                "context": 16384,
                "flags": "-ngl 99 -c 16384 --parallel 1 --cache-ram 0 --mmap -b 2048 -ub 512 -t 4 --flash-attn off --reasoning off",
                "why": "Thinking disabled. Answers directly in content, no reasoning_content. Faster for simple Q&A, tool calls, Ritu's agent.",
            },
            "nothink-fast": {
                "desc": "Thinking OFF, 8K ctx, small batch",
                "speed": "~47 tok/s",
                "context": 8192,
                "flags": "-ngl 99 -c 8192 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 --mmap -b 512 -ub 256 -t 4 --reasoning off",
                "why": "Thinking disabled + small batch workaround for flash-attn bug. Direct answers only.",
            },
        },
        "fits_8gb": True,
        "fits_12gb": True,
    },
    "qwen35-4b": {
        "file": "qwen3.5-4b-q4_k_m.gguf",
        "size_gb": 2.6,
        "default_mode": "default",
        "thinking": True,
        "modes": {
            "default": {
                "desc": "Thinking ON, 16K ctx",
                "speed": "~48 tok/s",
                "context": 16384,
                "flags": "-ngl 99 -c 16384 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 --mmap -b 2048 -ub 512 -t 4 --reasoning-format deepseek",
                "why": "Thinking enabled. Model reasons in reasoning_content then answers in content. Needs max_tokens ~200+ for short questions.",
            },
            "nothink": {
                "desc": "Thinking OFF, 16K ctx, faster",
                "speed": "~52 tok/s",
                "context": 16384,
                "flags": "-ngl 99 -c 16384 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 --mmap -b 2048 -ub 512 -t 4 --reasoning off",
                "why": "Thinking disabled. Answers directly without reasoning_content. Faster, fewer tokens, but less accurate on complex tasks.",
            },
        },
        "fits_8gb": True,
        "fits_12gb": True,
    },
    "qwen35-9b": {
        "file": "qwen3.5-9b-q4_k_m.gguf",
        "size_gb": 5.3,
        "default_mode": "default",
        "thinking": True,
        "modes": {
            "default": {
                "desc": "Thinking ON, ctx auto per GPU",
                "speed": "~46/37 tok/s",
                "context": 4096,
                "flags": "-ngl 99 -c 4096 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 --mmap -b 2048 -ub 512 -t 4 --reasoning-format deepseek",
                "flags_12gb": "-ngl 99 -c 16384 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 --mmap -b 2048 -ub 512 -t 4 --reasoning-format deepseek",
                "why": "Thinking enabled. 4K on 8GB, 16K on GPU 2 (12GB). Reasons before answering.",
            },
            "nothink": {
                "desc": "Thinking OFF, ctx auto per GPU",
                "speed": "~50/40 tok/s",
                "context": 4096,
                "flags": "-ngl 99 -c 4096 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 --mmap -b 2048 -ub 512 -t 4 --reasoning off",
                "flags_12gb": "-ngl 99 -c 16384 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 --mmap -b 2048 -ub 512 -t 4 --reasoning off",
                "why": "Thinking disabled. Answers directly. Faster for tool calling and simple Q&A.",
            },
        },
        "fits_8gb": True,
        "fits_12gb": True,
    },
    "mistral-nemo": {
        "file": "mistral-nemo-12b-instruct-q4_k_m.gguf",
        "size_gb": 7.0,
        "default_mode": "default",
        "modes": {
            "default": {
                "desc": "12B model, deep reasoning, 12GB GPU only",
                "speed": "~42 tok/s",
                "context": 8192,
                "flags": "-ngl 99 -c 8192 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 --mmap -b 2048 -ub 512 -t 4",
                "why": "7GB model only fits on GPU 2 (12GB). 8K context to leave room for KV cache.",
            },
        },
        "fits_8gb": False,
        "fits_12gb": True,
    },
    "qwen25-coder": {
        "file": "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "size_gb": 4.4,
        "default_mode": "default",
        "modes": {
            "default": {
                "desc": "Code-focused, original OpenClaw config",
                "speed": "~57 tok/s",
                "context": 16384,
                "flags": "-ngl 99 -c 16384 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 --mmap -b 2048 -ub 512 -t 4",
                "why": "Proven config from original OpenClaw session handoff.",
            },
        },
        "fits_8gb": True,
        "fits_12gb": True,
    },
}

# NEVER USE THESE FLAGS
BANNED_FLAGS = {"--mmproj", "--no-mmap"}
# Note: --flash-attn off is OK (used by gemma4 16k mode)
# Note: --flash-attn (on) is OK for non-Gemma models

# PCI address format validation
_PCI_RE = re.compile(r'^[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$')

# ══════════════════════════════════════════════════════════════════════════
# DATABASE — Single Source of Truth
# ══════════════════════════════════════════════════════════════════════════

DB_PATH = "/dev/shm/ai-rig.db"       # Runtime — fast writes, no USB wear
DB_PERSIST_PATH = "/opt/ai-rig.db"   # Persistent — synced periodically to USB

def _restore_db_from_persist():
    """On boot: copy persistent DB from USB to RAM if it exists."""
    if os.path.exists(DB_PERSIST_PATH) and not os.path.exists(DB_PATH):
        import shutil
        shutil.copy2(DB_PERSIST_PATH, DB_PATH)
        try: os.chmod(DB_PATH, 0o666)
        except: pass

def sync_db_to_persist():
    """Sync RAM DB to USB periodically (call every 5 min or on config changes)."""
    try:
        import shutil
        shutil.copy2(DB_PATH, DB_PERSIST_PATH + ".tmp")
        os.rename(DB_PERSIST_PATH + ".tmp", DB_PERSIST_PATH)
    except Exception:
        pass

def init_db():
    """Create SQLite database and tables if they don't exist.
    On boot: restores from persistent USB copy first."""
    _restore_db_from_persist()
    conn = sqlite3.connect(DB_PATH)
    # Ensure file is readable/writable by all (shell runs as root, dashboard as user)
    try: os.chmod(DB_PATH, 0o666)
    except: pass
    conn.execute("PRAGMA journal_mode=WAL")  # concurrent reads while writing
    conn.execute("PRAGMA busy_timeout=3000")  # wait up to 3s if locked
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS gpu_state (
            gpu_id      INTEGER PRIMARY KEY,
            card_name   TEXT,
            vram_mb     INTEGER,
            bar_size    TEXT,
            grade       TEXT,
            port        INTEGER,
            pid         INTEGER,
            status      TEXT,
            model_file  TEXT,
            model_name  TEXT,
            mode        TEXT,
            temperature REAL,
            vram_used   INTEGER,
            vram_total  INTEGER,
            gpu_temp    INTEGER,
            speed       REAL,
            flags       TEXT,
            context     INTEGER,
            power_watts REAL,
            perf_level  TEXT,
            pci_addr    TEXT,
            updated_at  TEXT
        );
        CREATE TABLE IF NOT EXISTS worklog (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            gpu_id      INTEGER,
            started_at  TEXT,
            ended_at    TEXT,
            client_ip   TEXT,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            speed       REAL,
            model_name  TEXT,
            status      TEXT,
            duration_ms INTEGER
        );
        CREATE TABLE IF NOT EXISTS events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            gpu_id      INTEGER,
            action      TEXT,
            detail      TEXT,
            source      TEXT,
            created_at  TEXT
        );
        CREATE TABLE IF NOT EXISTS system_state (
            id                  INTEGER PRIMARY KEY DEFAULT 1,
            ram_total           INTEGER,
            ram_used            INTEGER,
            ram_avail           INTEGER,
            nginx_active        INTEGER,
            uptime_sec          INTEGER,
            auto_sleep_minutes  INTEGER DEFAULT 0,
            auto_load           INTEGER DEFAULT 0,
            updated_at          TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_worklog_gpu ON worklog(gpu_id, started_at);
        CREATE INDEX IF NOT EXISTS idx_events_gpu ON events(gpu_id, created_at);
    """)
    # Migrate: add columns if missing (safe for existing DB on /dev/shm)
    existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(gpu_state)")}
    for col, typ in [("power_watts", "REAL"), ("perf_level", "TEXT"), ("pci_addr", "TEXT"),
                      ("vram_total", "INTEGER"), ("idle_since", "TEXT"),
                      ("default_model", "TEXT"), ("default_mode", "TEXT"), ("default_temp", "REAL"),
                      ("sleep_watts_cal", "REAL"), ("awake_watts_cal", "REAL")]:
        if col not in existing_cols:
            try: conn.execute(f"ALTER TABLE gpu_state ADD COLUMN {col} {typ}")
            except: pass

    # Migrate system_state columns
    sys_cols = {r[1] for r in conn.execute("PRAGMA table_info(system_state)")}
    for col, typ in [("auto_sleep_minutes", "INTEGER DEFAULT 0"), ("auto_load", "INTEGER DEFAULT 0")]:
        cname = col.split()[0] if " " in col else col
        if cname not in sys_cols:
            try: conn.execute(f"ALTER TABLE system_state ADD COLUMN {col} {typ}")
            except: pass

    # Migrate data from JSON file if it exists (one-time migration)
    try:
        import json as _json
        jpath = "/opt/ai-rig-defaults.json"
        if os.path.exists(jpath):
            with open(jpath) as f:
                jdata = _json.load(f)
            # Migrate auto_sleep and auto_load
            conn.execute("UPDATE system_state SET auto_sleep_minutes=?, auto_load=? WHERE id=1",
                (jdata.get("auto_sleep_minutes", 0), 1 if jdata.get("auto_load") else 0))
            # Migrate GPU defaults and calibration
            for gid_str, gdef in jdata.get("gpus", {}).items():
                if gdef:
                    conn.execute("UPDATE gpu_state SET default_model=?, default_mode=?, default_temp=? WHERE gpu_id=?",
                        (gdef.get("model"), gdef.get("mode"), gdef.get("temp"), int(gid_str)))
            for gid_str, sw in jdata.get("gpu_sleep_watts", {}).items():
                aw = jdata.get("gpu_awake_watts", {}).get(gid_str)
                conn.execute("UPDATE gpu_state SET sleep_watts_cal=?, awake_watts_cal=? WHERE gpu_id=?",
                    (sw, aw, int(gid_str)))
            conn.commit()
            # Rename JSON file so migration doesn't run again
            os.rename(jpath, jpath + ".migrated")
    except Exception:
        pass

    # Initialize gpu_state rows from GPU_MAP
    for gpu_id, info in GPU_MAP.items():
        conn.execute("""
            INSERT OR IGNORE INTO gpu_state (gpu_id, card_name, vram_mb, bar_size, grade, port, pci_addr)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (gpu_id, info["name"], info["vram_mb"], info["bar"], info.get("grade", "?"),
              BASE_PORT + gpu_id, info.get("pci", "")))
    conn.execute("INSERT OR IGNORE INTO system_state (id) VALUES (1)")
    conn.commit()
    conn.close()

def _db():
    """Get a database connection. Short-lived — open, use, close."""
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn

def db_write_gpu_state(gpu_id, **fields):
    """Update a single GPU's state in the database."""
    if not fields:
        return
    sets = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [datetime.now(timezone.utc).isoformat(), gpu_id]
    conn = _db()
    conn.execute(f"UPDATE gpu_state SET {sets}, updated_at = ? WHERE gpu_id = ?", vals)
    conn.commit()
    conn.close()

def db_write_all_gpu_states(instances):
    """Write ALL GPU states to SQLite — this is THE single source of truth.
    Dashboard reads ONLY from this database.
    Also handles: idle_since tracking, auto-sleep policy, GPU defaults."""
    conn = _db()
    now = datetime.now(timezone.utc).isoformat()

    # Load defaults for GPU default model info
    defaults = load_defaults_file()
    gpu_defaults = defaults.get("gpus", {})
    auto_sleep_min = defaults.get("auto_sleep_minutes", 0)

    for gpu_id, inst in instances.items():
        mode_info = _get_gpu_mode(gpu_id)
        model_name = mode_info.get("model") if mode_info else None
        mode = mode_info.get("mode") if mode_info else None
        temp_setting = mode_info.get("temp") if mode_info else None
        status = inst.get("status", "stopped")

        # Track idle_since: when a GPU goes from active to stopped, record timestamp
        # If GPU is running/loading/sleeping, clear idle_since
        if status in ("ready", "loading", "sleeping"):
            idle_since = None
        elif status == "stopped":
            # Check if it was already idle — keep existing timestamp
            existing = conn.execute("SELECT idle_since FROM gpu_state WHERE gpu_id = ?", (gpu_id,)).fetchone()
            idle_since = existing["idle_since"] if existing and existing["idle_since"] else now
        else:
            idle_since = None

        # GPU defaults from file
        gd = gpu_defaults.get(str(gpu_id))
        def_model = gd.get("model") if gd else None
        def_mode = gd.get("mode") if gd else None
        def_temp = gd.get("temp") if gd else None

        conn.execute("""
            UPDATE gpu_state SET
                pid = ?, status = ?, model_file = ?, model_name = ?, mode = ?,
                temperature = ?, vram_used = ?, vram_total = ?, gpu_temp = ?,
                power_watts = ?, perf_level = ?, idle_since = ?,
                default_model = ?, default_mode = ?, default_temp = ?,
                updated_at = ?
            WHERE gpu_id = ?
        """, (inst.get("pid"), status, inst.get("model"),
              model_name, mode, temp_setting,
              inst.get("vram_used"), inst.get("vram_total"),
              inst.get("temp"),
              inst.get("power_watts", 0), inst.get("perf_level", "on"),
              idle_since,
              def_model, def_mode, def_temp,
              now, gpu_id))

        # Auto-sleep policy: if GPU is idle for too long, sleep it
        if auto_sleep_min > 0 and status == "stopped" and idle_since:
            try:
                idle_dt = datetime.fromisoformat(idle_since)
                idle_secs = (datetime.fromisoformat(now) - idle_dt).total_seconds()
                if idle_secs > auto_sleep_min * 60:
                    set_gpu_power(gpu_id, "sleep")
            except:
                pass
    # Also update system state
    ram = get_ram_info()
    instances_ready = sum(1 for i in instances.values() if i.get("status") in ("ready", "loading"))
    uptime = int(float(Path("/proc/uptime").read_text().split()[0]))
    conn.execute("""
        UPDATE system_state SET
            ram_total = ?, ram_used = ?, ram_avail = ?,
            nginx_active = ?, uptime_sec = ?, updated_at = ?
        WHERE id = 1
    """, (ram["total"], ram["used"], ram["avail"], instances_ready, uptime, now))
    conn.commit()
    conn.close()

def db_record_event(gpu_id, action, detail="", source="shell"):
    """Record a GPU lifecycle event."""
    conn = _db()
    conn.execute(
        "INSERT INTO events (gpu_id, action, detail, source, created_at) VALUES (?, ?, ?, ?, ?)",
        (gpu_id, action, detail, source, datetime.now(timezone.utc).isoformat())
    )
    # Keep only last 500 events
    conn.execute("DELETE FROM events WHERE id NOT IN (SELECT id FROM events ORDER BY id DESC LIMIT 500)")
    conn.commit()
    conn.close()

def db_get_worklog(gpu_id=None, limit=50):
    """Query worklog entries."""
    conn = _db()
    if gpu_id is not None:
        rows = conn.execute(
            "SELECT * FROM worklog WHERE gpu_id = ? ORDER BY id DESC LIMIT ?", (gpu_id, limit)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM worklog ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def db_get_events(gpu_id=None, limit=20):
    """Query event history."""
    conn = _db()
    if gpu_id is not None:
        rows = conn.execute(
            "SELECT * FROM events WHERE gpu_id = ? ORDER BY id DESC LIMIT ?", (gpu_id, limit)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def db_get_gpu_states():
    """Read all GPU states from database."""
    conn = _db()
    rows = conn.execute("SELECT * FROM gpu_state ORDER BY gpu_id").fetchall()
    system = conn.execute("SELECT * FROM system_state WHERE id = 1").fetchone()
    conn.close()
    return [dict(r) for r in rows], dict(system) if system else {}

# Colors
R="\033[91m"; G="\033[92m"; Y="\033[93m"; B="\033[94m"; C="\033[96m"; W="\033[97m"
DIM="\033[2m"; BOLD="\033[1m"; NC="\033[0m"

def co(color, text):
    return f"{color}{text}{NC}"

# ══════════════════════════════════════════════════════════════════════════
# GPU DISCOVERY
# ══════════════════════════════════════════════════════════════════════════

def get_vram(gpu_idx):
    """Read VRAM usage from sysfs."""
    try:
        used = int(Path(f"/sys/class/drm/card{gpu_idx}/device/mem_info_vram_used").read_text().strip())
        total = int(Path(f"/sys/class/drm/card{gpu_idx}/device/mem_info_vram_total").read_text().strip())
        return used // (1024*1024), total // (1024*1024)
    except Exception:
        return 0, GPU_MAP.get(gpu_idx, {}).get("vram_mb", 0)

def get_gpu_temp(gpu_idx):
    """Read GPU temperature."""
    try:
        for hwmon in Path(f"/sys/class/drm/card{gpu_idx}/device/hwmon").iterdir():
            t_path = hwmon / "temp1_input"
            if t_path.exists():
                return int(t_path.read_text().strip()) // 1000
    except Exception:
        return None
    return None

def get_pid_on_port(port):
    """Get PID of process listening on port. Uses list args — no shell injection risk."""
    try:
        r = subprocess.run(
            ["lsof", "-t", f"-i:{port}"],
            capture_output=True, text=True, timeout=5
        )
        pid = r.stdout.strip().split("\n")[0]
        return int(pid) if pid else None
    except Exception:
        return None

def get_model_on_port(pid):
    """Get model name from process cmdline."""
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_text().replace("\0", " ")
        # Verify this is actually a llama-server process before trusting cmdline
        if "llama-server" not in cmdline:
            return None
        if "--model" in cmdline:
            path = cmdline.split("--model")[1].strip().split(" ")[0]
            return Path(path).name
    except Exception:
        pass
    return None

def get_process_state(pid):
    """Check if process is alive, zombie, or D-state. Returns (state, age_seconds)."""
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        state_char = None
        for line in status.split("\n"):
            if line.startswith("State:"):
                state_char = line.split()[1]
                break
        # Get process age in seconds
        age = None
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            starttime = int(stat.split(")")[1].split()[19])  # field 22 (0-indexed after comm)
            uptime = float(Path("/proc/uptime").read_text().split()[0])
            clk_tck = os.sysconf("SC_CLK_TCK")
            age = uptime - (starttime / clk_tck)
        except Exception:
            age = None
        if state_char == "Z": return "zombie", age
        if state_char == "D": return "d-state", age
        return "alive", age
    except Exception:
        return "dead", None
    return "dead", None

def check_health(port):
    """Check llama-server health endpoint."""
    try:
        resp = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
        data = json.loads(resp.read())
        return "ready" if data.get("status") == "ok" else "loading"
    except urllib.error.HTTPError as e:
        # 503 = model still loading, NOT a crash
        if e.code == 503:
            return "loading"
        # Other 5xx = actual server error
        if e.code >= 500:
            return "error"
        return "loading"
    except urllib.error.URLError:
        # Connection refused = server not listening yet (still starting)
        return "starting"
    except Exception:
        return "starting"

def _get_instance_worker(gpu):
    """Worker for parallel get_instances."""
    port = BASE_PORT + gpu
    bar = GPU_MAP.get(gpu, {}).get("bar", "?")

    # GATEKEEPER: Check sleep state via SAFE PCI bus path FIRST
    # /sys/bus/pci/devices/ is kernel PCI core — does NOT trigger amdgpu resume
    # Only if GPU is awake do we touch /sys/class/drm/ (which WOULD wake it)
    if _is_gpu_sleeping(gpu):
        return gpu, {
            "port": port, "pid": None, "model": None, "status": "sleeping",
            "vram_used": 0, "vram_total": GPU_MAP[gpu].get("vram_mb", 0),
            "temp": None, "bar": bar, "power_watts": 0, "perf_level": "sleep",
        }

    # GPU is awake — safe to probe sysfs
    pid = get_pid_on_port(port)
    model = get_model_on_port(pid) if pid else None
    vram_used, vram_total = get_vram(gpu)
    temp = get_gpu_temp(gpu)

    if pid:
        pstate, page = get_process_state(pid)
        if pstate == "zombie":
            status = "zombie"  # True zombie (Z-state), always bad
        elif pstate == "d-state":
            # D-state = uninterruptible sleep. NORMAL during Vulkan GPU init.
            # 256MB BAR GPUs can take 3-4 minutes to load a model.
            # Only call it zombie if stuck in D-state for >5 minutes.
            if page is not None and page < 300:
                status = "loading"  # Process is young, D-state is expected
            else:
                status = "zombie"  # Stuck >5 min, truly stuck
        elif pstate == "dead":
            status = "crashed"
        else:
            health = check_health(port)
            if health == "starting":
                # Process alive but port not bound yet — still loading model
                # 256MB BAR GPUs take 3-4 min, don't call it crashed
                status = "loading"
            else:
                status = health
    else:
        status = "stopped"

    pwr = get_gpu_power(gpu)
    return gpu, {
        "port": port, "pid": pid, "model": model, "status": status,
        "vram_used": vram_used, "vram_total": vram_total, "temp": temp,
        "bar": bar, "power_watts": pwr["power_watts"], "perf_level": pwr["perf_level"],
    }

def get_instances():
    """Get status of all GPU instances in parallel. Writes to SQLite as single source of truth.
    IMPORTANT: Check sleeping GPUs FIRST (via safe PCI path) before spawning any probe threads.
    This prevents DRM/hwmon probes from waking sleeping GPUs via shared PCIe bridges."""

    # Step 1: Identify sleeping GPUs using SAFE PCI bus path (no wake)
    sleeping = {}
    awake_gpus = []
    for gpu in range(NUM_GPUS):
        if _is_gpu_sleeping(gpu):
            sleeping[gpu] = (gpu, {
                "port": BASE_PORT + gpu, "pid": None, "model": None, "status": "sleeping",
                "vram_used": 0, "vram_total": GPU_MAP[gpu].get("vram_mb", 0),
                "temp": None, "bar": GPU_MAP[gpu].get("bar", "?"),
                "power_watts": 0, "perf_level": "sleep",
            })
        else:
            awake_gpus.append(gpu)

    # Step 2: Only probe AWAKE GPUs (safe to touch DRM/hwmon)
    with ThreadPoolExecutor(max_workers=max(len(awake_gpus), 1)) as executor:
        awake_results = list(executor.map(_get_instance_worker, awake_gpus))

    # Step 3: Merge sleeping + awake results
    instances = dict(sleeping.values())
    instances.update(dict(awake_results))
    # Write to database — this is THE source of truth
    try:
        db_write_all_gpu_states(instances)
    except Exception:
        pass  # DB write failure shouldn't break GPU detection
    return instances

# ══════════════════════════════════════════════════════════════════════════
# MODEL MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════

def get_models():
    """List GGUF models on disk."""
    models = []
    if MODEL_DIR.exists():
        for f in sorted(MODEL_DIR.glob("*.gguf")):
            if "mmproj" not in f.name:
                models.append({"name": f.name, "path": str(f), "size_gb": round(f.stat().st_size / (1024**3), 1)})
    return models

def model_fits(model_path, gpu_idx):
    """Check if model fits in GPU VRAM."""
    try:
        size_mb = os.path.getsize(model_path) / (1024**2)
    except OSError:
        return False, "file not found"
    if gpu_idx not in GPU_MAP:
        return False, "unknown GPU"
    vram_mb = GPU_MAP[gpu_idx]["vram_mb"]
    kv_mb = 700 if gpu_idx == 2 else 500
    needed = size_mb + kv_mb + 630
    return needed < vram_mb, f"{needed:.0f}/{vram_mb}MB"

# ══════════════════════════════════════════════════════════════════════════
# PROCESS MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════

def start_gpu(gpu_idx, model_path, flags=None):
    """Start llama-server on a GPU. Returns PID."""
    port = BASE_PORT + gpu_idx
    kill_port(port)

    # Wake GPU from D3hot sleep before loading — BACO → D0
    if _is_gpu_sleeping(gpu_idx):
        set_gpu_power(gpu_idx, "auto")
        time.sleep(2)  # Settle after wake from D3hot

    # Poll until port is released (GPU driver teardown can take a few seconds)
    for _ in range(50):
        if get_pid_on_port(port) is None:
            break
        time.sleep(0.1)

    if flags is None:
        # Legacy fallback — use gemma4 16k mode as default
        cfg = MODEL_CONFIGS.get("gemma4", {})
        mode = cfg.get("modes", {}).get("16k", cfg.get("modes", {}).get("default", {}))
        flags = mode.get("flags", "-ngl 99 -c 8192 --parallel 1 --cache-ram 0 --mmap -t 4")

    # Use list args — never split() a string containing user-supplied paths
    cmd = [LLAMA_BIN, "--model", model_path, "--host", "0.0.0.0", "--port", str(port)] + flags.split()

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LLAMA_LIB
    env["RADV_DEBUG"] = "nodcc"
    env["GGML_VK_VISIBLE_DEVICES"] = str(gpu_idx)

    # Open log files, then explicitly close handles in parent after fork
    out_f = open(f"/tmp/gpu{gpu_idx}.log", "w")
    err_f = open(f"/tmp/gpu{gpu_idx}_err.log", "w")
    try:
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=out_f,
            stderr=err_f,
            preexec_fn=os.setpgrp,
        )
    finally:
        # Always close in parent — child has its own copies via fork
        out_f.close()
        err_f.close()

    return proc.pid

# Alias for clarity
start_gpu_raw = start_gpu

def kill_port(port):
    """Kill process on a port gracefully (SIGTERM → wait → SIGKILL)."""
    pid = get_pid_on_port(port)
    if not pid:
        return
    try:
        os.kill(pid, signal.SIGTERM)
        # Wait up to 5s for graceful shutdown
        for _ in range(50):
            time.sleep(0.1)
            if get_process_state(pid)[0] == "dead":
                return
        # Force kill if still alive
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception:
        pass

def kill_zombies():
    """Kill zombie/D-state llama-server processes. Uses pgrep to avoid self-match."""
    try:
        r = subprocess.run(["pgrep", "-x", "llama-server"],
                           capture_output=True, text=True, timeout=5)
        pids = [int(p) for p in r.stdout.strip().split() if p]
    except Exception:
        return 0

    killed = 0
    for pid in pids:
        state, age = get_process_state(pid)
        if state in ("zombie", "d-state"):
            try:
                os.kill(pid, signal.SIGKILL)
                killed += 1
            except Exception:
                pass
    return killed

def disable_power_mgmt():
    """Ensure GPUs with loaded models stay awake (D0).
    Does NOT force-wake sleeping GPUs — BACO sleep is intentional."""
    # Only wake GPUs that have a model loaded (need to stay D0 for inference)
    for gpu_idx in range(NUM_GPUS):
        pid = get_pid_on_port(BASE_PORT + gpu_idx)
        if pid:
            # GPU has a running model — ensure it stays awake
            set_gpu_power(gpu_idx, "auto")

# ══════════════════════════════════════════════════════════════════════════
# GPU POWER CONTROL
# ══════════════════════════════════════════════════════════════════════════

# Safe PCI bus path for runtime PM checks — does NOT wake the GPU
# Uses /sys/bus/pci/devices/ (kernel PCI core) NOT /sys/class/drm/ (amdgpu driver)
def _pci_bus_path(gpu_idx):
    """Get the SAFE PCI bus sysfs path (won't trigger amdgpu resume)."""
    pci = GPU_MAP.get(gpu_idx, {}).get("pci", "")
    return Path(f"/sys/bus/pci/devices/0000:{pci}")

def _is_gpu_sleeping(gpu_idx):
    """Check if GPU is in D3hot (BACO) using SAFE PCI bus path.
    This reads kernel PCI core state, NOT amdgpu driver state — no wake triggered."""
    try:
        p = _pci_bus_path(gpu_idx)
        runtime = (p / "power" / "runtime_status").read_text().strip()
        return runtime == "suspended"
    except:
        return False

def get_gpu_power(gpu_idx):
    """Get GPU power state. Uses SAFE PCI bus path for sleep check,
    only touches DRM/hwmon if GPU is confirmed awake."""
    result = {"perf_level": "on", "clock_mhz": 0, "power_watts": 0}

    # SAFE CHECK via PCI bus path (no amdgpu driver wake)
    if _is_gpu_sleeping(gpu_idx):
        result["perf_level"] = "sleep"
        result["power_watts"] = 0
        return result

    # GPU is awake — safe to read DRM/hwmon
    card = f"card{gpu_idx}"
    base = Path(f"/sys/class/drm/{card}/device")
    try:
        for line in (base / "pp_dpm_sclk").read_text().splitlines():
            if "*" in line:
                result["clock_mhz"] = int(line.split(":")[1].strip().replace("Mhz", "").strip())
                break
    except: pass
    try:
        for hwmon in base.glob("hwmon/hwmon*"):
            pwr = (hwmon / "power1_average").read_text().strip()
            result["power_watts"] = round(int(pwr) / 1_000_000, 1)
            break
    except: pass
    return result

def _get_pci_addr(gpu_idx):
    """Get PCI sysfs path for a GPU. Uses os.path.realpath, NOT Path.resolve()."""
    symlink = f"/sys/class/drm/card{gpu_idx}/device"
    try:
        full = os.path.realpath(symlink)
        return Path(full)
    except Exception:
        pci = GPU_MAP.get(gpu_idx, {}).get("pci", "")
        return Path(f"/sys/bus/pci/devices/0000:{pci}")

def set_gpu_power(gpu_idx, level):
    """Set GPU power state via PCIe runtime PM using SAFE PCI bus path.
    level='sleep' → D3hot (BACO), GPU nearly off
    level='auto'  → D0 (full power, ready for use)
    Uses /sys/bus/pci/devices/ path (kernel PCI core, no amdgpu driver interaction)."""
    p = _pci_bus_path(gpu_idx)
    ctrl_path = p / "power" / "control"
    delay_path = p / "power" / "autosuspend_delay_ms"
    if not ctrl_path.exists():
        return False
    try:
        if level in ("sleep", "low"):
            # Set longer autosuspend delay to prevent flapping from brief probes
            if delay_path.exists():
                delay_path.write_text("10000")  # 10 seconds
            ctrl_path.write_text("auto")
            # Force immediate suspend by setting delay to 0 after control=auto
            time.sleep(0.5)
            if delay_path.exists():
                delay_path.write_text("0")
        else:  # "auto", "on", "wake"
            ctrl_path.write_text("on")
        return True
    except Exception:
        return False

def set_all_gpu_power(level, exclude=None):
    """Set power level on all GPUs, optionally excluding some."""
    exclude = exclude or []
    results = {}
    for gpu in range(NUM_GPUS):
        if gpu in exclude:
            results[gpu] = "skipped"
            continue
        results[gpu] = "ok" if set_gpu_power(gpu, level) else "failed"
    return results

# ══════════════════════════════════════════════════════════════════════════
# DEFAULTS FILE — persistent config, survives reboot
# ══════════════════════════════════════════════════════════════════════════

def load_defaults_file():
    """Load defaults from SQLite. Returns dict matching old JSON format for compatibility."""
    try:
        conn = _db()
        gpus = {}
        for r in conn.execute("SELECT gpu_id, default_model, default_mode, default_temp, sleep_watts_cal, awake_watts_cal FROM gpu_state"):
            gid = str(r["gpu_id"])
            if r["default_model"]:
                gpus[gid] = {"model": r["default_model"], "mode": r["default_mode"], "temp": r["default_temp"]}
            else:
                gpus[gid] = None
        sys = conn.execute("SELECT auto_sleep_minutes, auto_load FROM system_state WHERE id=1").fetchone()
        sleep_watts = {}
        awake_watts = {}
        for r in conn.execute("SELECT gpu_id, sleep_watts_cal, awake_watts_cal FROM gpu_state"):
            if r["sleep_watts_cal"] is not None:
                sleep_watts[str(r["gpu_id"])] = r["sleep_watts_cal"]
            if r["awake_watts_cal"] is not None:
                awake_watts[str(r["gpu_id"])] = r["awake_watts_cal"]
        conn.close()
        return {
            "version": 1,
            "auto_load": bool(sys["auto_load"]) if sys else False,
            "auto_sleep_minutes": sys["auto_sleep_minutes"] if sys else 0,
            "gpus": gpus,
            "gpu_sleep_watts": sleep_watts,
            "gpu_awake_watts": awake_watts,
        }
    except:
        return {"version": 1, "auto_load": False, "auto_sleep_minutes": 0, "gpus": {}, "gpu_sleep_watts": {}, "gpu_awake_watts": {}}

def save_defaults_file(data):
    """Save defaults to SQLite (replaces old JSON file)."""
    try:
        conn = _db()
        # Save system-level settings
        conn.execute("UPDATE system_state SET auto_sleep_minutes=?, auto_load=? WHERE id=1",
            (data.get("auto_sleep_minutes", 0), 1 if data.get("auto_load") else 0))
        # Save per-GPU defaults
        for gid_str, gdef in data.get("gpus", {}).items():
            if gdef:
                conn.execute("UPDATE gpu_state SET default_model=?, default_mode=?, default_temp=? WHERE gpu_id=?",
                    (gdef.get("model"), gdef.get("mode"), gdef.get("temp"), int(gid_str)))
            else:
                conn.execute("UPDATE gpu_state SET default_model=NULL, default_mode=NULL, default_temp=NULL WHERE gpu_id=?",
                    (int(gid_str),))
        # Save calibration data
        for gid_str, sw in data.get("gpu_sleep_watts", {}).items():
            aw = data.get("gpu_awake_watts", {}).get(gid_str)
            conn.execute("UPDATE gpu_state SET sleep_watts_cal=?, awake_watts_cal=? WHERE gpu_id=?",
                (sw, aw, int(gid_str)))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"  save_defaults error: {e}")

def get_gpu_sleep_watts(gpu_idx):
    """Get calibrated sleep power for a GPU, or estimate if not calibrated."""
    try:
        conn = _db()
        r = conn.execute("SELECT sleep_watts_cal FROM gpu_state WHERE gpu_id=?", (gpu_idx,)).fetchone()
        conn.close()
        if r and r["sleep_watts_cal"] is not None:
            return round(r["sleep_watts_cal"], 1), True
    except: pass
    est = 6.0 if GPU_MAP.get(gpu_idx, {}).get("arch") == "RDNA2" else 5.0
    return est, False

def _read_gpu_power_avg(gpu_idx, samples=20, interval=0.5):
    """Read power1_average N times and return mean. Only for awake GPUs."""
    card = f"card{gpu_idx}"
    readings = []
    for _ in range(samples):
        try:
            for hwmon in Path(f"/sys/class/drm/{card}/device").glob("hwmon/hwmon*"):
                pwr = int((hwmon / "power1_average").read_text().strip())
                readings.append(pwr / 1_000_000)
                break
        except:
            pass
        time.sleep(interval)
    return sum(readings) / len(readings) if readings else 0

def set_default(gpu_idx, model, mode, temp=None):
    """Set the default model for a GPU. Saved to SQLite + synced to USB."""
    try:
        conn = _db()
        conn.execute("UPDATE gpu_state SET default_model=?, default_mode=?, default_temp=? WHERE gpu_id=?",
            (model, mode, temp, gpu_idx))
        conn.commit()
        conn.close()
        sync_db_to_persist()
    except Exception as e:
        print(f"  set_default error: {e}")

def clear_default(gpu_idx):
    """Clear the default model for a GPU."""
    try:
        conn = _db()
        conn.execute("UPDATE gpu_state SET default_model=NULL, default_mode=NULL, default_temp=NULL WHERE gpu_id=?",
            (gpu_idx,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"  clear_default error: {e}")

def load_all_defaults():
    """Load all configured GPU defaults. Returns list of (gpu, model, mode, temp)."""
    defaults = load_defaults_file()
    gpu_defaults = defaults.get("gpus", {})
    loaded = []
    for gpu_str, cfg in gpu_defaults.items():
        if cfg is None: continue
        gpu = int(gpu_str)
        model = cfg.get("model")
        mode = cfg.get("mode")
        temp = cfg.get("temp")
        if not model: continue
        mcfg = MODEL_CONFIGS.get(model)
        if not mcfg:
            print(co(R, f"  GPU {gpu}: unknown model '{model}'")); continue
        if mode and mode not in mcfg["modes"]:
            mode = mcfg["default_mode"]
        if not mode:
            mode = mcfg["default_mode"]
        mode_cfg = mcfg["modes"][mode]
        flags = mode_cfg.get("flags_12gb", mode_cfg["flags"]) if gpu == 2 else mode_cfg["flags"]
        if temp is not None:
            flags += f" --temp {temp}"
        model_path = str(MODEL_DIR / mcfg["file"])
        if not Path(model_path).exists():
            print(co(R, f"  GPU {gpu}: model file not found: {mcfg['file']}")); continue
        pid = start_gpu(gpu, model_path, flags)
        temp_str = f" temp={temp}" if temp is not None else ""
        print(f"  GPU {gpu}: loading {co(C, model)} {mode}{temp_str} (PID {pid})")
        loaded.append((gpu, model, mode, temp))
        time.sleep(2)
    return loaded

def save_running_as_defaults():
    """Save whatever is currently running on each GPU as the default."""
    defaults = load_defaults_file()
    if "gpus" not in defaults: defaults["gpus"] = {}
    instances = get_instances()
    saved = 0
    for gpu in range(NUM_GPUS):
        inst = instances[gpu]
        if inst["status"] in ("ready", "loading"):
            mi = _get_gpu_mode(gpu)
            if mi:
                defaults["gpus"][str(gpu)] = {
                    "model": mi.get("model"),
                    "mode": mi.get("mode"),
                    "temp": mi.get("temp"),
                }
                saved += 1
            else:
                defaults["gpus"][str(gpu)] = None
        else:
            defaults["gpus"][str(gpu)] = None
    save_defaults_file(defaults)
    return saved

def cmd_defaults(args):
    """Manage GPU default model assignments."""
    defaults = load_defaults_file()
    gpu_defs = defaults.get("gpus", {})
    auto_load = defaults.get("auto_load", False)

    if not args:
        # Show defaults table
        print()
        print(co(BOLD, f"  {'GPU':>3}  {'Default Model':<16} {'Mode':<14} {'Temp':>6}  {'Auto-load':>9}"))
        print("  " + "─" * 60)
        for i in range(NUM_GPUS):
            gd = gpu_defs.get(str(i))
            if gd:
                m = gd.get("model", "—")
                mode = gd.get("mode", "—")
                t = f"{gd['temp']}" if gd.get("temp") is not None else "—"
                print(f"  {i:>3}  {co(C, m):<24} {mode:<14} {t:>6}")
            else:
                print(f"  {i:>3}  {co(DIM, '(none)')}")
        print(f"\n  Auto-load on startup: {co(G, 'ON') if auto_load else co(DIM, 'OFF')}")
        print(co(DIM, f"\n  Usage:"))
        print(co(DIM, f"    defaults set 0 gemma4 nothink [temp=0.2]"))
        print(co(DIM, f"    defaults clear 3"))
        print(co(DIM, f"    defaults load          — Load all defaults now"))
        print(co(DIM, f"    defaults save-running   — Save current running as defaults"))
        print(co(DIM, f"    defaults auto on/off    — Auto-load on startup"))
        print()
        return

    parts = args.strip().split()
    cmd = parts[0].lower()

    if cmd == "set" and len(parts) >= 3:
        try:
            gpu = int(parts[1])
        except:
            print(co(R, f"  Invalid GPU: {parts[1]}")); return
        model = parts[2].lower()
        if model not in MODEL_CONFIGS:
            print(co(R, f"  Unknown model: {model}")); return
        # Parse optional mode and temp
        mode = None; temp = None
        for p in parts[3:]:
            if p.startswith("temp="):
                try: temp = float(p.split("=")[1])
                except: pass
            elif mode is None:
                mode = p.lower()
        if not mode:
            mode = MODEL_CONFIGS[model]["default_mode"]
        if mode not in MODEL_CONFIGS[model]["modes"]:
            print(co(R, f"  Unknown mode '{mode}' for {model}")); return
        set_default(gpu, model, mode, temp)
        temp_str = f" temp={temp}" if temp is not None else ""
        print(f"  GPU {gpu}: default set to {co(C, model)} {mode}{temp_str}")

    elif cmd == "clear":
        if len(parts) < 2:
            print(co(R, "  Usage: defaults clear <gpu|all>")); return
        if parts[1] == "all":
            for g in range(NUM_GPUS): clear_default(g)
            print("  All defaults cleared")
        else:
            try:
                gpu = int(parts[1])
                clear_default(gpu)
                print(f"  GPU {gpu}: default cleared")
            except:
                print(co(R, f"  Invalid GPU: {parts[1]}")); return

    elif cmd == "load":
        print(f"\n  Loading GPU defaults...")
        loaded = load_all_defaults()
        if loaded:
            print(f"\n  {len(loaded)} GPU(s) loading. Use 'gpus' to check progress.")
        else:
            print(co(DIM, "  No defaults configured. Use 'defaults set <gpu> <model> [mode]'"))

    elif cmd == "save-running":
        saved = save_running_as_defaults()
        print(f"  Saved {saved} running GPU(s) as defaults")

    elif cmd == "auto":
        if len(parts) < 2:
            print(co(DIM, f"  Auto-load: {'ON' if auto_load else 'OFF'}")); return
        val = parts[1].lower()
        defaults["auto_load"] = val in ("on", "true", "1", "yes")
        save_defaults_file(defaults)
        print(f"  Auto-load: {co(G, 'ON') if defaults['auto_load'] else co(DIM, 'OFF')}")

    else:
        print(co(R, f"  Unknown: {cmd}. Use set/clear/load/save-running/auto"))
    print()

def rig_graceful_save():
    """Save running models as defaults before shutdown/suspend."""
    saved = save_running_as_defaults()
    if saved:
        print(f"  Saved {saved} running GPU(s) as defaults")
    return saved

def cmd_rig(args):
    """Rig power control: shutdown, reboot, suspend."""
    if not args:
        print()
        print(co(BOLD, "  Rig Power Control"))
        print()
        print(co(DIM, "  rig shutdown    — Save state + power off (S5). WoL to wake."))
        print(co(DIM, "  rig reboot      — Save state + reboot. Dashboard auto-starts."))
        print(co(DIM, "  rig suspend     — Save state + suspend to RAM (S3). WoL to wake."))
        print(co(DIM, "                    WARNING: S3 may not resume cleanly with 7 GPUs on risers."))
        print()
        return

    action = args.strip().lower()

    if action == "shutdown":
        print(co(Y, "\n  Shutting down rig (S5)..."))
        print("  Saving running models as defaults...")
        rig_graceful_save()
        save_state()
        print("  Stopping all GPU processes...")
        for gpu in range(NUM_GPUS):
            kill_port(BASE_PORT + gpu)
        print(co(R, "  Powering off. Send WoL to wake: python rig_client.py --wake"))
        time.sleep(1)
        os.system("sudo systemctl poweroff")

    elif action == "reboot":
        print(co(Y, "\n  Rebooting rig..."))
        print("  Saving running models as defaults...")
        rig_graceful_save()
        save_state()
        print("  Stopping all GPU processes...")
        for gpu in range(NUM_GPUS):
            kill_port(BASE_PORT + gpu)
        print(co(C, "  Rebooting. Dashboard will auto-start."))
        time.sleep(1)
        os.system("sudo reboot")

    elif action == "suspend":
        print(co(Y, "\n  Suspending rig to RAM (S3)..."))
        print(co(R, "  WARNING: Resume may fail with 7 GPUs on USB risers."))
        print(co(R, "  If resume fails, you'll need to power-cycle the rig."))
        resp = input("  Continue? (y/n): ").strip().lower()
        if resp != "y":
            print("  Cancelled."); return
        print("  Saving running models as defaults...")
        rig_graceful_save()
        save_state()
        print("  Stopping all GPU processes...")
        for gpu in range(NUM_GPUS):
            kill_port(BASE_PORT + gpu)
        print(co(C, "  Suspending. Send WoL to wake: python rig_client.py --wake"))
        time.sleep(1)
        os.system("sudo systemctl suspend")

    else:
        print(co(R, f"  Unknown action: {action}"))
        print(co(DIM, "  Use: rig shutdown, rig reboot, rig suspend"))
    print()

def calibrate_sleep_power():
    """Measure each GPU's D3hot sleep power using subtraction method.
    Sleeps one GPU at a time, measures power delta from remaining awake GPUs.
    Takes ~3 minutes for 7 GPUs. Returns dict of {gpu_id: watts}."""

    print(co(BOLD, "\n  ═══ POWER CALIBRATION ═══"))
    print(co(DIM, "  Measuring each GPU's D3hot sleep power via subtraction"))
    print(co(DIM, "  This takes ~3 minutes. All GPUs must be idle (no models loaded).\n"))

    # Check no models running
    instances = get_instances()
    active = [g for g, i in instances.items() if i["status"] in ("ready", "loading")]
    if active:
        print(co(R, f"  GPUs {active} have models loaded. Unload all first."))
        return None

    # Ensure all awake
    print("  Waking all GPUs...")
    for gpu in range(NUM_GPUS):
        set_gpu_power(gpu, "auto")
    time.sleep(5)

    # Step 1: Baseline — all 7 awake, 10 samples averaged
    print("  Reading baseline (all 7 awake, 10 samples)...")
    baseline_per = {}
    baseline_total = 0
    for gpu in range(NUM_GPUS):
        avg = _read_gpu_power_avg(gpu, samples=10, interval=0.3)
        baseline_per[gpu] = avg
        baseline_total += avg
        print(f"    GPU {gpu}: {avg:.1f}W")
    print(f"    Total: {baseline_total:.1f}W\n")

    # Step 2: Sleep each GPU one at a time, measure remaining
    results = {}
    for test_gpu in range(NUM_GPUS):
        print(f"  Calibrating GPU {test_gpu}...")
        gpu_awake_power = baseline_per[test_gpu]

        # Sleep this GPU
        set_gpu_power(test_gpu, "sleep")
        time.sleep(5)  # Wait for D3hot to settle

        # Verify it's sleeping
        if not _is_gpu_sleeping(test_gpu):
            print(co(Y, f"    GPU {test_gpu}: failed to sleep, skipping"))
            set_gpu_power(test_gpu, "auto")
            time.sleep(3)
            continue

        # Read remaining awake GPUs
        remaining_total = 0
        awake_list = [g for g in range(NUM_GPUS) if g != test_gpu]
        for gpu in awake_list:
            avg = _read_gpu_power_avg(gpu, samples=10, interval=0.3)
            remaining_total += avg

        # Calculate: disappeared power = what that GPU was contributing
        disappeared = baseline_total - remaining_total
        sleep_power = max(0, gpu_awake_power - disappeared)

        results[test_gpu] = round(sleep_power, 1)
        print(f"    GPU {test_gpu}: awake={gpu_awake_power:.1f}W, disappeared={disappeared:.1f}W → sleep≈{sleep_power:.1f}W")

        # Wake it back
        set_gpu_power(test_gpu, "auto")
        time.sleep(2)

    # Save to defaults file
    defaults = load_defaults_file()
    defaults["gpu_sleep_watts"] = {str(k): v for k, v in results.items()}
    defaults["gpu_awake_watts"] = {str(k): round(v, 1) for k, v in baseline_per.items()}
    defaults["calibrated_at"] = datetime.now().isoformat()
    save_defaults_file(defaults)

    # Summary
    print(co(BOLD, "\n  ═══ CALIBRATION RESULTS ═══"))
    total_awake = sum(baseline_per.values())
    total_sleep = sum(results.values())
    for gpu in range(NUM_GPUS):
        cal = results.get(gpu, "?")
        aw = baseline_per.get(gpu, "?")
        if isinstance(cal, (int, float)):
            saving = f"saves ~{aw - cal:.0f}W"
        else:
            saving = "failed"
        print(f"    GPU {gpu}: awake={aw:.1f}W → sleep≈{cal}W ({saving})")
    print(f"\n    All awake: ~{total_awake:.0f}W")
    print(f"    All sleep: ~{total_sleep:.0f}W")
    print(f"    Savings:   ~{total_awake - total_sleep:.0f}W")
    print(co(DIM, f"\n    Saved to {DEFAULTS_FILE}"))
    print()
    return results

def cmd_power(args):
    """GPU power control: sleep (D3hot BACO, ~0W) or wake (D0, full power)."""
    if not args:
        # Show current power state for all GPUs
        print()
        print(co(BOLD, f"  {'GPU':>3}  {'Card':<22} {'Power':>8}  {'State':>8}  {'Clock':>8}  {'Model':<8}"))
        print("  " + "─" * 65)
        instances = get_instances()
        total_watts = 0
        sleeping = 0
        sleep_total = 0
        for i in range(NUM_GPUS):
            info = GPU_MAP[i]
            pwr = get_gpu_power(i)
            st = instances[i]["status"]
            perf = pwr["perf_level"]
            clk = f"{pwr['clock_mhz']}MHz" if pwr["clock_mhz"] else "—"
            if perf == "sleep":
                sw, cal = get_gpu_sleep_watts(i)
                tag = "cal" if cal else "est"
                watts = co(G, f" ~{sw:.0f}W {tag}")
                perf_str = co(G, " sleep ")
                sleep_total += sw
                sleeping += 1
            else:
                watts = f"{pwr['power_watts']}W" if pwr["power_watts"] else "—"
                total_watts += pwr["power_watts"]
                perf_str = co(C, "   on  ")
            model = instances[i].get("model", "—") or "—"
            if len(model) > 8: model = model[:7] + "…"
            print(f"  {i:>3}  {info['name']:<22} {watts:>14}  {perf_str}  {clk:>8}  {model}")
        print(f"\n  Active: ~{total_watts:.0f}W | Sleeping: {sleeping} GPUs (~{sleep_total:.0f}W)")
        defaults = load_defaults_file()
        if defaults.get("calibrated_at"):
            print(co(DIM, f"  Calibrated: {defaults['calibrated_at']}"))
        else:
            print(co(Y, f"  Not calibrated — run 'power calibrate' for measured values"))
        auto_min = defaults.get("auto_sleep_minutes", 0)
        auto_str = f"{auto_min} min" if auto_min > 0 else "off"
        print(co(DIM, f"  Auto-sleep: {auto_str}"))
        print(co(DIM, f"\n  Usage: power <sleep|wake|calibrate|autosleep> [gpu|all]"))
        print(co(DIM, f"    power sleep all     — D3hot BACO on all idle GPUs"))
        print(co(DIM, f"    power wake all      — Restore all GPUs to full power"))
        print(co(DIM, f"    power calibrate     — Measure each GPU's D3hot power (~3 min)"))
        print(co(DIM, f"    power autosleep 5   — Auto-sleep idle GPUs after 5 minutes"))
        print(co(DIM, f"    power autosleep off — Disable auto-sleep"))
        print()
        return

    parts = args.strip().lower().split()
    level = parts[0]

    if level == "calibrate":
        calibrate_sleep_power()
        return

    if level == "autosleep":
        defaults = load_defaults_file()
        if len(parts) < 2:
            cur = defaults.get("auto_sleep_minutes", 0)
            print(f"\n  Auto-sleep: {co(C, str(cur) + ' min') if cur else co(DIM, 'off')}")
            print(co(DIM, "  Usage: power autosleep <minutes|off>"))
            print()
            return
        val = parts[1]
        if val == "off" or val == "0":
            defaults["auto_sleep_minutes"] = 0
            save_defaults_file(defaults)
            print(f"  Auto-sleep: {co(DIM, 'disabled')}")
        else:
            try:
                mins = int(val)
                if mins < 1 or mins > 1440:
                    print(co(R, "  Must be 1-1440 minutes")); return
                defaults["auto_sleep_minutes"] = mins
                save_defaults_file(defaults)
                print(f"  Auto-sleep: {co(G, f'{mins} minutes')} — idle GPUs will sleep automatically")
            except:
                print(co(R, f"  Invalid: {val}. Use a number or 'off'.")); return
        print()
        return

    if level in ("wake", "on"):
        level = "auto"
    elif level in ("sleep", "low", "off"):
        level = "sleep"
    else:
        print(co(R, f"  Invalid: {level}. Use 'sleep', 'wake', or 'calibrate'."))
        return

    # Parse GPU targets
    gpu_parts = parts[1:] if len(parts) > 1 else ["all"]
    if "all" in gpu_parts:
        gpus = list(range(NUM_GPUS))
    else:
        try:
            gpus = [int(x.strip(",")) for x in gpu_parts]
        except:
            print(co(R, f"  Invalid GPU: {gpu_parts}")); return

    instances = get_instances()
    for gpu in gpus:
        if gpu not in GPU_MAP:
            print(co(R, f"  Invalid GPU: {gpu}")); continue
        if level == "sleep" and instances.get(gpu, {}).get("status") in ("ready", "loading"):
            print(co(Y, f"  GPU {gpu}: has model loaded — unload first"))
            continue

        ok = set_gpu_power(gpu, level)
        if level == "sleep":
            label = co(G, "sleeping (D3hot BACO)")
        else:
            label = co(C, "awake (D0)")
        if ok:
            print(f"  GPU {gpu}: {label}")
        else:
            print(f"  GPU {gpu}: {co(R, 'failed')}")
    print()

# ══════════════════════════════════════════════════════════════════════════
# STATE PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════

def save_state():
    """Save current GPU assignments atomically with restricted permissions."""
    instances = get_instances()
    state = {"version": 1, "saved_at": datetime.now().isoformat(), "gpus": {}}
    for gpu in range(NUM_GPUS):
        inst = instances[gpu]
        if inst["model"] and inst["status"] in ("ready", "loading"):
            # Read actual model path from /proc/cmdline rather than guessing MODEL_DIR/name
            actual_path = None
            if inst["pid"]:
                try:
                    cmdline = Path(f"/proc/{inst['pid']}/cmdline").read_text().replace("\0", " ")
                    if "--model" in cmdline:
                        actual_path = cmdline.split("--model")[1].strip().split(" ")[0]
                except Exception:
                    pass
            if not actual_path:
                actual_path = str(MODEL_DIR / inst["model"])
            state["gpus"][str(gpu)] = {
                "model": inst["model"],
                "path": actual_path,
                "port": inst["port"],
            }
        else:
            state["gpus"][str(gpu)] = None

    tmp = STATE_FILE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.chmod(tmp, 0o600)
        os.rename(tmp, STATE_FILE)
    except Exception as e:
        print(co(R, f"  Warning: failed to save state: {e}"))

def load_state():
    """Load saved state."""
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════
# PERFORMANCE TRACKING
# ══════════════════════════════════════════════════════════════════════════

def _rotate_perf_log():
    """Keep only the last 10,000 lines of the performance log."""
    try:
        with open(PERF_LOG) as f:
            lines = f.readlines()
        with open(PERF_LOG, "w") as f:
            f.writelines(lines[-10000:])
    except Exception:
        pass

def log_perf(gpu, tokens, speed, elapsed, model, ptype="test"):
    """Log a request to performance file. Rotates at PERF_LOG_MAX_MB."""
    entry = {"ts": time.time(), "gpu": gpu, "tokens": tokens, "speed": round(speed, 1),
             "elapsed": round(elapsed, 2), "model": model, "type": ptype}
    try:
        with open(PERF_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
        if os.path.getsize(PERF_LOG) > PERF_LOG_MAX_MB * 1024 * 1024:
            _rotate_perf_log()
    except Exception:
        pass

def get_perf():
    """Get performance summary."""
    stats = {i: {"speeds": [], "tokens": 0, "reqs": 0} for i in range(NUM_GPUS)}
    try:
        with open(PERF_LOG) as f:
            for line in f:
                try:
                    e = json.loads(line.strip())
                    g = e.get("gpu")
                    if g is not None and g in stats:
                        stats[g]["speeds"].append(float(e["speed"]))
                        stats[g]["tokens"] += int(e["tokens"])
                        stats[g]["reqs"] += 1
                except Exception:
                    continue
    except Exception:
        pass
    return stats

# ══════════════════════════════════════════════════════════════════════════
# NGINX
# ══════════════════════════════════════════════════════════════════════════

def rebuild_nginx():
    """Rebuild nginx config from ready instances only (not loading)."""
    instances = get_instances()
    # Only include "ready" backends — "loading" ones reject requests
    servers = [f"    server 127.0.0.1:{inst['port']};" for gpu, inst in instances.items()
               if inst["status"] == "ready"]
    if not servers:
        print(co(R, "  No ready instances for nginx"))
        return False
    config = f"""upstream ai_rig_gpus {{
    least_conn;
{chr(10).join(servers)}
}}
server {{
    listen {LB_PORT};
    location / {{
        proxy_pass http://ai_rig_gpus;
        proxy_set_header Host $host;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        client_max_body_size 10m;
    }}
}}"""
    Path("/tmp/nginx_lb.conf").write_text(config)
    subprocess.run("sudo cp /tmp/nginx_lb.conf /etc/nginx/sites-available/ai-rig-lb.conf",
                   shell=True, timeout=10)
    subprocess.run("sudo ln -sf /etc/nginx/sites-available/ai-rig-lb.conf /etc/nginx/sites-enabled/",
                   shell=True, timeout=10)
    r = subprocess.run("sudo nginx -t && sudo systemctl reload nginx",
                       shell=True, capture_output=True, text=True, timeout=30)
    ok = r.returncode == 0
    print(co(G if ok else R, f"  nginx: {len(servers)} backends {'active' if ok else 'FAILED'}"))
    return ok

# ══════════════════════════════════════════════════════════════════════════
# DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════

def download_model(url_or_id):
    """Download a GGUF model from HuggingFace."""
    if not url_or_id:
        print(co(Y, "  Usage: models download <url or huggingface/repo-id>")); return

    if url_or_id.startswith("http"):
        url = url_or_id
        # Strip query strings and path components from filename
        filename = Path(urllib.parse.urlparse(url).path).name
        if not filename:
            print(co(R, "  Cannot determine filename from URL")); return
    else:
        # Model ID — list files
        api = f"https://huggingface.co/api/models/{url_or_id}/tree/main"
        try:
            resp = urllib.request.urlopen(api, timeout=15)
            files = json.loads(resp.read())
            ggufs = [f for f in files if f["path"].endswith(".gguf") and "mmproj" not in f["path"]]
            if not ggufs:
                print(co(R, "  No GGUF files found in that repo"))
                return
            print(f"\n  Files in {url_or_id}:")
            for i, f in enumerate(ggufs):
                print(f"    {i+1}. {f['path']} ({f.get('size',0)//(1024**2)}MB)")
            pick = input(f"\n  Download which? (1-{len(ggufs)}): ").strip()
            try:
                idx = int(pick) - 1
                if not (0 <= idx < len(ggufs)):
                    raise ValueError("out of range")
                chosen = ggufs[idx]["path"]
            except (ValueError, IndexError):
                print(co(R, "  Invalid choice")); return
            url = f"https://huggingface.co/{url_or_id}/resolve/main/{chosen}"
            # Strip any subdirectory components — store flat in MODEL_DIR
            filename = Path(chosen).name
        except Exception as e:
            print(co(R, f"  Error: {e}")); return

    dest = MODEL_DIR / filename
    if dest.exists():
        print(co(Y, f"  Already exists: {dest}")); return

    print(f"  Downloading {filename}...")
    part_path = str(dest) + ".part"
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "ai-rig-shell/1.0")
        resp = urllib.request.urlopen(req)
        total = int(resp.headers.get("Content-Length", 0))
        dl = 0
        with open(part_path, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                dl += len(chunk)
                if total:
                    pct = dl * 100 // total
                    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                    print(f"\r  [{bar}] {pct}% ({dl//(1024**2)}/{total//(1024**2)}MB)", end="", flush=True)
        # Detect truncated transfer before promoting to final name
        if total > 0 and dl < total:
            raise IOError(f"Incomplete download: received {dl}/{total} bytes")
        os.rename(part_path, str(dest))
        print(f"\n  Saved: {dest}")
    except BaseException as e:
        # Catch KeyboardInterrupt (BaseException) and clean up partial file
        print(co(R, f"\n  Download failed: {e}"))
        try:
            os.unlink(part_path)
        except Exception:
            pass
        if isinstance(e, KeyboardInterrupt):
            raise

# ══════════════════════════════════════════════════════════════════════════
# DIAGNOSE
# ══════════════════════════════════════════════════════════════════════════

def diagnose_gpu(gpu_idx):
    """Deep diagnosis of a GPU. Uses a healthy GPU's LLM to analyze."""
    print(f"\n  Collecting GPU {gpu_idx} data...")
    info = GPU_MAP.get(gpu_idx, {})
    port = BASE_PORT + gpu_idx
    pid = get_pid_on_port(port)
    vram_used, vram_total = get_vram(gpu_idx)
    temp = get_gpu_temp(gpu_idx)
    pstate, _page = get_process_state(pid) if pid else ("no process", None)

    # Crash log
    err_log = f"/tmp/gpu{gpu_idx}_err.log"
    crash_lines = ""
    try:
        with open(err_log) as f:
            lines = f.readlines()
            crash_lines = "".join(lines[-30:])
    except Exception:
        crash_lines = "(no log file)"

    # dmesg for this GPU — validate PCI address before embedding in shell command
    pci = info.get("pci", "")
    dmesg = ""
    if _PCI_RE.match(pci):
        try:
            r = subprocess.run(
                ["sh", "-c", f"dmesg | grep -i '{pci}\\|amdgpu' | tail -10"],
                capture_output=True, text=True, timeout=5
            )
            dmesg = r.stdout.strip()
        except Exception:
            dmesg = "(cant read dmesg)"
    else:
        dmesg = "(PCI address format unexpected — skipped)"

    # Power state
    power = "unknown"
    try:
        power = Path(f"/sys/class/drm/card{gpu_idx}/device/power/control").read_text().strip()
    except Exception:
        pass

    temp_str = f"{temp}C" if temp is not None else "unknown"

    report = f"""GPU {gpu_idx} Diagnostic Report:
Card: {info.get('name', '?')}
PCI Address: {pci}
PCIe BAR: {info.get('bar', '?')}
VRAM: {vram_used}MB / {vram_total}MB
Temperature: {temp_str}
Power control: {power}
Process PID: {pid} (state: {pstate})
Port: {port}

Last 30 lines of stderr:
{crash_lines}

dmesg (GPU related):
{dmesg}
"""
    print(report)

    # Find a healthy GPU to analyze
    instances = get_instances()
    analyzer = None
    for g, inst in instances.items():
        if g != gpu_idx and inst["status"] == "ready":
            analyzer = inst["port"]
            print(f"  Analyzing with GPU {g} (port {analyzer})...")
            break

    if not analyzer:
        print(co(Y, "  No healthy GPU available for AI analysis."))
        return

    # Truncate prompt to avoid context window overflow
    max_crash_chars = 2000
    truncated_report = report[:max_crash_chars] + ("...[truncated]" if len(report) > max_crash_chars else "")

    prompt = f"""You are a GPU diagnostics expert for AMD Radeon GPUs running llama.cpp with Vulkan backend on Ubuntu 22.04. Analyze this diagnostic data and tell me:
1. What is the exact problem?
2. What caused it?
3. How to fix it?

Be specific and actionable. Here is the data:

{truncated_report}"""

    try:
        data = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": 1000}).encode()
        req = urllib.request.Request(f"http://localhost:{analyzer}/v1/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=120)
        r = json.loads(resp.read())
        analysis = r["choices"][0]["message"].get("content", "")
        print(co(BOLD, "\n  AI DIAGNOSIS:"))
        print(f"  {analysis}\n")
    except Exception as e:
        print(co(R, f"  AI analysis failed: {e}"))

# ══════════════════════════════════════════════════════════════════════════
# GPU PROFILING
# ══════════════════════════════════════════════════════════════════════════

# Prompts covering a range of task types and output lengths
_PROFILE_PROMPTS = [
    {
        "name": "short_simple",
        "desc": "short simple",
        "payload": {"messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}], "max_tokens": 50},
    },
    {
        "name": "medium_code",
        "desc": "medium code",
        "payload": {"messages": [{"role": "user", "content": "Write a Python quicksort function with a docstring and example usage."}], "max_tokens": 300},
    },
    {
        "name": "json_extract",
        "desc": "JSON extraction",
        "payload": {"messages": [
            {"role": "system", "content": "Extract as JSON: {\"name\", \"age\", \"company\", \"salary\"}"},
            {"role": "user", "content": "Jane Doe is a 42-year-old VP of Engineering at Stripe, earning $380K."},
        ], "max_tokens": 150},
    },
    {
        "name": "long_analysis",
        "desc": "long analysis",
        "payload": {"messages": [{"role": "user", "content": "Explain transformer attention mechanisms and how GGUF quantization (Q4, Q5, Q8) affects model quality and speed. Be thorough."}], "max_tokens": 700},
    },
    {
        "name": "reasoning",
        "desc": "multi-step reasoning",
        "payload": {"messages": [{"role": "user", "content": "A store sells items at $5, $10, and $20. You have $100 and want exactly 10 items. List all valid combinations and show your work."}], "max_tokens": 600},
    },
]

def _collect_hw_info(gpu_idx):
    """Collect hardware info for a GPU from sysfs and system tools."""
    info = GPU_MAP.get(gpu_idx, {})
    hw = {
        "gpu_idx": gpu_idx,
        "name": info.get("name", "unknown"),
        "vram_mb": info.get("vram_mb", 0),
        "pci_bar": info.get("bar", "unknown"),
        "pci_addr": info.get("pci", "unknown"),
    }

    # Driver / kernel module version
    try:
        r = subprocess.run(["modinfo", "amdgpu"], capture_output=True, text=True, timeout=5)
        for line in r.stdout.split("\n"):
            if line.startswith("version:"):
                hw["driver_version"] = line.split(":", 1)[1].strip()
                break
        else:
            hw["driver_version"] = "unknown"
    except Exception:
        hw["driver_version"] = "unknown"

    # PCIe link speed and width
    try:
        sp = Path(f"/sys/class/drm/card{gpu_idx}/device/current_link_speed")
        wd = Path(f"/sys/class/drm/card{gpu_idx}/device/current_link_width")
        hw["pcie_speed"] = sp.read_text().strip() if sp.exists() else "unknown"
        hw["pcie_width"] = wd.read_text().strip() if wd.exists() else "unknown"
    except Exception:
        hw["pcie_speed"] = "unknown"
        hw["pcie_width"] = "unknown"

    # Power cap and current draw (hwmon)
    try:
        for hwmon in Path(f"/sys/class/drm/card{gpu_idx}/device/hwmon").iterdir():
            pc = hwmon / "power1_cap"
            pa = hwmon / "power1_average"
            hw["power_cap_w"] = int(pc.read_text().strip()) // 1_000_000 if pc.exists() else None
            hw["power_avg_w"] = int(pa.read_text().strip()) // 1_000_000 if pa.exists() else None
            break
    except Exception:
        hw["power_cap_w"] = None
        hw["power_avg_w"] = None

    # Current VRAM and temp
    vram_used, vram_total = get_vram(gpu_idx)
    hw["vram_used_mb"] = vram_used
    hw["vram_total_mb"] = vram_total
    hw["temp_c"] = get_gpu_temp(gpu_idx)

    # Vulkan device name via vulkaninfo (best-effort)
    try:
        r = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True, text=True, timeout=10,
            env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":0")}
        )
        for line in r.stdout.split("\n"):
            if "deviceName" in line and gpu_idx < line.count("deviceName") + 1:
                hw["vulkan_device"] = line.split("=", 1)[-1].strip()
                break
    except Exception:
        hw["vulkan_device"] = "unknown"

    return hw

def _run_profile_benchmarks(gpu_idx, port):
    """Run the profile benchmark suite. Returns list of result dicts."""
    results = []
    temps_during = []

    print(f"\n  Running {len(_PROFILE_PROMPTS)} benchmark prompts on GPU {gpu_idx}...")
    for i, bench in enumerate(_PROFILE_PROMPTS):
        print(f"    [{i+1}/{len(_PROFILE_PROMPTS)}] {bench['desc']}...", end="", flush=True)
        t_before = get_gpu_temp(gpu_idx)
        start = time.time()
        try:
            data = json.dumps(bench["payload"]).encode()
            req = urllib.request.Request(
                f"http://localhost:{port}/v1/chat/completions", data=data,
                headers={"Content-Type": "application/json"}
            )
            resp = urllib.request.urlopen(req, timeout=180)
            r = json.loads(resp.read())
            elapsed = time.time() - start
            tokens = r.get("usage", {}).get("completion_tokens", 0)
            speed = r.get("timings", {}).get("predicted_per_second", 0.0)
            t_after = get_gpu_temp(gpu_idx)
            if t_after is not None:
                temps_during.append(t_after)
            results.append({
                "name": bench["name"],
                "desc": bench["desc"],
                "tokens": tokens,
                "tok_per_s": round(speed, 1),
                "elapsed_s": round(elapsed, 2),
                "temp_before_c": t_before,
                "temp_after_c": t_after,
                "ok": True,
            })
            print(f" {co(G, 'OK')} {tokens}tok {speed:.0f}t/s {elapsed:.1f}s"
                  + (f" {t_after}°C" if t_after else ""))
        except Exception as e:
            elapsed = time.time() - start
            results.append({"name": bench["name"], "desc": bench["desc"],
                            "ok": False, "error": str(e)[:80], "elapsed_s": round(elapsed, 2)})
            print(f" {co(R, 'FAIL')} {str(e)[:50]}")

    return results, temps_during

def profile_gpu(gpu_idx):
    """Collect full hardware profile + AI-generated analysis for one GPU."""
    print(f"\n{co(BOLD, f'  ═══ Profiling GPU {gpu_idx} ═══')}")

    inst = get_instances()[gpu_idx]
    if inst["status"] != "ready":
        print(co(Y, f"  GPU {gpu_idx} is not ready (status: {inst['status']})"))
        print(co(Y,  "  Load a model first with: load {gpu} <model#>"))
        return None

    port = inst["port"]
    model_name = inst["model"] or "unknown"

    # 1. Collect hardware info
    print(f"  Collecting hardware info...")
    hw = _collect_hw_info(gpu_idx)

    # 2. Run benchmarks
    bench_results, temps = _run_profile_benchmarks(gpu_idx, port)

    # 3. Build summary stats
    ok_results = [r for r in bench_results if r["ok"]]
    speeds = [r["tok_per_s"] for r in ok_results]
    avg_speed = round(sum(speeds) / len(speeds), 1) if speeds else 0
    peak_speed = round(max(speeds), 1) if speeds else 0
    max_temp = max(temps) if temps else None

    profile = {
        "gpu_idx": gpu_idx,
        "profiled_at": datetime.now().isoformat(),
        "model_used_for_profiling": model_name,
        "hardware": hw,
        "benchmarks": bench_results,
        "summary": {
            "avg_tok_per_s": avg_speed,
            "peak_tok_per_s": peak_speed,
            "benchmarks_passed": len(ok_results),
            "benchmarks_total": len(bench_results),
            "max_temp_c": max_temp,
            "vram_total_mb": hw["vram_total_mb"],
            "pci_bar": hw["pci_bar"],
        },
        "ai_analysis": None,
    }

    # 4. Feed to AI for recommendations
    instances = get_instances()
    analyzer_port = None
    for g, ainst in instances.items():
        if g != gpu_idx and ainst["status"] == "ready":
            analyzer_port = ainst["port"]
            print(f"\n  AI analysis via GPU {g} (port {analyzer_port})...")
            break

    if not analyzer_port:
        # Self-analyze if this is the only ready GPU
        if inst["status"] == "ready":
            analyzer_port = port
            print(f"\n  AI self-analysis (GPU {gpu_idx})...")

    if analyzer_port:
        bench_summary = "\n".join(
            f"  - {r['desc']}: {r['tok_per_s']}t/s, {r['tokens']}tok, {r['elapsed_s']}s"
            if r["ok"] else f"  - {r['desc']}: FAILED ({r.get('error','')})"
            for r in bench_results
        )
        ai_prompt = f"""You are an expert in GPU hardware and LLM inference optimization. Analyze this GPU profile and provide a comprehensive hardware profile report.

GPU: {hw['name']}
VRAM: {hw['vram_total_mb']}MB total
PCIe BAR size: {hw['pci_bar']} (affects large model loading — 256MB BAR limits non-resizable memory)
PCIe link: {hw.get('pcie_speed','?')} x{hw.get('pcie_width','?')}
Driver: {hw.get('driver_version','?')}
Power cap: {hw.get('power_cap_w','?')}W

Benchmark results (model: {model_name}):
{bench_summary}

Peak temp under load: {max_temp}°C
Average throughput: {avg_speed} tok/s
Peak throughput: {peak_speed} tok/s

Provide a structured analysis with these sections:
1. STRENGTHS: What this GPU does well for LLM inference
2. LIMITATIONS: Key hardware constraints (VRAM, BAR, PCIe bandwidth, etc.)
3. RECOMMENDED WORKLOADS: What model sizes/quantizations work best (e.g., "Q4_K_M up to 7B fits well")
4. MAX MODEL SIZE: Largest model that will fit (with reasoning)
5. ESTIMATED PERFORMANCE: tok/s estimates for common configurations (3B Q4, 7B Q4, 13B Q4 if applicable)
6. QUIRKS: Any hardware-specific issues to watch for (thermal throttling, BAR limitations, driver issues)
7. OPTIMIZATION TIPS: Specific flags or settings to improve performance on this GPU

Be specific and actionable. Base estimates on the benchmark data provided."""

        try:
            data = json.dumps({
                "messages": [{"role": "user", "content": ai_prompt}],
                "max_tokens": 1500
            }).encode()
            req = urllib.request.Request(
                f"http://localhost:{analyzer_port}/v1/chat/completions", data=data,
                headers={"Content-Type": "application/json"}
            )
            resp = urllib.request.urlopen(req, timeout=180)
            r = json.loads(resp.read())
            profile["ai_analysis"] = r["choices"][0]["message"].get("content", "")
        except Exception as e:
            print(co(R, f"  AI analysis failed: {e}"))
            profile["ai_analysis"] = f"AI analysis failed: {e}"
    else:
        print(co(Y, "  No GPU available for AI analysis — saving hardware data only."))

    # 5. Save to profiles file
    try:
        existing = {}
        if os.path.exists(PROFILES_FILE):
            try:
                with open(PROFILES_FILE) as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        existing[str(gpu_idx)] = profile
        tmp = PROFILES_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(existing, f, indent=2)
        os.chmod(tmp, 0o600)
        os.rename(tmp, PROFILES_FILE)
        print(co(G, f"  Profile saved → {PROFILES_FILE}"))
    except Exception as e:
        print(co(R, f"  Failed to save profile: {e}"))

    # 6. Display formatted profile
    _print_profile(profile)
    return profile

def _print_profile(profile):
    """Print a GPU profile in a readable format."""
    hw = profile["hardware"]
    s = profile["summary"]
    print()
    print(co(BOLD, f"  ┌─── GPU {profile['gpu_idx']} PROFILE {'─'*50}"))
    print(co(BOLD,  f"  │ {hw['name']}"))
    print(f"  │ VRAM: {hw['vram_total_mb']}MB  BAR: {hw['pci_bar']}  PCIe: {hw.get('pcie_speed','?')} x{hw.get('pcie_width','?')}")
    print(f"  │ Driver: {hw.get('driver_version','?')}  Power cap: {hw.get('power_cap_w','?')}W")
    print(f"  │ Profiled with: {profile['model_used_for_profiling']}")
    print(f"  ├─── Benchmark Results {'─'*46}")
    for r in profile["benchmarks"]:
        if r["ok"]:
            temp_str = f"  {r['temp_after_c']}°C" if r.get("temp_after_c") else ""
            print(f"  │ {r['desc']:<22} {r['tok_per_s']:>6.0f} t/s  {r['tokens']:>4}tok  {r['elapsed_s']:>5.1f}s{temp_str}")
        else:
            print(f"  │ {r['desc']:<22} {co(R, 'FAILED')}")
    print(f"  ├─── Summary {'─'*56}")
    print(f"  │ Avg: {s['avg_tok_per_s']} t/s  Peak: {s['peak_tok_per_s']} t/s  "
          f"Passed: {s['benchmarks_passed']}/{s['benchmarks_total']}"
          + (f"  Max temp: {s['max_temp_c']}°C" if s['max_temp_c'] else ""))
    if profile.get("ai_analysis"):
        print(f"  ├─── AI Analysis {'─'*52}")
        for line in profile["ai_analysis"].split("\n"):
            print(f"  │ {line}")
    print(f"  └{'─'*68}")
    print()

def cmd_profile(args):
    """Profile one or all GPUs — hardware info + benchmarks + AI analysis."""
    if not args:
        print(co(Y, "  Usage: profile <gpu|all>")); return
    gpus = list(range(NUM_GPUS)) if args.strip() == "all" else []
    if not gpus:
        try:
            gpus = [int(x) for x in args.split(",")]
        except Exception:
            print(co(R, f"  Invalid: {args}")); return
    invalid = [g for g in gpus if g not in GPU_MAP]
    if invalid:
        print(co(R, f"  Invalid GPU index (valid: 0–{NUM_GPUS-1}): {invalid}")); return
    for gpu in gpus:
        profile_gpu(gpu)

# ══════════════════════════════════════════════════════════════════════════
# COMMANDS
# ══════════════════════════════════════════════════════════════════════════

def _get_gpu_mode(gpu_idx):
    """Read the model+mode that was loaded on this GPU."""
    try:
        with open(f"/tmp/gpu{gpu_idx}_mode.json") as f:
            return json.load(f)
    except:
        return None

def cmd_gpus(auto_refresh=True):
    instances = get_instances()
    ram = get_ram_info()
    print()
    print(co(BOLD, f"  {'GPU':>3} {'Card':<22} {'VRAM':>12} {'Temp':>4} {'Port':>5} {'Status':<9} {'Model':<16} {'Mode':<8} {'PID':>6} {'BAR':>7}"))
    print("  " + "─" * 100)
    for i in range(NUM_GPUS):
        info = GPU_MAP[i]
        inst = instances[i]
        vu, vt = inst["vram_used"], inst["vram_total"]
        vram_str = f"{vu}M/{vt}M"
        temp_str = f"{inst['temp']}C" if inst['temp'] is not None else " —"
        st = inst["status"]
        if st == "ready":     st_str = co(G, "ready  ")
        elif st == "loading" and info["bar"] == "256MB":
                              st_str = co(Y, "load▒▒▒")
        elif st == "loading": st_str = co(Y, "load▒  ")
        elif st == "starting" and info["bar"] == "256MB":
                              st_str = co(Y, "start▒▒")
        elif st == "starting":st_str = co(Y, "start  ")
        elif st == "error":   st_str = co(R, "ERROR  ")
        elif st == "crashed": st_str = co(R, "CRASHED")
        elif st == "zombie":  st_str = co(R, "ZOMBIE ")
        elif st == "sleeping":st_str = co(G, "sleep💤")
        else:                 st_str = co(DIM, "off    ")

        # Get model name, mode, and temperature
        mode_info = _get_gpu_mode(i)
        if mode_info and inst["status"] != "stopped":
            model_short = mode_info.get("model", "—")
            t = mode_info.get("temp")
            mode_str = mode_info.get("mode", "—")
            if t is not None:
                mode_str += f" t={t}"
        else:
            model_short = "—"
            mode_str = "—"

        pid = str(inst["pid"] or "—")
        bar = info["bar"]
        print(f"  {i:>3} {info['name']:<22} {vram_str:>12} {temp_str:>4} {inst['port']:>5} {st_str} {model_short:<16} {mode_str:<8} {pid:>6} {co(DIM, bar):>15}")
    print(f"\n  RAM: {ram['used']}M / {ram['total']}M ({ram['avail']}M avail)")
    if ram["avail"] < 2048:
        print(co(R, "  ⚠ LOW RAM — may cause OOM kills"))

    # Auto-refresh only if enabled (disabled when called from load/unload)
    if not auto_refresh:
        print()
        return

    # If any GPU is in a transitional state, keep refreshing
    transitional = {"loading", "starting"}
    active_states = {inst["status"] for inst in instances.values()}
    if active_states & transitional:
        count = sum(1 for inst in instances.values() if inst["status"] in transitional)
        print(co(Y, f"\n  {count} GPU(s) still loading — auto-refreshing every 5s (Ctrl+C to stop)"))
        try:
            while True:
                time.sleep(5)
                instances = get_instances()
                ram = get_ram_info()

                # Move cursor up to overwrite the table
                lines_up = NUM_GPUS + 5  # header + rows + ram + message
                print(f"\033[{lines_up}A", end="")

                print(co(BOLD, f"  {'GPU':>3} {'Card':<22} {'VRAM':>12} {'Temp':>4} {'Port':>5} {'Status':<9} {'Model':<16} {'Mode':<8} {'PID':>6} {'BAR':>7}"))
                print("  " + "─" * 100)
                for i in range(NUM_GPUS):
                    info = GPU_MAP[i]
                    inst = instances[i]
                    vu, vt = inst["vram_used"], inst["vram_total"]
                    vram_str = f"{vu}M/{vt}M"
                    temp_str = f"{inst['temp']}C" if inst['temp'] is not None else " —"
                    st = inst["status"]
                    if st == "ready":     st_str = co(G, "ready  ")
                    elif st == "loading" and info["bar"] == "256MB":
                                          st_str = co(Y, "load▒▒▒")
                    elif st == "loading": st_str = co(Y, "load▒  ")
                    elif st == "starting" and info["bar"] == "256MB":
                                          st_str = co(Y, "start▒▒")
                    elif st == "starting":st_str = co(Y, "start  ")
                    elif st == "error":   st_str = co(R, "ERROR  ")
                    elif st == "crashed": st_str = co(R, "CRASHED")
                    elif st == "zombie":  st_str = co(R, "ZOMBIE ")
                    elif st == "sleeping":st_str = co(G, "sleep💤")
                    else:                 st_str = co(DIM, "off    ")

                    mode_info = _get_gpu_mode(i)
                    if mode_info and inst["status"] != "stopped":
                        model_short = mode_info.get("model", "—")
                        mode_str = mode_info.get("mode", "—")
                    else:
                        model_short = "—"
                        mode_str = "—"

                    pid = str(inst["pid"] or "—")
                    bar = info["bar"]
                    print(f"  {i:>3} {info['name']:<22} {vram_str:>12} {temp_str:>4} {inst['port']:>5} {st_str} {model_short:<16} {mode_str:<8} {pid:>6} {co(DIM, bar):>15}")

                print(f"\n  RAM: {ram['used']}M / {ram['total']}M ({ram['avail']}M avail)")
                if ram["avail"] < 2048:
                    print(co(R, "  ⚠ LOW RAM — may cause OOM kills"))

                # Check if all settled
                still_loading = sum(1 for inst in instances.values() if inst["status"] in transitional)
                if still_loading == 0:
                    ready = sum(1 for inst in instances.values() if inst["status"] == "ready")
                    failed = sum(1 for inst in instances.values() if inst["status"] in ("crashed", "error", "zombie"))
                    print(co(G if failed == 0 else Y, f"\n  All settled: {ready} ready, {failed} failed"))
                    break
                else:
                    print(co(Y, f"\n  {still_loading} GPU(s) still loading — refreshing...            "))

        except KeyboardInterrupt:
            print(co(DIM, "\n  Stopped auto-refresh"))
    print()

def cmd_models():
    # Show configured models with modes
    print()
    print(co(BOLD, "  Configured Models (proven stable):"))
    print()
    print(co(BOLD, f"  {'Name':<16} {'Mode':<10} {'Speed':>10} {'Ctx':>6} {'Size':>6} {'Fits 8GB':>8}  Description"))
    print("  " + "─" * 95)
    for name, cfg in MODEL_CONFIGS.items():
        for mode, mcfg in cfg["modes"].items():
            default = " *" if mode == cfg["default_mode"] else ""
            fits = co(G, "yes") if cfg["fits_8gb"] else co(R, "12GB")
            print(f"  {name:<16} {mode + default:<10} {mcfg['speed']:>10} {mcfg['context']:>5} {cfg['size_gb']:>5.1f}G {fits:>16}  {mcfg['desc']}")
    print()
    print(co(DIM, "  * = default mode. Usage: load <gpu|all> <model> [mode]"))
    print(co(DIM, "  Example: load all gemma4 16k"))
    print()

    # Also show raw GGUF files on disk
    raw = get_models()
    if raw:
        print(co(BOLD, "  GGUF files on disk:"))
        for i, m in enumerate(raw):
            configured = any(cfg["file"] == m["name"] for cfg in MODEL_CONFIGS.values())
            tag = co(G, " (configured)") if configured else co(DIM, " (raw)")
            print(f"    {i+1}. {m['name']} ({m['size_gb']}G){tag}")
        print()

def cmd_load(args):
    if not args:
        print(co(Y, "  Usage: load <gpu|all> <model> [mode]"))
        print(co(Y, "  Example: load all gemma4 16k"))
        print(co(Y, "  Example: load 0,1,2 qwen35-4b"))
        print(co(Y, "  Type 'models' to see available models and modes"))
        return

    # Parse flexibly: accept "0,1 gemma4", "0 1 gemma4", "0, 1 gemma4", "all gemma4 16k"
    # Strategy: collect GPU numbers/all from the left, model name is the first non-number token
    parts = args.replace(",", " ").split()
    if len(parts) < 2:
        print(co(Y, "  Usage: load <gpu|all> <model> [mode]")); return

    # Collect GPU tokens (numbers or "all") from the left
    gpu_tokens = []
    model_idx = None
    for i, p in enumerate(parts):
        if p.lower() == "all":
            gpu_tokens.append("all")
        elif p.isdigit() and int(p) < NUM_GPUS:
            gpu_tokens.append(p)
        else:
            model_idx = i
            break

    if not gpu_tokens or model_idx is None:
        print(co(Y, "  Usage: load <gpu|all> <model> [mode] [temp=X]"))
        print(co(Y, "  Example: load 0 1 2 gemma4 16k temp=0.2")); return

    target = ",".join(gpu_tokens)
    model_name = parts[model_idx].lower()

    # Parse remaining args: mode and temp=X can be in any order
    remaining = [p.lower() for p in parts[model_idx + 1:]]
    mode = None
    temp = None
    for r in remaining:
        if r.startswith("temp="):
            try:
                temp = float(r.split("=", 1)[1])
                if not 0.0 <= temp <= 2.0:
                    print(co(R, f"  Temperature must be 0.0-2.0, got {temp}")); return
            except ValueError:
                print(co(R, f"  Invalid temperature: {r}")); return
        elif mode is None:
            mode = r

    # Find model config
    cfg = None
    matched_name = None
    for name, c in MODEL_CONFIGS.items():
        if model_name == name or model_name in name:
            cfg = c
            matched_name = name
            break

    if not cfg:
        print(co(R, f"  Model not found: {model_name}"))
        print(co(DIM, "  Available: " + ", ".join(MODEL_CONFIGS.keys())))
        return

    # Resolve mode
    if mode is None:
        mode = cfg["default_mode"]
    if mode not in cfg["modes"]:
        print(co(R, f"  Unknown mode '{mode}' for {matched_name}"))
        print(co(DIM, f"  Available modes: {', '.join(cfg['modes'].keys())}"))
        return

    mode_cfg = cfg["modes"][mode]
    model_path = str(MODEL_DIR / cfg["file"])

    if not Path(model_path).exists():
        print(co(R, f"  Model file not found: {model_path}")); return

    # Resolve GPUs
    gpus = list(range(NUM_GPUS)) if target == "all" else []
    if not gpus:
        try:
            gpus = [int(x) for x in target.split(",")]
        except Exception:
            print(co(R, f"  Invalid GPU: {target}")); return

    # Validate GPU indices
    invalid = [g for g in gpus if g not in GPU_MAP]
    if invalid:
        print(co(R, f"  Invalid GPU (valid: 0-{NUM_GPUS-1}): {invalid}")); return

    temp_str = f" temp={temp}" if temp is not None else ""
    print(f"\n  Loading {co(C, matched_name)} mode={co(C, mode)}{co(C, temp_str)} onto GPU {gpus}")
    print(co(DIM, f"  {mode_cfg['desc']} | {mode_cfg['speed']} | ctx={mode_cfg['context']}"))

    for gpu in gpus:
        # Check VRAM fit
        if gpu != 2 and not cfg["fits_8gb"]:
            print(co(R, f"  ✗ GPU {gpu}: {matched_name} needs 12GB GPU (use GPU 2)"))
            continue

        # Get flags — use 12GB override if available for GPU 2
        flags = mode_cfg.get("flags_12gb", mode_cfg["flags"]) if gpu == 2 else mode_cfg["flags"]

        # Append temperature if specified
        if temp is not None:
            flags += f" --temp {temp}"

        pid = start_gpu(gpu, model_path, flags)
        # Save mode info for gpus display
        mode_file = f"/tmp/gpu{gpu}_mode.json"
        try:
            with open(mode_file, "w") as mf:
                info = {"model": matched_name, "mode": mode, "file": cfg["file"]}
                if temp is not None:
                    info["temp"] = temp
                json.dump(info, mf)
        except: pass
        print(f"  GPU {gpu} → port {BASE_PORT + gpu}: {co(Y, 'starting')} (PID {pid})")
        time.sleep(2)

    save_state()

    # Auto-save as GPU defaults (last-run = new default)
    for gpu in gpus:
        set_default(gpu, matched_name, mode, temp)

    # Show slow BAR warning
    slow_gpus = [g for g in gpus if GPU_MAP.get(g, {}).get("bar") == "256MB"]
    if slow_gpus:
        print(co(Y, f"\n  ⏳ GPUs {slow_gpus} have 256MB BAR — will take 3-4 min to load"))
    fast_gpus = [g for g in gpus if g not in slow_gpus]
    if fast_gpus:
        print(co(DIM, f"  GPUs {fast_gpus} have full BAR — will load in ~30s"))

    print()
    # Show one-shot status snapshot, no auto-refresh (return prompt immediately)
    cmd_gpus(auto_refresh=False)
    print(co(DIM, "  Run 'gpus' to watch loading progress"))

def cmd_unload(args):
    if not args:
        print(co(Y, "  Usage: unload <gpu|all>")); return
    gpus = list(range(NUM_GPUS)) if args.strip() == "all" else []
    if not gpus:
        try:
            gpus = [int(x) for x in args.split(",")]
        except Exception:
            print(co(R, f"  Invalid: {args}")); return
    for gpu in gpus:
        kill_port(BASE_PORT + gpu)
        print(f"  GPU {gpu}: {co(DIM, 'stopped')}")
    save_state()
    print()

def cmd_verify(verbose=False):
    """Wait for all loading GPUs, then run 3-level tests on each."""
    print(f"\n  Waiting for all GPUs to load (up to 5 min)...")
    start = time.time()
    max_wait = 300

    while time.time() - start < max_wait:
        instances = get_instances()
        loading = [g for g, inst in instances.items() if inst["status"] in ("loading", "starting")]
        ready = [g for g, inst in instances.items() if inst["status"] == "ready"]
        failed = [g for g, inst in instances.items() if inst["status"] in ("crashed", "zombie", "error")]

        if not loading and not ready:
            print(co(DIM, "  No GPUs loading or running")); return

        elapsed = int(time.time() - start)
        print(f"\r  [{elapsed:>3}s] Ready: {len(ready)}  Loading: {len(loading)}  Failed: {len(failed)}  ", end="", flush=True)

        if not loading:
            break
        time.sleep(5)

    print("\n")
    instances = get_instances()

    # Load tests from /opt/ai-rig-verify.json
    VERIFY_FILE = "/opt/ai-rig-verify.json"
    long_job = "Senior Machine Learning Engineer at Pfizer Inc., New York, NY 10017. Full-time. $180K-$220K. PhD in CS required, 5+ years Python, PyTorch, TensorFlow, distributed training, LLMs. " * 20

    # All tests use objective facts with verifiable correct answers
    # No opinions, no brand names to parrot — purely factual
    long_text = "The speed of light in a vacuum is approximately 299,792,458 meters per second. " * 30 + "Water boils at 100 degrees Celsius at standard atmospheric pressure. The chemical formula for water is H2O. The Earth orbits the Sun at an average distance of about 149.6 million kilometers. " * 10

    DEFAULT_TESTS = {
        "tests": [
            {
                "name": "Basic",
                "desc": "Arithmetic — verifiable correct answer",
                "type": "math",
                "prompt": {"messages": [{"role": "user", "content": "What is 37 * 43? Show only the final number."}], "max_tokens": 500},
                "expect_all": ["1591"],
                "timeout": 30
            },
            {
                "name": "JSON",
                "desc": "Structured extraction — all fields must be present and correct",
                "type": "extraction",
                "prompt": {"messages": [
                    {"role": "system", "content": "Extract these facts as JSON only, no explanation: {\"element\", \"symbol\", \"atomic_number\", \"state_at_room_temp\", \"group\"}"},
                    {"role": "user", "content": "Gold is a chemical element with the symbol Au and atomic number 79. It is a solid at room temperature and belongs to group 11 of the periodic table."}
                ], "max_tokens": 2000},
                "expect_all": ["Au", "79", "solid", "11", "Gold"],
                "timeout": 60
            },
            {
                "name": "Code",
                "desc": "Function generation — must have correct structure and logic",
                "type": "code",
                "prompt": {"messages": [{"role": "user", "content": "Write a Python function called celsius_to_fahrenheit that converts Celsius to Fahrenheit using the formula F = C * 9/5 + 32. Include a docstring. Show a test: celsius_to_fahrenheit(100) should return 212."}], "max_tokens": 2000},
                "expect_all": ["def ", "celsius_to_fahrenheit", "return", "212", "32"],
                "timeout": 60
            },
            {
                "name": "Reasoning",
                "desc": "Multi-step math — verifiable intermediate and final answers",
                "type": "reasoning",
                "prompt": {"messages": [{"role": "user", "content": "A rectangle has length 15 cm and width 8 cm. Calculate: 1) The area. 2) The perimeter. 3) The length of the diagonal (round to 2 decimal places). Show each step."}], "max_tokens": 2000},
                "expect_all": ["120", "46", "17"],
                "timeout": 60
            },
            {
                "name": "Stress",
                "desc": "Large prompt (~3000 tokens) — tests context limit without crash",
                "type": "stress",
                "prompt": {"messages": [
                    {"role": "system", "content": "Answer only: What is the boiling point of water in Celsius? And what is the chemical formula for water?"},
                    {"role": "user", "content": long_text + " Based on the text above, answer the system's questions."}
                ], "max_tokens": 500},
                "expect_all": ["100", "H2O"],
                "timeout": 120
            }
        ]
    }

    try:
        with open(VERIFY_FILE) as f:
            test_data = json.load(f)
        TESTS = test_data.get("tests", DEFAULT_TESTS["tests"])
    except FileNotFoundError:
        TESTS = DEFAULT_TESTS["tests"]
        with open(VERIFY_FILE, "w") as f:
            json.dump(DEFAULT_TESTS, f, indent=2)
        print(co(DIM, f"  Created test file: {VERIFY_FILE}"))
        print(co(DIM, f"  Edit it to add/change/remove tests.\n"))

    # Results table: gpu -> {test_name: {status, speed, tokens}}
    results = {}

    for test in TESTS:
        tname = test.get("name", "?")
        tdesc = test.get("desc", "")
        tprompt = test.get("prompt", {})
        texpect = test.get("expect_contains", "")
        ttimeout = test.get("timeout", 60)

        # Show what we're sending and what we expect
        expect_all = test.get("expect_all", [])
        user_msg = ""
        sys_msg = ""
        for m in tprompt.get("messages", []):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
            elif m.get("role") == "system":
                sys_msg = m.get("content", "")

        print(co(BOLD, f"  ── {tname}: {tdesc} ──"))
        if verbose:
            if sys_msg:
                sys_preview = sys_msg[:120] + ("..." if len(sys_msg) > 120 else "")
                print(co(DIM, f"    System: {sys_preview}"))
            user_preview = user_msg[:200] + ("..." if len(user_msg) > 200 else "")
            print(co(DIM, f"    Prompt: {user_preview}"))
            if expect_all:
                print(co(DIM, f"    Expect: ALL of {expect_all}"))
            elif texpect:
                print(co(DIM, f"    Expect: contains \"{texpect}\""))
            print()

        for gpu in range(NUM_GPUS):
            inst = instances[gpu]
            if inst["status"] != "ready":
                if gpu not in results:
                    results[gpu] = {"status": inst["status"]}
                continue

            if gpu not in results:
                results[gpu] = {}

            port = inst["port"]
            t0 = time.time()
            try:
                data = json.dumps(tprompt).encode()
                req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                             data=data, headers={"Content-Type": "application/json"})
                resp = urllib.request.urlopen(req, timeout=ttimeout)
                r = json.loads(resp.read())
                elapsed = time.time() - t0
                speed = r["timings"]["predicted_per_second"]
                tokens = r["usage"]["completion_tokens"]
                prompt_tokens = r["usage"]["prompt_tokens"]
                content = r["choices"][0]["message"].get("content", "")
                # Check both content AND thinking (thinking models put answers in reasoning_content)
                thinking = r["choices"][0]["message"].get("reasoning_content", "")
                full_output = (content + " " + thinking).lower()

                # expect_all = ALL keywords must be present (strict)
                # expect_contains = any single keyword (legacy, lenient)
                expect_all = test.get("expect_all", [])
                if expect_all:
                    missing = [kw for kw in expect_all if kw.lower() not in full_output]
                    passed = len(missing) == 0
                elif texpect:
                    passed = texpect.lower() in full_output
                else:
                    passed = len(content.strip()) > 5

                status_str = co(G, "PASS") if passed else co(Y, "WEAK")
                results[gpu][tname] = {"ok": passed, "speed": speed, "tokens": tokens, "prompt_tok": prompt_tokens, "time": elapsed}

                # Always show actual response (cleaned up)
                clean = content.replace("\n", " ").replace("```json", "").replace("```python", "").replace("```", "").strip()
                if len(clean) > 250:
                    preview = clean[:247] + "..."
                else:
                    preview = clean

                print(f"    GPU {gpu}: {status_str} {speed:>5.0f}t/s {prompt_tokens:>4}→{tokens:>4}tok {elapsed:>5.1f}s")
                print(f"             {co(C, 'Response:')} {preview}")
                if not passed and expect_all:
                    print(f"             {co(R, 'Missing:')} {', '.join(missing)}")
            except Exception as e:
                elapsed = time.time() - t0
                results[gpu][tname] = {"ok": False, "speed": 0, "tokens": 0, "prompt_tok": 0, "time": elapsed, "error": str(e)[:40]}
                print(f"    GPU {gpu}: {co(R, 'CRASH')} {elapsed:.1f}s — {str(e)[:50]}")

                # Auto-reload: get the model+mode that was loaded, restart, wait
                mi = _get_gpu_mode(gpu)
                if mi:
                    model_name = mi.get("model", "")
                    mode_name = mi.get("mode", "")
                    model_file = mi.get("file", "")
                    model_path = str(MODEL_DIR / model_file) if model_file else None

                    if model_path and os.path.exists(model_path):
                        # Get flags for this model+mode
                        cfg = MODEL_CONFIGS.get(model_name, {})
                        mode_cfg = cfg.get("modes", {}).get(mode_name, {})
                        flags = mode_cfg.get("flags_12gb", mode_cfg.get("flags")) if gpu == 2 else mode_cfg.get("flags")

                        if flags:
                            print(f"    GPU {gpu}: {co(Y, 'reloading')} {model_name} {mode_name}...")
                            start_gpu(gpu, model_path, flags)

                            # Wait for it to come back (up to 4 min for 256MB BAR)
                            max_reload = 240
                            for w in range(max_reload):
                                time.sleep(1)
                                h = check_health(inst["port"])
                                if h == "ready":
                                    print(f"    GPU {gpu}: {co(G, 'reloaded')} ({w+1}s)")
                                    instances[gpu]["status"] = "ready"
                                    break
                                if w > 0 and w % 30 == 0:
                                    print(f"    GPU {gpu}: {co(Y, 'still loading')} ({w}s)...")
                            else:
                                print(f"    GPU {gpu}: {co(R, 'reload timeout')} — may need more time")
                        else:
                            print(f"    GPU {gpu}: {co(Y, 'no flags found for reload')}")
                    else:
                        print(f"    GPU {gpu}: {co(Y, 'model file not found for reload')}")
                else:
                    print(f"    GPU {gpu}: {co(Y, 'no mode info — manual reload needed')}")
        print()

    # Summary table
    print(co(BOLD, "  ═══ VERIFICATION SUMMARY ═══"))
    print()

    mode_info = {}
    for gpu in range(NUM_GPUS):
        mode_info[gpu] = _get_gpu_mode(gpu)

    # Build header with fixed column widths
    test_names = [t.get("name", "?") for t in TESTS]
    test_cols = "".join(f" {n:>9}" for n in test_names)
    print(co(BOLD, f"  GPU  Card                 VRAM   BAR  Grade  Model          Mode   {test_cols}  Avg t/s  Verdict"))
    print("  " + "─" * (85 + 10 * len(TESTS)))

    production_ready = 0
    for gpu in range(NUM_GPUS):
        info = GPU_MAP[gpu]
        mi = mode_info.get(gpu) or {}
        model = mi.get("model", "—")
        mode = mi.get("mode", "—")
        t = mi.get("temp")
        if t is not None:
            mode = f"{mode} t={t}"

        vram_gb = f"{info['vram_mb']//1024}G"
        bar = f"{info['bar']:>5}"
        grade = info.get("grade", "?")
        if "A+" in grade:   grade_str = co(G, " A+")
        elif "A" in grade:  grade_str = co(G, "  A")
        elif grade == "B":  grade_str = co(Y, "  B")
        else:               grade_str = co(DIM, "  ?")

        # Fixed prefix
        prefix = f"  {gpu:>3}  {info['name']:<20} {vram_gb:>4} {bar:>5} {grade_str}  {model:<14} {mode:<6}"

        if "status" in results.get(gpu, {}):
            st = results[gpu]["status"]
            print(f"{prefix} {co(DIM, st)}")
            continue

        speeds = []
        all_pass = True
        test_results_str = ""
        for test in TESTS:
            tname = test.get("name", "?")
            tr = results.get(gpu, {}).get(tname, {})
            if tr.get("ok"):
                test_results_str += f"  {co(G, '   PASS')}"
                speeds.append(tr.get("speed", 0))
            elif "error" in tr:
                test_results_str += f"  {co(R, '  CRASH')}"
                all_pass = False
            elif tr:
                test_results_str += f"  {co(Y, '   WEAK')}"
            else:
                test_results_str += f"  {co(DIM, '     —')}"

        avg = sum(speeds) / len(speeds) if speeds else 0
        if all_pass and len(speeds) == len(TESTS):
            verdict = co(G, "  READY")
            production_ready += 1
        elif speeds:
            verdict = co(Y, "PARTIAL")
        else:
            verdict = co(R, " FAILED")

        print(f"{prefix}{test_results_str}  {avg:>6.0f}  {verdict}")

    # Re-check instances after all tests (some may have crashed)
    instances_after = get_instances()
    total_before = sum(1 for inst in instances.values() if inst["status"] in ("ready", "loading", "starting"))
    crashed_during = []
    for gpu in range(NUM_GPUS):
        was_ready = instances.get(gpu, {}).get("status") == "ready"
        now_dead = instances_after.get(gpu, {}).get("status") in ("stopped", "crashed", "zombie")
        if was_ready and now_dead:
            crashed_during.append(gpu)

    print()
    print(f"  Production ready: {production_ready}/{total_before}")
    if production_ready == total_before and total_before > 0:
        print(co(G, "  ✓ All GPUs verified — ready for production!"))
    elif production_ready > 0:
        print(co(Y, f"  ⚠ {production_ready} GPUs ready, {total_before - production_ready} need attention"))

    if crashed_during:
        mi = _get_gpu_mode(crashed_during[0]) or {}
        current_mode = mi.get("mode", "?")
        current_model = mi.get("model", "?")
        print()
        print(co(R, f"  ✗ GPUs {crashed_during} crashed during testing (mode: {current_mode})"))
        print(co(R, f"    These GPUs are now OFF and need to be reloaded."))
        if current_mode == "fast":
            print(co(Y, f"\n    The 'fast' mode crashes on large prompts (Stress test)."))
            print(co(Y, f"    Fix: reload with 16k mode:"))
            print(co(C, f"      load {','.join(str(g) for g in crashed_during)} {current_model} 16k"))
        else:
            print(co(Y, f"\n    Reload them:"))
            print(co(C, f"      load {','.join(str(g) for g in crashed_during)} {current_model} {current_mode}"))

    test_names = " → ".join(t.get("name", "?") for t in TESTS)
    print(co(DIM, f"\n  Tests: {test_names}"))
    print(co(DIM, f"  Test data: /opt/ai-rig-verify.json (edit to add/change/remove tests)"))
    print()

def cmd_test(args):
    if not args:
        print(co(Y, "  Usage: test <gpu|all>")); return
    gpus = list(range(NUM_GPUS)) if args.strip() == "all" else []
    if not gpus:
        try:
            gpus = [int(x) for x in args.split(",")]
        except Exception:
            print(co(R, f"  Invalid: {args}")); return

    prompt = {"messages": [{"role":"system","content":"Extract as JSON: {\"title\",\"company\",\"location\"}"},
                           {"role":"user","content":"Senior ML Engineer at Pfizer, New York NY. $200K. Python, TensorFlow, 5+ years."}],
              "max_tokens": 500}
    print()
    for gpu in gpus:
        port = BASE_PORT + gpu
        start = time.time()
        try:
            data = json.dumps(prompt).encode()
            req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=data,
                                         headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req, timeout=60)
            r = json.loads(resp.read())
            elapsed = time.time() - start
            tokens = r.get("usage", {}).get("completion_tokens", 0)
            speed = r.get("timings", {}).get("predicted_per_second", 0.0)
            content = r["choices"][0]["message"].get("content", "")[:80]
            print(f"  GPU {gpu}: {co(G, 'OK')} {tokens}tok {speed:.0f}t/s {elapsed:.1f}s → {content}")
            if tokens > 0 and speed > 0:
                log_perf(gpu, tokens, speed, elapsed, get_instances()[gpu].get("model", "?"))
        except urllib.error.HTTPError as e:
            print(f"  GPU {gpu}: {co(Y, f'HTTP {e.code} (loading?)')}")
        except Exception as e:
            print(f"  GPU {gpu}: {co(R, f'FAIL — {str(e)[:40]}')}")
    print()

def cmd_bench(args):
    if not args:
        print(co(Y, "  Usage: bench <gpu>")); return
    try:
        gpu = int(args.strip())
    except Exception:
        print(co(R, "Invalid GPU")); return
    if gpu not in GPU_MAP:
        print(co(R, f"  Invalid GPU index (valid: 0–{NUM_GPUS-1})")); return

    port = BASE_PORT + gpu
    # Snapshot model name once — don't call get_instances() inside the benchmark loop
    model_name = get_instances()[gpu].get("model", "?")

    prompts = [
        {"messages": [{"role":"user","content":"What is 2+2? One word."}], "max_tokens": 100},
        {"messages": [{"role":"user","content":"Write a Python string reversal function."}], "max_tokens": 500},
        {"messages": [{"role":"system","content":"JSON: {\"name\",\"age\"}"},{"role":"user","content":"John Smith, 35"}], "max_tokens": 200},
        {"messages": [{"role":"user","content":"TCP vs UDP in 3 sentences."}], "max_tokens": 500},
        {"messages": [{"role":"user","content":"Store 20% off, item $80. What do you pay? Show math."}], "max_tokens": 1000},
    ]
    print(f"\n  Benchmarking GPU {gpu}...")
    speeds = []
    for i, p in enumerate(prompts):
        start = time.time()
        try:
            data = json.dumps(p).encode()
            req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=data,
                                         headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req, timeout=120)
            r = json.loads(resp.read())
            elapsed = time.time() - start
            speed = r.get("timings", {}).get("predicted_per_second", 0.0)
            tokens = r.get("usage", {}).get("completion_tokens", 0)
            speeds.append(speed)
            print(f"    [{i+1}/5] {co(G, 'OK')} {tokens:>4}tok {speed:>5.0f}t/s {elapsed:>5.1f}s")
            if tokens > 0 and speed > 0:
                log_perf(gpu, tokens, speed, elapsed, model_name, "bench")
        except Exception as e:
            print(f"    [{i+1}/5] {co(R, f'FAIL: {str(e)[:40]}')}")
    if speeds:
        avg = sum(speeds) / len(speeds)
        print(f"\n  Avg: {avg:.0f} tok/s ({len(speeds)}/5 passed)")
    print()

def cmd_perf():
    stats = get_perf()
    print()
    print(co(BOLD, f"  {'GPU':>3} {'Avg Speed':>10} {'Requests':>10} {'Total Tokens':>14}"))
    print("  " + "─" * 42)
    total_tok = 0
    total_req = 0
    for g in range(NUM_GPUS):
        s = stats[g]
        avg = sum(s["speeds"]) / len(s["speeds"]) if s["speeds"] else 0
        print(f"  {g:>3} {avg:>8.0f}t/s {s['reqs']:>10} {s['tokens']:>14,}")
        total_tok += s["tokens"]
        total_req += s["reqs"]
    print("  " + "─" * 42)
    print(f"  {'TOT':>3} {'':>10} {total_req:>10} {total_tok:>14,}")
    print()

def cmd_logs(args):
    if not args:
        print(co(Y, "  Usage: logs <gpu>")); return
    try:
        gpu = int(args.strip())
    except Exception:
        print(co(R, "Invalid GPU")); return
    err = Path(f"/tmp/gpu{gpu}_err.log")
    if err.exists():
        print(co(BOLD, f"\n  === GPU {gpu} stderr (last 30 lines) ==="))
        try:
            lines = err.read_text(errors="replace").splitlines()
            for line in lines[-30:]:
                print(f"  {line}")
        except Exception as e:
            print(co(R, f"  Error reading log: {e}"))
    else:
        print(co(DIM, f"  No log for GPU {gpu}"))
    print()

def cmd_chat(args):
    if not args:
        print(co(Y, "  Usage: chat <gpu>")); return
    try:
        gpu = int(args.strip())
    except Exception:
        print(co(R, "Invalid GPU")); return
    port = BASE_PORT + gpu
    msgs = []
    print(f"  Chat with GPU {gpu}. Type 'exit' to quit.\n")
    while True:
        try:
            user = input("  you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in ("exit", "quit", "/exit"):
            break
        if not user:
            continue
        msgs.append({"role": "user", "content": user})
        try:
            data = json.dumps({"messages": msgs, "max_tokens": 4000}).encode()
            req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=data,
                                         headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req, timeout=120)
            r = json.loads(resp.read())
            content = r["choices"][0]["message"].get("content", "")
            speed = r.get("timings", {}).get("predicted_per_second", 0.0)
            print(f"  {co(C, 'ai>')} {content}")
            if speed:
                print(co(DIM, f"  ({speed:.0f} tok/s)"))
            msgs.append({"role": "assistant", "content": content})
        except Exception as e:
            print(co(R, f"  Error: {e}"))
            # Only remove the user message if we know the request never reached the model
            # (network/connection errors). Malformed responses leave history intact.
            if msgs and msgs[-1]["role"] == "user":
                msgs.pop()
    print()

def cmd_resume():
    state = load_state()
    if not state:
        print(co(DIM, "  No saved state")); return
    print(f"  Resuming from {state.get('saved_at', '?')}...")
    for gpu_str, info in state.get("gpus", {}).items():
        if info:
            try:
                gpu = int(gpu_str)
            except ValueError:
                continue
            if gpu not in GPU_MAP:
                print(co(R, f"  GPU {gpu_str}: invalid GPU index — skipped"))
                continue
            model_path = info.get("path", "")
            # Validate path before passing to start_gpu
            if not model_path.endswith(".gguf"):
                print(co(R, f"  GPU {gpu}: suspicious path rejected: {model_path}"))
                continue
            if os.path.exists(model_path):
                pid = start_gpu(gpu, model_path)
                print(f"  GPU {gpu}: loading {info['model']} (PID {pid})")
                time.sleep(2)
            else:
                print(co(R, f"  GPU {gpu}: model not found {model_path}"))
    print(co(Y, f"\n  Loading... Check with 'gpus'"))

def cmd_diagnose(args):
    if not args:
        print(co(Y, "  Usage: diagnose <gpu|all>")); return
    gpus = list(range(NUM_GPUS)) if args.strip() == "all" else []
    if not gpus:
        try:
            gpus = [int(x) for x in args.split(",")]
        except Exception:
            print(co(R, f"  Invalid: {args}")); return
    invalid = [g for g in gpus if g not in GPU_MAP]
    if invalid:
        print(co(R, f"  Invalid GPU index: {invalid}")); return
    for gpu in gpus:
        diagnose_gpu(gpu)

def cmd_kill_zombies():
    killed = kill_zombies()
    print(f"  Killed {killed} zombie/D-state processes" if killed else "  No zombies found")

def cmd_ports():
    print()
    for gpu in range(NUM_GPUS):
        print(f"    GPU {gpu} → port {BASE_PORT + gpu}")
    print(f"    nginx LB → port {LB_PORT}")
    print()

def cmd_status():
    """Curses live dashboard."""
    try:
        curses.wrapper(_dashboard)
    except curses.error:
        _dashboard_simple()

def _dashboard(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    curses.init_pair(3, curses.COLOR_RED, -1)
    curses.init_pair(4, curses.COLOR_CYAN, -1)

    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        instances = get_instances()
        ram = get_ram_info()

        stdscr.addstr(0, 0, "═══ AI RIG DASHBOARD ═══", curses.A_BOLD)
        stdscr.addstr(0, 26, f"  {time.strftime('%H:%M:%S')}  |  LB: http://localhost:{LB_PORT}")
        stdscr.addstr(2, 0, f"{'GPU':>3} {'Card':<24} {'VRAM':>11} {'Temp':>5} {'Port':>5} {'Status':<8} {'Model':<28} {'PID':>6}", curses.A_BOLD)
        stdscr.addstr(3, 0, "─" * min(w-1, 105))

        for i in range(NUM_GPUS):
            if 4+i >= h-4:
                break
            info = GPU_MAP[i]
            inst = instances[i]
            vu, vt = inst["vram_used"], inst["vram_total"]
            temp_s = f"{inst['temp']}C" if inst['temp'] is not None else "  —"
            st = inst["status"]
            model = (inst["model"] or "—")[:26]
            pid = str(inst["pid"] or "—")
            bar_pct = min(10, int(vu * 10 / vt)) if vt else 0
            vram_bar = "█" * bar_pct + "░" * (10 - bar_pct)

            color = (curses.color_pair(1) if st == "ready" else
                     curses.color_pair(2) if st == "loading" else
                     curses.color_pair(3) if st == "crashed" else
                     curses.A_DIM)

            try:
                stdscr.addstr(4+i, 0, f"{i:>3} {info['name']:<24} {vram_bar} {temp_s:>5} {inst['port']:>5} ")
                stdscr.addstr(f"{st:<8}", color)
                stdscr.addstr(f" {model:<28} {pid:>6}")
            except curses.error:
                pass

        row = 4 + NUM_GPUS + 1
        if row < h - 2:
            pct = ram["used"] * 100 // ram["total"] if ram["total"] else 0
            ram_bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            try:
                stdscr.addstr(row, 0, f"  RAM: [{ram_bar}] {ram['used']}M / {ram['total']}M ({ram['avail']}M free)")
                if ram["avail"] < 2048:
                    stdscr.addstr(row, 60, "⚠ LOW", curses.color_pair(3))
                active = sum(1 for inst in instances.values() if inst["status"] in ("ready", "loading"))
                stdscr.addstr(row+1, 0, f"  nginx: {active}/{NUM_GPUS} backends  |  Press 'q' to exit")
            except curses.error:
                pass

        stdscr.refresh()
        for _ in range(50):
            ch = stdscr.getch()
            if ch == ord('q') or ch == 27:
                return
            time.sleep(0.1)

def _dashboard_simple():
    """Fallback dashboard without curses."""
    print(co(DIM, "  Simple dashboard. Ctrl+C to exit."))
    try:
        while True:
            os.system("clear")
            cmd_gpus()
            time.sleep(5)
    except KeyboardInterrupt:
        print()

# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def get_ram_info():
    """Read system RAM info. Returns zeros on failure — never crashes callers."""
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                key = parts[0].rstrip(":")
                if key in ("MemTotal", "MemAvailable", "MemFree", "SwapTotal", "SwapFree"):
                    info[key] = int(parts[1]) // 1024
        return {"total": info.get("MemTotal", 0), "avail": info.get("MemAvailable", 0),
                "used": info.get("MemTotal", 0) - info.get("MemAvailable", 0)}
    except Exception:
        return {"total": 0, "avail": 0, "used": 0}

# ══════════════════════════════════════════════════════════════════════════
# MAIN SHELL
# ══════════════════════════════════════════════════════════════════════════

# ── HELP SYSTEM ──────────────────────────────────────────────────────────
# 'help' = short overview, 'help <command>' = detailed info, 'help all' = full reference

HELP_SHORT = f"""
  {co(BOLD, 'AI Rig Shell v1.4')}  |  ↑↓ history  |  Tab autocomplete

  {co(C, 'gpus')} [{co(DIM, '-r')}]             GPU status  ({co(DIM, '-r')} = auto-refresh until settled)
  {co(C, 'models')}                 List models and modes
  {co(C, 'load')} <gpu|all> <model> [{co(DIM, 'mode')}] [{co(DIM, 'temp=X')}]
                                    Load model onto GPU(s)
                                    Models: gemma4, qwen35-4b, qwen35-9b, mistral-nemo, qwen25-coder
                                    Thinking: gemma4/qwen35 have thinking ON by default
                                              Use {co(C, 'nothink')} mode to disable thinking
                                    Temp:   0.0=deterministic, 0.2=factual, 0.8=default, 1.0+=creative
  {co(C, 'unload')} <gpu|all>       Stop GPU(s)
  {co(C, 'verify')}                 Wait for load + test each GPU
  {co(C, 'test')} <gpu|all>         Quick test prompt
  {co(C, 'bench')} <gpu>            5-prompt benchmark
  {co(C, 'chat')} <gpu>             Interactive chat
  {co(C, 'status')}                 Live dashboard (q to exit)
  {co(C, 'perf')}                   Performance history
  {co(C, 'logs')} <gpu>             Show error log
  {co(C, 'diagnose')} <gpu|all>     Deep GPU diagnosis (uses AI)
  {co(C, 'nginx reload')}           Update load balancer
  {co(C, 'resume')}                 Reload from saved state
  {co(C, 'rig')} <shutdown|reboot|suspend>  Rig power (saves state first)
  {co(C, 'defaults')}                 GPU default models (set/load/save)
  {co(C, 'power')} [sleep|wake|...] GPU power + auto-sleep policy
  {co(C, 'kill-zombies')}           Kill stuck processes
  {co(C, 'ports')}                  Port assignments
  {co(C, 'models download')} <url>  Download from HuggingFace
  {co(C, 'help')} <command>         Detailed help for a command
  {co(C, 'help all')}               Full reference
  {co(C, 'exit')}                   Exit (GPUs keep running)
"""

HELP_DETAIL = {}

HELP_DETAIL["gpus"] = f"""
  {co(BOLD, 'gpus')} [{co(C, '-r')}]

  Show all 7 GPUs: card name, VRAM used/total, temperature, port, status,
  loaded model, mode, PID, and PCIe BAR size.

  Status values:
    {co(G, 'ready')}     Model loaded, accepting requests
    {co(Y, 'load▒')}     Loading on fast BAR GPU (~30s)
    {co(Y, 'load▒▒▒')}   Loading on 256MB BAR GPU (~3-4 min)
    {co(R, 'CRASHED')}   Process died unexpectedly
    {co(R, 'ZOMBIE')}    Process stuck in D-state (use kill-zombies)
    {co(DIM, 'off')}       No model loaded

  {co(C, 'gpus')}  Auto-refreshes every 5s until all GPUs reach a final state
             (ready, crashed, or off). Ctrl+C to stop. Use after 'load'.
"""

HELP_DETAIL["models"] = f"""
  {co(BOLD, 'models')}

  Lists all configured models with their modes, speeds, and context sizes.
  Also shows raw GGUF files on disk.

  {co(BOLD, 'What is a mode?')}
  A mode is a set of tested, proven flags for a specific use case.
  Modes control: context size, thinking on/off, flash attention, batch size.

  {co(BOLD, 'Thinking models:')}
  Gemma 4, Qwen 3.5 4B, and Qwen 3.5 9B have a thinking/reasoning mode.
  When thinking is ON, the model reasons internally before answering.
    - Thinking ON:  more accurate, uses more tokens, slower effective speed
    - Thinking OFF: direct answers, faster, better for tool calls & simple Q&A
  Use {co(C, 'nothink')} mode to disable thinking. Default modes have thinking ON.

  {co(BOLD, 'Which modes exist for which model:')}

    gemma4        → {co(C, 'fast')}, {co(C, '16k')}*, {co(C, 'nothink')}, {co(C, 'nothink-fast')}
    qwen35-4b     → {co(C, 'default')}*, {co(C, 'nothink')}
    qwen35-9b     → {co(C, 'default')}*, {co(C, 'nothink')}
    mistral-nemo  → {co(C, 'default')}* (no thinking)
    qwen25-coder  → {co(C, 'default')}* (no thinking)

  {co(BOLD, 'Models and their modes:')}

  {co(BOLD, co(C, 'gemma4'))} — Gemma 4 E4B (4.7 GB, Google, thinking model)

    {co(C, 'fast')}          ~47 tok/s | 8K ctx  | thinking ON
      Flash attention workaround (-b 512). Thinks then answers.

    {co(C, '16k')} *         ~48 tok/s | 16K ctx | thinking ON  [DEFAULT]
      Flash attention OFF. Best for long prompts. Thinks then answers.

    {co(C, 'nothink')}       ~48 tok/s | 16K ctx | thinking OFF
      Direct answers, no reasoning. Best for Ritu's agent, tool calls.

    {co(C, 'nothink-fast')}  ~47 tok/s | 8K ctx  | thinking OFF
      Direct answers + flash attention workaround.

  {co(BOLD, co(C, 'qwen35-4b'))} — Qwen 3.5 4B (2.6 GB, Alibaba, thinking model)

    {co(C, 'default')} *  ~48 tok/s | 16K ctx | thinking ON
      Reasons internally before answering. Needs max_tokens ~200+.

    {co(C, 'nothink')}   ~52 tok/s | 16K ctx | thinking OFF
      Direct answers. Faster for simple tasks and tool calling.

  {co(BOLD, co(C, 'qwen35-9b'))} — Qwen 3.5 9B (5.3 GB, Alibaba, thinking model)

    {co(C, 'default')} *  ~46/37 tok/s | 4K/16K ctx | thinking ON
      4K on 8GB GPUs, 16K on GPU 2 (12GB). Reasons before answering.

    {co(C, 'nothink')}   ~50/40 tok/s | 4K/16K ctx | thinking OFF
      Direct answers. Faster for simple tasks.

  {co(BOLD, co(C, 'mistral-nemo'))} — Mistral Nemo 12B (7.0 GB, Mistral AI, no thinking)

    {co(C, 'default')} *  ~42 tok/s | 8K ctx | GPU 2 only (needs 12GB)
      Standard instruct model. No thinking architecture.

  {co(BOLD, co(C, 'qwen25-coder'))} — Qwen 2.5 Coder 7B (4.4 GB, code-focused, no thinking)

    {co(C, 'default')} *  ~57 tok/s | 16K ctx | any GPU
      Code generation & review. Original OpenClaw config.

  * = default mode (used when you don't specify a mode)
"""

HELP_DETAIL["load"] = f"""
  {co(BOLD, 'load')} <gpu|all> <model> [mode] [temp=X]

  Load a model onto one or more GPUs.

  {co(BOLD, 'Parameters:')}
    gpu         GPU number(s): 0, 1, 0,1,2, 0 1 2, or all
    model       Model name: gemma4, qwen35-4b, qwen35-9b, mistral-nemo, qwen25-coder
    mode        Optional mode. See 'help models' for all modes per model.
    temp=X      Optional. Temperature 0.0-2.0. Controls randomness.

  {co(BOLD, 'Thinking modes:')}
    Gemma 4 and Qwen 3.5 models have thinking (reasoning) ON by default.
    Use {co(C, 'nothink')} mode to disable thinking for direct answers:

    load 0 gemma4 nothink           Gemma 4, thinking OFF, direct answers
    load 0 gemma4 16k               Gemma 4, thinking ON (default)
    load all qwen35-4b nothink      Qwen 3.5 4B, thinking OFF, all GPUs

  {co(BOLD, 'Temperature guide:')}
    temp=0.0    Deterministic — same input always gives same output
    temp=0.2    Low randomness — best for factual extraction, JSON, Ritu agent
    temp=0.6    Balanced — good for general chat
    temp=0.8    Default if not specified — creative, varied responses
    temp=1.0+   High randomness — brainstorming, creative writing

  {co(BOLD, 'Examples:')}
    load 0 gemma4                      Gemma 4 default (16k, thinking ON)
    load 0 gemma4 nothink              Gemma 4, thinking OFF
    load all gemma4 16k temp=0.2       All GPUs, 16k, thinking ON, factual
    load all gemma4 nothink temp=0     All GPUs, no thinking, deterministic
    load 0 1 2 qwen35-4b nothink      Qwen 3.5 4B, thinking OFF
    load 2 mistral-nemo                Mistral Nemo on GPU 2 (no thinking mode)
    load 3 4 5 6 gemma4 nothink-fast   GPUs 3-6, no thinking, small batch

  {co(BOLD, 'Loading times:')}
    GPUs 0,1 (8GB BAR):       ~30 seconds
    GPU 2 (16GB BAR):         ~30 seconds
    GPUs 3,4,5,6 (256MB BAR): ~3-4 minutes

  Use {co(C, 'models')} to see all available modes per model.
  Use {co(C, 'verify')} to run tests on all loaded GPUs.
"""

HELP_DETAIL["unload"] = f"""
  {co(BOLD, 'unload')} <gpu|all>

  Stop the model on one or more GPUs.

    unload 0        Stop GPU 0
    unload 0,1,3    Stop GPUs 0, 1, 3
    unload all      Stop all 7 GPUs
"""

HELP_DETAIL["verify"] = f"""
  {co(BOLD, 'verify')} [{co(C, '-v')}]

  Runs 5 tests on every loaded GPU: Basic (math), JSON (extraction), Code
  (function), Reasoning (multi-step), Stress (large prompt).

  Auto-reloads any GPU that crashes during testing. Shows results table with
  PASS/WEAK/CRASH per test per GPU.

  {co(C, 'verify')}       Run tests, show results and responses
  {co(C, 'verify -v')}    Verbose — also show the prompt sent and expected answer for each test

  Tests are loaded from /opt/ai-rig-verify.json (editable).
  Use after {co(C, 'load all')} to confirm everything is working.
"""

HELP_DETAIL["test"] = f"""
  {co(BOLD, 'test')} <gpu|all>

  Send a JSON extraction test prompt to GPU(s). Reports speed and output.

    test 0       Test GPU 0
    test all     Test all running GPUs
"""

HELP_DETAIL["bench"] = f"""
  {co(BOLD, 'bench')} <gpu>

  Run 5 varied prompts (math, code, JSON, reasoning) on one GPU.
  Reports per-prompt speed and overall average.

    bench 0      Benchmark GPU 0
"""

HELP_DETAIL["chat"] = f"""
  {co(BOLD, 'chat')} <gpu>

  Interactive multi-turn chat with a GPU. Type messages, get responses.
  Type 'exit' to return to the shell.

    chat 0       Chat with GPU 0
"""

HELP_DETAIL["status"] = f"""
  {co(BOLD, 'status')}

  Live curses dashboard. Shows all GPUs with VRAM bars, temps, RAM usage,
  and nginx backend status. Refreshes every 5s. Press 'q' to exit.
"""

HELP_DETAIL["perf"] = f"""
  {co(BOLD, 'perf')}

  Performance history from test/bench/chat commands.
  Shows per-GPU: avg tok/s, total requests, total tokens generated.
  Data logged to /opt/ai-rig-perf.jsonl.
"""

HELP_DETAIL["logs"] = f"""
  {co(BOLD, 'logs')} <gpu>

  Show last 30 lines of a GPU's stderr log (/tmp/gpu<N>_err.log).
  Useful for finding crash messages like "double free or corruption".

    logs 0       Show GPU 0 error log
"""

HELP_DETAIL["diagnose"] = f"""
  {co(BOLD, 'diagnose')} <gpu|all>

  Deep diagnosis: collects driver state, VRAM, temp, PCIe BAR, link speed,
  process state, and crash log. If another GPU has a model running, sends
  all the data to it for AI-powered analysis and fix suggestions.

    diagnose 4       Diagnose GPU 4
    diagnose all     Diagnose all GPUs
"""

HELP_DETAIL["nginx"] = f"""
  {co(BOLD, 'nginx reload')}

  Rebuild nginx load balancer from currently running GPU instances.
  Detects which ports are active and generates upstream config.
  Load balancer: http://localhost:{LB_PORT}/v1/chat/completions
"""

HELP_DETAIL["resume"] = f"""
  {co(BOLD, 'resume')}

  Reload models from saved state (/opt/ai-rig-state.json).
  Restores GPU→model assignments from the previous session.
  Use after a reboot.
"""

HELP_DETAIL["power"] = f"""
  {co(BOLD, 'power')} [{co(C, 'low')}|{co(C, 'on')}] [gpu|all]

  Control GPU power state to save electricity when GPUs are idle.

  {co(BOLD, 'Without arguments:')} show current power state for all GPUs.

  {co(BOLD, 'Levels:')}
    {co(C, 'low')}    Set GPU to lowest clock (300MHz). Saves ~5W per GPU.
    {co(C, 'on')}     Restore to auto (full performance). Required before loading models.

  {co(BOLD, 'Examples:')}
    power                Show power state for all GPUs
    power low all        Set all GPUs to low power
    power low 3 4 5 6    Set GPUs 3-6 to low power (256MB BAR, slow anyway)
    power on all         Restore full performance on all GPUs
    power on 0           Restore GPU 0 before loading a model

  {co(BOLD, 'Notes:')}
  - Loading a model ({co(C, 'load')}) auto-restores power to 'auto' on that GPU.
  - Idle GPUs draw ~10-12W each. Low mode drops to ~7-8W.
  - For 6 idle GPUs, 'power low all' saves ~20-30W total.
  - GPUs with models loaded will have degraded performance in low mode.
"""

HELP_DETAIL["kill-zombies"] = f"""
  {co(BOLD, 'kill-zombies')}

  Force-kill zombie/D-state llama-server processes that can't be stopped
  with 'unload'. These are processes stuck in GPU driver I/O.
"""

HELP_DETAIL["ports"] = f"""
  {co(BOLD, 'ports')}

  GPU 0 → port {BASE_PORT}
  GPU 1 → port {BASE_PORT+1}
  GPU 2 → port {BASE_PORT+2}
  GPU 3 → port {BASE_PORT+3}
  GPU 4 → port {BASE_PORT+4}
  GPU 5 → port {BASE_PORT+5}
  GPU 6 → port {BASE_PORT+6}
  nginx LB → port {LB_PORT} (routes to all running GPUs)
"""

def _build_help_all():
    """Full reference — all commands with details."""
    parts = [HELP_SHORT, co(BOLD, "\n  ═══ FULL REFERENCE ═══\n")]
    for cmd_name in ["gpus", "models", "load", "unload", "verify", "test", "bench",
                     "chat", "status", "perf", "logs", "diagnose", "nginx", "resume",
                     "kill-zombies", "ports"]:
        if cmd_name in HELP_DETAIL:
            parts.append(HELP_DETAIL[cmd_name])
    return "\n".join(parts)

def cmd_help(args=""):
    """Show help — short, per-command, or full."""
    args = args.strip().lower()
    if not args:
        print(HELP_SHORT)
    elif args == "all":
        # Page through with less-like behavior
        full = _build_help_all()
        print(full)
    elif args in HELP_DETAIL:
        print(HELP_DETAIL[args])
    elif args.replace("-", "") in HELP_DETAIL:
        print(HELP_DETAIL[args.replace("-", "")])
    else:
        print(co(Y, f"  No help for '{args}'. Try: help, help <command>, or help all"))
        print(co(DIM, f"  Commands: {', '.join(sorted(HELP_DETAIL.keys()))}"))
    print()

def _sigterm_handler(signum, frame):
    """Clean up all GPU processes on SIGTERM (e.g. systemctl stop)."""
    print("\n  SIGTERM received — stopping all GPUs...")
    for gpu in range(NUM_GPUS):
        kill_port(BASE_PORT + gpu)
    sys.exit(0)

def main():
    signal.signal(signal.SIGTERM, _sigterm_handler)
    disable_power_mgmt()
    init_db()  # Create SQLite database — single source of truth

    # Enable readline: up/down arrow history, Ctrl+R search, persistent history file
    HISTORY_FILE = "/opt/.ai-rig-history"
    try:
        import readline
        readline.set_history_length(500)
        try:
            readline.read_history_file(HISTORY_FILE)
        except FileNotFoundError:
            pass
        import atexit
        atexit.register(readline.write_history_file, HISTORY_FILE)

        # Tab completion: context-aware for load command
        BASE_COMMANDS = ["gpus", "models", "models download", "load", "load all",
                    "unload", "unload all", "test", "test all", "bench", "status",
                    "perf", "logs", "chat", "diagnose", "diagnose all", "verify",
                    "power", "power sleep all", "power wake all", "power sleep", "power wake", "power calibrate",
                    "rig shutdown", "rig reboot", "rig suspend",
                    "power autosleep", "power autosleep off",
                    "defaults", "defaults load", "defaults save-running", "defaults auto on", "defaults auto off",
                    "nginx reload", "resume", "kill-zombies", "ports", "help", "exit"]
        MODEL_NAMES = list(MODEL_CONFIGS.keys())

        def completer(text, state):
            line = readline.get_line_buffer()
            parts = line.split()

            if len(parts) >= 2 and parts[0] == "load":
                # After "load <gpu(s)>", complete model names
                # Check if we have GPU spec(s) and need model completion
                gpu_parts = []
                model_pos = None
                for i, p in enumerate(parts[1:], 1):
                    if p.lower() == "all" or p.replace(",", "").isdigit():
                        gpu_parts.append(p)
                    else:
                        model_pos = i
                        break

                if gpu_parts and model_pos is None:
                    # Have GPU spec, need model name — complete model names
                    prefix = text.lower()
                    matches = [n for n in MODEL_NAMES if n.startswith(prefix)]
                elif gpu_parts and model_pos is not None:
                    # Have GPU + model, complete modes
                    model_text = parts[model_pos].lower()
                    # Find matching model config
                    matched_cfg = None
                    for n, c in MODEL_CONFIGS.items():
                        if model_text == n or model_text in n:
                            matched_cfg = c
                            break
                    if matched_cfg and model_pos == len(parts) - 1 and not line.endswith(" "):
                        # Still typing model name
                        prefix = text.lower()
                        matches = [n for n in MODEL_NAMES if n.startswith(prefix)]
                    elif matched_cfg:
                        # Model done, complete mode names
                        prefix = text.lower()
                        modes = list(matched_cfg["modes"].keys())
                        matches = [m for m in modes if m.startswith(prefix)]
                    else:
                        prefix = text.lower()
                        matches = [n for n in MODEL_NAMES if n.startswith(prefix)]
                else:
                    matches = [c for c in BASE_COMMANDS if c.startswith(line)]
            elif len(parts) >= 2 and parts[0] == "unload":
                # Complete GPU numbers or "all"
                matches = [c for c in BASE_COMMANDS if c.startswith(line)]
            else:
                # Default: complete base commands
                matches = [c for c in BASE_COMMANDS if c.startswith(line)]

            return matches[state] if state < len(matches) else None

        readline.set_completer(completer)
        readline.parse_and_bind("tab: complete")
    except ImportError:
        pass  # readline not available, basic input still works

    print(f"""
  {co(BOLD, '╔══════════════════════════════════════╗')}
  {co(BOLD, '║       AI RIG MANAGEMENT SHELL        ║')}
  {co(BOLD, '╚══════════════════════════════════════╝')}

  Type {co(C, 'help')} for commands  |  ↑↓ history  |  Tab autocomplete
""")

    state = load_state()
    if state and any(v for v in state.get("gpus", {}).values()):
        print(co(Y, "  Previous session had models loaded. Type 'resume' to reload.\n"))

    while True:
        try:
            raw = input(co(B, "ai-rig") + co(W, "> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!"); break
        if not raw:
            continue

        parts = raw.split(None, 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd in ("exit", "quit", "q"):
            print("Bye!"); break
        elif cmd in ("gpus", "gpu"):
            cmd_gpus()
        elif cmd in ("models", "model"):
            if args.startswith("download"):
                sub = args.split(None, 1)
                download_model(sub[1] if len(sub) > 1 else "")
            else:
                cmd_models()
        elif cmd == "load":
            cmd_load(args)
        elif cmd in ("unload", "stop"):
            cmd_unload(args)
        elif cmd == "verify":
            cmd_verify(verbose=("-v" in args or "--verbose" in args))
        elif cmd == "test":
            cmd_test(args)
        elif cmd == "bench":
            cmd_bench(args)
        elif cmd in ("status", "dashboard"):
            cmd_status()
        elif cmd == "perf":
            cmd_perf()
        elif cmd in ("logs", "log"):
            cmd_logs(args)
        elif cmd == "chat":
            cmd_chat(args)
        elif cmd == "diagnose":
            cmd_diagnose(args)
        elif cmd == "profile":
            cmd_profile(args)
        elif cmd == "resume":
            cmd_resume()
        elif cmd == "kill-zombies":
            cmd_kill_zombies()
        elif cmd == "power":
            cmd_power(args)
        elif cmd == "rig":
            cmd_rig(args)
        elif cmd in ("defaults", "default"):
            cmd_defaults(args)
        elif cmd == "ports":
            cmd_ports()
        elif cmd == "nginx":
            if args.strip() == "reload":
                rebuild_nginx()
            else:
                print(co(Y, "  Usage: nginx reload"))
        elif cmd in ("help", "?"):
            cmd_help(args)
        else:
            print(co(R, f"  Unknown: {cmd}. Type 'help'"))

if __name__ == "__main__":
    main()
