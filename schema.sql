-- AI Rig SQLite Schema
-- Database: /dev/shm/ai-rig.db (tmpfs, created fresh on boot)
-- Writer: ai_rig_shell.py (the only writer)
-- Reader: dashboard/app.py (read-only)

PRAGMA journal_mode=WAL;        -- concurrent reads while writing
PRAGMA busy_timeout=3000;       -- wait up to 3s if locked

-- ═══════════════════════════════════════════════════════════════
-- GPU STATE — real-time status of all 7 GPUs
-- Updated every 2 seconds by get_instances() → db_write_all_gpu_states()
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS gpu_state (
    gpu_id          INTEGER PRIMARY KEY,    -- 0-6
    card_name       TEXT,                   -- e.g. "RX 5700 XT (ASUS)"
    vram_mb         INTEGER,                -- total VRAM in MB (from GPU_MAP)
    bar_size        TEXT,                   -- "8GB", "16GB", "256MB"
    grade           TEXT,                   -- "A+", "A", "B"
    port            INTEGER,                -- llama-server port (9080-9086)
    pid             INTEGER,                -- llama-server process ID (null if stopped)
    status          TEXT,                   -- ready, loading, stopped, sleeping, crashed, zombie
    model_file      TEXT,                   -- e.g. "gemma-4-E4B-it-Q4_K_M.gguf"
    model_name      TEXT,                   -- e.g. "gemma4" (from MODEL_CONFIGS)
    mode            TEXT,                   -- e.g. "nothink", "16k", "fast"
    temperature     REAL,                   -- temperature setting (0.0-2.0)
    vram_used       INTEGER,                -- current VRAM used in MB
    vram_total      INTEGER,                -- current VRAM total in MB
    gpu_temp        INTEGER,                -- GPU temperature in Celsius
    speed           REAL,                   -- last inference speed (tok/s)
    flags           TEXT,                   -- llama-server CLI flags used
    context         INTEGER,                -- context window size
    power_watts     REAL,                   -- current power draw (from hwmon)
    perf_level      TEXT,                   -- "on" or "sleep" (D3hot BACO)
    pci_addr        TEXT,                   -- PCI address e.g. "03:00.0"
    idle_since      TEXT,                   -- ISO timestamp when GPU became idle (for auto-sleep)
    default_model   TEXT,                   -- default model name (from /opt/ai-rig-defaults.json)
    default_mode    TEXT,                   -- default mode
    default_temp    REAL,                   -- default temperature
    updated_at      TEXT                    -- ISO timestamp of last update
);

-- ═══════════════════════════════════════════════════════════════
-- WORKLOG — per-request inference history
-- Written by nginx access log parsing (future) or direct logging
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS worklog (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    gpu_id              INTEGER,            -- which GPU handled this request
    started_at          TEXT,               -- ISO timestamp
    ended_at            TEXT,               -- ISO timestamp
    client_ip           TEXT,               -- requester IP
    prompt_tokens       INTEGER,            -- input tokens
    completion_tokens   INTEGER,            -- output tokens
    speed               REAL,               -- tok/s for this request
    model_name          TEXT,               -- model used
    status              TEXT,               -- "ok", "error", "timeout"
    duration_ms         INTEGER             -- total request duration
);

-- ═══════════════════════════════════════════════════════════════
-- EVENTS — GPU lifecycle events (load, unload, crash, sleep, wake)
-- Last 500 events kept, older ones pruned automatically
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    gpu_id      INTEGER,                    -- which GPU
    action      TEXT,                       -- "load", "unload", "crash", "sleep", "wake"
    detail      TEXT,                       -- human-readable details
    source      TEXT,                       -- "shell", "dashboard", "auto-sleep"
    created_at  TEXT                        -- ISO timestamp
);

-- ═══════════════════════════════════════════════════════════════
-- SYSTEM STATE — rig-level metrics (RAM, uptime, etc)
-- Single row (id=1), updated every 2 seconds alongside gpu_state
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS system_state (
    id              INTEGER PRIMARY KEY DEFAULT 1,
    ram_total       INTEGER,                -- total RAM in MB
    ram_used        INTEGER,                -- used RAM in MB
    ram_avail       INTEGER,                -- available RAM in MB
    nginx_active    INTEGER,                -- number of active GPU backends in nginx
    uptime_sec      INTEGER,                -- system uptime in seconds
    updated_at      TEXT                    -- ISO timestamp
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_worklog_gpu ON worklog(gpu_id, started_at);
CREATE INDEX IF NOT EXISTS idx_events_gpu ON events(gpu_id, created_at);

-- Initialize system_state row
INSERT OR IGNORE INTO system_state (id) VALUES (1);
