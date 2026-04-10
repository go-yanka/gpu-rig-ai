# GPU Rig AI - 7-GPU AMD Inference Server

Convert a mining rig into a local AI inference server. Manage 7 AMD GPUs running llama.cpp with a terminal shell and web dashboard.

## What This Is

A complete management system for running AI models on a 7-GPU AMD mining rig:

- **AI Rig Shell** (`ai_rig_shell.py`) - Terminal-based GPU management with 30+ commands
- **Web Dashboard** (`dashboard/`) - Real-time monitoring, testing, and control via browser
- **Power Management** - D3hot BACO sleep/wake with calibrated power measurement (~59W savings)
- **Model Testing** - 5-level verification suite (Basic, JSON, Code, Reasoning, Stress)
- **GPU Defaults** - Persistent per-GPU model assignments, auto-load on startup

## Hardware

| Component | Details |
|-----------|---------|
| **GPUs** | 6x RX 5700 XT (8GB RDNA1) + 1x RX 6700 XT (12GB RDNA2) |
| **Motherboard** | ASRock H110 Pro BTC+ (mining board, USB risers) |
| **RAM** | 16GB DDR4 |
| **Storage** | 16GB USB boot drive |
| **OS** | Ubuntu 22.04 LTS, kernel 5.15 |
| **Backend** | llama.cpp Vulkan (llama-server-rocm build b8665) |

## Models

| Model | Size | 8GB GPU | Thinking | Speed |
|-------|------|:-------:|:--------:|------:|
| Gemma 4 E4B | 4.7G | Yes | On/Off | ~48 t/s |
| Qwen 3.5 4B | 2.6G | Yes | On/Off | ~48 t/s |
| Qwen 3.5 9B | 5.3G | Yes (4K ctx) | On/Off | ~46 t/s |
| Mistral Nemo 12B | 7.0G | 12GB only | No | ~42 t/s |
| Qwen 2.5 Coder 7B | 4.4G | Yes | No | ~57 t/s |

Each model supports multiple modes: thinking on/off, context sizes, temperature control.

## Quick Start

```bash
# On the rig — start management shell
sudo python3 /opt/ai-rig-shell.py

# Load models
ai-rig> load all gemma4 nothink
ai-rig> gpus
ai-rig> verify

# Power management
ai-rig> power sleep all        # D3hot BACO, saves ~59W
ai-rig> power calibrate        # Measure actual sleep watts
ai-rig> power autosleep 5      # Auto-sleep after 5min idle

# GPU defaults
ai-rig> defaults set 0 gemma4 nothink temp=0.2
ai-rig> defaults set 2 mistral-nemo
ai-rig> defaults load

# Dashboard
cd /opt/dashboard && sudo python3 app.py
# Open http://192.168.1.107:8080
```

## Architecture

```
Shell (brain) ──writes──> SQLite (/dev/shm/ai-rig.db) <──reads── Dashboard (display)

/opt/ai-rig-defaults.json  ← persistent config (model defaults, power calibration)
```

Single source of truth: Shell writes to SQLite, Dashboard reads from SQLite only.

## Dashboard

- 3-state GPU cards: Active (green), Idle (dim), Sleeping (compact)
- Actions dropdown on every card — context-aware per state
- Power panel with savings, auto-sleep timer
- Testing panel — Models/GPUs/Custom with verbose output
- GPU detail modal — configure default model per GPU
- Rig control — Reboot/Sleep/Shutdown from browser

## Power Management

D3hot BACO sleep with calibrated per-GPU measurements:

| State | Total Power | Savings |
|-------|:-----------:|:-------:|
| All awake | ~66W | — |
| All sleeping | ~7W | **~59W** |

## License

MIT
