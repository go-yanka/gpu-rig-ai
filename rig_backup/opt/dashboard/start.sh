#!/usr/bin/env bash
# Start the AI Rig Dashboard
# Usage: ./start.sh [port]

PORT="${1:-8080}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export DASHBOARD_PORT="$PORT"
export AI_MODEL_DIR="${AI_MODEL_DIR:-/opt/ai-models}"
export AI_STATE_FILE="${AI_STATE_FILE:-/opt/ai-rig-state.json}"
export AI_PERF_LOG="${AI_PERF_LOG:-/opt/ai-rig-perf.jsonl}"
export AI_BASE_PORT="${AI_BASE_PORT:-9080}"

echo ""
echo "  AI Rig Dashboard"
echo "  ─────────────────────────────"
echo "  http://192.168.1.107:${PORT}"
echo "  http://localhost:${PORT}"
echo "  Press Ctrl+C to stop"
echo ""

cd "$SCRIPT_DIR"
exec python3 app.py
