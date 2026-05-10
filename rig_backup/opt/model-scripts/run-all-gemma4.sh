#!/bin/bash
# Load Gemma 4 E4B on ALL 7 GPUs — PROVEN CONFIG (16k mode)
# Usage: sudo ./run-all-gemma4.sh [MODE]
#   MODE: 16k (default) or fast

MODE=${1:-16k}
echo "=== Loading Gemma 4 E4B on ALL 7 GPUs (mode: $MODE) ==="
for GPU in 0 1 2 3 4 5 6; do
    PORT=$((9080 + GPU))
    lsof -t -i:$PORT 2>/dev/null | xargs kill -9 2>/dev/null
done
sleep 3

for GPU in 0 1 2 3 4 5 6; do
    PORT=$((9080 + GPU))
    /opt/model-scripts/run-gemma4-e4b.sh $GPU $PORT $MODE &
    echo "  GPU $GPU -> port $PORT"
    sleep 2
done

echo ""
echo "All 7 starting. 256MB BAR GPUs (3-6) take 3-4 min."
echo "Check: ai-rig> gpus"
