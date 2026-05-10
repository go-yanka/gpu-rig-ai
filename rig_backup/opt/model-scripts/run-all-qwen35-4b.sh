#!/bin/bash
# Load Qwen 3.5 4B on ALL 7 GPUs — PROVEN CONFIG
# Usage: sudo ./run-all-qwen35-4b.sh

echo "=== Loading Qwen 3.5 4B on ALL 7 GPUs ==="
for GPU in 0 1 2 3 4 5 6; do
    PORT=$((9080 + GPU))
    lsof -t -i:$PORT 2>/dev/null | xargs kill -9 2>/dev/null
done
sleep 3

for GPU in 0 1 2 3 4 5 6; do
    PORT=$((9080 + GPU))
    /opt/model-scripts/run-qwen35-4b.sh $GPU $PORT &
    echo "  GPU $GPU -> port $PORT"
    sleep 2
done

echo ""
echo "All 7 starting."
echo "Check: ai-rig> gpus"
