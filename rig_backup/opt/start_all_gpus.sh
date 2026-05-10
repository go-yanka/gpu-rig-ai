#!/bin/bash
# PROVEN CONFIG — do not change flags without testing
# Flags from working handoff: -b 2048 -ub 512 -t 4
export LD_LIBRARY_PATH=/opt/llama-server
export RADV_DEBUG=nodcc
LLAMA=/opt/llama-server/llama-server-rocm
MODEL=/opt/ai-models/gemma-4-E4B-it-Q4_K_M.gguf

# Disable GPU power management FIRST
for d in /sys/class/drm/card*/device/power/control; do echo on > $d 2>/dev/null; done

# Enable WoL
ethtool -s eth0 wol g 2>/dev/null

# Kill old instances
for PORT in 9080 9081 9082 9083 9084 9085 9086; do
    PID=$(lsof -t -i:$PORT 2>/dev/null)
    [ -n "$PID" ] && kill -9 $PID 2>/dev/null
done
sleep 3

echo "=== AI RIG — PROVEN CONFIG ==="
STARTED=0
for GPU in 0 1 2 3 4 5 6; do
    PORT=$((9080 + GPU))
    GGML_VK_VISIBLE_DEVICES=$GPU nohup $LLAMA \
        --model $MODEL --host 0.0.0.0 --port $PORT \
        -ngl 99 -c 16384 --parallel 1 \
        --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
        --mmap -b 2048 -ub 512 -t 4 \
        > /dev/null 2>&1 &
    PID=$!
    for i in $(seq 1 240); do
        sleep 1
        if curl -s --max-time 2 http://localhost:$PORT/health 2>/dev/null | grep -q "ok"; then
            STARTED=$((STARTED + 1))
            echo "[GPU $GPU] port $PORT READY (${i}s)"
            break
        fi
        if ! kill -0 $PID 2>/dev/null; then
            echo "[GPU $GPU] port $PORT FAILED"
            break
        fi
    done
done
echo "=== $STARTED/7 GPUs ==="
