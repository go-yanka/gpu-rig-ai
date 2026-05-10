#!/bin/bash
# Gemma 4 E4B — PROVEN STABLE CONFIG (tested Apr 9, 2026)
# Mode: 16k (default) — flash-attn off fixes llama.cpp issue #21336
# NOTE: No KV quant (--cache-type-k/v) when using --flash-attn off
# Usage: sudo ./run-gemma4-e4b.sh <GPU_NUMBER> [PORT] [MODE]
#   MODE: 16k (default, 16K ctx ~48 tok/s) or fast (8K ctx ~47 tok/s)

GPU=${1:?Usage: $0 <GPU_NUMBER> [PORT] [MODE]}
PORT=${2:-$((9080 + GPU))}
MODE=${3:-16k}
MODEL=/opt/ai-models/gemma-4-E4B-it-Q4_K_M.gguf

for d in /sys/class/drm/card*/device/power/control; do echo on > $d 2>/dev/null; done
export LD_LIBRARY_PATH=/opt/llama-server
export RADV_DEBUG=nodcc
export GGML_VK_VISIBLE_DEVICES=$GPU

if [ "$MODE" = "fast" ]; then
    echo "Starting Gemma 4 E4B on GPU $GPU, port $PORT (fast mode: 8K ctx)"
    exec /opt/llama-server/llama-server-rocm         --model $MODEL --host 0.0.0.0 --port $PORT         -ngl 99 -c 8192 --parallel 1         --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0         --mmap -b 512 -ub 256 -t 4
else
    echo "Starting Gemma 4 E4B on GPU $GPU, port $PORT (16k mode: 16K ctx, flash-attn off)"
    exec /opt/llama-server/llama-server-rocm         --model $MODEL --host 0.0.0.0 --port $PORT         -ngl 99 -c 16384 --parallel 1         --cache-ram 0 --mmap -b 2048 -ub 512 -t 4         --flash-attn off
fi
