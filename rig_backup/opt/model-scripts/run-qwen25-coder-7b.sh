#!/bin/bash
# Qwen 2.5 Coder 7B — PROVEN STABLE (original OpenClaw config from handoff)
# Usage: sudo ./run-qwen25-coder-7b.sh <GPU_NUMBER> [PORT]

GPU=${1:?Usage: $0 <GPU_NUMBER> [PORT]}
PORT=${2:-$((9080 + GPU))}
MODEL=/opt/ai-models/qwen2.5-coder-7b-instruct-q4_k_m.gguf

for d in /sys/class/drm/card*/device/power/control; do echo on > $d 2>/dev/null; done
export LD_LIBRARY_PATH=/opt/llama-server
export RADV_DEBUG=nodcc
export GGML_VK_VISIBLE_DEVICES=$GPU

echo "Starting Qwen 2.5 Coder 7B on GPU $GPU, port $PORT"
echo "Model: $MODEL (4.4 GB)"
echo "Context: 16384 | Batch: 2048 | Micro-batch: 512"

exec /opt/llama-server/llama-server-rocm     --model $MODEL     --host 0.0.0.0 --port $PORT     -ngl 99 -c 16384 --parallel 1     --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0     --mmap -b 2048 -ub 512 -t 4
