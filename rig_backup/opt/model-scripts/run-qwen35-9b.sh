#!/bin/bash
# Qwen 3.5 9B — PROVEN STABLE on 12GB GPU (GPU 2)
# Usage: sudo ./run-qwen35-9b.sh <GPU_NUMBER> [PORT]
# NOTE: On 8GB GPUs, context limited to 4096. On 12GB GPU 2, use 16384.

GPU=${1:?Usage: $0 <GPU_NUMBER> [PORT]}
PORT=${2:-$((9080 + GPU))}
MODEL=/opt/ai-models/qwen3.5-9b-q4_k_m.gguf

CTX=4096
[ "$GPU" = "2" ] && CTX=16384

for d in /sys/class/drm/card*/device/power/control; do echo on > $d 2>/dev/null; done
export LD_LIBRARY_PATH=/opt/llama-server
export RADV_DEBUG=nodcc
export GGML_VK_VISIBLE_DEVICES=$GPU

echo "Starting Qwen 3.5 9B on GPU $GPU, port $PORT"
echo "Model: $MODEL (5.3 GB)"
echo "Context: $CTX | Batch: 2048 | Micro-batch: 512"

exec /opt/llama-server/llama-server-rocm     --model $MODEL     --host 0.0.0.0 --port $PORT     -ngl 99 -c $CTX --parallel 1     --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0     --mmap -b 2048 -ub 512 -t 4
