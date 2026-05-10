#!/bin/bash
# Mistral Nemo 12B — TESTED STABLE on 12GB GPU (GPU 2 only)
# Usage: sudo ./run-mistral-nemo-12b.sh <GPU_NUMBER> [PORT]
# WARNING: 7GB model — only fits on GPU 2 (12GB VRAM)

GPU=${1:?Usage: $0 <GPU_NUMBER> [PORT]}
PORT=${2:-$((9080 + GPU))}
MODEL=/opt/ai-models/mistral-nemo-12b-instruct-q4_k_m.gguf

if [ "$GPU" != "2" ]; then
    echo "WARNING: Mistral Nemo 12B (7GB) may not fit on 8GB GPU $GPU"
    echo "Recommended: use GPU 2 (12GB RX 6700 XT)"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

for d in /sys/class/drm/card*/device/power/control; do echo on > $d 2>/dev/null; done
export LD_LIBRARY_PATH=/opt/llama-server
export RADV_DEBUG=nodcc
export GGML_VK_VISIBLE_DEVICES=$GPU

echo "Starting Mistral Nemo 12B on GPU $GPU, port $PORT"
echo "Model: $MODEL (7.0 GB)"
echo "Context: 8192 | Batch: 2048 | Micro-batch: 512"

exec /opt/llama-server/llama-server-rocm     --model $MODEL     --host 0.0.0.0 --port $PORT     -ngl 99 -c 8192 --parallel 1     --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0     --mmap -b 2048 -ub 512 -t 4
