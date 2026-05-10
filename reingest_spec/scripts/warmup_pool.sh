#!/usr/bin/env bash
# Shared warmup function for embed pool and reranker.
# Called by: resource_preflight.sh, gate_preflight.sh, bge-reranker ExecStartPost.
# Codified 2026-04-26 (RG demand: every script that touches GPUs must warm them).
#
# Why: per-GPU solo warmup (in embedder_direct.py) primes individual cards,
# but pool-LEVEL concurrent traffic primes the inter-GPU dispatch + JIT shader cache
# under realistic load. Without this, first 50-100 concurrent queries show 200-500ms
# outliers polluting eval timing windows.

set -e
N_CALLS=${1:-30}
COLLECTION=${2:-cbic_v1}

echo "[warmup] $N_CALLS concurrent /retrieve calls on $COLLECTION ..."
for i in $(seq 1 $N_CALLS); do
  curl -s -m 30 -X POST http://127.0.0.1:9500/retrieve \
    -H 'content-type: application/json' \
    -d "{\"question\":\"warmup query $i CGST refund SEZ valuation\",\"k\":3,\"collection\":\"$COLLECTION\"}" \
    > /dev/null &
done
wait
echo "[warmup] embed-pool steady-state complete"

# Reranker warmup: small batch directly to :9085
echo "[warmup] reranker (5 calls) ..."
for i in $(seq 1 5); do
  curl -s -m 15 -X POST http://127.0.0.1:9085/v1/rerank \
    -H 'content-type: application/json' \
    -d '{"model":"bge","query":"GST refund SEZ","documents":["procedure for refund","unrelated text","SEZ unit definition"]}' \
    > /dev/null &
done
wait
echo "[warmup] reranker steady-state complete"
