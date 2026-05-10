#!/usr/bin/env bash
# MANDATORY preflight before launching ANY trust gate (G1/G2/G3/G4/G5).
# Refuses launch if another gate PYTHON process is running.
# Codified 2026-04-26 after CP-1 G3+G4 concurrent contention wasted ~45min.
# Reference: MEMORY.md GATE CONCURRENCY LESSON 2026-04-25.
#
# 2026-04-26 (later): also runs pool+reranker warmup. RG demand: every entry point
# that touches GPUs enforces sequential cold-load + concurrent warmup.
set -e
GATE_NAME="${1:-unspecified}"

# Match python processes whose argv contains 'gate_g[1-5]' (the actual evaluator scripts)
RUNNING=$(ps -eo pid,cmd | awk '/^[ \t]*[0-9]+ +(\/usr\/bin\/)?python[0-9.]* .*gate_g[1-5][a-z_]*\.py/ && !/awk/ && !/bash/ {print}' || true)
if [ -n "$RUNNING" ]; then
  echo "REFUSE: another gate evaluator is running. Wait for it to finish." >&2
  echo "$RUNNING" >&2
  exit 1
fi

curl -fsS -m 3 http://127.0.0.1:9085/health >/dev/null 2>&1 || { echo 'REFUSE: bge-reranker:9085 not healthy' >&2; exit 2; }
curl -fsS -m 3 http://127.0.0.1:9082/health >/dev/null 2>&1 || { echo 'REFUSE: qwen3:9082 not healthy' >&2; exit 3; }
# 2026-05-08: also check cbic-rag-api itself — gate evaluators hit /retrieve and /query
# on this API, so a dead API silently fails ALL queries (380 errors in CP-2 first run today).
curl -fsS -m 3 http://127.0.0.1:9500/health >/dev/null 2>&1 || { echo 'REFUSE: cbic-rag-api:9500 not healthy — start it before running gates' >&2; exit 4; }

# Pool+reranker warmup — drives steady-state JIT/shader cache before gate measurements.
# Non-fatal (log-and-continue) since gate eval can self-warm if needed, but pre-warm gives
# more stable rerank latency under the gate's parallel /retrieve workers.
bash /opt/indian-legal-ai/reingest_spec/scripts/warmup_pool.sh 20 cbic_v1 2>&1 | tail -3 || echo "WARN warmup failed (non-fatal)"

echo "PREFLIGHT_OK gate=$GATE_NAME ts=$(date +%s)"
