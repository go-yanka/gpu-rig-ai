#!/usr/bin/env bash
# MANDATORY preflight before launching ANY ingest run.
# REFUSES launch if EMBED_GPUS is not maximum-feasible.
# Codified 2026-04-26 after RG flagged for the Nth time that we keep defaulting
# to {4,5,6} despite GPUs 0, 1, 3 being usable. Same enforcement model as gate_preflight.sh.
#
# Reserved-from-pool: GPU 2 (qwen3-14b on :9082, single-slot).
# GPU 0 (bge-reranker) coexists fine with BGE-M3 (12GB VRAM, both <3GB combined).
# Expected pool: 0,1,3,4,5,6 (all 7 GPUs minus GPU 2).
#
# 2026-04-26 (later): also runs pool+reranker concurrent-warmup AFTER config check.
# Codified RG lesson: warmed-up GPUs at steady-state JIT/shader cache give consistent
# throughput; cold pool produces 200-500ms outliers in first 50-100 queries.

set -e

ALL_GPUS="0 1 2 3 4 5 6"
RESERVED_GPUS="2"

EXPECTED_LIST=""
for g in $ALL_GPUS; do
  skip=0
  for r in $RESERVED_GPUS; do [ "$g" = "$r" ] && skip=1 && break; done
  [ $skip -eq 0 ] && EXPECTED_LIST="$EXPECTED_LIST,$g"
done
EXPECTED=${EXPECTED_LIST#,}

CURRENT=$(grep -oP 'EMBED_GPUS=\$\{EMBED_GPUS:-\K[0-9,]+' /opt/indian-legal-ai/rag/cbic_rag/bin/start_api.sh | head -1 || echo "")

norm() { echo "$1" | tr ',' '\n' | sort -n | paste -sd, ; }
EXPECTED_NORM=$(norm "$EXPECTED")
CURRENT_NORM=$(norm "$CURRENT")

if [ "$CURRENT_NORM" != "$EXPECTED_NORM" ]; then
  echo "REFUSE: EMBED_GPUS=$CURRENT but expected $EXPECTED (all non-reserved GPUs)" >&2
  echo "  Reserved (host other models, can't be in embed pool): $RESERVED_GPUS" >&2
  echo "  Fix: edit /opt/indian-legal-ai/rag/cbic_rag/bin/start_api.sh" >&2
  echo "       AND /etc/systemd/system/cbic-rag-api.service.d/*.conf (drop-ins)" >&2
  echo "       set EMBED_GPUS=$EXPECTED, then 'systemctl daemon-reload && systemctl restart cbic-rag-api'" >&2
  exit 1
fi

for g in $(echo $EXPECTED | tr ',' ' '); do
  f="/sys/class/drm/card$g/device/gpu_busy_percent"
  if [ ! -r "$f" ]; then
    echo "REFUSE: GPU $g not visible in /sys/class/drm/" >&2
    exit 2
  fi
done

LIVE=$(journalctl -u cbic-rag-api --since '5 min ago' 2>/dev/null | grep -oP "ready: \[\K[0-9, ]+" | tail -1)
if [ -n "$LIVE" ]; then
  LIVE_NORM=$(norm "$(echo $LIVE | tr -d ' ')")
  if [ "$LIVE_NORM" != "$EXPECTED_NORM" ]; then
    echo "WARN: API last logged pool [$LIVE] != expected [$EXPECTED]. Restart cbic-rag-api to pick up config." >&2
  fi
fi

echo "RESOURCE_OK embed_pool=$EXPECTED (max feasible, all non-reserved); reserved=$RESERVED_GPUS"

# Pool + reranker concurrent warmup (RG-codified 2026-04-26).
# Sequential cold-load is in embed_pool_profiles.json (sequential_cold_load=true) — safe.
# This adds the steady-state warmup step under concurrent load.
bash /opt/indian-legal-ai/reingest_spec/scripts/warmup_pool.sh 30 cbic_v1
