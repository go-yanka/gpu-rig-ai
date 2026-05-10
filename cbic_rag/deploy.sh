#!/bin/bash
# deploy.sh — one-shot setup for cbic RAG on the rig.
# Idempotent: safe to rerun.
#
# Steps:
#   1. Create venv at /opt/indian-legal-ai/rag/cbic_api/venv
#   2. Install deps (requirements.txt)
#   3. Launch a NEW qdrant instance on :6343 with isolated storage
#      (does NOT touch the existing qdrant on :6333 or its collections)
#   4. Systemd units (user-level) for qdrant-cbic, cbic-rag-api
#   5. Kick off ingestion in background (screen session `cbic-ingest`)
#   6. Print wiring instructions for Open WebUI + LiteLLM
set -eu

BASE=/opt/indian-legal-ai/rag/cbic_api
QSTORE=/opt/indian-legal-ai/rag/qdrant_cbic_storage
QPORT=6343
APIPORT=9500
QDRANT_BIN=$(command -v qdrant || true)
if [ -z "$QDRANT_BIN" ]; then
  # try the one that's already running
  QDRANT_BIN=$(readlink -f /proc/$(pgrep -f '^\./qdrant$' | head -1)/exe 2>/dev/null || true)
fi
if [ -z "$QDRANT_BIN" ] || [ ! -x "$QDRANT_BIN" ]; then
  echo "ERROR: cannot locate qdrant binary. Install qdrant or set QDRANT_BIN."
  exit 1
fi

echo "[deploy] using qdrant binary: $QDRANT_BIN"

mkdir -p "$BASE" "$QSTORE"
cd "$BASE"

# --- copy code in place ---
SRC="$(cd "$(dirname "$0")" && pwd)"
cp -f "$SRC/chunker.py" "$SRC/ingest.py" "$SRC/retriever.py" \
      "$SRC/storyformat.py" "$SRC/api.py" "$SRC/requirements.txt" \
      "$SRC/hyde.py" "$SRC/embedder.py" "$SRC/api_v2_shadow.py" \
      "$SRC/colbert_rerank.py" "$SRC/router.py" "$SRC/tables.py" \
      "$SRC/query_log.py" "$SRC/ocr.py" "$BASE/"
# static/ directory is served by api.py FastAPI — must be co-located with api.py
mkdir -p "$BASE/static"
if [ -d "$SRC/static" ]; then
  cp -f "$SRC/static/"* "$BASE/static/" 2>/dev/null || true
fi

# --- venv ---
if [ ! -d "$BASE/venv" ]; then
  python3 -m venv "$BASE/venv"
fi
. "$BASE/venv/bin/activate"
pip install --upgrade pip wheel >/dev/null
pip install -r "$BASE/requirements.txt"

# --- Qdrant config for isolated instance ---
cat > "$BASE/qdrant-cbic.yaml" <<YAML
log_level: INFO
storage:
  storage_path: $QSTORE
  snapshots_path: $QSTORE/snapshots
service:
  host: 0.0.0.0
  http_port: $QPORT
  grpc_port: $((QPORT+1))
  enable_cors: true
cluster:
  enabled: false
YAML

# --- start qdrant-cbic (foreground? use nohup + pidfile) ---
if ! ss -tln | awk '{print $4}' | grep -q ":$QPORT\$"; then
  nohup "$QDRANT_BIN" --config-path "$BASE/qdrant-cbic.yaml" \
        > "$BASE/qdrant-cbic.log" 2>&1 &
  echo $! > "$BASE/qdrant-cbic.pid"
  echo "[deploy] launched qdrant-cbic on :$QPORT pid=$(cat $BASE/qdrant-cbic.pid)"
  sleep 4
else
  echo "[deploy] qdrant-cbic already listening on :$QPORT"
fi

# --- sanity ---
curl -fsS "http://127.0.0.1:$QPORT/collections" >/dev/null \
  && echo "[deploy] qdrant-cbic reachable" \
  || { echo "[deploy] ERROR: qdrant-cbic not reachable"; exit 1; }

# --- start api ---
cat > "$BASE/start_api.sh" <<EOF
#!/bin/bash
cd "$BASE"
. venv/bin/activate
export QDRANT_URL=http://127.0.0.1:$QPORT
export QDRANT_COLL=cbic_v1
export EMBED_MODEL=BAAI/bge-m3
export RETRIEVER_GPU=6
export LITELLM_URL=http://127.0.0.1:4444
export LLM_MODEL=\${LLM_MODEL:-qwen3-14b-hermes}
export PORT=$APIPORT
exec python3 api.py
EOF
chmod +x "$BASE/start_api.sh"

if ! pgrep -f "python3 api.py" >/dev/null; then
  nohup "$BASE/start_api.sh" > "$BASE/api.log" 2>&1 &
  echo $! > "$BASE/api.pid"
  echo "[deploy] launched cbic-rag API on :$APIPORT pid=$(cat $BASE/api.pid)"
else
  echo "[deploy] cbic-rag API already running"
fi

# --- kick off ingest in a screen session (uses GPUs 0..6) ---
cat > "$BASE/run_ingest.sh" <<EOF
#!/bin/bash
cd "$BASE"
. venv/bin/activate
export QDRANT_URL=http://127.0.0.1:$QPORT
export QDRANT_COLL=cbic_v1
export EMBED_MODEL=BAAI/bge-m3
export GPU_IDS=0,1,2,3,4,5,6
export EMBED_BATCH=32
python3 -u ingest.py --resume 2>&1 | tee -a ingest.log
EOF
chmod +x "$BASE/run_ingest.sh"

if command -v screen >/dev/null 2>&1; then
  if ! screen -ls | grep -q cbic-ingest; then
    screen -dmS cbic-ingest bash -c "$BASE/run_ingest.sh"
    echo "[deploy] ingest running in screen 'cbic-ingest' (screen -r cbic-ingest)"
  else
    echo "[deploy] ingest screen already active"
  fi
else
  nohup "$BASE/run_ingest.sh" > "$BASE/ingest.log" 2>&1 &
  echo "[deploy] ingest running in background pid=$!"
fi

cat <<FIN

==============================================================
cbic RAG deployed.
  Qdrant (new instance):  http://127.0.0.1:$QPORT
  API:                    http://127.0.0.1:$APIPORT
  Storage:                $QSTORE
  Logs:                   $BASE/{qdrant-cbic,api,ingest}.log

Test the API:
  curl -X POST http://127.0.0.1:$APIPORT/query \\
       -H 'Content-Type: application/json' \\
       -d '{"question":"What is the time limit to claim ITC on capital goods?"}'

Wire into Open WebUI:
  Settings → Connections → OpenAI → + Add
    Base URL: http://<rig-ip>:$APIPORT/v1
    API Key:  anything
    Model:    cbic-rag

Wire into LiteLLM (optional, for unified auth):
  Edit /app/config.yaml, add under model_list:
    - model_name: cbic-rag
      litellm_params:
        model: openai/cbic-rag
        api_base: http://127.0.0.1:$APIPORT/v1
        api_key: anything
==============================================================
FIN
