#!/usr/bin/env bash
# =============================================================================
# Indian Legal AI — Overnight Setup Script v2
# - Downloads all datasets
# - Starts Qdrant
# - Builds RAG index using llama-server /v1/embeddings (already running)
# - Starts RAG API
# Run: sudo bash /opt/indian-legal-ai/scripts/overnight_setup.sh
# Log: /opt/indian-legal-ai/logs/overnight.log
# =============================================================================

set -euo pipefail

WORK_DIR="/opt/indian-legal-ai"
LOG="$WORK_DIR/logs/overnight.log"
DATASETS="$WORK_DIR/datasets"
SCRIPTS="$WORK_DIR/scripts"
RAG_DIR="$WORK_DIR/rag"
MODELS_DIR="$WORK_DIR/models"

mkdir -p "$DATASETS" "$SCRIPTS" "$RAG_DIR" "$MODELS_DIR" "$WORK_DIR/logs"
exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "======================================================"
log " Indian Legal AI — Overnight Setup v2 Starting"
log "======================================================"

# ===========================================================================
# PHASE 1 — Verify dependencies (already installed)
# ===========================================================================
log ""
log "PHASE 1: Verifying Python dependencies..."
python3 -c "import huggingface_hub, datasets, qdrant_client, fastapi, uvicorn, requests, tqdm, pandas; print('All OK')"
log "Dependencies OK."

# ===========================================================================
# PHASE 2 — Download datasets from HuggingFace
# ===========================================================================
log ""
log "PHASE 2: Downloading datasets from HuggingFace..."
log "This will take 2-6 hours depending on internet speed."
log "Total: ~35GB for core datasets (skipping 50GB NyayaAnumana for now)"

python3 /opt/indian-legal-ai/scripts/download_datasets.py
log "Dataset downloads complete."

# ===========================================================================
# PHASE 3 — Download Indian Legal 8B GGUF model
# ===========================================================================
log ""
log "PHASE 3: Downloading Indian Legal 8B GGUF model (4.92GB)..."

python3 << 'PYEOF'
import os
from huggingface_hub import hf_hub_download
import datetime

def log(m): print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)

dest = "/opt/indian-legal-ai/models/indian-legal-8b-q4_k_m.gguf"
if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000_000:
    log(f"Model already exists ({os.path.getsize(dest)//1024//1024}MB), skipping.")
else:
    log("Downloading Ambuj-Tripathi-Llama-3.1-8B-IndianLegal-GGUF...")
    try:
        path = hf_hub_download(
            repo_id="invincibleambuj/Ambuj-Tripathi-Llama-3.1-8B-IndianLegal-GGUF",
            filename="meta-llama-3.1-8b-instruct.Q4_K_M.gguf",
            local_dir="/opt/indian-legal-ai/models",
            local_dir_use_symlinks=False,
        )
        log(f"Downloaded: {path}")
    except Exception as e:
        log(f"WARN: Model download failed — {e}")
        log("Continuing without it — will use gemma4 as LLM brain instead.")
PYEOF

# ===========================================================================
# PHASE 4 — Scrape CBIC GST FAQs
# ===========================================================================
log ""
log "PHASE 4: Scraping CBIC GST FAQs..."
python3 /opt/indian-legal-ai/scripts/scrape_gst_faqs.py
log "GST FAQ scraping done."

# ===========================================================================
# PHASE 5 — Start Qdrant
# ===========================================================================
log ""
log "PHASE 5: Starting Qdrant vector database..."

if curl -s http://localhost:6333/healthz >/dev/null 2>&1; then
    log "Qdrant already running."
else
    log "Starting Qdrant via Docker..."
    docker run -d \
        --name qdrant \
        --restart unless-stopped \
        -p 6333:6333 \
        -p 6334:6334 \
        -v /opt/indian-legal-ai/rag/qdrant_storage:/qdrant/storage \
        qdrant/qdrant:latest

    log "Waiting for Qdrant to start..."
    for i in {1..20}; do
        if curl -s http://localhost:6333/healthz >/dev/null 2>&1; then
            log "Qdrant is up."
            break
        fi
        sleep 3
    done
fi

# Final check
if ! curl -s http://localhost:6333/healthz >/dev/null 2>&1; then
    log "ERROR: Qdrant failed to start. Check docker logs qdrant"
    exit 1
fi

# ===========================================================================
# PHASE 6 — Build RAG index
# ===========================================================================
log ""
log "PHASE 6: Building RAG index..."
log "Uses llama-server /v1/embeddings — no extra GPU needed."
python3 /opt/indian-legal-ai/scripts/build_rag_index.py
log "RAG index complete."

# ===========================================================================
# PHASE 7 — Start RAG API
# ===========================================================================
log ""
log "PHASE 7: Starting RAG API server on port 7000..."
pkill -f rag_api.py 2>/dev/null || true
sleep 1
nohup python3 /opt/indian-legal-ai/scripts/rag_api.py \
    > /opt/indian-legal-ai/logs/rag_api.log 2>&1 &

sleep 5
if curl -s http://localhost:7000/health >/dev/null 2>&1; then
    log "RAG API running on http://192.168.1.107:7000"
else
    log "WARN: RAG API slow to start. Check /opt/indian-legal-ai/logs/rag_api.log"
fi

# ===========================================================================
# PHASE 8 — Smoke test
# ===========================================================================
log ""
log "PHASE 8: Running smoke test..."
python3 << 'PYEOF'
import urllib.request, json, datetime

def log(m): print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)

test_q = "What is the deduction limit under Section 80C of the Income Tax Act?"
log(f"Test question: {test_q}")

try:
    data = json.dumps({"question": test_q}).encode()
    req = urllib.request.Request("http://localhost:7000/ask", data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=120) as r:
        result = json.loads(r.read())
    log(f"Answer received ({len(result['answer'])} chars)")
    log(f"Sources used: {len(result['sources'])}")
    log(f"Answer preview: {result['answer'][:300]}...")
    log("SMOKE TEST PASSED")
except Exception as e:
    log(f"SMOKE TEST WARN: {e}")
    log("RAG may still be indexing — check logs and retry.")
PYEOF

# ===========================================================================
# DONE
# ===========================================================================
log ""
log "======================================================"
log " SETUP COMPLETE"
log "======================================================"
log " RAG API:    http://192.168.1.107:7000"
log " API Docs:   http://192.168.1.107:7000/docs"
log " WebUI:      http://192.168.1.107:3000"
log " Dashboard:  http://192.168.1.107:8080"
log ""
log " Test:"
log "   curl -X POST http://192.168.1.107:7000/ask \\"
log "     -H 'Content-Type: application/json' \\"
log "     -d '{\"question\": \"What is Section 80C deduction limit?\"}'"
log "======================================================"
