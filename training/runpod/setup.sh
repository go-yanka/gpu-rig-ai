#!/usr/bin/env bash
# setup.sh — idempotent pod bootstrap for BGE-M3 fine-tune.
# Run after SSH'ing into the RunPod pod.
#
# Safe to re-run: every step checks before acting.
set -euo pipefail

WORKDIR=/workspace/cbic
HF_CACHE=/workspace/hf_cache
MODEL_ID="BAAI/bge-m3"

echo "[setup] starting at $(date -Is)"

mkdir -p "$WORKDIR" "$HF_CACHE"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_ENABLE_HF_TRANSFER=1

# --- 1. System deps (almost always already present in the pytorch template) ---
if ! command -v git >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq git wget pigz
fi
command -v pigz >/dev/null 2>&1 || apt-get install -y -qq pigz

# --- 2. Python deps (pin versions; check before install) ---
need_install=0
python -c "import sentence_transformers, sys; sys.exit(0 if sentence_transformers.__version__.startswith('3.0') else 1)" 2>/dev/null || need_install=1
python -c "import datasets, sys; sys.exit(0 if datasets.__version__.startswith('2.21') else 1)" 2>/dev/null || need_install=1
python -c "import accelerate, sys; sys.exit(0 if accelerate.__version__.startswith('0.34') else 1)" 2>/dev/null || need_install=1
python -c "import hf_transfer" 2>/dev/null || need_install=1

if [ "$need_install" = "1" ]; then
  echo "[setup] installing Python deps"
  pip install --quiet -U \
    "sentence-transformers==3.0.1" \
    "datasets==2.21.0" \
    "accelerate==0.34.2" \
    "hf_transfer==0.1.8"
else
  echo "[setup] Python deps already pinned, skipping"
fi

# --- 3. Prime HF cache with BGE-M3 base (so training starts fast) ---
if [ ! -d "$HF_CACHE/hub/models--BAAI--bge-m3" ]; then
  echo "[setup] downloading $MODEL_ID into $HF_CACHE"
  python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="BAAI/bge-m3",
    cache_dir=os.environ["HF_HOME"] + "/hub",
    allow_patterns=["*.json", "*.txt", "*.model", "sentencepiece.bpe.model",
                    "tokenizer*", "pytorch_model.bin", "model.safetensors",
                    "1_Pooling/*", "config_sentence_transformers.json",
                    "modules.json", "sentence_bert_config.json"],
)
print("[setup] bge-m3 cached")
PY
else
  echo "[setup] bge-m3 already in cache, skipping download"
fi

# --- 4. GPU sanity check ---
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available on this pod!"
print(f"[gpu] {torch.cuda.get_device_name(0)}  vram={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
print(f"[gpu] torch={torch.__version__}  cuda={torch.version.cuda}")
PY

cd "$WORKDIR"
echo "[setup] done at $(date -Is). Workdir: $WORKDIR"
echo "[setup] next: upload curated_pairs.jsonl + hard_negatives.jsonl + finetune_bge_m3.py + prep_pairs.py + run_training.sh"
