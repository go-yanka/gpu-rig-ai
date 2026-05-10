#!/usr/bin/env bash
# run_training.sh — one-shot training on the RunPod pod.
# Assumes setup.sh has run and uploaded files are in /workspace/cbic/.
set -euo pipefail

WORKDIR=/workspace/cbic
OUT="$WORKDIR/bge-m3-cbic-v1"
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export TOKENIZERS_PARALLELISM=false

cd "$WORKDIR"

t0=$(date +%s)
echo "[run] starting at $(date -Is)"

# --- 1. Prep train/val splits from curated pairs ---
# prep_pairs.py expects the Gemini schema (records with .questions[]).
# curated_pairs.jsonl already IS the (query, positive, chunk_id, ...) flat form,
# so we copy it straight to train.jsonl and carve a tiny val split for safety.
if [ -f curated_pairs.jsonl ]; then
  echo "[run] building train/val from curated_pairs.jsonl"
  python - <<'PY'
import json, random, pathlib
rows = [json.loads(l) for l in open("curated_pairs.jsonl", encoding="utf-8") if l.strip()]
# group by chunk_id to prevent leakage
by = {}
for r in rows:
    by.setdefault(str(r.get("chunk_id", r.get("query"))), []).append(r)
keys = sorted(by); random.Random(42).shuffle(keys)
n_val = max(1, int(round(len(keys) * 0.05)))
val_keys = set(keys[:n_val])
train, val = [], []
for k, v in by.items():
    (val if k in val_keys else train).extend(v)
with open("train.jsonl", "w", encoding="utf-8") as f:
    for r in train: f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open("val.jsonl", "w", encoding="utf-8") as f:
    for r in val: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[prep] train={len(train)}  val={len(val)}  chunks={len(by)}")
PY
else
  echo "[run] ERROR: curated_pairs.jsonl not found"; exit 1
fi

t1=$(date +%s)
echo "[run] prep done in $((t1-t0))s"

# --- 2. Fine-tune ---
GOLD_ARG=""
if [ -f gold.jsonl ]; then GOLD_ARG="--gold gold.jsonl"; fi

HN_ARG=""
if [ -f hard_negatives.jsonl ]; then HN_ARG="--hard-negatives hard_negatives.jsonl"; fi

python finetune_bge_m3.py \
    --train train.jsonl \
    --val val.jsonl \
    $HN_ARG \
    $GOLD_ARG \
    --base-model BAAI/bge-m3 \
    --out "$OUT" \
    --epochs 3 \
    --batch-size 32 \
    --lr 2e-5 \
    --max-seq-length 512 \
    --warmup-ratio 0.1 \
    --fp16

t2=$(date +%s)
echo "[run] training done in $((t2-t1))s  (wall $((t2-t0))s)"

# --- 3. Pack for download ---
cd "$WORKDIR"
tar --use-compress-program=pigz -cf bge-m3-cbic-v1.tar.gz bge-m3-cbic-v1
sha256sum bge-m3-cbic-v1.tar.gz | tee bge-m3-cbic-v1.tar.gz.sha256
ls -lh bge-m3-cbic-v1.tar.gz

t3=$(date +%s)
echo "[run] tar done in $((t3-t2))s"
echo "[run] TOTAL wall-clock: $((t3-t0))s"
echo "[run] Download with:  scp -P <port> root@<pod-ip>:$WORKDIR/bge-m3-cbic-v1.tar.gz ."
