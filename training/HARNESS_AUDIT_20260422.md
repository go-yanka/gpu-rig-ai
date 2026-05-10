# BGE-M3 Fine-tune Harness Audit — 2026-04-22

## TL;DR
The original harness was written for the Gemini raw-pairs schema only. It
silently dropped all Sonnet HIGH-complex questions (key mismatch), ignored
the curator and hard-negs pipeline, and never ran the IR evaluator against
`gold_set.yaml`. Patched end-to-end; ready for RunPod.

## What was wrong

### `prep_pairs.py`
1. Only read `qobj["q"]`. Sonnet HIGH-complex (`pairs_opus_highcomplex.jsonl`)
   uses `qobj["question"]` → every HIGH-complex pair was silently dropped.
2. Single `--in` input; couldn't merge the three raw pair files (Gemini +
   Sonnet HC + Claude Opus) in one pass.
3. No support for the curator output (`curated_pairs.jsonl`) schema
   (`question`, `positive_chunk_id`, `chunk_text`, `qa_scores`, …).
4. No hard-negatives path at all.

### `finetune_bge_m3.py`
1. Had a `--gold` flag but never used it.
2. Assumed `--gold` would be JSONL pair format; our gold is **YAML** with
   `expected_sections` / `expected_rules` / `expected_notifications`, no
   direct `chunk_id`.
3. `evaluator` only ran on `val.jsonl` (tiny, leakage-controlled, but not
   the real target). Gold-set recall was never measured during training.
4. Per-epoch callback didn't log metrics anywhere — only final loss visible.

### `README.md`
Assumed single Gemini JSONL input; no mention of curator/HN or gold eval.

## What was fixed

### `prep_pairs.py` (rewritten)
- `_extract_question` accepts both `q` and `question` keys.
- `--mode {auto,raw,curated}` with auto-detect from first record shape.
- `--in` now `nargs="+"` so all three raw files can be merged.
- New `flatten_curated()` for curator schema.
- New `attach_hard_negatives()` that reads the HN JSONL
  (`{question, positive_chunk_id, hard_negs:[{chunk_id,text_snippet}]}`) and
  emits one triplet row per negative (key `negative` in output).
- Stats now include `n_triplets`, `n_pairs_only`, `by_complexity`.
- Paths/args via CLI and env vars (`HARD_NEGATIVES`, `PREP_OUT_DIR`).
- Cross-file dedup by normalized query hash.

### `finetune_bge_m3.py` (rewritten)
- Detects optional `negative` field per row → builds MNRL triplets
  (InputExample with 3 texts). MNRL in sentence-transformers 3.x natively
  treats the 3rd text as an extra hard negative on top of in-batch negatives.
- New `--gold-yaml` + `--chunk-corpus` flags. Builds an
  `InformationRetrievalEvaluator` over the full Qdrant chunk dump by mapping
  `expected_sections` / `expected_rules` (regex-extracted tokens) and
  `expected_notifications` (doc_number match) to `chunk_id`s. Items with
  zero matches are dropped with a warning count.
- IR evaluator reports `accuracy@k`, `precision_recall@k`, `mrr@10`,
  `ndcg@10` for `k ∈ {5,10,20}`.
- `SequentialEvaluator` runs both val and gold evaluators each epoch.
- Per-epoch callback appends `{epoch, steps, score}` to
  `<out>/eval_gold.jsonl` AND prints to stdout, so RunPod logs show recall
  trending, not just loss.
- All paths parameterizable via env: `TRAIN_JSONL`, `VAL_JSONL`,
  `GOLD_YAML`, `CHUNK_CORPUS`, `OUT_DIR`, `BASE_MODEL`, `EPOCHS`,
  `BATCH_SIZE`, `LR`.

## What remains questionable / needs the user's attention

1. **Gold → chunk_id mapping is heuristic.** `_SECTION_TOKEN` regex matches
   things like `10(1)(a)`, `12(7)`, `54`. It works for the gst_pos_* style
   `expected_sections: ["10(1)(a) IGST"]` but may miss exotic formats
   (e.g., purely prose refs). Run the mapping once locally and check the
   "dropped" count printed by `build_gold_ir_inputs`; if >20/170, widen the
   regex or add per-item overrides.
2. **Chunk corpus dump doesn't exist yet.** Need a one-shot script that
   scrolls Qdrant `cbic_v1` and writes JSONL with fields
   `{chunk_id, text, section_ref, category, doc_number, doc_type}`.
   This is a separate 10-line script and is NOT in this audit's scope.
3. **Hard-negs miner** isn't written yet (per plan, it's a downstream
   deliverable). Harness accepts its output format as spec'd — no change
   needed on this side when it lands.
4. **Sonnet HC complexity field** is uppercase `"HIGH"`; `prep_pairs.py`
   lowercases via `.lower()` so stats merge correctly. Spot-check
   `stats.json → by_complexity` shows `{high, medium, low, ?}` after merge.

## How to run it on RunPod

```bash
# 0. Pod setup (PyTorch 2.3 / CUDA 12 image)
pip install "sentence-transformers>=3.0,<4" "torch>=2.2" "transformers>=4.40" pyyaml

# 1. Upload (from laptop)
scp -r D:/_gpu_rig_ai/training/{prep_pairs.py,finetune_bge_m3.py} \
       D:/_gpu_rig_ai/eval/gold_set.yaml \
       D:/_gpu_rig_ai/eval/training_pairs/pairs_*.jsonl \
       pod:/workspace/

# 2a. (Interim) Build splits from the 3 raw pair files:
cd /workspace
python prep_pairs.py --mode raw \
    --in pairs_2000_20260422.jsonl pairs_opus_highcomplex.jsonl pairs_claude_opus.jsonl \
    --out-dir .

# 2b. (Preferred once curator + HN are ready):
python prep_pairs.py --mode curated \
    --in curated_pairs.jsonl \
    --hard-negatives hard_negatives.jsonl \
    --out-dir .

# 3. Dump chunk corpus from rig's Qdrant (one-shot, separate script;
#    needed by --chunk-corpus). scp the resulting JSONL to /workspace/.

# 4. Smoke-test harness without spending GPU time:
python finetune_bge_m3.py --dry-run \
    --train train.jsonl --val val.jsonl \
    --gold-yaml gold_set.yaml --chunk-corpus cbic_chunks.jsonl \
    --out bge-m3-cbic-v1

# 5. Real run (~15–20 min on A100 40GB, ~$0.50):
python finetune_bge_m3.py \
    --train train.jsonl --val val.jsonl \
    --gold-yaml gold_set.yaml --chunk-corpus cbic_chunks.jsonl \
    --out bge-m3-cbic-v1 \
    --epochs 3 --batch-size 32 --lr 2e-5 --fp16

# 6. Watch recall trend:
tail -f bge-m3-cbic-v1/eval_gold.jsonl
# Also check the sentence-transformers auto-written:
#   bge-m3-cbic-v1/eval/Information-Retrieval_evaluation_cbic-gold_results.csv

# 7. Pull model down:
scp -r pod:/workspace/bge-m3-cbic-v1 ./
```

Sentinel for success: `cbic-gold` `accuracy@5` > 0.40 by epoch 3 (baseline
was 0.1235). If it plateaus below 0.25, re-run with more HN (expand miner
top-k) before throwing more epochs at it.
