# BGE-M3 Domain Adaptation for CBIC

Fine-tune `BAAI/bge-m3` on CBIC (Indian indirect-tax law) query/chunk pairs to
fix the catastrophic recall@5 = 12.35% measured against the out-of-domain base
model.

## Chosen approach

**sentence-transformers + MultipleNegativesRankingLoss (MNRL)** with in-batch
negatives.

Why this over the alternatives:

| Option | Verdict | Reason |
| --- | --- | --- |
| **sentence-transformers MNRL** | **Picked** | Canonical recipe for (q, pos) pair data. Stable, well-documented, debuggable. Produces a vanilla SentenceTransformer dir that drops straight into the existing embedder. |
| FlagEmbedding `finetune.py` | Rejected for v1 | More machinery (HN mining, teacher distill), heavier schema, harder to debug on ROCm. Revisit for v2 once we have HN mining. |
| LoRA adapter on BGE-M3 | Rejected | Overkill for single-domain adapt; full FT of 568M params fits in 24 GB. Adapters also need a custom serving path (PEFT merge) we don't currently have. |

## Files

```
training/
├── prep_pairs.py          # JSONL (gen_training_pairs.py output) -> train/val.jsonl
├── finetune_bge_m3.py     # MNRL training harness
├── README.md              # this file
├── train.jsonl            # (generated)
├── val.jsonl              # (generated)
├── stats.json             # (generated)
└── bge-m3-cbic-v1/        # (generated model dir, scp this to rig)
```

## End-to-end workflow

### 1. Wait for pair generation (Gemini 2.5 Pro, running)

```bash
# Background: scripts/gen_training_pairs.py writes to
#   D:\_gpu_rig_ai\eval\training_pairs\pairs_2000_20260422.jsonl
# ~5-10 hours for 2000 chunks, yielding ~4000-6000 (q, chunk) pairs
tail -f D:/_gpu_rig_ai/eval/training_pairs/gen_2000.log
wc -l D:/_gpu_rig_ai/eval/training_pairs/pairs_2000_20260422.jsonl
```

### 2. Prep splits

`prep_pairs.py` has two modes (auto-detected):

**Raw mode** (merge Gemini + Sonnet-HC + Claude-Opus raw pair files):

```bash
cd D:/_gpu_rig_ai/training
python prep_pairs.py --mode raw \
    --in ../eval/training_pairs/pairs_2000_20260422.jsonl \
         ../eval/training_pairs/pairs_opus_highcomplex.jsonl \
         ../eval/training_pairs/pairs_claude_opus.jsonl \
    --out-dir .
```

**Curated mode** (preferred for v1 run — reads `curated_pairs.jsonl` from the
curator and optionally merges hard negatives):

```bash
python prep_pairs.py --mode curated \
    --in curated_pairs.jsonl \
    --hard-negatives hard_negatives.jsonl \
    --out-dir .
```

Prints stats and writes `train.jsonl`, `val.jsonl`, `stats.json`.
Split is **by chunk_id**, not by question — no leakage. 95/5 train/val. When
hard-negs are supplied, output rows include a `negative` field so
`finetune_bge_m3.py` builds MNRL triplets.

Note: the HIGH-complex Sonnet file uses key `"question"` (not `"q"`);
`prep_pairs.py` handles both.

### 3. Train

Local (Windows laptop, CPU) smoke pass:

```bash
python finetune_bge_m3.py \
    --train train.jsonl --val val.jsonl \
    --out bge-m3-cbic-v1 \
    --epochs 1 --batch-size 8 \
    --dry-run         # just load model + data, verify no errors
```

Real training on rig or cloud (see "Where to train" below):

```bash
python finetune_bge_m3.py \
    --train train.jsonl --val val.jsonl \
    --gold-yaml ../eval/gold_set.yaml \
    --chunk-corpus cbic_chunks.jsonl \
    --out bge-m3-cbic-v1 \
    --epochs 3 --batch-size 32 --lr 2e-5 --fp16
```

`--gold-yaml` + `--chunk-corpus` together enable the real metric: per-epoch
recall@{5,10,20} + mrr@10 on the 170-item gold set, evaluated against the
full chunk corpus. Watch `bge-m3-cbic-v1/eval_gold.jsonl` and
`bge-m3-cbic-v1/eval/Information-Retrieval_evaluation_cbic-gold_results.csv`.

`cbic_chunks.jsonl` must be produced separately by scrolling Qdrant with
fields `{chunk_id, text, section_ref, category, doc_number, doc_type}`
(trivial ~10-line script — not yet in the repo).

### 4. Deploy to rig

```bash
# from laptop
scp -r bge-m3-cbic-v1 rig:/tmp/bge-m3-cbic-v1
ssh rig 'sudo mv /tmp/bge-m3-cbic-v1 /opt/ai-models/bge-m3-cbic-v1 && sudo chown -R ai-models:ai-models /opt/ai-models/bge-m3-cbic-v1'

# Update embedder config to point at the new model path.
# The llama-embedding-server instances use GGUF, so for the fine-tuned model
# we have two options:
#  (a) Quickest: run a parallel sentence-transformers HTTP endpoint on rig
#      (FastAPI + st.encode) on one GPU, swap the ingest + query path to it.
#  (b) Convert the fine-tuned model to GGUF with `convert-hf-to-gguf.py`
#      from llama.cpp, then reuse the existing llama-embedding-server wrapper.
#
# Decision: ship (a) first for speed, then do (b) once recall is proven.
```

### 5. Re-ingest + re-audit

```bash
# re-embed the cbic_v1 Qdrant collection into cbic_v2 using the new model
ssh rig 'cd /opt/cbic && python tools/reembed_collection.py --src cbic_v1 --dst cbic_v2 --embedder http://127.0.0.1:<new-port>'

# re-run recall audit
cd D:/_gpu_rig_ai/eval
python recall_audit_rig.py --collection cbic_v2 --out recall_audit_cbic_v2.jsonl
python run_eval.py --run recall_audit_cbic_v2.jsonl
```

Expected: recall@5 12.35% -> 40-65% with 4-6k pairs and 3 epochs. If the jump is
<20 points, mine hard negatives (top-5 wrong retrievals per query) and re-train.

## Estimated training time (2000 chunks -> ~5000 pairs, 3 epochs, bs=32)

| Hardware | Steps/epoch | Minutes/epoch | Total (3 epochs) | Cost |
| --- | --- | --- | --- | --- |
| RunPod A100 40GB | 156 | 4-6 min | **15-20 min** | ~$0.50 |
| RunPod RTX 4090 24GB | 156 | 8-12 min | **25-40 min** | ~$0.30 |
| rig RX 6700 XT (ROCm, fp16) | 156 | 25-45 min (if stable) | **1.5-2.5 h** | free |
| Laptop CPU | 156 | 2-4 h | **8-14 h** | free |

## Where to train

**Primary: RunPod A100 40GB** — ~$0.50 for the full run, and the stack (CUDA
12, torch 2.3, sentence-transformers) is known-good. Flow:

1. Spin up a PyTorch pod.
2. `pip install sentence-transformers==3.0.*`
3. `scp train.jsonl val.jsonl finetune_bge_m3.py pod:/workspace/`
4. `python finetune_bge_m3.py ... --fp16`
5. `scp -r pod:/workspace/bge-m3-cbic-v1 ./`

**Fallback: rig ROCm.** Works but fragile. Use `--fp16` (not bf16 — RDNA2
doesn't have real bf16). If OOM, drop `--batch-size` to 16. If `torch` ROCm
segfaults during `model.fit`, fall back to CPU training for a one-off pass.

**Last resort: laptop CPU.** Set `--batch-size 8 --epochs 1` just to validate
the pipeline end-to-end before spending money.

## Python dependencies

```
pip install "sentence-transformers>=3.0,<4" "torch>=2.2" "transformers>=4.40"
```

FlagEmbedding is NOT required for this MNRL path; only install it if we
graduate to the FlagEmbedding trainer in v2.

## Smoke-test log

`prep_pairs.py` was smoke-tested on 2026-04-22 with the first 10 records in
`pairs_2000_20260422.jsonl`. See `stats.json` after running.

## Roadmap to v2 (post-recall-check)

1. Hard-negative mining: for each train query, run the v1 model against the full
   corpus, take top-5 non-gold hits, feed as explicit negatives via
   `MultipleNegativesRankingLoss` with triplets `(q, pos, neg)`.
2. Contrastive distillation from a reranker (bge-reranker-v2-m3) for pair
   re-weighting.
3. Rebuild on the full 108k-chunk corpus after pairs_10k dataset lands.
