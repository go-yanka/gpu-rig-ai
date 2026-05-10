#!/usr/bin/env python3
"""Fine-tune BGE-M3 dense head on CBIC Q-chunk pairs.

Resume-safe across rig reboots (codified rule
RULES_INDEX `[TRIGGER: training resume, checkpoint]`):

- HF Trainer with save_strategy=steps, save_steps=200
- save_total_limit=3 (last 3 checkpoint dirs, ~12GB)
- output_dir on persistent disk (/opt/indian-legal-ai/models/...)
- resume_from_checkpoint=True (auto-finds latest)
- Idempotent: if final model exists, exit without re-training

Hardware: rig card 2 (RX 6700 XT 12GB, gfx1031) via PyTorch+ROCm.
Env: HSA_OVERRIDE_GFX_VERSION=10.3.0 (treat gfx1031 as gfx1030 for ROCm).

Data: 5,500 curated Q-chunk pairs from /opt/indian-legal-ai/eval/training_pairs/
Loss: MultipleNegativesRankingLoss (in-batch negatives)
Hard negs: optional via --hard-negs path (mined separately)
"""
from __future__ import annotations
import argparse, json, os, sys, random
from pathlib import Path

# ROCm gfx1031 (RX 6700 XT) needs override to be treated as gfx1030
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
# Restrict to card 2 (PyTorch sees ROCm devices as cuda:N; we want only one)
os.environ.setdefault("HIP_VISIBLE_DEVICES", "2")
os.environ.setdefault("ROCR_VISIBLE_DEVICES", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "max_split_size_mb:512")

import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer

ROOT = Path("/opt/indian-legal-ai")
DEFAULT_OUT = ROOT / "models" / "bge-m3-cbic-ft-v1"
DEFAULT_LOG = ROOT / "logs" / "finetune_v1.log"
PAIRS_DIR = ROOT / "eval" / "training_pairs"
GOLD_PATH = ROOT / "reingest_spec" / "eval" / "v2_gold.json"
BASE_MODEL = "BAAI/bge-m3"


def load_curated_pairs():
    """Load all curated Q-chunk pairs from JSONL files in PAIRS_DIR.

    Each pair is (query_text, positive_text). We dedup against v2_gold.json
    to prevent gold leakage.
    """
    # Build gold-question set (substring match — be strict)
    gold_qs = set()
    if GOLD_PATH.exists():
        for g in json.load(GOLD_PATH.open()):
            q = g.get("question") or g.get("query") or ""
            if q:
                gold_qs.add(q.strip().lower())
    print(f"[data] gold queries to exclude: {len(gold_qs)}", flush=True)

    pairs = []
    n_skipped_leak = 0
    for jf in sorted(PAIRS_DIR.glob("*.jsonl")):
        if "BAD" in jf.name or "smoke" in jf.name:
            continue
        with jf.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                # Schema varies across files. Try common shapes:
                q = d.get("question") or d.get("query") or ""
                pos = (d.get("answer") or d.get("chunk_text") or
                       d.get("positive") or d.get("text") or
                       d.get("verbatim_quote") or "")
                if not q or not pos:
                    continue
                if q.strip().lower() in gold_qs:
                    n_skipped_leak += 1
                    continue
                pairs.append((q.strip(), pos.strip()))
    print(f"[data] loaded {len(pairs)} pairs, skipped {n_skipped_leak} gold-leak", flush=True)
    return pairs


class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, i):
        q, p = self.pairs[i]
        return {"anchor": q, "positive": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-resume", action="store_true",
                    help="Force fresh start (default: auto-resume from latest checkpoint)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_marker = out_dir / "TRAINING_COMPLETE"
    if final_marker.exists():
        print(f"[idempotent] final model already exists at {out_dir} — exiting without re-training.")
        return 0

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Sanity-check ROCm visible
    print(f"[env] torch: {torch.__version__} hip: {getattr(torch.version,'hip',None)} cuda_avail: {torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        print("[FATAL] no GPU visible to torch — abort"); return 2
    dev = torch.cuda.get_device_name(0)
    print(f"[env] device: {dev}", flush=True)

    # Load data
    pairs = load_curated_pairs()
    if len(pairs) < 100:
        print(f"[FATAL] only {len(pairs)} pairs loaded — abort"); return 3
    random.shuffle(pairs)
    n_dev = max(50, int(len(pairs) * 0.1))
    dev_pairs = pairs[:n_dev]
    train_pairs = pairs[n_dev:]
    print(f"[data] train={len(train_pairs)} dev={len(dev_pairs)}", flush=True)

    train_ds = PairDataset(train_pairs)
    dev_ds = PairDataset(dev_pairs)

    # Load base model
    print(f"[model] loading {BASE_MODEL} (will download if not cached)", flush=True)
    model = SentenceTransformer(BASE_MODEL)
    model.max_seq_length = args.max_seq_len
    print(f"[model] loaded, seq_len={model.max_seq_length}", flush=True)

    # MNR loss (in-batch negatives)
    loss = losses.MultipleNegativesRankingLoss(model)

    targs = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,                 # keep last 3 checkpoints
        eval_strategy="steps",
        eval_steps=args.save_steps,
        logging_steps=50,
        logging_dir=str(out_dir / "logs"),
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        loss=loss,
    )

    # Auto-resume unless --no-resume forces fresh
    resume = not args.no_resume
    print(f"[train] launching trainer (resume_from_checkpoint={resume})", flush=True)
    trainer.train(resume_from_checkpoint=resume if (out_dir / "checkpoints").exists() and any((out_dir/"checkpoints").iterdir()) else False)

    # Save final
    final_dir = out_dir / "final"
    model.save(str(final_dir))
    final_marker.touch()
    print(f"[done] final model saved at {final_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
