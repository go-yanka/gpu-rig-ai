"""
Fine-tune BGE-M3 on (query, positive) pair data or (query, positive, negative)
triplet data using sentence-transformers MultipleNegativesRankingLoss.

What changed in the 20260422 audit:
  - Training data may now contain an optional `negative` field (per row). When
    present, we build triplet InputExamples and MNRL treats the extra text as
    an additional hard negative on top of in-batch negatives.
  - New `--gold-yaml` flag: parses `D:/_gpu_rig_ai/eval/gold_set.yaml`
    (170 items with expected_sections/rules/notifications) and runs an
    InformationRetrievalEvaluator AGAINST A QDRANT CHUNK CORPUS per epoch,
    printing recall@5 / recall@10 / mrr@10.
  - Per-epoch callback logs metric-only lines to stdout so a RunPod operator
    sees recall trending, not just loss.

Inputs:
  train.jsonl / val.jsonl from prep_pairs.py
  Each line may be a pair:    {"query": "...", "positive": "..."}
  OR a triplet:                {"query": "...", "positive": "...", "negative": "..."}

Gold evaluation:
  --gold-yaml <path>            # gold_set.yaml (170 items)
  --chunk-corpus <path>         # JSONL dumped from Qdrant:
                                #   {"chunk_id": ..., "text": "...",
                                #    "section_ref": "...", "category": "...",
                                #    "doc_number": "...", "doc_type": "..."}

Outputs:
  <out-dir>/                    (final model)
  <out-dir>/checkpoints/        (per-epoch)
  <out-dir>/eval_gold.jsonl     (per-epoch evaluator scores)

Usage:
  python finetune_bge_m3.py \
      --train train.jsonl --val val.jsonl \
      --gold-yaml /workspace/gold_set.yaml \
      --chunk-corpus /workspace/cbic_chunks.jsonl \
      --out bge-m3-cbic-v1 \
      --epochs 3 --batch-size 32 --lr 2e-5 --fp16

Env vars honored: TRAIN_JSONL, VAL_JSONL, GOLD_YAML, CHUNK_CORPUS, OUT_DIR,
BASE_MODEL, EPOCHS, BATCH_SIZE, LR.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=os.environ.get("TRAIN_JSONL", "train.jsonl"))
    ap.add_argument("--val",   default=os.environ.get("VAL_JSONL",   "val.jsonl"))
    ap.add_argument("--gold-yaml", default=os.environ.get("GOLD_YAML"),
                    help="path to eval/gold_set.yaml (170 items)")
    ap.add_argument("--chunk-corpus", default=os.environ.get("CHUNK_CORPUS"),
                    help="JSONL dump of corpus chunks (chunk_id,text,section_ref,...)")
    ap.add_argument("--base-model", default=os.environ.get("BASE_MODEL", "BAAI/bge-m3"))
    ap.add_argument("--out", default=os.environ.get("OUT_DIR", "bge-m3-cbic-v1"))
    ap.add_argument("--epochs",     type=int,   default=int(os.environ.get("EPOCHS", "3")))
    ap.add_argument("--batch-size", type=int,   default=int(os.environ.get("BATCH_SIZE", "32")))
    ap.add_argument("--lr",         type=float, default=float(os.environ.get("LR", "2e-5")))
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--warmup-ratio",   type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--save-each-epoch", action="store_true", default=True)
    ap.add_argument("--gold-recall-k", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--dry-run", action="store_true",
                    help="load data + model, do NOT train")
    return ap.parse_args()


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------- Gold set -> IR evaluator ----------

_SECTION_TOKEN = re.compile(r"(\d+[A-Za-z]?(?:\(\d+\)(?:\([a-z]\))?)?)")


def _norm_token(s: str) -> str:
    return re.sub(r"\s+", "", s.lower())


def build_gold_ir_inputs(gold_yaml_path: Path, chunk_corpus_path: Path
                         ) -> tuple[dict, dict, dict]:
    """
    Parse gold_set.yaml, load chunk corpus, produce IR evaluator inputs:
       queries       = {qid: question_text}
       corpus        = {chunk_id_str: chunk_text}
       relevant_docs = {qid: {chunk_id_str, ...}}

    Gold item -> relevant chunks: any chunk whose `section_ref` contains a
    token from expected_sections/expected_rules, or whose `doc_number` matches
    expected_notifications. Items with zero matches are dropped.
    """
    import yaml
    gold = yaml.safe_load(gold_yaml_path.read_text(encoding="utf-8"))
    items = gold.get("items") or []

    corpus_records = load_jsonl(str(chunk_corpus_path))
    corpus: dict[str, str] = {}
    section_index: dict[str, set[str]] = {}
    docnum_index: dict[str, set[str]] = {}
    for rec in corpus_records:
        cid = str(rec.get("chunk_id"))
        if not cid or cid == "None":
            continue
        corpus[cid] = rec.get("text") or ""
        sec = rec.get("section_ref") or ""
        for tok in _SECTION_TOKEN.findall(sec):
            section_index.setdefault(_norm_token(tok), set()).add(cid)
        dn = rec.get("doc_number") or ""
        if dn:
            docnum_index.setdefault(_norm_token(dn), set()).add(cid)

    queries: dict[str, str] = {}
    relevant: dict[str, set[str]] = {}
    dropped = 0
    for it in items:
        qid = str(it.get("id"))
        qtext = (it.get("question") or "").strip()
        if not qid or not qtext:
            continue
        rel: set[str] = set()
        for s in it.get("expected_sections") or []:
            for tok in _SECTION_TOKEN.findall(str(s)):
                rel |= section_index.get(_norm_token(tok), set())
        for s in it.get("expected_rules") or []:
            for tok in _SECTION_TOKEN.findall(str(s)):
                rel |= section_index.get(_norm_token(tok), set())
        for n in it.get("expected_notifications") or []:
            rel |= docnum_index.get(_norm_token(str(n)), set())
        if not rel:
            dropped += 1
            continue
        queries[qid] = qtext
        relevant[qid] = rel

    print(f"[gold] mapped {len(queries)}/{len(items)} gold queries to chunks "
          f"({dropped} dropped). corpus_size={len(corpus)}")
    return queries, corpus, relevant


def main():
    args = parse_args()

    from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
    from torch.utils.data import DataLoader
    import torch

    train_rows = load_jsonl(args.train)
    val_rows   = load_jsonl(args.val)
    if not train_rows:
        print("ERROR: no training rows", file=sys.stderr); sys.exit(1)
    n_trip = sum(1 for r in train_rows if r.get("negative"))
    print(f"[data] train={len(train_rows)} ({n_trip} triplets, "
          f"{len(train_rows)-n_trip} pairs)  val={len(val_rows)}")

    train_examples = []
    for r in train_rows:
        q, p = r.get("query"), r.get("positive")
        if not q or not p:
            continue
        neg = r.get("negative")
        if neg:
            train_examples.append(InputExample(texts=[q, p, neg]))
        else:
            train_examples.append(InputExample(texts=[q, p]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}  (torch={torch.__version__})")
    if device == "cpu":
        print("[warn] training on CPU; expect it to be slow")

    base = args.resume_from or args.base_model
    print(f"[model] loading {base}")
    model = SentenceTransformer(base, device=device)
    model.max_seq_length = args.max_seq_length

    # ---------- Evaluators ----------
    evaluators = []
    if val_rows:
        vq = {str(i): r["query"] for i, r in enumerate(val_rows) if r.get("query")}
        vc = {str(i): r["positive"] for i, r in enumerate(val_rows) if r.get("positive")}
        vr = {str(i): {str(i)} for i in range(len(val_rows))}
        evaluators.append(evaluation.InformationRetrievalEvaluator(
            queries=vq, corpus=vc, relevant_docs=vr,
            name="cbic-val",
            mrr_at_k=[10], ndcg_at_k=[10],
            accuracy_at_k=args.gold_recall_k,
            precision_recall_at_k=args.gold_recall_k,
            show_progress_bar=False, corpus_chunk_size=2000,
        ))

    if args.gold_yaml and args.chunk_corpus:
        gy = Path(args.gold_yaml); cc = Path(args.chunk_corpus)
        if gy.exists() and cc.exists():
            gq, gc, gr = build_gold_ir_inputs(gy, cc)
            if gq:
                evaluators.append(evaluation.InformationRetrievalEvaluator(
                    queries=gq, corpus=gc,
                    relevant_docs={k: set(v) for k, v in gr.items()},
                    name="cbic-gold",
                    mrr_at_k=[10], ndcg_at_k=[10],
                    accuracy_at_k=args.gold_recall_k,
                    precision_recall_at_k=args.gold_recall_k,
                    show_progress_bar=True, corpus_chunk_size=2000,
                ))
        else:
            print(f"[warn] gold-yaml or chunk-corpus missing: "
                  f"gold_exists={gy.exists()} corpus_exists={cc.exists()}")

    evaluator = (evaluation.SequentialEvaluator(evaluators)
                 if len(evaluators) > 1 else (evaluators[0] if evaluators else None))

    if args.dry_run:
        print(f"[dry-run] ok. train_examples={len(train_examples)} "
              f"evaluators={len(evaluators)}")
        return

    train_loader = DataLoader(train_examples, shuffle=True,
                              batch_size=args.batch_size, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model)

    steps_per_epoch = max(1, len(train_loader))
    warmup_steps = math.ceil(steps_per_epoch * args.epochs * args.warmup_ratio)
    print(f"[train] epochs={args.epochs} steps/epoch={steps_per_epoch} "
          f"warmup={warmup_steps}")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(out_dir / "checkpoints") if args.save_each_epoch else None
    if checkpoint_path:
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    def _epoch_cb(score, epoch, steps):
        print(f"[epoch {epoch}] evaluator_score={score:.4f} steps={steps}", flush=True)
        with (out_dir / "eval_gold.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "steps": steps,
                                 "score": float(score)}) + "\n")

    model.fit(
        train_objectives=[(train_loader, loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        use_amp=(args.fp16 or args.bf16),
        output_path=str(out_dir),
        checkpoint_path=checkpoint_path,
        checkpoint_save_steps=steps_per_epoch if checkpoint_path else 0,
        checkpoint_save_total_limit=args.epochs if checkpoint_path else 0,
        show_progress_bar=True,
        callback=_epoch_cb,
        evaluation_steps=0,
    )

    (out_dir / "TRAINING_INFO.json").write_text(json.dumps({
        "base_model": args.base_model,
        "n_train": len(train_rows), "n_triplets": n_trip,
        "n_val": len(val_rows),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "max_seq_length": args.max_seq_length,
        "fp16": args.fp16, "bf16": args.bf16,
        "gold_yaml": args.gold_yaml, "chunk_corpus": args.chunk_corpus,
    }, indent=2), encoding="utf-8")
    print(f"[done] model saved to {out_dir}")


if __name__ == "__main__":
    main()
