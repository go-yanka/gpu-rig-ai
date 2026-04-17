#!/usr/bin/env python3
"""
Indian Legal AI — Dataset Downloader
=====================================
Downloads HuggingFace datasets and a GGUF model to /opt/indian-legal-ai/.

Usage:
    python3 /opt/indian-legal-ai/scripts/download_datasets.py

Features:
- Skip already-downloaded files (idempotent / resume-safe)
- Streaming download for large datasets (low RAM usage)
- Per-dataset row limits where specified
- Size-check guard for the InJudgements dataset (skip if >5GB)
- GGUF model download via huggingface_hub
- Detailed timestamped logging
"""

import json
import os
import sys
import datetime
import traceback

# ── Config ────────────────────────────────────────────────────────────────────
WORK_DIR     = "/opt/indian-legal-ai"
DATASETS_DIR = f"{WORK_DIR}/datasets"
MODELS_DIR   = f"{WORK_DIR}/models"
LOGS_DIR     = f"{WORK_DIR}/logs"

# (repo_id, split, local_dir_name, subset_or_None, max_rows_or_None)
DATASETS = [
    # 24K legal Q&A pairs
    ("viber1/indian-law-dataset",                        "train", "indian_law",        None, None),
    # 34K Indian Acts and Regulations
    ("mratanusarkar/Indian-Laws-and-Regulations",        "train", "indian_laws_acts",  None, None),
    # 1116 curated tax FAQs
    ("Prarabdha/India_Tax_FAQs",                         "train", "india_tax_faqs",    None, None),
    # 13K court judgements — guard against very large variants
    ("InJudgements/InJudgements",                        "train", "injudgements",      None, None),
    # Small Indian law Q&A dataset
    ("tomoe/indian-law-qa",                              "train", "tomoe_law",         None, None),
    # 6M SFT examples — we only need 100K samples
    ("Prarabdha/Prarabdha-SFT-V1",                       "train", "prarabdha_sft",     None, 100_000),
]

GGUF_MODEL = {
    "repo_id":  "invincibleambuj/Ambuj-Tripathi-Llama-3.1-8B-IndianLegal-GGUF",
    "filename": "meta-llama-3.1-8b-instruct.Q4_K_M.gguf",
    "local_dir": MODELS_DIR,
}

# InJudgements: skip if estimated total download > this many bytes
INJUDGEMENTS_MAX_BYTES = 5 * 1024 ** 3  # 5 GB

# Minimum line count to consider a JSONL "already done"
MIN_LINES_DONE = 100

# Minimum model size to consider it "already downloaded"
MIN_MODEL_BYTES = 1 * 1024 ** 3  # 1 GB


# ── Logging ───────────────────────────────────────────────────────────────────
def log(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def log_sep(title: str):
    log("")
    log("=" * 60)
    log(f"  {title}")
    log("=" * 60)


# ── Helpers ───────────────────────────────────────────────────────────────────
def count_lines(path: str) -> int:
    """Count lines in a file efficiently (no full load into RAM)."""
    n = 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in f:
                n += 1
    except Exception:
        pass
    return n


def already_downloaded(jsonl_path: str, min_lines: int = MIN_LINES_DONE) -> bool:
    """Return True if JSONL exists and has enough rows."""
    if not os.path.isfile(jsonl_path):
        return False
    n = count_lines(jsonl_path)
    if n >= min_lines:
        log(f"  Already have {n:,} rows → skipping")
        return True
    log(f"  Exists but only {n} lines (< {min_lines}) → re-downloading")
    return False


def flatten_row(row: dict) -> dict:
    """
    HuggingFace datasets can have varying schemas.
    Normalise to a flat dict with text/question/answer keys.
    Keep all original keys too — nothing is discarded.
    """
    # Many legal datasets use 'instruction'/'output' or 'question'/'answer'
    flat = dict(row)
    # Ensure there is always a 'text' field for the RAG chunker
    if "text" not in flat:
        parts = []
        for key in ("instruction", "question", "input", "prompt"):
            if flat.get(key):
                parts.append(str(flat[key]))
        for key in ("output", "answer", "response", "completion"):
            if flat.get(key):
                parts.append(str(flat[key]))
        if parts:
            flat["text"] = " ".join(parts)
        else:
            # last resort: join all string values
            flat["text"] = " ".join(
                str(v) for v in flat.values() if isinstance(v, str)
            )
    return flat


def estimate_dataset_size(repo_id: str) -> int:
    """
    Try to estimate total parquet download size via HuggingFace Hub API.
    Returns size in bytes, or 0 if unable to determine.
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        total = 0
        for f in files:
            if f.endswith(".parquet") or f.endswith(".jsonl") or f.endswith(".arrow"):
                try:
                    info = api.get_paths_info(
                        repo_id=repo_id,
                        paths=[f],
                        repo_type="dataset",
                        expand=True,
                    )
                    for item in info:
                        size = getattr(item, "size", None)
                        if size:
                            total += size
                except Exception:
                    pass
        return total
    except Exception as e:
        log(f"  Could not estimate size for {repo_id}: {e}")
        return 0


# ── Core download logic ───────────────────────────────────────────────────────
def download_dataset(
    repo_id: str,
    split: str,
    name: str,
    subset: str | None,
    max_rows: int | None,
) -> bool:
    """
    Download one HuggingFace dataset and save as JSONL.
    Returns True on success, False on failure/skip.
    """
    out_dir  = os.path.join(DATASETS_DIR, name)
    out_path = os.path.join(out_dir, "data.jsonl")

    log_sep(f"Dataset: {name}")
    log(f"  Repo   : {repo_id}")
    log(f"  Split  : {split}")
    log(f"  Subset : {subset or 'default'}")
    log(f"  Max rows: {max_rows:,}" if max_rows else "  Max rows: all")

    if already_downloaded(out_path):
        return True

    # Special guard for InJudgements — can be extremely large
    if "InJudgements" in repo_id:
        log("  Checking dataset size before downloading...")
        size_bytes = estimate_dataset_size(repo_id)
        if size_bytes > INJUDGEMENTS_MAX_BYTES:
            log(f"  SKIP: estimated size {size_bytes / 1024**3:.1f} GB > 5 GB limit")
            return False
        if size_bytes > 0:
            log(f"  Estimated size: {size_bytes / 1024**2:.0f} MB — proceeding")
        else:
            log("  Could not determine size — will attempt streaming (safe)")

    os.makedirs(out_dir, exist_ok=True)

    try:
        from datasets import load_dataset

        log("  Loading dataset (streaming=True) ...")
        ds_kwargs = {
            "streaming": True,
            "trust_remote_code": True,
        }
        if subset:
            ds_kwargs["name"] = subset

        ds = load_dataset(repo_id, split=split, **ds_kwargs)

        written = 0
        tmp_path = out_path + ".tmp"

        with open(tmp_path, "w", encoding="utf-8") as f:
            for row in ds:
                flat = flatten_row(dict(row))
                # Inject metadata
                flat.setdefault("dataset", name)
                flat.setdefault("source", repo_id)
                f.write(json.dumps(flat, ensure_ascii=False) + "\n")
                written += 1

                if written % 10_000 == 0:
                    log(f"  ... {written:,} rows written")

                if max_rows and written >= max_rows:
                    log(f"  Reached max_rows={max_rows:,} — stopping")
                    break

        # Atomic rename
        os.replace(tmp_path, out_path)
        log(f"  Done: {written:,} rows → {out_path}")
        return True

    except Exception as e:
        log(f"  ERROR downloading {repo_id}: {e}")
        traceback.print_exc()
        # Clean up partial temp file
        tmp_path = out_path + ".tmp"
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        return False


# ── Model download ────────────────────────────────────────────────────────────
def download_gguf_model() -> bool:
    """
    Download the Indian Legal LLaMA 8B GGUF model via huggingface_hub.
    Skips if file already exists and is > 1 GB.
    """
    log_sep("GGUF Model: Indian Legal LLaMA 8B")
    repo_id  = GGUF_MODEL["repo_id"]
    filename = GGUF_MODEL["filename"]
    local_dir = GGUF_MODEL["local_dir"]
    dest_path = os.path.join(local_dir, filename)

    log(f"  Repo    : {repo_id}")
    log(f"  File    : {filename}")
    log(f"  Dest    : {dest_path}")

    if os.path.isfile(dest_path):
        size = os.path.getsize(dest_path)
        if size >= MIN_MODEL_BYTES:
            log(f"  Already exists ({size / 1024**3:.2f} GB) → skipping")
            return True
        else:
            log(f"  Exists but only {size / 1024**2:.0f} MB — re-downloading")

    os.makedirs(local_dir, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download

        log("  Downloading GGUF (~4.92 GB) — this will take a while...")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        size = os.path.getsize(path)
        log(f"  Downloaded: {path} ({size / 1024**3:.2f} GB)")
        return True

    except Exception as e:
        log(f"  ERROR downloading GGUF model: {e}")
        traceback.print_exc()
        return False


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log_sep("Indian Legal AI — Dataset Downloader")
    log(f"Datasets dir : {DATASETS_DIR}")
    log(f"Models dir   : {MODELS_DIR}")

    os.makedirs(DATASETS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR,   exist_ok=True)
    os.makedirs(LOGS_DIR,     exist_ok=True)

    results = {}

    # Download datasets
    for (repo_id, split, name, subset, max_rows) in DATASETS:
        success = download_dataset(repo_id, split, name, subset, max_rows)
        results[name] = "ok" if success else "FAILED/SKIPPED"

    # Download GGUF model
    model_ok = download_gguf_model()
    results["gguf_model"] = "ok" if model_ok else "FAILED"

    # Summary
    log_sep("Summary")
    ok_count   = sum(1 for v in results.values() if v == "ok")
    fail_count = len(results) - ok_count
    for name, status in results.items():
        marker = "OK  " if status == "ok" else "FAIL"
        log(f"  [{marker}] {name}")
    log("")
    log(f"  {ok_count}/{len(results)} completed successfully")

    if fail_count:
        log(f"  {fail_count} failed — check logs above for details")
        log("  Re-run the script to retry failed items (idempotent)")

    log("")
    log("Next step: python3 /opt/indian-legal-ai/scripts/build_rag_index.py")
    log("")


if __name__ == "__main__":
    main()
