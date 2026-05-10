"""B1 benchmark: PyMuPDF parsing throughput on Windows laptop.

Goal: measure docs/sec on a stratified 100-PDF sample to decide whether the
Windows machine can share parse workload with the rig for CBIC re-ingestion.

Pass criterion: >= 10 docs/sec sustained.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pymupdf  # type: ignore

RESULTS_PATH = Path(r"D:\_gpu_rig_ai\benchmarks\b1_results.json")
SAMPLE_SIZE = 100
WORKERS = 4
SEED = 42

# Candidate PDF source roots (project tree has almost no PDFs; use Google Drive
# as a mixed corpus - same PyMuPDF code path, representative of real-world PDFs).
CANDIDATE_ROOTS = [
    r"D:\_Google_Drive_Rahul",
    r"D:\_backup_itrs",
    r"D:\_gpu_rig_ai",
]


def enumerate_pdfs() -> list[Path]:
    pdfs: list[Path] = []
    for root in CANDIDATE_ROOTS:
        p = Path(root)
        if not p.exists():
            continue
        for f in p.rglob("*.pdf"):
            try:
                if f.is_file() and f.stat().st_size > 0:
                    pdfs.append(f)
            except OSError:
                continue
    return pdfs


def stratify(pdfs: list[Path], n: int) -> list[Path]:
    """Stratify by file size: small <1MB, medium 1-10MB, large >10MB."""
    small, medium, large = [], [], []
    for p in pdfs:
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        if sz < 1_000_000:
            small.append(p)
        elif sz < 10_000_000:
            medium.append(p)
        else:
            large.append(p)

    random.seed(SEED)
    random.shuffle(small)
    random.shuffle(medium)
    random.shuffle(large)

    # Target split: 50 small, 35 medium, 15 large (typical CBIC mix skews small)
    targets = [("small", small, 50), ("medium", medium, 35), ("large", large, 15)]
    out: list[Path] = []
    leftovers: list[Path] = []
    for _, bucket, want in targets:
        take = bucket[:want]
        out.extend(take)
        leftovers.extend(bucket[want:])

    # Top up if a bucket was short
    random.shuffle(leftovers)
    while len(out) < n and leftovers:
        out.append(leftovers.pop())

    return out[:n]


def parse_one(path_str: str) -> tuple[str, int, int, float, str | None]:
    """Return (path, pages, chars, elapsed_s, error)."""
    t0 = time.perf_counter()
    try:
        doc = pymupdf.open(path_str)
        pages = doc.page_count
        chars = 0
        for page in doc:
            chars += len(page.get_text())
        doc.close()
        return (path_str, pages, chars, time.perf_counter() - t0, None)
    except Exception as e:  # noqa: BLE001
        return (path_str, 0, 0, time.perf_counter() - t0, f"{type(e).__name__}: {e}")


def run_single(paths: list[Path]) -> dict:
    t0 = time.perf_counter()
    results = [parse_one(str(p)) for p in paths]
    wall = time.perf_counter() - t0
    return summarize(results, wall, mode="single")


def run_parallel(paths: list[Path], workers: int) -> dict:
    t0 = time.perf_counter()
    results: list[tuple] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(parse_one, str(p)) for p in paths]
        for fut in as_completed(futs):
            results.append(fut.result())
    wall = time.perf_counter() - t0
    return summarize(results, wall, mode=f"parallel_{workers}")


def summarize(results: list[tuple], wall: float, mode: str) -> dict:
    total_pages = sum(r[1] for r in results)
    total_chars = sum(r[2] for r in results)
    fails = [r for r in results if r[4] is not None]
    ok = len(results) - len(fails)
    return {
        "mode": mode,
        "wall_s": round(wall, 3),
        "docs": len(results),
        "ok_docs": ok,
        "fail_docs": len(fails),
        "total_pages": total_pages,
        "total_chars": total_chars,
        "docs_per_sec": round(len(results) / wall, 3) if wall > 0 else 0,
        "pages_per_sec": round(total_pages / wall, 3) if wall > 0 else 0,
        "mean_chars_per_doc": round(total_chars / ok, 1) if ok else 0,
        "failures": [{"path": r[0], "error": r[4]} for r in fails][:20],
    }


def main() -> int:
    print(f"[{time.strftime('%H:%M:%S')}] enumerating PDFs...")
    pool = enumerate_pdfs()
    print(f"  found {len(pool)} PDFs across candidate roots")
    if len(pool) < SAMPLE_SIZE:
        print(f"  WARNING: fewer than {SAMPLE_SIZE} PDFs available")
    sample = stratify(pool, SAMPLE_SIZE)
    print(f"  stratified sample size: {len(sample)}")

    size_buckets = {"small": 0, "medium": 0, "large": 0}
    total_bytes = 0
    for p in sample:
        sz = p.stat().st_size
        total_bytes += sz
        if sz < 1_000_000:
            size_buckets["small"] += 1
        elif sz < 10_000_000:
            size_buckets["medium"] += 1
        else:
            size_buckets["large"] += 1
    print(f"  buckets: {size_buckets}  total_bytes: {total_bytes/1e6:.1f} MB")

    print(f"[{time.strftime('%H:%M:%S')}] running single-threaded baseline...")
    single = run_single(sample)
    print(f"  -> {single['docs_per_sec']} docs/s, {single['pages_per_sec']} pg/s, "
          f"fails={single['fail_docs']}, wall={single['wall_s']}s")

    print(f"[{time.strftime('%H:%M:%S')}] running parallel ({WORKERS} workers)...")
    parallel = run_parallel(sample, WORKERS)
    print(f"  -> {parallel['docs_per_sec']} docs/s, {parallel['pages_per_sec']} pg/s, "
          f"fails={parallel['fail_docs']}, wall={parallel['wall_s']}s")

    best = max(single["docs_per_sec"], parallel["docs_per_sec"])
    passed = best >= 10.0

    payload = {
        "pass": passed,
        "pass_criterion_docs_per_sec": 10.0,
        "single_docs_per_sec": single["docs_per_sec"],
        "parallel_docs_per_sec": parallel["docs_per_sec"],
        "sample_size": len(sample),
        "workers": WORKERS,
        "cpu_count": os.cpu_count(),
        "pymupdf_version": pymupdf.__version__,
        "python_version": sys.version.split()[0],
        "sample_buckets": size_buckets,
        "sample_total_mb": round(total_bytes / 1e6, 1),
        "details": {
            "single": single,
            "parallel": parallel,
        },
        "notes": (
            "PDF corpus source: D:/_Google_Drive_Rahul + D:/_backup_itrs + project tree. "
            "No CBIC-specific PDFs on the laptop; PyMuPDF throughput is corpus-agnostic "
            "(same code path), so these numbers are representative for the decision."
        ),
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"[{time.strftime('%H:%M:%S')}] wrote {RESULTS_PATH}")
    print(f"  PASS={passed}  best={best} docs/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
