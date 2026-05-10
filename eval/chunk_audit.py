"""Chunk-quality audit for CBIC Qdrant collection.

Scrolls all points, computes length/category/orphan/table/dup/metadata stats,
writes JSON + markdown summary with honest red flags.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

QDRANT = "http://192.168.1.107:6343"
COLLECTION = "cbic_v1"
BATCH = 500

OUT_JSON = Path(r"D:/_gpu_rig_ai/eval/chunk_audit.json")
OUT_MD = Path(r"D:/_gpu_rig_ai/consults/chunk_audit_20260422.md")


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{QDRANT}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def scroll_all():
    offset = None
    total = 0
    while True:
        body = {
            "limit": BATCH,
            "with_payload": True,
            "with_vector": False,
        }
        if offset is not None:
            body["offset"] = offset
        resp = post(f"/collections/{COLLECTION}/points/scroll", body)
        result = resp["result"]
        points = result.get("points", [])
        if not points:
            break
        for p in points:
            yield p
            total += 1
        offset = result.get("next_page_offset")
        if offset is None:
            break
        if total % 10000 == 0:
            print(f"  scrolled {total}...", file=sys.stderr)


def bucket(n: int) -> str:
    if n < 100:
        return "0-100"
    if n < 300:
        return "100-300"
    if n < 600:
        return "300-600"
    if n < 1200:
        return "600-1200"
    if n < 2400:
        return "1200-2400"
    return "2400+"


def pct(n, d):
    return round(100.0 * n / d, 2) if d else 0.0


def main():
    t0 = time.time()
    print(f"scrolling {COLLECTION} @ {QDRANT}", file=sys.stderr)

    length_buckets = Counter()
    cat_counts = Counter()
    cat_lengths = defaultdict(list)
    orphan_count = 0
    orphan_examples = []
    table_lengths = []
    table_count = 0
    empty_count = 0
    near_empty_count = 0
    dup_hashes = Counter()
    dup_samples = {}
    missing_section = 0
    missing_parent = 0
    missing_category = 0
    parent_acts = Counter()
    total_len_chars = 0
    all_lens = []
    total_points = 0

    for p in scroll_all():
        total_points += 1
        payload = p.get("payload") or {}
        text = payload.get("text") or ""
        text_full = payload.get("text_full") or ""
        use_text = text_full if len(text_full) > len(text) else text
        n = len(use_text or "")
        all_lens.append(n)
        total_len_chars += n
        length_buckets[bucket(n)] += 1

        category = payload.get("category") or ""
        subcat = payload.get("subcategory") or ""
        doc_type = payload.get("doc_type") or ""
        doc_number = payload.get("doc_number") or ""
        section_ref = payload.get("section_ref") or ""
        parent_act = payload.get("parent_act") or ""
        title = payload.get("title") or ""
        is_table = bool(payload.get("is_table"))

        if category:
            cat_counts[category] += 1
            cat_lengths[category].append(n)
        else:
            missing_category += 1
        if not section_ref:
            missing_section += 1
        if not parent_act:
            missing_parent += 1
        else:
            parent_acts[parent_act] += 1

        # orphan heading: short + has title + has section_ref
        if n < 80 and title.strip() and section_ref.strip():
            orphan_count += 1
            if len(orphan_examples) < 5:
                orphan_examples.append({
                    "id": p.get("id"),
                    "text": use_text,
                    "title": title,
                    "section_ref": section_ref,
                    "doc_number": doc_number,
                    "len": n,
                })

        if is_table:
            table_count += 1
            table_lengths.append(n)

        if use_text is None or n == 0:
            empty_count += 1
        if n < 20:
            near_empty_count += 1

        if 100 <= n <= 500:
            h = hashlib.md5(use_text[:200].lower().strip().encode("utf-8")).hexdigest()
            dup_hashes[h] += 1
            if h not in dup_samples:
                dup_samples[h] = {"text": use_text[:200], "doc_numbers": []}
            if len(dup_samples[h]["doc_numbers"]) < 5:
                dup_samples[h]["doc_numbers"].append(doc_number)

    elapsed = round(time.time() - t0, 1)
    print(f"done: {total_points} points in {elapsed}s", file=sys.stderr)

    # Finalize
    median_len = statistics.median(all_lens) if all_lens else 0
    mean_len = round(statistics.mean(all_lens), 1) if all_lens else 0

    cat_summary = {}
    for cat, lens in cat_lengths.items():
        cat_summary[cat] = {
            "count": len(lens),
            "median_len": int(statistics.median(lens)),
            "mean_len": round(statistics.mean(lens), 1),
        }

    table_stats = {
        "count": table_count,
        "median_len": int(statistics.median(table_lengths)) if table_lengths else 0,
        "mean_len": round(statistics.mean(table_lengths), 1) if table_lengths else 0,
    }

    top_dups = dup_hashes.most_common(10)
    top_dups_out = []
    for h, c in top_dups:
        if c < 2:
            continue
        s = dup_samples.get(h, {})
        top_dups_out.append({
            "hash": h,
            "count": c,
            "sample_text": s.get("text", ""),
            "sample_doc_numbers": s.get("doc_numbers", []),
        })

    top_parents = parent_acts.most_common(20)

    stats = {
        "collection": COLLECTION,
        "total_points": total_points,
        "elapsed_sec": elapsed,
        "length": {
            "median_chars": int(median_len),
            "mean_chars": mean_len,
            "buckets": dict(length_buckets),
            "buckets_pct": {k: pct(v, total_points) for k, v in length_buckets.items()},
            "approx_tokens_median": int(median_len // 4),
        },
        "categories": cat_summary,
        "orphan_headings": {
            "count": orphan_count,
            "pct": pct(orphan_count, total_points),
            "examples": orphan_examples,
        },
        "tables": table_stats,
        "empty": {"zero_len": empty_count, "under_20_chars": near_empty_count},
        "near_duplicates_top10_100_500_chars": top_dups_out,
        "missing_metadata": {
            "section_ref": missing_section,
            "section_ref_pct": pct(missing_section, total_points),
            "parent_act": missing_parent,
            "parent_act_pct": pct(missing_parent, total_points),
            "category": missing_category,
            "category_pct": pct(missing_category, total_points),
        },
        "top_parent_acts": [{"parent_act": k, "count": v} for k, v in top_parents],
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    # Red flags (honest)
    red_flags = []
    pct_tiny = stats["length"]["buckets_pct"].get("0-100", 0)
    pct_huge = stats["length"]["buckets_pct"].get("2400+", 0)
    if pct_tiny >= 5:
        red_flags.append(f"{pct_tiny}% of chunks are under 100 chars (likely noise/headings).")
    if pct_huge >= 10:
        red_flags.append(f"{pct_huge}% of chunks are over 2400 chars (imprecise retrieval, may exceed effective BGE-M3 window).")
    if stats["orphan_headings"]["pct"] >= 2:
        red_flags.append(f"Orphan-heading chunks: {orphan_count} ({stats['orphan_headings']['pct']}%) — short text with title+section_ref, likely headings with no body.")
    if top_dups_out and top_dups_out[0]["count"] >= 50:
        td = top_dups_out[0]
        red_flags.append(f"Boilerplate contamination: top duplicate prefix repeats {td['count']}x. Sample: {td['sample_text'][:120]!r}")
    if stats["missing_metadata"]["section_ref_pct"] >= 30:
        red_flags.append(f"{stats['missing_metadata']['section_ref_pct']}% of chunks missing section_ref (weak citation grounding).")
    if stats["missing_metadata"]["parent_act_pct"] >= 30:
        red_flags.append(f"{stats['missing_metadata']['parent_act_pct']}% of chunks missing parent_act (weak filtering/disambiguation).")
    if near_empty_count >= 100:
        red_flags.append(f"{near_empty_count} chunks are under 20 chars (effectively empty).")

    # Markdown
    lines = []
    lines.append(f"# CBIC Chunk Audit — {COLLECTION}")
    lines.append("")
    lines.append(f"Date: 2026-04-22  |  Points: {total_points:,}  |  Scroll time: {elapsed}s")
    lines.append("")
    lines.append("## Red Flags")
    lines.append("")
    if red_flags:
        for rf in red_flags:
            lines.append(f"- {rf}")
    else:
        lines.append("- Clean, no major issues detected against the audit thresholds.")
    lines.append("")
    lines.append("## Length distribution")
    lines.append("")
    lines.append(f"Median: {int(median_len)} chars (~{int(median_len)//4} tokens)  |  Mean: {mean_len} chars")
    lines.append("")
    lines.append("| Bucket (chars) | Count | % |")
    lines.append("|---|---:|---:|")
    for b in ["0-100", "100-300", "300-600", "600-1200", "1200-2400", "2400+"]:
        c = length_buckets.get(b, 0)
        lines.append(f"| {b} | {c:,} | {pct(c, total_points)}% |")
    lines.append("")
    lines.append("## Categories")
    lines.append("")
    lines.append("| Category | Count | Median len | Mean len |")
    lines.append("|---|---:|---:|---:|")
    for cat, d in sorted(cat_summary.items(), key=lambda x: -x[1]["count"]):
        lines.append(f"| {cat} | {d['count']:,} | {d['median_len']} | {d['mean_len']} |")
    lines.append("")
    lines.append("## Orphan headings (<80 chars + title + section_ref)")
    lines.append("")
    lines.append(f"Count: {orphan_count:,} ({stats['orphan_headings']['pct']}%)")
    lines.append("")
    for ex in orphan_examples:
        lines.append(f"- `{ex['section_ref']}` / `{ex['title'][:60]}` / doc `{ex['doc_number']}` — len {ex['len']}: {ex['text']!r}")
    lines.append("")
    lines.append("## Tables")
    lines.append("")
    lines.append(f"Count: {table_stats['count']:,}  |  Median len: {table_stats['median_len']}  |  Mean len: {table_stats['mean_len']}")
    lines.append("")
    lines.append("## Empty / near-empty")
    lines.append("")
    lines.append(f"- Zero-length text: {empty_count:,}")
    lines.append(f"- Under 20 chars: {near_empty_count:,}")
    lines.append("")
    lines.append("## Near-duplicate prefixes (len 100-500, hash of text[:200])")
    lines.append("")
    lines.append("| Count | Sample text (first 160 chars) | Sample doc_numbers |")
    lines.append("|---:|---|---|")
    for td in top_dups_out:
        sample = td["sample_text"][:160].replace("|", "\\|").replace("\n", " ")
        docs = ", ".join(str(d) for d in td["sample_doc_numbers"][:3])
        lines.append(f"| {td['count']} | {sample} | {docs} |")
    lines.append("")
    lines.append("## Missing metadata")
    lines.append("")
    mm = stats["missing_metadata"]
    lines.append(f"- section_ref missing: {mm['section_ref']:,} ({mm['section_ref_pct']}%)")
    lines.append(f"- parent_act missing:  {mm['parent_act']:,} ({mm['parent_act_pct']}%)")
    lines.append(f"- category missing:    {mm['category']:,} ({mm['category_pct']}%)")
    lines.append("")
    lines.append("## Top 20 parent_acts")
    lines.append("")
    lines.append("| parent_act | count |")
    lines.append("|---|---:|")
    for k, v in top_parents:
        lines.append(f"| {k} | {v:,} |")
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(f"red_flags: {len(red_flags)}")
    for rf in red_flags:
        print(f"  - {rf}")


if __name__ == "__main__":
    main()
