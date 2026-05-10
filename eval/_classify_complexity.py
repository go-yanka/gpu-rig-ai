#!/usr/bin/env python3
"""Classify gold set items into LOW/MEDIUM/HIGH complexity and cross-ref recall."""
import json
import re
from pathlib import Path
from collections import defaultdict, Counter

import yaml

GOLD = Path(r"D:\_gpu_rig_ai\eval\gold_set.yaml")
RECALL = Path(r"D:\_gpu_rig_ai\eval\recall_audit_20260422.jsonl")
OUT_JSONL = Path(r"D:\_gpu_rig_ai\eval\gold_complexity_20260422.jsonl")
OUT_MD = Path(r"D:\_gpu_rig_ai\consults\gold_complexity_20260422.md")

SCENARIO_WORDS = [
    r"\bif\b", r"\bwhen\b", r"\bsuppose\b", r"\bmr\.", r"\bms\.", r"\bpvt\b",
    r"\bltd\b", r"\bregistered in\b", r"\bsupplier in\b", r"\bbuyer in\b",
    r"\bclient in\b", r"\binstructs?\b", r"\bwhether\b", r"\bcan\b",
    r"\bshould\b", r"\bavail\b", r"\bclaim\b",
]
INDIAN_STATES = [
    "Maharashtra", "Karnataka", "Delhi", "Gujarat", "Tamil Nadu", "Chennai",
    "Mumbai", "Bengaluru", "Bangalore", "Pune", "Kolkata", "Hyderabad",
    "Rajasthan", "Punjab", "Haryana", "Kerala", "Odisha", "UP", "Uttar Pradesh",
    "MP", "Madhya Pradesh", "Telangana", "Goa", "Bihar", "Assam", "Jharkhand",
    "Chhattisgarh", "Uttarakhand", "Ahmedabad", "Noida", "Gurgaon", "Jaipur",
]
SCENARIO_RE = re.compile("|".join(SCENARIO_WORDS), re.IGNORECASE)
MULTI_STEP_HINTS = re.compile(
    r"bill-to-ship-to|bill[- ]to|ship[- ]to|each leg|multiple|cross[- ]|"
    r"interaction|simultaneous|both|and also|first.*then",
    re.IGNORECASE,
)
LOOKUP_HINTS = re.compile(
    r"^what is the (gst rate|rate of|definition|meaning|threshold|time limit|due date)|"
    r"^what is section|^define |^when was |^who is ",
    re.IGNORECASE,
)


def count_states(q: str) -> int:
    hits = set()
    for s in INDIAN_STATES:
        if re.search(rf"\b{re.escape(s)}\b", q, re.IGNORECASE):
            hits.add(s.lower())
    return len(hits)


def classify(item: dict):
    q = item["question"]
    n_entities = (
        len(item.get("expected_sections") or [])
        + len(item.get("expected_rules") or [])
        + len(item.get("expected_notifications") or [])
    )
    q_len = len(q.split())
    scenario_hits = len(SCENARIO_RE.findall(q))
    states = count_states(q)
    multi_step = bool(MULTI_STEP_HINTS.search(q))
    lookup = bool(LOOKUP_HINTS.search(q))
    diff = (item.get("difficulty") or "").lower()

    reasons = []
    score = 0  # higher -> more complex

    if n_entities >= 4:
        score += 3; reasons.append(f"n_entities={n_entities}>=4")
    elif n_entities >= 2:
        score += 1; reasons.append(f"n_entities={n_entities}")
    else:
        reasons.append(f"n_entities={n_entities}")

    if states >= 2:
        score += 2; reasons.append(f"states={states}")
    elif states == 1:
        score += 1; reasons.append("state=1")

    if multi_step:
        score += 2; reasons.append("multi_step_phrase")

    if scenario_hits >= 3:
        score += 1; reasons.append(f"scenario_words={scenario_hits}")
    elif scenario_hits >= 1:
        reasons.append(f"scenario_words={scenario_hits}")

    if q_len >= 40:
        score += 1; reasons.append(f"q_len={q_len}")

    if lookup and n_entities <= 1 and states == 0:
        score -= 2; reasons.append("lookup_pattern")

    if diff == "complex":
        score += 1; reasons.append("diff=complex")
    elif diff == "basic":
        score -= 1; reasons.append("diff=basic")

    if score <= 0:
        bucket = "LOW"
    elif score <= 3:
        bucket = "MEDIUM"
    else:
        bucket = "HIGH"

    return bucket, n_entities, q_len, reasons, score


def main():
    data = yaml.safe_load(GOLD.read_text(encoding="utf-8"))
    items = data["items"]

    classified = []
    for it in items:
        bucket, n_ent, ql, reasons, score = classify(it)
        classified.append({
            "id": it["id"],
            "category": it.get("category", "unknown"),
            "question": it["question"],
            "complexity": bucket,
            "n_entities": n_ent,
            "q_len": ql,
            "score": score,
            "reasons": reasons,
        })

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for row in classified:
            f.write(json.dumps({k: v for k, v in row.items() if k != "question"}) + "\n")

    # Load recall
    recall_by_id = {}
    with RECALL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            recall_by_id[r["id"]] = r

    # Distributions
    total = Counter(x["complexity"] for x in classified)
    by_cat = defaultdict(Counter)
    for x in classified:
        by_cat[x["category"]][x["complexity"]] += 1

    # Recall per bucket
    bucket_recall = defaultdict(lambda: {"n": 0, "hit_k": 0, "hit_1": 0, "missing": 0})
    for x in classified:
        r = recall_by_id.get(x["id"])
        b = x["complexity"]
        if r is None:
            bucket_recall[b]["missing"] += 1
            continue
        bucket_recall[b]["n"] += 1
        if r.get("hit_at_k"):
            bucket_recall[b]["hit_k"] += 1
        if r.get("hit_at_1"):
            bucket_recall[b]["hit_1"] += 1

    # Examples
    examples = {"LOW": [], "MEDIUM": [], "HIGH": []}
    for x in classified:
        if len(examples[x["complexity"]]) < 3:
            examples[x["complexity"]].append(x)

    # Build markdown
    N = len(classified)
    lines = []
    lines.append("# Gold Set Complexity Audit — 2026-04-22")
    lines.append("")
    lines.append(f"Input: `{GOLD}` ({N} items)")
    lines.append(f"Recall data: `{RECALL}`")
    lines.append("")
    lines.append("## Total distribution")
    lines.append("")
    lines.append("| Bucket | N | % |")
    lines.append("|---|---:|---:|")
    for b in ("LOW", "MEDIUM", "HIGH"):
        n = total[b]
        lines.append(f"| {b} | {n} | {100*n/N:.1f}% |")
    lines.append(f"| **Total** | **{N}** | **100.0%** |")
    lines.append("")

    lines.append("## Per-category distribution")
    lines.append("")
    cats = sorted(by_cat.keys())
    lines.append("| Category | LOW | MEDIUM | HIGH | Total |")
    lines.append("|---|---:|---:|---:|---:|")
    for c in cats:
        low = by_cat[c]["LOW"]; med = by_cat[c]["MEDIUM"]; hi = by_cat[c]["HIGH"]
        lines.append(f"| {c} | {low} | {med} | {hi} | {low+med+hi} |")
    lines.append("")

    lines.append("## Examples per bucket")
    lines.append("")
    for b in ("LOW", "MEDIUM", "HIGH"):
        lines.append(f"### {b}")
        for ex in examples[b]:
            lines.append(f"- **{ex['id']}** (cat={ex['category']}, n_ent={ex['n_entities']}, q_len={ex['q_len']}, score={ex['score']}): {ex['question']}")
        lines.append("")

    lines.append("## Recall@5 by complexity bucket")
    lines.append("")
    lines.append("| Bucket | N evaluated | recall@5 | recall@1 | missing from audit |")
    lines.append("|---|---:|---:|---:|---:|")
    for b in ("LOW", "MEDIUM", "HIGH"):
        br = bucket_recall[b]
        n = br["n"]
        r5 = 100*br["hit_k"]/n if n else 0
        r1 = 100*br["hit_1"]/n if n else 0
        lines.append(f"| {b} | {n} | {r5:.1f}% | {r1:.1f}% | {br['missing']} |")
    lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    overall_n = sum(bucket_recall[b]["n"] for b in bucket_recall)
    overall_hit = sum(bucket_recall[b]["hit_k"] for b in bucket_recall)
    overall = 100*overall_hit/overall_n if overall_n else 0
    lines.append(f"- Overall recall@5 across audited items: {overall:.1f}% ({overall_hit}/{overall_n})")
    lr = bucket_recall["LOW"]; mr = bucket_recall["MEDIUM"]; hr = bucket_recall["HIGH"]
    def pct(br):
        return 100*br["hit_k"]/br["n"] if br["n"] else 0
    low_r, med_r, high_r = pct(lr), pct(mr), pct(hr)
    lines.append(f"- LOW vs HIGH delta: {low_r - high_r:+.1f} pp (LOW {low_r:.1f}% vs HIGH {high_r:.1f}%)")
    if abs(low_r - high_r) < 5:
        lines.append("- **Failure is approximately uniform across complexity buckets** — retrieval is broadly broken, not just on hard queries.")
    elif high_r < low_r:
        lines.append("- **Failure concentrates on HIGH complexity** — training data should over-sample multi-step/scenario queries.")
    else:
        lines.append("- **Failure concentrates on LOW complexity** — unexpected; likely a term-mismatch / entity-resolution issue, not a reasoning issue.")
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {OUT_JSONL}")
    print(f"Wrote {OUT_MD}")
    print(f"Total: {dict(total)}")
    print(f"Recall per bucket: LOW={low_r:.1f}% MED={med_r:.1f}% HIGH={high_r:.1f}%")


if __name__ == "__main__":
    main()
