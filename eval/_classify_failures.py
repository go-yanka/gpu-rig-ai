"""Classify failure modes from recall audit JSONL.
Heuristic rules over question text, expected entities, and top-k metadata.
"""
import json, re, os
from collections import Counter, defaultdict

IN_PATH  = r"D:\_gpu_rig_ai\eval\recall_audit_20260422.jsonl"
OUT_JSONL = r"D:\_gpu_rig_ai\eval\failure_modes_20260422.jsonl"
OUT_MD    = r"D:\_gpu_rig_ai\consults\failure_modes_20260422.md"

ACT_TOKENS = {
    "cgst":   ["cgst", "central goods and services", "central gst"],
    "sgst":   ["sgst", "state goods and services"],
    "igst":   ["igst", "integrated goods and services", "integrated gst", "inter-state", "inter state", "place of supply"],
    "utgst":  ["utgst", "union territory goods"],
    "customs":["customs", "bcd", "basic customs", "tariff act", "customs tariff"],
    "cess":   ["cess", "compensation"],
    "income": ["income tax", "income-tax", " ita "],
}

PROCEDURAL_Q = [
    "how to file", "which form", "return form", "due date", "gstr", "itc claim",
    "filing", "invoice format", "e-way bill", "registration procedure", "refund application",
    "tran-1", "tran-2", "form no", "annual return", "compliance",
]
PROCEDURAL_T = ["gstr", "return", "form ", "procedure", "filing", "eway", "e-way",
                "invoice", "refund application", "application for", "rule "]
SUBSTANTIVE_Q = ["place of supply", "levy", "chargeable", "taxable event", "exemption",
                 "rate of tax", "classification", "valuation", "time of supply",
                 "scope of supply", "composite supply", "mixed supply", "input tax credit eligibility"]

RATE_Q_MARKERS = ["rate", "tax rate", "gst rate", "what is the rate", "hsn", "sac",
                  "% gst", "percent gst", "bcd rate", "cess rate", "notification"]

def act_of_text(s):
    s = (s or "").lower()
    hits = set()
    for act, toks in ACT_TOKENS.items():
        for t in toks:
            if t in s:
                hits.add(act); break
    return hits

def act_of_expected(exp_keys):
    hits = set()
    for k in exp_keys:
        hits |= act_of_text(k)
    return hits

def act_of_topk(topk):
    hits = set()
    for m in topk:
        blob = " ".join(str(m.get(x,"")) for x in ("title","doc_id","doc_number","section_ref"))
        hits |= act_of_text(blob)
    return hits

SECTION_RE = re.compile(r"\b(\d{1,3}[A-Z]?)\s*(?:\(\s*\d+[A-Za-z]?\s*\))*", re.I)
def sections_in_expected(exp_keys):
    out = set()
    for k in exp_keys:
        # extract leading number tokens like "10(1)(a)" or "Section 10"
        m = re.findall(r"\b(\d{1,3}[A-Z]?)(?:\([^)]+\))*", k)
        for x in m:
            out.add(x.upper())
    return out

def sections_in_topk(topk):
    out = set()
    for m in topk:
        sr = str(m.get("section_ref","") or "")
        if sr:
            mm = re.findall(r"\b(\d{1,3}[A-Z]?)(?:\([^)]+\))*", sr)
            for x in mm: out.add(x.upper())
    return out

def notation_possible(item):
    """Did any top-k chunk contain the expected section in section_ref or title?"""
    exp_keys = list(item["per_entity_hits"].keys())
    exp_sections = sections_in_expected(exp_keys)
    if not exp_sections: return False
    for m in item["top_k_meta"]:
        blob = " ".join(str(m.get(x,"")) for x in ("section_ref","title","doc_number"))
        blob_u = blob.upper()
        for s in exp_sections:
            # match section token with word boundary
            if re.search(r"\b"+re.escape(s)+r"\b", blob_u):
                return True
    return False

def is_procedural_q(q):
    ql = q.lower()
    return any(p in ql for p in PROCEDURAL_Q)

def is_substantive_q(q):
    ql = q.lower()
    return any(p in ql for p in SUBSTANTIVE_Q)

def topk_procedural_share(topk):
    n = max(1, len(topk))
    hits = 0
    for m in topk:
        t = (m.get("title","") or "").lower()
        if any(p in t for p in PROCEDURAL_T): hits += 1
    return hits / n

def is_rate_q(q):
    ql = q.lower()
    return any(p in ql for p in RATE_Q_MARKERS)

def expected_is_notification(exp_keys):
    for k in exp_keys:
        kl = k.lower()
        if "notification" in kl or re.search(r"\b\d{1,3}\s*/\s*20\d{2}", kl):
            return True
    return False

def topk_is_act_rule(topk):
    """majority of top-k look like Act sections, not notifications"""
    n = max(1, len(topk))
    act_like = 0
    for m in topk:
        sr = str(m.get("section_ref","") or "")
        did = str(m.get("doc_id","") or "").lower()
        if sr or "act" in did or "rule" in did:
            act_like += 1
    return act_like / n >= 0.6

def classify(item):
    q = item["question"]
    exp_keys = list(item["per_entity_hits"].keys())
    topk = item["top_k_meta"]

    exp_acts  = act_of_expected(exp_keys) | act_of_text(q)
    topk_acts = act_of_topk(topk)

    evidence = []

    # 3) Notation mismatch — expected section appears in section_ref/title of a top-k chunk
    if notation_possible(item):
        evidence.append(f"expected section(s) {sorted(sections_in_expected(exp_keys))} present in top-k section_ref/title")
        return "Notation_mismatch", evidence

    # 5) Rate notification miss — rate question, expected notification, got Act/Rule
    if (is_rate_q(q) or expected_is_notification(exp_keys)) and expected_is_notification(exp_keys) and topk_is_act_rule(topk):
        evidence.append("rate/notif question; expected notification; top-k dominated by Act/Rule chunks")
        return "Rate_notification_miss", evidence

    # 1) Wrong domain — expected acts disjoint from top-k acts (and both non-empty)
    if exp_acts and topk_acts and not (exp_acts & topk_acts):
        evidence.append(f"expected_acts={sorted(exp_acts)} topk_acts={sorted(topk_acts)} disjoint")
        return "Wrong_domain", evidence

    # 6) Procedural vs substantive confusion
    proc_share = topk_procedural_share(topk)
    if is_substantive_q(q) and proc_share >= 0.6:
        evidence.append(f"substantive question; {proc_share:.0%} of top-k look procedural")
        return "Procedural_vs_substantive_confusion", evidence
    if is_procedural_q(q) and proc_share <= 0.2:
        evidence.append(f"procedural question; only {proc_share:.0%} of top-k look procedural")
        return "Procedural_vs_substantive_confusion", evidence

    # 2) Right act wrong section — overlap of acts, but expected sections not in top-k
    if exp_acts & topk_acts:
        exp_sec = sections_in_expected(exp_keys)
        top_sec = sections_in_topk(topk)
        if exp_sec and not (exp_sec & top_sec):
            evidence.append(f"act overlap {sorted(exp_acts & topk_acts)}; expected_sec={sorted(exp_sec)} top_sec={sorted(top_sec)}")
            return "Right_act_wrong_section", evidence

    # Low max score => likely corpus gap
    max_score = max((m.get("score",0) for m in topk), default=0)
    if max_score < 0.35:
        evidence.append(f"low max score {max_score:.2f}; retrieval confidence poor")
        return "No_authoritative_chunk_exists", evidence

    # Default fallback
    evidence.append("no rule matched; default bucket")
    return "Right_act_wrong_section", evidence


def main():
    items = []
    with open(IN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            items.append(json.loads(line))

    misses = [x for x in items if not x.get("hit_at_k")]
    print(f"total={len(items)} misses={len(misses)}")

    rows = []
    for it in misses:
        mode, ev = classify(it)
        top_titles = [ (m.get("title","") or "")[:80] for m in it["top_k_meta"][:5] ]
        rows.append({
            "id": it["id"],
            "category": it.get("category",""),
            "question_short": (it["question"][:160] + ("…" if len(it["question"])>160 else "")),
            "expected": list(it["per_entity_hits"].keys()),
            "top_titles": top_titles,
            "failure_mode": mode,
            "evidence": ev,
        })

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Aggregates
    mode_counts = Counter(r["failure_mode"] for r in rows)
    per_cat = defaultdict(Counter)
    for r in rows:
        per_cat[r["category"]][r["failure_mode"]] += 1

    N = len(rows)
    lines = []
    lines.append(f"# Failure-Mode Taxonomy — Recall Audit 2026-04-22\n")
    lines.append(f"Source: `{IN_PATH}`  \nMisses analyzed: **{N}** (of 170 items)\n")

    lines.append("## Taxonomy table\n")
    lines.append("| Failure mode | Count | % of misses |")
    lines.append("|---|---:|---:|")
    for m, c in mode_counts.most_common():
        lines.append(f"| {m} | {c} | {c/N:.1%} |")
    lines.append("")

    lines.append("## Per-category breakdown\n")
    cats = sorted(per_cat.keys())
    all_modes = [m for m,_ in mode_counts.most_common()]
    header = "| category | total | " + " | ".join(all_modes) + " |"
    sep = "|" + "---|"*(len(all_modes)+2)
    lines.append(header); lines.append(sep)
    for c in cats:
        total = sum(per_cat[c].values())
        cells = [str(per_cat[c].get(m,0)) for m in all_modes]
        lines.append(f"| {c} | {total} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Examples (up to 5 per mode)\n")
    by_mode = defaultdict(list)
    for r in rows: by_mode[r["failure_mode"]].append(r)
    for m in all_modes:
        lines.append(f"### {m}  ({mode_counts[m]})\n")
        for r in by_mode[m][:5]:
            lines.append(f"- **{r['id']}** ({r['category']}): {r['question_short']}")
            lines.append(f"  - expected: `{r['expected']}`")
            lines.append(f"  - top-3 titles:")
            for t in r["top_titles"][:3]:
                lines.append(f"    - {t}")
            lines.append(f"  - evidence: {r['evidence'][0] if r['evidence'] else ''}")
        lines.append("")

    lines.append("## Intervention fix-rate estimate\n")
    lines.append("| Intervention | Wrong_domain | Right_act_wrong_section | Notation_mismatch | Rate_notification_miss | Procedural_vs_substantive | No_authoritative_chunk |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append("| **meta_filter** (parent_act/doc_type) | FIX (high) | partial | partial | FIX (filter to notification) | partial | no |")
    lines.append("| **HyDE** (hypothetical doc expansion) | partial | FIX (high) | partial | partial | FIX (clarifies intent) | no |")
    lines.append("| **Fine-tune** embedder on CBIC pairs | FIX | FIX | small | partial | FIX | no |")
    lines.append("| **Hard negatives** (same-act wrong-section) | small | FIX (high) | no | no | partial | no |")
    lines.append("| **Matcher fix** (multi-field substring) | no | no | FIX (complete) | no | no | no |")
    lines.append("| **Corpus ingest** (missing docs) | no | no | no | partial | no | FIX |")
    lines.append("")

    # Recommendation: top 2 modes by count excluding No_authoritative_chunk
    ranked = [m for m,_ in mode_counts.most_common() if m != "No_authoritative_chunk_exists"]
    top2 = ranked[:2]
    lines.append("## Recommendation — target these first\n")
    lines.append(f"1. **{top2[0]}** ({mode_counts[top2[0]]} misses, {mode_counts[top2[0]]/N:.0%}) — see fix-rate row above for best intervention.")
    if len(top2) > 1:
        lines.append(f"2. **{top2[1]}** ({mode_counts[top2[1]]} misses, {mode_counts[top2[1]]/N:.0%}) — second priority.")
    lines.append("")
    # Quick heuristic recommendation text
    rec = {
        "Notation_mismatch": "Matcher fix is near-zero-cost and fully recovers these; do it before any model work.",
        "Right_act_wrong_section": "Hard-negative mining within the same parent_act + HyDE give the largest lift.",
        "Wrong_domain": "meta_filter on parent_act/doc_type cuts this to near zero with no training.",
        "Rate_notification_miss": "meta_filter on doc_type=notification + rate-keyword routing.",
        "Procedural_vs_substantive_confusion": "HyDE + doc_type filter (Act vs Rule vs Form).",
    }
    for m in top2:
        if m in rec:
            lines.append(f"- {m}: {rec[m]}")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"wrote {OUT_JSONL}")
    print(f"wrote {OUT_MD}")
    print("mode counts:", mode_counts)

if __name__ == "__main__":
    main()
