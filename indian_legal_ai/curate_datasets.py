#!/usr/bin/env python3
"""
Curate the raw downloaded datasets into a clean corpus for RAG indexing.

Input:  /opt/indian-legal-ai/datasets/<dataset_name>/data.jsonl
Output: /opt/indian-legal-ai/datasets/_curated/<dataset_name>.jsonl
        each row: {"text": str, "source": str, "dataset": str, "tier": int}

- Tier 1 = authoritative statutory text (prefer this at retrieval)
- Tier 2 = case law / judgments
- Tier 3 = Q&A pairs (useful context but not primary source)
- Tier 4 = DROPPED (synthetic, corrupted, or irrelevant)

Transformations:
- Strips Llama-2 [INST]...[/INST] wrappers (ipc_sections, deshmukh_law, jizzu_law_v4)
- Cleans common OCR artefacts in courts_cases (company* -> con*)
- Drops rows shorter than MIN_LEN or longer than MAX_LEN
- Deduplicates within each dataset by text hash
"""

import os, json, re, hashlib, datetime, sys

DATASETS_DIR = "/opt/indian-legal-ai/datasets"
OUT_DIR      = f"{DATASETS_DIR}/_curated"
MIN_LEN      = 40
MAX_LEN      = 50_000   # drop absurdly long rows

TIERS = {
    # ---- Tier 1: authoritative statutory text ----
    "indian_laws":          1,   # 34k - act_title/section/law — gold
    "constitution_s":       1,   # 454 - Constitution articles
    "ipc_sections":         1,   # 512 - IPC (needs [INST] unwrap)
    "indian_law_complete":  1,   # 13k - needs inspection but looks statutory

    # ---- Tier 2: case law ----
    "courts_cases":         2,   # 28k - SC/HC judgments (OCR noise)
    "sc_chunked":           2,   # 10k - SC judgments chunked
    "judgment_summaries":   2,   # 6.9k - judgment text
    "legal_abs":            2,   # 3.6k - judgment bodies
    "sc_en_ta":             2,   # 20k - SC EN+TA (keep English portion only)

    # ---- Tier 3: Q&A / SFT material ----
    "prarabdha_sft":        3,
    "kshitij_law":          3,
    "indian_law":           3,
    "indian_law_9b":        3,
    "deshmukh_law":         3,   # [INST] unwrap
    "tech_legal":           3,
    "jizzu_law_v4":         3,   # [INST] unwrap
    "ipc_insights":         3,   # instruction/response but response has IPC text
    "varma_law":            3,
    "shreyas_legal":        3,
    "indian_lawyer":        3,
    "traffic_law":          3,
    "lawyer_gpt":           3,
    "legal_asst":           3,
    "karthi_law":           3,
    "gst_faqs":             3,

    # ---- Tier 4: DROPPED ----
    # "income_tax":       4,  # ITR form data, not tax law
    # "court_cases_m":    4,  # synthetic fake cases
    # "constitution_k":   4,  # corrupted (movie script leak)
    # "legal_ner":        4,  # NER tags, not useful text
    # "legal_corpus":     4,  # mostly abbreviations list
}

# ---------- Text cleaners -----------------------------------------------------
INST_RE  = re.compile(r"<s>\s*\[INST\](?:\s*<<SYS>>.*?<</SYS>>)?\s*(.*?)\s*\[/INST\]\s*(.*?)(?:</s>|$)", re.DOTALL)
SYS_RE   = re.compile(r"<<SYS>>.*?<</SYS>>", re.DOTALL)
TAG_RE   = re.compile(r"</?s>|\[/?INST\]")
WS_RE    = re.compile(r"[ \t]+")
MULTI_NL = re.compile(r"\n{3,}")

# OCR artefacts in courts_cases dataset — "con" -> "company" replacement pattern
OCR_FIXES = [
    (re.compile(r"\bcompany([a-z])"), r"con\1"),     # companyducted -> conducted
    (re.compile(r"\bCompany([a-z])"), r"Con\1"),
    (re.compile(r"\bnumber([a-z]{2,})"), r"non\1"),  # numberice? — less common
]

def strip_inst(text: str) -> str:
    """Extract meaningful content from Llama-2 [INST] wrapped text."""
    m = INST_RE.search(text)
    if m:
        q, a = m.group(1).strip(), m.group(2).strip()
        # drop system prompt fluff inside q
        q = SYS_RE.sub("", q).strip()
        if q and a:
            return f"Q: {q}\nA: {a}"
        return q or a
    # Fallback: just strip tags
    return TAG_RE.sub("", SYS_RE.sub("", text)).strip()

def fix_ocr(text: str) -> str:
    for pat, repl in OCR_FIXES:
        text = pat.sub(repl, text)
    return text

def normalize_ws(text: str) -> str:
    text = WS_RE.sub(" ", text)
    text = MULTI_NL.sub("\n\n", text)
    return text.strip()

def clean_for_dataset(dataset: str, text: str) -> str:
    if dataset in ("ipc_sections", "deshmukh_law", "jizzu_law_v4"):
        text = strip_inst(text)
    if dataset == "courts_cases":
        text = fix_ocr(text)
    if dataset == "sc_en_ta":
        # Heuristic: drop rows that are mostly non-ASCII (likely pure Tamil)
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
        if ascii_ratio < 0.7:
            return ""
    return normalize_ws(text)

# ---------- Main --------------------------------------------------------------
def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def curate_one(dataset: str, tier: int):
    in_path  = os.path.join(DATASETS_DIR, dataset, "data.jsonl")
    out_path = os.path.join(OUT_DIR, f"{dataset}.jsonl")
    if not os.path.exists(in_path) or os.path.getsize(in_path) == 0:
        log(f"  [SKIP] {dataset} — empty or missing")
        return 0, 0
    seen = set()
    kept = dropped = 0
    with open(in_path, "r", encoding="utf-8") as f_in, \
         open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            raw = row.get("text") or ""
            text = clean_for_dataset(dataset, raw)
            if not text or len(text) < MIN_LEN or len(text) > MAX_LEN:
                dropped += 1
                continue
            h = hashlib.md5(text.encode("utf-8", "ignore")).digest()
            if h in seen:
                dropped += 1
                continue
            seen.add(h)
            out = {
                "text":    text,
                "source":  row.get("source", dataset),
                "dataset": dataset,
                "tier":    tier,
            }
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1
    log(f"  [{dataset:22s}] tier={tier}  kept={kept:>7,}  dropped={dropped:>6,}")
    return kept, dropped

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log("=" * 64)
    log(" Curating datasets into Tier-tagged corpus")
    log("=" * 64)

    total_kept = total_dropped = 0
    by_tier = {1: 0, 2: 0, 3: 0}
    for dataset, tier in TIERS.items():
        k, d = curate_one(dataset, tier)
        total_kept += k
        total_dropped += d
        by_tier[tier] = by_tier.get(tier, 0) + k

    log("")
    log("=" * 64)
    log(" Summary")
    log("=" * 64)
    log(f"  Tier 1 (statutory):  {by_tier.get(1,0):>8,} rows")
    log(f"  Tier 2 (case law):   {by_tier.get(2,0):>8,} rows")
    log(f"  Tier 3 (Q&A):        {by_tier.get(3,0):>8,} rows")
    log(f"  Total kept:          {total_kept:>8,} rows")
    log(f"  Total dropped:       {total_dropped:>8,} rows")
    log(f"  Output: {OUT_DIR}")

if __name__ == "__main__":
    main()
