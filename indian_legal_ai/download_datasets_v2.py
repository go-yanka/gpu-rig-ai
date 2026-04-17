#!/usr/bin/env python3
"""
Download Indian legal datasets from HuggingFace.
Only uses datasets that have been verified to exist via `api.list_datasets`.
"""

import os, json, datetime, traceback, sys

DATASETS_DIR = "/opt/indian-legal-ai/datasets"

# Verified working datasets (searched via HfApi)
DATASETS = [
    # (repo_id, local_dir, max_rows, description)
    # --- Already working ---
    ("viber1/indian-law-dataset",               "indian_law",          None,   "24K Q&A"),
    ("kshitij230/Indian-Law",                   "kshitij_law",         None,   "Indian Law Q&A"),
    ("Mukesh555/indian_lawyer_dataset",         "indian_lawyer",       None,   "Lawyer dataset"),
    ("Sahi19/IndianLawComplete",                "indian_law_complete", None,   "Complete Indian law"),
    ("mratanusarkar/Indian-Laws",               "indian_laws",         None,   "Laws and acts"),
    ("Ghost222/Indian_Law_9Brainz",             "indian_law_9b",       None,   "9B format"),
    ("Pravincoder/Indian_traffic_law_QA",       "traffic_law",         None,   "Traffic law Q&A"),
    ("AgamiAI/Indian-Income-Tax-Returns",       "income_tax",          None,   "Income Tax Returns"),
    ("jizzu/llama2_indian_law_v4",              "jizzu_law_v4",        None,   "Llama2 v4 format"),

    # --- High-value verified datasets (searched via HfApi) ---
    ("Prarabdha/indian-legal-supervised-fine-tuning-data",  "prarabdha_sft",     200000, "Prarabdha legal SFT data (cap 200K)"),
    ("bharatgenai/BhashaBench-Legal",            "bhasha_bench",       None,   "BhashaBench Legal benchmark"),
    ("opennyaiorg/InJudgements_dataset",         "injudgements",       50000,  "OpenNyAI court judgments (cap 50K)"),
    ("vihaannnn/Indian-Supreme-Court-Judgements-Chunked", "sc_chunked",  50000,  "SC judgments chunked (cap 50K)"),
    ("santoshtyss/indian_courts_cases",          "courts_cases",       100000, "Court cases (cap 100K)"),
    ("ninadn/indian-legal",                      "ninadn_legal",       None,   "Indian legal data"),
    ("harshitv804/Indian_Penal_Code",            "ipc",                None,   "Indian Penal Code"),
    ("sairamn/indian-penal-code",                "ipc_sairamn",        None,   "IPC"),
    ("Sharathhebbar24/Indian-Constitution",      "constitution_s",     None,   "Indian Constitution"),
    ("MrTryAll/IndianConstitution",              "constitution_m",     None,   "Indian Constitution v2"),
    ("ramandeepSingh03/company_law_India_2013",  "companies_act",      None,   "Companies Act 2013"),
    ("Techmaestro369/indian-legal-texts-finetuning", "tech_legal",     None,   "Legal texts finetuning"),
    ("geekyrakshit/indian-legal-acts",           "legal_acts",         None,   "Legal acts"),
    ("ShreyasP123/Legal-Dataset-for-india",      "shreyas_legal",      None,   "Legal dataset"),
    ("nisaar/Lawyer_GPT_India",                  "lawyer_gpt",         None,   "Lawyer GPT"),
    ("jft-ai-team/IndianPenalCode-Legal-Insights", "ipc_insights",    None,   "IPC Legal Insights"),
    ("sujantkumarkv/indian_legal_corpus",        "legal_corpus",       None,   "Legal corpus"),
    ("AjayMukundS/Indian_Legal_NER_Dataset",     "legal_ner",          None,   "Legal NER data"),
    ("ayush0504/IndianPenalCodeSections",        "ipc_sections",       None,   "IPC Sections"),
    ("rishiai/indian-court-judgements-and-its-summaries", "judgment_summaries", None, "Judgment summaries"),
    ("Narenameme/indian_supreme_court_judgements_en_ta", "sc_en_ta",  20000,  "SC judgments EN+TA (cap 20K)"),
    ("maheshCoder/indian_court_cases",           "court_cases_m",      50000,  "Court cases (cap 50K)"),
    ("varma007ut/Indian_Law_Dataset_MinorProject", "varma_law",        None,   "Law dataset"),
    ("22deshmukh/indian-law-dataset",            "deshmukh_law",       None,   "Indian law"),
    ("KarthiDreamr/indian-law-dataset",          "karthi_law",         None,   "Indian law"),
    ("vnovaai/LEGAL_ASSISTANT_DATASET_JSONL_INDIA", "legal_asst",     None,   "Legal assistant"),
    ("Kuberwastaken/Indian-Constitution",        "constitution_k",     None,   "Indian Constitution"),
    ("Yashaswat/Indian-Legal-Text-ABS",          "legal_abs",          None,   "Legal text ABS"),
]

os.makedirs(DATASETS_DIR, exist_ok=True)

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def normalise_row(row: dict) -> str:
    """Extract `text` content from any dataset row schema."""
    # Try common field combinations
    for fields in [
        ("instruction", "output"),
        ("Instruction", "Response"),
        ("question", "answer"),
        ("Question", "Answer"),
        ("prompt", "response"),
        ("input", "output"),
        ("text",),
        ("content",),
    ]:
        vals = [str(row.get(f, "")).strip() for f in fields]
        if all(vals):
            if len(fields) == 1:
                return vals[0]
            return f"Q: {vals[0]}\nA: {vals[1]}"

    # Fallback: concatenate any non-empty string fields
    parts = [f"{k}: {v}" for k, v in row.items() if isinstance(v, str) and v.strip()]
    return "\n".join(parts) if parts else ""

def download_one(repo_id: str, local_dir: str, max_rows: int = None):
    from datasets import load_dataset

    target_dir  = os.path.join(DATASETS_DIR, local_dir)
    target_file = os.path.join(target_dir, "data.jsonl")
    os.makedirs(target_dir, exist_ok=True)

    if os.path.exists(target_file) and os.path.getsize(target_file) > 1024:
        with open(target_file) as f:
            lines = sum(1 for _ in f)
        if lines > 10:
            log(f"  [SKIP] {repo_id} — already have {lines} rows at {target_file}")
            return lines

    log(f"  Loading {repo_id}...")
    ds = load_dataset(repo_id, split="train", streaming=True)

    tmp_file = target_file + ".tmp"
    count = 0
    with open(tmp_file, "w") as out:
        for row in ds:
            text = normalise_row(row)
            if not text or len(text) < 20:
                continue
            record = {"text": text, "source": repo_id, "dataset": local_dir}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if count % 1000 == 0:
                log(f"    {count:,} rows...")
            if max_rows and count >= max_rows:
                break

    os.replace(tmp_file, target_file)
    log(f"  [OK]  {repo_id} → {count:,} rows")
    return count

def main():
    log("=" * 56)
    log(" Download Indian Legal Datasets (verified IDs)")
    log("=" * 56)

    total = 0
    results = {}
    for repo_id, local_dir, max_rows, desc in DATASETS:
        log("")
        log(f"► {desc}")
        try:
            n = download_one(repo_id, local_dir, max_rows)
            total += n
            results[repo_id] = ("OK", n)
        except Exception as e:
            log(f"  [FAIL] {repo_id}: {e}")
            results[repo_id] = ("FAIL", str(e))

    log("")
    log("=" * 56)
    log(" Summary")
    log("=" * 56)
    for repo_id, (status, info) in results.items():
        log(f"  [{status}] {repo_id}: {info}")
    log("")
    log(f"Total: {total:,} rows downloaded")

if __name__ == "__main__":
    main()
