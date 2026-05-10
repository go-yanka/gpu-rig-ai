#!/usr/bin/env python3
"""Eval harness for indian_legal_t1_v2.

Stages:
  1. sample  - sample N chunks from Qdrant, balanced by dataset + status
  2. draft   - for each chunk, ask LLM to draft a question + gold section/act
  3. run     - call /ask for each question, capture answer + sources
  4. judge   - LLM-judge scores (citation_accuracy, relevance, refusal_correctness)
  5. report  - aggregate + markdown scoreboard

All stages checkpoint to /opt/indian-legal-ai/eval/v2/{stage}.jsonl so restarts
resume where they left off.
"""
import argparse, json, os, re, sys, time, random, hashlib, urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

QDRANT = "http://localhost:6333"
COLL = "indian_legal_t1_v2"
API_URL = "http://localhost:8090"
LLM_URL = "http://localhost:9086/v1/chat/completions"
LLM_MODEL = "meta-llama-3.1-8b-instruct.Q4_K_M.gguf"
EVAL_DIR = "/opt/indian-legal-ai/eval/v2"

os.makedirs(EVAL_DIR, exist_ok=True)

# ---- HTTP ----
def http_json(url, body=None, method="POST", timeout=120):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode() if body else None, method=method
    )
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def call_llm(system, user, max_tokens=400, temperature=0.2):
    body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = http_json(LLM_URL, body, timeout=90)
    return r["choices"][0]["message"]["content"].strip()

# ---- Checkpoint helpers ----
def jsonl_load(path):
    if not os.path.exists(path): return []
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try: out.append(json.loads(line))
            except: pass
    return out

def jsonl_append(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ---- Stage 1: sample ----
def cmd_sample(args):
    """Sample N chunks balanced by dataset + status."""
    out_path = os.path.join(EVAL_DIR, "samples.jsonl")
    if jsonl_load(out_path) and not args.force:
        print(f"[sample] already have {len(jsonl_load(out_path))} samples at {out_path}; use --force to redo")
        return

    # scroll collection and bucket by (dataset, status)
    print(f"[sample] scrolling {COLL}...")
    buckets = defaultdict(list)
    offset = None
    total = 0
    while True:
        body = {"limit": 500, "with_payload": True, "with_vector": False}
        if offset is not None: body["offset"] = offset
        r = http_json(f"{QDRANT}/collections/{COLL}/points/scroll", body)["result"]
        for p in r["points"]:
            pay = p.get("payload", {})
            ds = pay.get("dataset", "?")
            st = pay.get("status", "?")
            sec = pay.get("section_no") or ""
            text = pay.get("text", "")
            # only keep chunks that look like real statute (has a section number and enough text)
            if len(text) < 200 or len(text) > 1500:
                continue
            if not sec and not pay.get("chapter_no"):
                continue
            buckets[(ds, st)].append({
                "id": p["id"],
                "payload": pay,
            })
        total += len(r["points"])
        if not r.get("next_page_offset"):
            break
        offset = r["next_page_offset"]
    print(f"[sample] scanned {total} points, {len(buckets)} buckets")

    # stratified sample
    random.seed(42)
    per_bucket = max(1, args.n // max(1, len(buckets)))
    samples = []
    for k, pts in buckets.items():
        random.shuffle(pts)
        take = pts[:per_bucket]
        for p in take:
            p["bucket"] = {"dataset": k[0], "status": k[1]}
            samples.append(p)
    # trim to n
    random.shuffle(samples)
    samples = samples[:args.n]
    # wipe + write
    open(out_path, "w").close()
    for s in samples:
        jsonl_append(out_path, s)
    print(f"[sample] wrote {len(samples)} samples -> {out_path}")
    cnt = Counter((s["payload"].get("dataset"), s["payload"].get("status")) for s in samples)
    for k,v in sorted(cnt.items()): print(f"   {k}: {v}")


# ---- Stage 2: draft questions ----
DRAFT_SYSTEM = """You are building an evaluation set for an Indian-law retrieval system.

Given a statute chunk, generate ONE factual lookup question that:
- Has a concrete, unambiguous answer contained in the chunk
- A practicing lawyer might actually ask
- Uses natural language, NOT a copy of the section text
- Does not leak the exact section number in the question

Reply with JSON only, on a single line:
{"question": "...", "expected_section": "<section_no>", "expected_act": "<act_name>", "gold_span": "<short quote from chunk containing the answer, max 150 chars>"}

If the chunk is not suitable (too vague, boilerplate, table-of-contents, or table of section headers), reply exactly: {"skip": true}"""

def cmd_draft(args):
    """Generate question per sample chunk via LLM."""
    samples = jsonl_load(os.path.join(EVAL_DIR, "samples.jsonl"))
    out_path = os.path.join(EVAL_DIR, "drafts.jsonl")
    done_ids = {d["id"] for d in jsonl_load(out_path)}
    print(f"[draft] samples={len(samples)} already_done={len(done_ids)}")

    for i, s in enumerate(samples):
        if s["id"] in done_ids: continue
        p = s["payload"]
        head = f"Act: {p.get('act_name','?')}\nSection: {p.get('section_no','')}\nStatus: {p.get('status','?')}\nText:\n{p.get('text','')[:1200]}"
        try:
            raw = call_llm(DRAFT_SYSTEM, head, max_tokens=220, temperature=0.3)
        except Exception as e:
            print(f"  [draft {i}] LLM err: {e}; skipping")
            continue
        # extract JSON
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            print(f"  [draft {i}] no JSON in: {raw[:120]!r}")
            continue
        try:
            q = json.loads(m.group(0))
        except Exception as e:
            print(f"  [draft {i}] JSON parse fail: {e}")
            continue
        if q.get("skip"):
            jsonl_append(out_path, {"id": s["id"], "skipped": True, "bucket": s.get("bucket")})
            continue
        rec = {
            "id": s["id"],
            "question": q.get("question", "").strip(),
            "expected_section": q.get("expected_section"),
            "expected_act": q.get("expected_act"),
            "gold_span": q.get("gold_span"),
            "source_payload": {k: p.get(k) for k in ("act_name","section_no","chapter_no","status","dataset","file","act_year","is_amendment","type")},
            "bucket": s.get("bucket"),
        }
        if not rec["question"]:
            continue
        jsonl_append(out_path, rec)
        if (i+1) % 10 == 0:
            print(f"  [draft] {i+1}/{len(samples)}")
    drafts = [d for d in jsonl_load(out_path) if not d.get("skipped")]
    print(f"[draft] kept {len(drafts)} questions")


# ---- Stage 3: run through /ask ----
def cmd_run(args):
    drafts = [d for d in jsonl_load(os.path.join(EVAL_DIR, "drafts.jsonl")) if not d.get("skipped")]
    out_path = os.path.join(EVAL_DIR, "runs.jsonl")
    done_ids = {r["id"] for r in jsonl_load(out_path)}
    print(f"[run] drafts={len(drafts)} already_done={len(done_ids)}")
    for i, d in enumerate(drafts):
        if d["id"] in done_ids: continue
        t0 = time.time()
        try:
            r = http_json(f"{API_URL}/ask", {"q": d["question"], "top_k": 5}, timeout=120)
        except Exception as e:
            print(f"  [run {i}] err: {e}")
            jsonl_append(out_path, {"id": d["id"], "error": str(e)})
            continue
        rec = {
            "id": d["id"],
            "question": d["question"],
            "expected_section": d.get("expected_section"),
            "expected_act": d.get("expected_act"),
            "gold_span": d.get("gold_span"),
            "source_payload": d.get("source_payload"),
            "answer": r.get("answer",""),
            "raw_answer": r.get("raw_answer",""),
            "sources": r.get("sources",[]),
            "refused": r.get("refused"),
            "stripped": r.get("stripped",[]),
            "elapsed_ms": r.get("elapsed_ms"),
            "wall_ms": int((time.time()-t0)*1000),
        }
        jsonl_append(out_path, rec)
        if (i+1) % 5 == 0:
            print(f"  [run] {i+1}/{len(drafts)}  last={rec['wall_ms']}ms refused={rec['refused']}")
    runs = jsonl_load(out_path)
    print(f"[run] total runs={len(runs)}")


# ---- Stage 4: LLM judge ----
JUDGE_SYSTEM = """You are grading an Indian-law retrieval system against a gold answer.

Given a question, the gold-truth section/act/span, and the system's answer+sources, output JSON only:
{"hit_in_sources": 0 or 1,                    // did the retrieval surface the gold section?
 "citation_accuracy": 0 or 1,                 // does the answer cite the right section (or accept when sources contain it)?
 "answer_correct": 0 | 1 | 2,                 // 0=wrong/contradictory, 1=partially right, 2=fully right
 "refusal_appropriate": 0 or 1 or null,       // null if not refused; 1 if refused AND gold not in sources; 0 if wrongly refused
 "notes": "<brief>"
}

Scoring rules:
- hit_in_sources=1 if any source has the same section_no AND act_name matches (case-insensitive substring either way)
- If system refused: hit_in_sources=0, citation_accuracy=0, answer_correct=0, refusal_appropriate=(1 if gold_span not clearly retrievable else 0)
- answer_correct=2 requires the answer to contain the key fact from gold_span

Do NOT include anything outside the JSON object."""

def build_judge_user(r):
    src_lines = []
    for i, s in enumerate(r.get("sources",[])[:5]):
        src_lines.append(f"  [{i+1}] act={s.get('act')} sec={s.get('section')} status={s.get('status')} | {(s.get('text') or '')[:200]}")
    return (
        f"QUESTION: {r['question']}\n\n"
        f"GOLD:\n  expected_act: {r.get('expected_act')}\n  expected_section: {r.get('expected_section')}\n  gold_span: {r.get('gold_span')}\n\n"
        f"SYSTEM ANSWER: {r.get('answer','')}\n\n"
        f"SYSTEM SOURCES:\n" + ("\n".join(src_lines) or "  (none)") + "\n\n"
        f"REFUSED: {r.get('refused')}\n"
    )

def cmd_judge(args):
    runs = [r for r in jsonl_load(os.path.join(EVAL_DIR, "runs.jsonl")) if "error" not in r]
    out_path = os.path.join(EVAL_DIR, "judged.jsonl")
    done_ids = {j["id"] for j in jsonl_load(out_path)}
    print(f"[judge] runs={len(runs)} already_done={len(done_ids)}")
    for i, r in enumerate(runs):
        if r["id"] in done_ids: continue
        try:
            raw = call_llm(JUDGE_SYSTEM, build_judge_user(r), max_tokens=200, temperature=0)
        except Exception as e:
            print(f"  [judge {i}] LLM err: {e}")
            continue
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            print(f"  [judge {i}] no JSON: {raw[:120]!r}")
            jsonl_append(out_path, {"id": r["id"], "parse_error": True, "raw": raw[:500]})
            continue
        try:
            score = json.loads(m.group(0))
        except Exception as e:
            print(f"  [judge {i}] parse fail: {e}")
            continue
        rec = {"id": r["id"], "question": r["question"], "score": score,
               "expected_section": r.get("expected_section"),
               "expected_act": r.get("expected_act"),
               "refused": r.get("refused"),
               "bucket": r.get("source_payload", {}).get("dataset")}
        jsonl_append(out_path, rec)
        if (i+1) % 10 == 0:
            print(f"  [judge] {i+1}/{len(runs)}")
    print(f"[judge] done")


# ---- Stage 5: report ----
def cmd_report(args):
    judged = [j for j in jsonl_load(os.path.join(EVAL_DIR, "judged.jsonl")) if "score" in j]
    out_md = os.path.join(EVAL_DIR, "SCOREBOARD.md")
    n = len(judged)
    if n == 0:
        print("[report] no judged records")
        return
    hit = sum(1 for j in judged if j["score"].get("hit_in_sources") == 1)
    cite = sum(1 for j in judged if j["score"].get("citation_accuracy") == 1)
    full = sum(1 for j in judged if j["score"].get("answer_correct") == 2)
    part = sum(1 for j in judged if j["score"].get("answer_correct") == 1)
    wrong = sum(1 for j in judged if j["score"].get("answer_correct") == 0)
    refused = sum(1 for j in judged if j.get("refused"))

    per_bucket = defaultdict(lambda: {"n": 0, "hit": 0, "full": 0})
    for j in judged:
        b = j.get("bucket") or "?"
        per_bucket[b]["n"] += 1
        if j["score"].get("hit_in_sources") == 1: per_bucket[b]["hit"] += 1
        if j["score"].get("answer_correct") == 2: per_bucket[b]["full"] += 1

    lines = []
    lines.append(f"# V2 Eval Scoreboard\n")
    lines.append(f"- collection: `{COLL}`")
    lines.append(f"- model: `{LLM_MODEL}`")
    lines.append(f"- n = {n}\n")
    lines.append(f"## Aggregate\n")
    lines.append(f"| metric | count | rate |")
    lines.append(f"|---|---|---|")
    lines.append(f"| hit_in_sources | {hit} | {hit/n:.2%} |")
    lines.append(f"| citation_accuracy | {cite} | {cite/n:.2%} |")
    lines.append(f"| answer_correct=2 (full) | {full} | {full/n:.2%} |")
    lines.append(f"| answer_correct=1 (partial) | {part} | {part/n:.2%} |")
    lines.append(f"| answer_correct=0 (wrong) | {wrong} | {wrong/n:.2%} |")
    lines.append(f"| refused | {refused} | {refused/n:.2%} |")
    lines.append(f"\n## Per-dataset\n")
    lines.append(f"| dataset | n | hit@5 | full_correct |")
    lines.append(f"|---|---|---|---|")
    for b, v in sorted(per_bucket.items()):
        lines.append(f"| {b} | {v['n']} | {v['hit']/max(v['n'],1):.0%} | {v['full']/max(v['n'],1):.0%} |")
    md = "\n".join(lines)
    with open(out_md, "w") as f: f.write(md)
    print(md)
    print(f"\n[report] saved -> {out_md}")


# ---- Driver ----
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("sample"); s1.add_argument("--n", type=int, default=120); s1.add_argument("--force", action="store_true")
    s2 = sub.add_parser("draft")
    s3 = sub.add_parser("run")
    s4 = sub.add_parser("judge")
    s5 = sub.add_parser("report")
    s6 = sub.add_parser("all"); s6.add_argument("--n", type=int, default=120); s6.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.cmd == "sample": cmd_sample(args)
    elif args.cmd == "draft": cmd_draft(args)
    elif args.cmd == "run": cmd_run(args)
    elif args.cmd == "judge": cmd_judge(args)
    elif args.cmd == "report": cmd_report(args)
    elif args.cmd == "all":
        cmd_sample(args); cmd_draft(args); cmd_run(args); cmd_judge(args); cmd_report(args)

if __name__ == "__main__":
    main()
