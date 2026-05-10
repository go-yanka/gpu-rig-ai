"""
Generate (question, source_chunk) training pairs using Claude CLI Opus.

Mirrors gen_training_pairs_v2.py (Gemini) but calls claude -p with --model opus.
Different seed sampling to get DIFFERENT chunks from Gemini run — so the two sets
are complementary, not overlapping.

Usage:
    python gen_training_pairs_claude.py --n 2000 \\
        --out D:\\_gpu_rig_ai\\eval\\training_pairs\\pairs_claude_2000.jsonl \\
        --workers 3
"""
from __future__ import annotations
import argparse, json, os, random, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import httpx

QDRANT_URL = os.environ.get("QDRANT_URL", "http://192.168.1.107:6343")
QDRANT_COLL = "cbic_v1"


PROMPT_TMPL = """TASK: Generate 2 practitioner questions for a training dataset. RESPOND WITH ONLY THE JSON OBJECT BELOW. DO NOT ASK QUESTIONS. DO NOT RESPOND CONVERSATIONALLY.

Context: This is a retrieval-training dataset for Indian indirect-tax law (CBIC: GST, Customs, Central Excise, Service Tax). The retrieval system finds the correct authoritative chunk when a tax practitioner asks a real-world question.

Below is ONE chunk from the corpus. Generate 2 realistic questions that an Indian tax practitioner (CA, advocate, or in-house tax counsel) might ask, for which THIS chunk is the correct authoritative answer source.

REQUIREMENTS for each question:
1. Realistic practitioner phrasing — how someone actually asks, not textbook prose. Examples: "Can ITC be claimed on CSR expenses?", "Is reverse charge applicable on manpower supply from a proprietorship to a company?", "What is the time limit for issuing invoice for continuous supply of services?"
2. The question must be answerable PRIMARILY from this chunk (not requiring synthesis across many chunks).
3. Questions should differ — one more specific/technical, one more scenario-based if possible. Aim for complexity diversity: one LOW/MEDIUM (direct lookup or simple rule application) and one MEDIUM/HIGH (scenario with specific facts).
4. NO trivial paraphrases of the chunk text. Do not ask "what does Section 16 say?" — ask what Section 16 would actually be cited to answer.
5. The question should have a SEMANTIC link specifically to this chunk. A generic question that could be answered by many chunks is BAD training data.
6. Avoid referencing the section/rule number in the question unless it's something a practitioner would naturally mention — users usually describe a situation, not quote a section.
7. If the chunk is a pure table header, a page number, a TOC entry, or otherwise not substantive enough to anchor 2 real questions, return an empty questions array and explain in reasoning.

Return ONLY valid JSON, no markdown fences, no prose outside JSON:
{{"reasoning": "<1-2 sentence rationale: what does this chunk authoritatively cover?>", "questions": [{{"q": "<question 1>", "why_this_chunk": "<1 sentence: which specific fact/rule in the chunk answers this>", "complexity": "low|medium|high"}}, {{"q": "<question 2>", "why_this_chunk": "<1 sentence>", "complexity": "low|medium|high"}}]}}

CHUNK METADATA:
  category: {category}
  subcategory: {subcategory}
  doc_type: {doc_type}
  doc_number: {doc_number}
  section_ref: {section_ref}
  title: {title}

CHUNK TEXT:
{text}

Return only the JSON object, nothing else."""


def _resolve_claude_cmd() -> str:
    if os.name == "nt":
        c = Path(os.environ.get("APPDATA", "")) / "npm" / "claude.cmd"
        if c.exists(): return str(c)
    return "claude"

CLAUDE_CMD = _resolve_claude_cmd()


def call_claude(prompt: str, model: str = "opus", timeout: int = 180) -> str | None:
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
    try:
        r = subprocess.run(
            [CLAUDE_CMD, "-p", prompt, "--output-format", "json", "--model", model],
            env=env, shell=False, capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
        if r.returncode != 0: return None
        # Parse JSON envelope and extract the result field
        try:
            envelope = json.loads(r.stdout)
            return envelope.get("result", "").strip()
        except Exception:
            return r.stdout.strip()
    except Exception:
        return None


def repair_json(text: str) -> dict | None:
    if not text: return None
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try: return json.loads(t)
    except Exception: pass
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        try: return json.loads(m.group(0))
        except Exception: pass
    # try to salvage trailing
    mm = re.search(r'\{\s*"reasoning"\s*:\s*"([^"]*)"\s*,\s*"questions"\s*:\s*\[(.*)', t, re.DOTALL)
    if mm:
        reasoning = mm.group(1); qblob = mm.group(2)
        qp = re.compile(r'\{\s*"q"\s*:\s*"([^"]+)"\s*,\s*"why_this_chunk"\s*:\s*"([^"]+)"\s*(?:,\s*"complexity"\s*:\s*"([^"]+)"\s*)?\}', re.DOTALL)
        qs = []
        for mm2 in qp.finditer(qblob):
            q = {"q": mm2.group(1), "why_this_chunk": mm2.group(2)}
            if mm2.group(3): q["complexity"] = mm2.group(3)
            qs.append(q)
        if qs: return {"reasoning": reasoning, "questions": qs}
    return None


def sample_chunks(n: int, seed: int = 99) -> list[dict]:
    """Seeded sampling so Claude run selects DIFFERENT chunks than Gemini (seed=42)."""
    rng = random.Random(seed)
    r = httpx.get(f"{QDRANT_URL}/collections/{QDRANT_COLL}", timeout=10.0)
    total = r.json()["result"]["points_count"]
    print(f"[sample] collection has {total} points", file=sys.stderr)
    candidates, offset, batches = [], None, 0
    target = n * 4
    while len(candidates) < target and batches < 30:
        body = {"limit": 500, "with_payload": True, "with_vector": False}
        if offset is not None: body["offset"] = offset
        r = httpx.post(f"{QDRANT_URL}/collections/{QDRANT_COLL}/points/scroll",
                       json=body, timeout=30.0)
        data = r.json()["result"]
        pts = data["points"]; next_off = data.get("next_page_offset")
        for p in pts:
            pl = p.get("payload", {})
            text = (pl.get("text") or "").strip()
            if len(text) < 200: continue
            if pl.get("is_table"): continue
            if text.count("\n") > 20 and len(text)/max(text.count("\n"),1) < 30: continue
            candidates.append({
                "chunk_id": p.get("id"),
                "doc_id": pl.get("doc_id"),
                "category": pl.get("category"),
                "subcategory": pl.get("subcategory"),
                "doc_type": pl.get("doc_type"),
                "doc_number": pl.get("doc_number"),
                "section_ref": pl.get("section_ref"),
                "title": pl.get("title"),
                "text": text,
            })
        batches += 1
        if not next_off: break
        offset = next_off
        if rng.random() < 0.6 and total > 2000:
            try: offset = rng.randint(0, max(0, total - 1000))
            except Exception: pass
    rng.shuffle(candidates)
    chosen = candidates[:n]
    print(f"[sample] {len(candidates)} cand, returning {len(chosen)}", file=sys.stderr)
    return chosen


def worker(c: dict, model: str) -> dict:
    prompt = PROMPT_TMPL.format(
        category=c.get("category") or "?",
        subcategory=c.get("subcategory") or "?",
        doc_type=c.get("doc_type") or "?",
        doc_number=c.get("doc_number") or "?",
        section_ref=c.get("section_ref") or "?",
        title=(c.get("title") or "")[:200],
        text=c["text"][:3000],
    )
    raw = call_claude(prompt, model=model)
    parsed = repair_json(raw) if raw else None
    return {"chunk": c, "result": parsed, "raw": raw if parsed is None else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--model", default="opus")
    ap.add_argument("--exclude", help="existing pairs JSONL whose chunk_ids should be excluded")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try: done_ids.add(str(json.loads(line).get("chunk_id")))
            except Exception: pass
        print(f"[resume] {len(done_ids)} existing records in out", file=sys.stderr)
    if args.exclude and Path(args.exclude).exists():
        n_before = len(done_ids)
        for line in Path(args.exclude).read_text(encoding="utf-8").splitlines():
            try: done_ids.add(str(json.loads(line).get("chunk_id")))
            except Exception: pass
        print(f"[exclude] added {len(done_ids)-n_before} chunk_ids from {args.exclude}", file=sys.stderr)

    chunks = sample_chunks(args.n + len(done_ids), args.seed)
    chunks = [c for c in chunks if str(c["chunk_id"]) not in done_ids][:args.n]
    print(f"[queue] {len(chunks)} new chunks to process (model={args.model}, workers={args.workers})", file=sys.stderr)

    t0 = time.time()
    write_lock = Lock()
    counters = {"ok": 0, "skip": 0, "err": 0}

    with out_path.open("a", encoding="utf-8") as f, \
         ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(worker, c, args.model): c for c in chunks}
        done = 0
        for fut in as_completed(futs):
            done += 1
            r = fut.result()
            c = r["chunk"]; res = r["result"]
            if res is None:
                counters["err"] += 1
                if done % 10 == 0 or done <= 5:
                    print(f"[{done}/{len(chunks)}] ERR raw={(r.get('raw') or '')[:80]!r}", file=sys.stderr)
                continue
            qs = res.get("questions") or []
            if not qs:
                counters["skip"] += 1; continue
            rec = {**c, "claude_reasoning": res.get("reasoning",""), "questions": qs, "generator": f"claude-{args.model}"}
            with write_lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            counters["ok"] += 1
            if done % 10 == 0 or done <= 5:
                rate = done / max(time.time()-t0, 1)
                eta = (len(chunks) - done) / max(rate, 0.01)
                print(f"[{done}/{len(chunks)}] ok={counters['ok']} skip={counters['skip']} err={counters['err']}  rate={rate:.2f}/s  eta={eta/60:.1f}m",
                      file=sys.stderr)

    print(f"\nDONE: ok={counters['ok']} skip={counters['skip']} err={counters['err']}  elapsed={time.time()-t0:.0f}s  out={out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
