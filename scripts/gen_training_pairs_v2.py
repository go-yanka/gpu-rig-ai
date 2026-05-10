"""
Gen training pairs v2 — gemini-2.5-flash, concurrent, JSON-repair, thinking disabled.

Fixes vs v1:
- gemini-2.5-flash (3-5x faster, cheaper, no heavy thinking)
- maxOutputTokens=4000 (was 1200; truncation was root cause of ~50% failures)
- ThreadPoolExecutor with 8 concurrent workers
- JSON repair: if raw parse fails, try to salvage via trailing-brace trim / partial array extraction
- Resume-safe: if output file exists, skip chunk_ids already present
"""
import argparse, json, os, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import httpx

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_KEY:
    print("ERROR: GEMINI_API_KEY not set", file=sys.stderr); sys.exit(1)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://192.168.1.107:6343")
QDRANT_COLL = "cbic_v1"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


PROMPT_TMPL = """You are helping build a training dataset for a retrieval system over Indian indirect-tax law (CBIC: GST, Customs, Central Excise, Service Tax). The retrieval system's job is to find the correct authoritative chunk when a tax practitioner asks a real-world question.

Below is ONE chunk from the corpus. Your job: generate 2 realistic questions that an Indian tax practitioner (CA, advocate, or in-house tax counsel) might ask, for which THIS chunk would be the correct authoritative answer source.

REQUIREMENTS:
1. Realistic practitioner phrasing (not textbook prose).
2. Answerable primarily from this chunk.
3. Two different questions — one specific/technical, one scenario-based.
4. NO trivial paraphrases of chunk text.
5. Semantic link must be specifically to this chunk.
6. If chunk is non-substantive (pure header/TOC/page#), return questions: [].

Return ONLY valid JSON, no markdown fences:
{{
  "reasoning": "<1-2 sentence rationale>",
  "questions": [
    {{"q": "<question 1>", "why_this_chunk": "<1 sentence>"}},
    {{"q": "<question 2>", "why_this_chunk": "<1 sentence>"}}
  ]
}}

CHUNK METADATA:
  category: {category}
  subcategory: {subcategory}
  doc_type: {doc_type}
  doc_number: {doc_number}
  section_ref: {section_ref}
  title: {title}

CHUNK TEXT:
{text}
"""


def sample_chunks(n: int, seed: int = 42) -> list[dict]:
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
        pts = data["points"]
        next_off = data.get("next_page_offset")
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


def repair_json(text: str) -> dict | None:
    """Try multiple strategies to recover JSON from truncated/malformed Gemini output."""
    # 1) raw parse
    try: return json.loads(text)
    except Exception: pass
    # 2) strip markdown fences
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try: return json.loads(t)
    except Exception: pass
    # 3) trim to last full question object and close JSON
    #    Look for pattern: "questions": [ { ... }, { ... partial? ]
    m = re.search(r'\{\s*"reasoning"\s*:\s*"([^"]*)"\s*,\s*"questions"\s*:\s*\[(.*)', t, re.DOTALL)
    if m:
        reasoning = m.group(1)
        qblob = m.group(2)
        # find complete question objects
        q_pattern = re.compile(r'\{\s*"q"\s*:\s*"([^"]+)"\s*,\s*"why_this_chunk"\s*:\s*"([^"]+)"\s*\}', re.DOTALL)
        qs = [{"q": mm.group(1), "why_this_chunk": mm.group(2)} for mm in q_pattern.finditer(qblob)]
        if qs:
            return {"reasoning": reasoning, "questions": qs}
    # 4) give up
    return None


def call_gemini(prompt: str, retries: int = 3) -> dict | None:
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "topP": 0.9,
            "maxOutputTokens": 4000,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    params = {"key": GEMINI_KEY}
    last_err = None
    for attempt in range(retries):
        try:
            r = httpx.post(GEMINI_URL, json=payload, params=params, timeout=60.0)
            if r.status_code == 429:
                time.sleep(5 + attempt * 5); continue
            r.raise_for_status()
            data = r.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            out = repair_json(text)
            if out is not None: return out
            last_err = "unparseable after repair"
        except Exception as e:
            last_err = str(e)
        time.sleep(1 + attempt)
    return {"__error__": str(last_err)}


def worker(c: dict) -> dict:
    prompt = PROMPT_TMPL.format(
        category=c.get("category") or "?",
        subcategory=c.get("subcategory") or "?",
        doc_type=c.get("doc_type") or "?",
        doc_number=c.get("doc_number") or "?",
        section_ref=c.get("section_ref") or "?",
        title=(c.get("title") or "")[:200],
        text=c["text"][:3000],
    )
    result = call_gemini(prompt)
    return {"chunk": c, "result": result}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # resume: collect chunk_ids already in output
    done_ids = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try: done_ids.add(str(json.loads(line).get("chunk_id")))
            except Exception: pass
        print(f"[resume] {len(done_ids)} existing records", file=sys.stderr)

    chunks = sample_chunks(args.n, args.seed)
    chunks = [c for c in chunks if str(c["chunk_id"]) not in done_ids]
    print(f"[queue] {len(chunks)} new chunks to process", file=sys.stderr)

    t0 = time.time()
    write_lock = Lock()
    counters = {"ok": 0, "skip": 0, "err": 0}

    with out_path.open("a", encoding="utf-8") as f, \
         ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(worker, c): c for c in chunks}
        done = 0
        for fut in as_completed(futs):
            done += 1
            r = fut.result()
            c = r["chunk"]; res = r["result"]
            if res is None or "__error__" in res:
                counters["err"] += 1
                if done % 10 == 0 or done <= 5:
                    print(f"[{done}/{len(chunks)}] ERR {res.get('__error__','?') if res else 'None'}", file=sys.stderr)
                continue
            qs = res.get("questions") or []
            if not qs:
                counters["skip"] += 1; continue
            rec = {**c, "gemini_reasoning": res.get("reasoning",""), "questions": qs}
            with write_lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            counters["ok"] += 1
            if done % 25 == 0 or done <= 5:
                rate = done / max(time.time()-t0, 1)
                eta = (len(chunks) - done) / max(rate, 0.01)
                print(f"[{done}/{len(chunks)}] ok={counters['ok']} skip={counters['skip']} err={counters['err']}  rate={rate:.1f}/s  eta={eta/60:.1f}m",
                      file=sys.stderr)

    print(f"\nDONE: ok={counters['ok']} skip={counters['skip']} err={counters['err']}  elapsed={time.time()-t0:.0f}s  out={out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
