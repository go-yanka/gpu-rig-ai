"""
Generate (question, source_chunk) training pairs from cbic_v1 using Gemini 2.5 Pro.

Usage:
    python gen_training_pairs.py --n 50 --out ../training_pairs/sample_v1_raw.jsonl

Pulls random chunks from Qdrant cbic_v1 (via rig SSH-forwarded port or direct LAN),
asks Gemini to generate 2 realistic practitioner questions per chunk, writes JSONL.

Output schema per line:
  {chunk_id, doc_id, category, subcategory, section_ref, title, text,
   questions: [{q, reasoning}, ...]}
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

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
    print("ERROR: GEMINI_API_KEY not set", file=sys.stderr)
    sys.exit(1)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://192.168.1.107:6343")
QDRANT_COLL = "cbic_v1"
GEMINI_MODEL = "gemini-2.5-pro"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


PROMPT_TMPL = """You are helping build a training dataset for a retrieval system over Indian indirect-tax law (CBIC: GST, Customs, Central Excise, Service Tax). The retrieval system's job is to find the correct authoritative chunk when a tax practitioner asks a real-world question.

Below is ONE chunk from the corpus. Your job: generate 2 realistic questions that an Indian tax practitioner (CA, advocate, or in-house tax counsel) might ask, for which THIS chunk would be the correct authoritative answer source.

REQUIREMENTS for each question:
1. Realistic practitioner phrasing — how someone actually asks, not textbook prose. Examples: "Can ITC be claimed on CSR expenses?", "Is reverse charge applicable on manpower supply by a proprietorship to a company?", "What is the GST rate on sale of used motor vehicle by a registered dealer?"
2. The question must be answerable PRIMARILY from this chunk (not requiring synthesis across many chunks).
3. Questions should be different from each other — one more specific/technical, one more scenario-based if possible.
4. NO trivial paraphrases of the chunk text. Don't ask "what does Section 16 say?" — ask what Section 16 would actually be cited to answer.
5. Avoid questions that could be answered by any generic tax chunk. The question should have a SEMANTIC link specifically to this chunk's content.
6. If the chunk is a pure table header, a page number, a TOC entry, or otherwise not substantive enough to anchor 2 real questions, return questions: [] and explain in reasoning.

Return ONLY valid JSON (no markdown fences, no prose outside JSON):
{{
  "reasoning": "<1-2 sentence rationale: what does this chunk authoritatively cover?>",
  "questions": [
    {{"q": "<question 1>", "why_this_chunk": "<1 sentence: which specific fact/rule in the chunk answers this>"}},
    {{"q": "<question 2>", "why_this_chunk": "<1 sentence: which specific fact/rule in the chunk answers this>"}}
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
    """Random-sample n substantive chunks (text >= 200 chars, not pure table)."""
    rng = random.Random(seed)
    # Get total points
    r = httpx.get(f"{QDRANT_URL}/collections/{QDRANT_COLL}", timeout=10.0)
    total = r.json()["result"]["points_count"]
    print(f"[sample] collection has {total} points", file=sys.stderr)

    # We need to scroll; Qdrant scroll is sequential. Pull batches & sample.
    # Strategy: scroll with random offset via batched scrolls until we have 3*n candidates.
    candidates = []
    offset = None
    batches = 0
    target = n * 4  # oversample, filter
    while len(candidates) < target and batches < 30:
        body = {
            "limit": 500,
            "with_payload": True,
            "with_vector": False,
        }
        if offset:
            body["offset"] = offset
        r = httpx.post(
            f"{QDRANT_URL}/collections/{QDRANT_COLL}/points/scroll",
            json=body,
            timeout=30.0,
        )
        data = r.json()["result"]
        pts = data["points"]
        next_off = data.get("next_page_offset")
        for p in pts:
            pl = p.get("payload", {})
            text = (pl.get("text") or "").strip()
            if len(text) < 200:
                continue
            if pl.get("is_table"):
                continue
            if text.count("\n") > 20 and len(text) / max(text.count("\n"), 1) < 30:
                # looks like a list/table mis-tagged
                continue
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
        if not next_off:
            break
        offset = next_off
        # Skip ahead randomly to spread the sample
        if rng.random() < 0.6 and total > 2000:
            # random restart: skip a random number of points
            # (simulate by using a random integer offset if ids are int)
            try:
                jump = rng.randint(0, max(0, total - 1000))
                offset = jump
            except Exception:
                pass

    rng.shuffle(candidates)
    chosen = candidates[:n]
    print(f"[sample] {len(candidates)} candidates found, returning {len(chosen)}", file=sys.stderr)
    return chosen


def call_gemini(prompt: str, retries: int = 3) -> dict | None:
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "topP": 0.9,
            "maxOutputTokens": 1200,
            "responseMimeType": "application/json",
        },
    }
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_KEY}
    last_err = None
    for attempt in range(retries):
        try:
            r = httpx.post(GEMINI_URL, json=payload, headers=headers, params=params, timeout=60.0)
            if r.status_code == 429:
                time.sleep(5 + attempt * 5)
                continue
            r.raise_for_status()
            data = r.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
        except Exception as e:
            last_err = e
            time.sleep(2 + attempt * 2)
    print(f"  [gemini ERR] {last_err}", file=sys.stderr)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of chunks to sample")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = sample_chunks(args.n, args.seed)
    if len(chunks) < args.n:
        print(f"[warn] only got {len(chunks)} chunks (wanted {args.n})", file=sys.stderr)

    t0 = time.time()
    ok = skipped = err = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, c in enumerate(chunks, 1):
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
            if result is None:
                err += 1
                print(f"[{i}/{len(chunks)}] FAIL", file=sys.stderr)
                continue
            qs = result.get("questions") or []
            if not qs:
                skipped += 1
                print(f"[{i}/{len(chunks)}] skipped (no questions): {result.get('reasoning','')[:80]}", file=sys.stderr)
                continue
            record = {
                **c,
                "gemini_reasoning": result.get("reasoning", ""),
                "questions": qs,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            ok += 1
            print(f"[{i}/{len(chunks)}] ok ({len(qs)} Qs)  elapsed={time.time()-t0:.0f}s", file=sys.stderr)

    print(f"\nDONE: ok={ok} skipped={skipped} err={err} out={out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
