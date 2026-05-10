"""
Generate HIGH-complexity practitioner training pairs using Claude Opus via CLI.

Target: 200 chunks × 2 questions = 400 pairs, all multi-step / scenario-based.

Why a separate script: the generic pair-gen prompt produces mostly LOW/MEDIUM
questions. For the rare HIGH complexity band (6.5% of our gold set) we need a
much more directive prompt that explicitly requires scenario facts, multiple
entities, and legal reasoning.
"""
from __future__ import annotations
import argparse, json, os, random, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import httpx

QDRANT_URL = os.environ.get("QDRANT_URL", "http://192.168.1.107:6343")
QDRANT_COLL = "cbic_v1"


PROMPT_TMPL = """TASK: Generate 2 HIGH-COMPLEXITY practitioner questions. RESPOND WITH ONLY THE JSON BELOW. DO NOT ASK CLARIFICATIONS. DO NOT SPEAK CONVERSATIONALLY.

Context: training data for a retrieval system over Indian indirect-tax law (CBIC). We already have ample LOW/MEDIUM data; we need HIGH-complexity questions — the hardest kind of practitioner query.

A HIGH-complexity question must have AT LEAST 2 of these traits:
1. **Scenario facts**: names specific parties, GSTINs, states, goods/services, amounts, dates (e.g. "Quantum Tech Pvt Ltd in Karnataka with GSTIN 29AAA…, sells to buyer in Delhi on 15-Apr-2025…")
2. **Multi-entity reasoning**: requires linking 2+ provisions (a section + a rule, or two sections, or a section + a notification)
3. **Cross-act interaction**: interplay between CGST and IGST, or between Customs and GST, or between a notification and its parent rule
4. **Multi-party/multi-jurisdiction**: bill-to vs ship-to, place-of-supply with 3 parties, export/SEZ/EOU complexity
5. **Legal interpretation**: requires interpreting ambiguous text, applying a proviso, or reasoning about a carve-out

Below is ONE chunk from the corpus. Generate 2 HIGH-complexity questions for which THIS chunk is the correct authoritative source. The chunk must realistically be ONE of the authoritative sources — other provisions may also be needed, but the answer relies materially on this chunk.

HARD CONSTRAINTS:
- Each question must read like a real CA / advocate / in-house counsel asking a client problem, not a textbook exercise.
- Use specific names (Pvt Ltd / LLP / proprietorship), specific states/cities, specific goods or services, realistic amounts.
- Do NOT reference the chunk's section number in the question — practitioners describe situations, not cite sections.
- Each question must have a semantic link SPECIFICALLY to this chunk (not answerable by any generic chunk).
- If this chunk is too narrow/procedural/tabular to anchor 2 HIGH-complexity questions, return an empty questions array in reasoning.
- Return STRICTLY valid JSON, no fences.

Output JSON schema:
{{"reasoning": "<1-2 sentences on what this chunk authoritatively covers and why it supports HIGH-complexity queries>", "questions": [{{"q": "<HIGH-complexity scenario question 1>", "why_this_chunk": "<which specific fact/rule/proviso in the chunk is indispensable to answering this>", "complexity_traits": ["scenario","multi_entity", "..."]}}, {{"q": "<HIGH-complexity scenario question 2>", "why_this_chunk": "<...>", "complexity_traits": [...]}}]}}

CHUNK METADATA:
  category: {category}
  subcategory: {subcategory}
  doc_type: {doc_type}
  doc_number: {doc_number}
  section_ref: {section_ref}
  parent_act: {parent_act}
  title: {title}

CHUNK TEXT:
{text}

Return only the JSON object."""


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
            [CLAUDE_CMD, "-p", "--output-format", "json", "--model", model],
            env=env, shell=False, input=prompt, capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
        if r.returncode != 0: return None
        try:
            env_out = json.loads(r.stdout)
            return env_out.get("result", "").strip()
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
    return None


def sample_chunks(n: int, seed: int, exclude_ids: set[str]) -> list[dict]:
    """Sample substantive chunks — bias toward longer text (more likely to support HIGH complexity)."""
    rng = random.Random(seed)
    r = httpx.get(f"{QDRANT_URL}/collections/{QDRANT_COLL}", timeout=10.0)
    total = r.json()["result"]["points_count"]
    print(f"[sample] collection has {total} points", file=sys.stderr)
    candidates, offset, batches = [], None, 0
    target = max(n * 6, 2000)
    while len(candidates) < target and batches < 300:
        body = {"limit": 500, "with_payload": True, "with_vector": False}
        if offset is not None: body["offset"] = offset
        r = httpx.post(f"{QDRANT_URL}/collections/{QDRANT_COLL}/points/scroll",
                       json=body, timeout=30.0)
        data = r.json()["result"]
        pts = data["points"]; next_off = data.get("next_page_offset")
        for p in pts:
            pl = p.get("payload", {})
            if str(p.get("id")) in exclude_ids: continue
            text = (pl.get("text") or "").strip()
            # Bias toward richer chunks: ≥ 400 chars of prose text, not tables
            if len(text) < 400: continue
            if pl.get("is_table"): continue
            if text.count("\n") > 20 and len(text)/max(text.count("\n"),1) < 30: continue
            # Prefer chunks with a meaningful section_ref (more likely to anchor multi-entity questions)
            # but don't strictly require it
            candidates.append({
                "chunk_id": p.get("id"),
                "doc_id": pl.get("doc_id"),
                "category": pl.get("category"),
                "subcategory": pl.get("subcategory"),
                "doc_type": pl.get("doc_type"),
                "doc_number": pl.get("doc_number"),
                "section_ref": pl.get("section_ref"),
                "parent_act": pl.get("parent_act"),
                "title": pl.get("title"),
                "text": text,
            })
        batches += 1
        if not next_off: break
        offset = next_off
    rng.shuffle(candidates)
    chosen = candidates[:n]
    print(f"[sample] {len(candidates)} cand (text ≥600 chars), returning {len(chosen)}", file=sys.stderr)
    return chosen


def worker(c: dict, model: str) -> dict:
    prompt = PROMPT_TMPL.format(
        category=c.get("category") or "?",
        subcategory=c.get("subcategory") or "?",
        doc_type=c.get("doc_type") or "?",
        doc_number=c.get("doc_number") or "?",
        section_ref=c.get("section_ref") or "?",
        parent_act=c.get("parent_act") or "?",
        title=(c.get("title") or "")[:200],
        text=c["text"][:3500],
    )
    raw = call_claude(prompt, model=model)
    parsed = repair_json(raw) if raw else None
    return {"chunk": c, "result": parsed, "raw": raw if parsed is None else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of chunks (2 Qs each → n*2 pairs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--model", default="opus")
    ap.add_argument("--exclude", action="append", default=[], help="existing pairs JSONL(s) to exclude")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    exclude_ids = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try: exclude_ids.add(str(json.loads(line).get("chunk_id")))
            except Exception: pass
    for ex in args.exclude:
        ep = Path(ex)
        if ep.exists():
            before = len(exclude_ids)
            for line in ep.read_text(encoding="utf-8").splitlines():
                try: exclude_ids.add(str(json.loads(line).get("chunk_id")))
                except Exception: pass
            print(f"[exclude] +{len(exclude_ids)-before} from {ex}", file=sys.stderr)
    print(f"[exclude] total {len(exclude_ids)} excluded chunk_ids", file=sys.stderr)

    chunks = sample_chunks(args.n, args.seed, exclude_ids)
    print(f"[queue] {len(chunks)} chunks (model={args.model}, workers={args.workers})", file=sys.stderr)

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
            # Claude sometimes returns bare list of questions instead of {reasoning, questions}
            if isinstance(res, list):
                res = {"reasoning": "", "questions": res}
            if not isinstance(res, dict):
                counters["err"] += 1; continue
            qs = res.get("questions") or []
            if not qs:
                counters["skip"] += 1; continue
            rec = {**c, "claude_reasoning": res.get("reasoning",""), "questions": qs, "generator": f"claude-{args.model}-highcomplex"}
            with write_lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            counters["ok"] += 1
            if done % 5 == 0 or done <= 3:
                rate = done / max(time.time()-t0, 1)
                eta = (len(chunks) - done) / max(rate, 0.01)
                print(f"[{done}/{len(chunks)}] ok={counters['ok']} skip={counters['skip']} err={counters['err']}  rate={rate:.2f}/s  eta={eta/60:.1f}m  pairs_so_far={counters['ok']*2}",
                      file=sys.stderr)

    print(f"\nDONE: ok={counters['ok']} skip={counters['skip']} err={counters['err']}  pairs={counters['ok']*2}  elapsed={time.time()-t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
