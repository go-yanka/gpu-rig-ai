#!/usr/bin/env python3
"""Recall@5 for 4 retrieval variants, using the live /query API only.

Variants: baseline, hyde, meta_filter, wider_rerank.
wider_rerank relies on env swap done externally; this script is variant-agnostic.
"""
import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import httpx
import yaml

API_URL = "http://127.0.0.1:9500"
LLM_URL = "http://127.0.0.1:9082"
LLM_MODEL = "qwen3-14b"

HYDE_SYS = (
    "You are a concise Indian tax-law expert. Given a question, write a 2-3 sentence "
    "hypothetical answer that names the likely Act, section, rule or notification. "
    "Do NOT hedge. Output only the hypothetical answer."
)

META_PATTERNS = [
    (re.compile(r"\bIGST\b", re.I), "Act: IGST."),
    (re.compile(r"\bCGST\b|\bSGST\b", re.I), "Act: CGST."),
    (re.compile(r"\bcustoms\b", re.I), "Act: Customs."),
    (re.compile(r"\bcentral excise\b|\bcenvat\b|\bexcise\b", re.I), "Act: Central Excise."),
    (re.compile(r"\bservice tax\b|\bfinance act\b", re.I), "Act: Service Tax."),
]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def citation_contains(c: dict, needle: str) -> bool:
    n = norm(needle)
    if not n:
        return False
    fields = [
        c.get("title", ""),
        c.get("doc_id", ""),
        c.get("number", ""),
        c.get("subcategory", ""),
        c.get("category", ""),
        (c.get("excerpt") or "")[:2000],
        (c.get("text_full") or "")[:2000],
    ]
    hay = " ".join(str(f) for f in fields)
    return n in norm(hay)


def apply_meta_filter(q: str) -> str:
    prefixes = []
    for pat, label in META_PATTERNS:
        if pat.search(q):
            prefixes.append(label)
            break  # only one
    if not prefixes:
        return q
    return " ".join(prefixes) + " " + q


async def hyde_expand(client: httpx.AsyncClient, question: str) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": HYDE_SYS + " /no_think"},
            {"role": "user", "content": question + " /no_think"},
        ],
        "temperature": 0.0,
        "max_tokens": 200,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        r = await client.post(f"{LLM_URL}/v1/chat/completions",
                              json=payload, timeout=60.0)
        r.raise_for_status()
        j = r.json()
        hyp = j["choices"][0]["message"]["content"].strip()
        # strip any <think> blocks just in case
        hyp = re.sub(r"<think>.*?</think>", "", hyp, flags=re.S).strip()
        return hyp
    except Exception as e:
        print(f"  hyde err: {e}", file=sys.stderr)
        return ""


async def do_query(client: httpx.AsyncClient, question: str, k: int = 5) -> dict:
    last_exc = None
    for attempt in range(6):
        try:
            r = await client.post(f"{API_URL}/query",
                                  json={"question": question, "k": k},
                                  timeout=httpx.Timeout(300.0, connect=10.0))
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            # exponential backoff: 3, 6, 12, 24, 48, 96 s
            wait = min(96, 3 * (2 ** attempt))
            await asyncio.sleep(wait)
    raise last_exc  # type: ignore[misc]


def evaluate(item: dict, citations: list, k: int = 5):
    expected = (
        (item.get("expected_sections") or [])
        + (item.get("expected_rules") or [])
        + (item.get("expected_notifications") or [])
    )
    if not expected:
        expected = item.get("expected_conclusion_keywords") or []

    top = citations[:k]
    per_entity = {}
    hit1 = hit5 = False
    for ent in expected:
        ranks = []
        for rank, c in enumerate(citations, 1):
            if citation_contains(c, ent):
                ranks.append(rank)
        per_entity[ent] = ranks[:5]
        if ranks:
            if ranks[0] == 1:
                hit1 = True
            if ranks[0] <= k:
                hit5 = True
    return {
        "n_expected": len(expected),
        "hit_at_1": hit1,
        "hit_at_5": hit5,
        "per_entity": per_entity,
        "top_k_meta": [
            {
                "rank": j + 1,
                "doc_id": c.get("doc_id"),
                "number": c.get("number"),
                "title": (c.get("title") or "")[:100],
                "subcategory": c.get("subcategory"),
                "score": c.get("score"),
            }
            for j, c in enumerate(top)
        ],
    }


async def run_one(sem, client, item, variant, out_fh):
    async with sem:
        q0 = item["question"]
        t0 = time.perf_counter()
        try:
            if variant == "baseline":
                q = q0
                hyp = None
            elif variant == "hyde":
                hyp = await hyde_expand(client, q0)
                q = (hyp + " " + q0).strip() if hyp else q0
            elif variant == "meta_filter":
                q = apply_meta_filter(q0)
                hyp = None
            elif variant == "wider_rerank":
                q = q0
                hyp = None
            else:
                raise ValueError(variant)

            resp = await do_query(client, q, k=5)
            cits = resp.get("citations", []) or []
            ev = evaluate(item, cits, k=5)
            elapsed = time.perf_counter() - t0
            rec = {
                "variant": variant,
                "id": item["id"],
                "category": item.get("category"),
                "subcategory": item.get("subcategory"),
                "question": q0,
                "used_question": q,
                "hyde_text": hyp,
                "n_citations": len(cits),
                "elapsed_s": round(elapsed, 2),
                **ev,
            }
        except Exception as e:
            elapsed = time.perf_counter() - t0
            rec = {
                "variant": variant,
                "id": item["id"],
                "category": item.get("category"),
                "question": q0,
                "error": str(e),
                "elapsed_s": round(elapsed, 2),
                "hit_at_1": False,
                "hit_at_5": False,
                "n_expected": 0,
            }
        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_fh.flush()
        return rec


async def run_variant(variant, items, out_path, concurrency=2):
    sem = asyncio.Semaphore(concurrency)
    results = []
    with open(out_path, "a", encoding="utf-8") as fh:
        async with httpx.AsyncClient() as client:
            tasks = [run_one(sem, client, it, variant, fh) for it in items]
            done_n = 0
            total = len(tasks)
            for coro in asyncio.as_completed(tasks):
                r = await coro
                results.append(r)
                done_n += 1
                if done_n % 10 == 0 or done_n == total:
                    hits = sum(1 for x in results if x.get("hit_at_5"))
                    print(f"  [{variant}] {done_n}/{total} hit@5 running={hits}",
                          file=sys.stderr)
    return results


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", required=True,
                    choices=["baseline", "hyde", "meta_filter", "wider_rerank"])
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    gold = yaml.safe_load(Path(args.gold).read_text(encoding="utf-8"))
    items = gold.get("items") if isinstance(gold, dict) else gold
    if args.limit:
        items = items[: args.limit]

    t0 = time.perf_counter()
    results = await run_variant(args.variant, items, args.out,
                                 concurrency=args.concurrency)
    elapsed = time.perf_counter() - t0
    n = len(results)
    h1 = sum(1 for r in results if r.get("hit_at_1"))
    h5 = sum(1 for r in results if r.get("hit_at_5"))
    errs = sum(1 for r in results if r.get("error"))
    print(f"\n=== {args.variant} ===", file=sys.stderr)
    print(f"items: {n}  hit@1: {h1} ({100*h1/n:.2f}%)  "
          f"hit@5: {h5} ({100*h5/n:.2f}%)  errs: {errs}  "
          f"elapsed: {elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
