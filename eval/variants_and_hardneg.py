"""
Integrated: run all 4 retrieval variants against 170 gold items via live /query,
AND emit hard negatives from the baseline retrieval for later training.

Local script — talks to http://192.168.1.107:9500/query. Parallel workers.
"""
import argparse, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import httpx, yaml

API = os.environ.get("CBIC_API", "http://192.168.1.107:9500")
QWEN = os.environ.get("CBIC_LLM", "http://192.168.1.107:9082")


def norm(s): return re.sub(r"\s+", " ", (s or "").lower()).strip()


def match_entity(entity: str, chunk: dict) -> bool:
    """Best-effort match: section code + act name separately, looser than pure substring."""
    e = norm(entity)
    if not e: return False
    fields = [
        chunk.get("section_ref",""), chunk.get("doc_number",""),
        chunk.get("hierarchy",""), chunk.get("title",""),
        chunk.get("parent_act",""), (chunk.get("text") or "")[:3000],
        (chunk.get("text_full") or "")[:3000],
    ]
    hay = norm(" ".join(str(f) for f in fields))
    if e in hay: return True
    # split "10(1)(a) IGST" into code + act
    m = re.match(r"^(?:rule|section|sec\.?)?\s*([0-9a-zA-Z()\.\-]+)(?:\s+(cgst|igst|sgst|customs|excise|service tax|finance|st))?$", e)
    if m:
        code, act = m.group(1), m.group(2)
        sec = norm(chunk.get("section_ref") or "")
        if code and sec and (code == sec or sec.startswith(code) or code.startswith(sec)):
            if not act or act in hay: return True
    # notification: NNN/YYYY style
    mn = re.search(r"(\d+)[/\-](\d{2,4})", e)
    if mn and mn.group(0) in norm(chunk.get("doc_number") or ""):
        return True
    return False


def hyde_expand(question: str) -> str:
    """Ask qwen3 for a brief hypothetical answer to augment query embedding."""
    body = {
        "model": "qwen3-14b",
        "messages": [
            {"role":"system","content":"You are an Indian indirect-tax expert. Write a terse 2-sentence hypothetical answer to the question, citing likely sections/rules. /no_think"},
            {"role":"user","content":question},
        ],
        "temperature":0.0, "max_tokens":140, "stream":False,
    }
    try:
        r = httpx.post(f"{QWEN}/v1/chat/completions", json=body, timeout=40.0)
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


def detect_act(question: str) -> str | None:
    q = question.lower()
    if "igst" in q: return "IGST"
    if "cgst" in q: return "CGST"
    if "gst" in q and "cgst" not in q and "igst" not in q: return "CGST"
    if "customs" in q or "bcd " in q or "igcr" in q: return "Customs"
    if "service tax" in q: return "Service Tax"
    if "excise" in q: return "Excise"
    return None


def query_api(question: str, k: int = 8) -> list[dict]:
    """POST /query, return retrieved chunks. This is slow (does generation) but reliable."""
    body = {"question": question, "k": k}
    r = httpx.post(f"{API}/query", json=body, timeout=600.0)
    d = r.json()
    # citations field holds the retrieved chunks
    cites = d.get("citations") or []
    # Normalize: each citation should have some metadata
    return cites


def run_variant(item: dict, variant: str, k: int) -> dict:
    q = item["question"]
    if variant == "baseline":
        qtext = q
        chunks = query_api(qtext, k=k)
    elif variant == "hyde":
        hypo = hyde_expand(q)
        qtext = q + "\n" + hypo if hypo else q
        chunks = query_api(qtext, k=k)
    elif variant == "meta_filter":
        act = detect_act(q)
        qtext = (f"Under {act}, " if act else "") + q
        chunks = query_api(qtext, k=k)
    else:
        raise ValueError(variant)

    expected = (item.get("expected_sections",[]) or []) + \
               (item.get("expected_rules",[]) or []) + \
               (item.get("expected_notifications",[]) or [])
    if not expected:
        expected = item.get("expected_conclusion_keywords",[]) or []

    hit_k1 = hit_k5 = False
    per_entity = {}
    for ent in expected:
        ranks = []
        for i, c in enumerate(chunks[:k], 1):
            if match_entity(ent, c): ranks.append(i)
        per_entity[ent] = ranks
        if ranks:
            if ranks[0] == 1: hit_k1 = True
            if ranks[0] <= 5: hit_k5 = True

    return {
        "id": item["id"], "variant": variant, "category": item.get("category"),
        "question": q, "expected": expected,
        "hit_at_1": hit_k1, "hit_at_5": hit_k5,
        "per_entity": per_entity,
        "top_k": [{
            "rank": i+1, "doc_id": c.get("doc_id"), "doc_number": c.get("doc_number"),
            "section_ref": c.get("section_ref"), "parent_act": c.get("parent_act"),
            "title": (c.get("title") or "")[:100],
            "text_snippet": (c.get("text_full") or c.get("text") or c.get("excerpt") or "")[:300],
            "score": c.get("score"),
        } for i, c in enumerate(chunks[:k])],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=r"D:\_gpu_rig_ai\eval\gold_set.yaml")
    ap.add_argument("--out", default=r"D:\_gpu_rig_ai\eval\variants_results.jsonl")
    ap.add_argument("--summary", default=r"D:\_gpu_rig_ai\consults\variants_20260422.md")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--variants", default="baseline,hyde,meta_filter")
    args = ap.parse_args()

    gold = yaml.safe_load(Path(args.gold).read_text(encoding="utf-8"))
    items = gold.get("items") if isinstance(gold, dict) else gold
    variants = [v.strip() for v in args.variants.split(",")]

    # Resume: skip (id, variant) pairs already graded
    done = set()
    out_path = Path(args.out)
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line); done.add((d["id"], d["variant"]))
            except Exception: pass
    print(f"[resume] {len(done)} already done", file=sys.stderr)

    tasks = [(item, v) for v in variants for item in items if (item["id"], v) not in done]
    print(f"[queue] {len(tasks)} (item, variant) pairs  workers={args.workers}", file=sys.stderr)

    t0 = time.time()
    lock = Lock()
    counters = {"ok": 0, "err": 0}

    with out_path.open("a", encoding="utf-8") as f, \
         ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_variant, item, v, args.k): (item, v) for item, v in tasks}
        n = len(tasks); i = 0
        for fut in as_completed(futs):
            i += 1
            item, v = futs[fut]
            try:
                res = fut.result()
                counters["ok"] += 1
                with lock:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
            except Exception as e:
                counters["err"] += 1
                with lock:
                    f.write(json.dumps({"id":item["id"],"variant":v,"error":str(e)}) + "\n"); f.flush()
            if i % 10 == 0 or i <= 3:
                rate = i / max(time.time()-t0, 1)
                eta = (n - i) / max(rate, 0.001)
                print(f"[{i}/{n}] ok={counters['ok']} err={counters['err']}  rate={rate:.2f}/s  eta={eta/60:.1f}m",
                      file=sys.stderr)

    # Summarize
    rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_variant = {}
    for r in rows:
        if "error" in r: continue
        v = r["variant"]
        by_variant.setdefault(v, {"n":0,"k1":0,"k5":0,"by_cat":{}})
        by_variant[v]["n"] += 1
        by_variant[v]["k1"] += int(r.get("hit_at_1", False))
        by_variant[v]["k5"] += int(r.get("hit_at_5", False))
        c = r.get("category") or "?"
        d = by_variant[v]["by_cat"].setdefault(c, {"n":0,"k5":0})
        d["n"] += 1; d["k5"] += int(r.get("hit_at_5", False))

    # Write markdown summary
    lines = [f"# Recall variants — {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"Gold: {len(items)} items | API: {API} | k={args.k}", "",
             "## Headline", "", "| Variant | @1 | @5 |", "|---|---:|---:|"]
    for v, d in by_variant.items():
        p1 = 100*d["k1"]/d["n"] if d["n"] else 0
        p5 = 100*d["k5"]/d["n"] if d["n"] else 0
        lines.append(f"| {v} | {d['k1']}/{d['n']} ({p1:.1f}%) | {d['k5']}/{d['n']} ({p5:.1f}%) |")
    lines += ["", "## Per category (@5)", "", "| Category | " + " | ".join(by_variant.keys()) + " |",
              "|---|" + "|".join(["---:"]*len(by_variant)) + "|"]
    cats = sorted({c for v in by_variant.values() for c in v["by_cat"]})
    for c in cats:
        row = [c]
        for v in by_variant:
            d = by_variant[v]["by_cat"].get(c, {"n":0,"k5":0})
            row.append(f"{d['k5']}/{d['n']} ({100*d['k5']/d['n']:.0f}%)" if d["n"] else "-")
        lines.append("| " + " | ".join(row) + " |")
    Path(args.summary).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:30]), file=sys.stderr)


if __name__ == "__main__":
    main()
