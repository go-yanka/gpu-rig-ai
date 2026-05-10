#!/usr/bin/env python3
"""Recall@k variant sweep — run on rig.

Variants:
  A) baseline        retrieve(k=5)
  B) wider+colbert   retrieve(k=50) -> colbert rerank -> top-5
  C) meta_filter     regex-detect parent_act in Q, retrieve(k=5, filter=parent_act)
  D) hyde            qwen3 hypothetical answer -> retrieve(hypothetical, k=5)

Output: JSONL per item + summary. Run each variant serially over 170 gold items.
"""
import argparse, json, os, re, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, "/opt/indian-legal-ai/rag/cbic_rag")
import yaml
import retriever
try:
    from colbert_rerank import rerank_colbert
except Exception:
    rerank_colbert = None

LLAMA = os.environ.get("QWEN_URL", "http://127.0.0.1:9082")


def norm(s): return re.sub(r"\s+", " ", (s or "").lower()).strip()

def chunk_contains(chunk, needle):
    n = norm(needle)
    if not n: return False
    fields = [chunk.get("section_ref",""), chunk.get("doc_number",""),
              chunk.get("hierarchy",""), chunk.get("title",""),
              chunk.get("parent_act",""), chunk.get("text","")[:2000]]
    return n in norm(" ".join(str(f) for f in fields))

def detect_parent_act(q):
    ql = q.lower()
    # Rough mapping — parent_act field values vary, use contains
    if "igst" in ql: return "IGST"
    if "cgst" in ql or "gst" in ql: return "CGST"
    if "customs" in ql or "bcd" in ql or "igcr" in ql: return "Customs"
    if "service tax" in ql or "finance act" in ql: return "Finance"
    if "excise" in ql: return "Excise"
    return None

def hyde(q, max_tokens=120):
    body = {
        "model": "qwen3-14b",
        "messages":[
            {"role":"system","content":"You are an Indian indirect-tax expert. Given a question, write a brief (2-3 sentences) hypothetical answer citing likely sections/rules. /no_think"},
            {"role":"user","content":q},
        ],
        "temperature":0.0, "max_tokens":max_tokens, "stream":False,
    }
    req = urllib.request.Request(f"{LLAMA}/v1/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"].strip()

def eval_hits(chunks, expected, k):
    hit_k1, hit_k5 = False, False
    for ent in expected:
        for rank, c in enumerate(chunks, 1):
            if chunk_contains(c, ent):
                if rank == 1: hit_k1 = True
                if rank <= k: hit_k5 = True
                break
    return hit_k1, hit_k5

def run_variant(items, variant, k):
    results = []
    n = len(items)
    agg_k1 = agg_k5 = 0
    for i, item in enumerate(items, 1):
        q = item["question"]
        try:
            if variant == "baseline":
                chunks = retriever.retrieve(q, k=k)[:k]
            elif variant == "wider_colbert":
                pool = retriever.retrieve(q, k=50)
                if rerank_colbert and pool:
                    try:
                        chunks = rerank_colbert(q, pool, top_n=k)[:k]
                    except Exception:
                        chunks = pool[:k]
                else:
                    chunks = pool[:k]
            elif variant == "meta_filter":
                pact = detect_parent_act(q)
                filt = None
                if pact:
                    # Try as substring-match isn't supported by Qdrant direct; use exact match
                    # We'll try with a MatchValue on parent_act being the detected string; if no hits fall back
                    try:
                        chunks = retriever.retrieve(q, k=k, filters={"parent_act": pact})[:k]
                        if not chunks:
                            chunks = retriever.retrieve(q, k=k)[:k]
                    except Exception:
                        chunks = retriever.retrieve(q, k=k)[:k]
                else:
                    chunks = retriever.retrieve(q, k=k)[:k]
            elif variant == "hyde":
                try:
                    hypo = hyde(q)
                    q2 = q + " " + hypo
                except Exception as e:
                    q2 = q
                chunks = retriever.retrieve(q2, k=k)[:k]
            else:
                raise ValueError(variant)
        except Exception as e:
            print(f"[{variant} {i}/{n}] {item['id']} ERR: {e}", file=sys.stderr)
            results.append({"id":item["id"],"variant":variant,"error":str(e)})
            continue

        expected = (item.get("expected_sections",[]) or []) + \
                   (item.get("expected_rules",[]) or []) + \
                   (item.get("expected_notifications",[]) or [])
        if not expected:
            expected = item.get("expected_conclusion_keywords",[]) or []
        hit_k1, hit_k5 = eval_hits(chunks, expected, k)
        agg_k1 += int(hit_k1); agg_k5 += int(hit_k5)
        results.append({
            "id":item["id"],"variant":variant,"category":item.get("category"),
            "hit_at_1":hit_k1,"hit_at_k":hit_k5,
            "top_k_meta":[{"rank":j+1,"section_ref":c.get("section_ref"),
                           "doc_number":c.get("doc_number"),
                           "title":(c.get("title") or "")[:80]}
                          for j,c in enumerate(chunks)]
        })
        if i % 20 == 0 or i == n:
            print(f"[{variant} {i}/{n}] @1={agg_k1} @{k}={agg_k5}", file=sys.stderr)
    return results, agg_k1, agg_k5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--variants", default="baseline,wider_colbert,meta_filter,hyde")
    args = ap.parse_args()
    gold = yaml.safe_load(Path(args.gold).read_text(encoding="utf-8"))
    items = gold.get("items") if isinstance(gold, dict) else gold

    all_results = []
    summary = {}
    for v in args.variants.split(","):
        v = v.strip()
        print(f"\n=== {v} ===", file=sys.stderr)
        t0 = time.time()
        res, k1, k5 = run_variant(items, v, args.k)
        dt = time.time() - t0
        n = len(items)
        summary[v] = {"n":n, "hit_at_1":k1, "hit_at_k":k5,
                      "pct_at_1":round(100*k1/n,2), "pct_at_k":round(100*k5/n,2),
                      "elapsed_s":round(dt,1)}
        all_results.extend(res)
        print(f"[{v}] done @1={k1}/{n} ({100*k1/n:.2f}%) @{args.k}={k5}/{n} ({100*k5/n:.2f}%) in {dt:.1f}s", file=sys.stderr)

    with open(args.out,"w",encoding="utf-8") as f:
        for r in all_results: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    with open(args.out + ".summary.json","w") as f:
        json.dump(summary,f,indent=2)
    print("\n=== SUMMARY ===", file=sys.stderr)
    for v,s in summary.items():
        print(f"  {v:20s} @1={s['pct_at_1']:5.2f}%  @{args.k}={s['pct_at_k']:5.2f}%  ({s['elapsed_s']}s)", file=sys.stderr)

if __name__ == "__main__":
    main()
