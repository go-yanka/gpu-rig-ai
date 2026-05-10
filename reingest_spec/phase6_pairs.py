"""phase6_pairs — per-chunk training-pair generator.

Reads operational params from /opt/indian-legal-ai/reingest_spec/DECISIONS.yaml.
Writes append-only to /opt/indian-legal-ai/data/training_corpus/cbic_pairs_v2.jsonl
+ per-scope snapshot. See PAIR_GEN_SPEC.md for design.

Generator mix B (locked 2026-04-25):
  - qwen3-14b @ 9082  : 100% of chunks, 12 q each
  - gemini-flash       : 20% of chunks (hash%5==0), 4 q each
  - claude -p (CLI)    : 10% of chunks (hash%10==0), 2 adversarial q each (free, Max plan)
  - hard negatives     : inline, cosine band 0.60–0.85, k=5, same-doc only
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import argparse, hashlib, json, os, subprocess, sys, time, uuid, re, traceback
from pathlib import Path
import requests
import yaml

ROOT = "/opt/indian-legal-ai"
DECISIONS = f"{ROOT}/reingest_spec/DECISIONS.yaml"
TRAINING_DIR = f"{ROOT}/data/training_corpus"
CANONICAL = f"{TRAINING_DIR}/cbic_pairs_v2.jsonl"
REJECTS   = f"{TRAINING_DIR}/cbic_pairs_v2_rejects.jsonl"
QDRANT_HOST = os.environ.get("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6343"))
QWEN_URL    = os.environ.get("QWEN_URL", "http://127.0.0.1:9082/v1/chat/completions")

os.makedirs(TRAINING_DIR, exist_ok=True)

# ---------------- DECISIONS.yaml ----------------
def load_decisions():
    raw = open(DECISIONS, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()[:16]
    return yaml.safe_load(raw), sha

# ---------------- Generators ----------------
QWEN_PROMPT = """You are generating training queries for a CBIC (Indian tax law) RAG system.

CHUNK META:
- doc_id: {doc_id}
- section: {section_ref}
- category: {category} / {subcategory}
- title: {title}

CHUNK TEXT:
{text}

Generate exactly 12 questions an Indian tax practitioner or CA might ask whose ANSWER is grounded in this chunk. Mix the query types as: 4 factual, 3 scenario (real-world client situation), 2 definition, 2 procedural (how-to), 1 multi_hop (requires combining with another section).

Difficulty mix: 4 basic, 5 medium, 3 hard.

Output STRICT JSON only, no preamble, no markdown:
{{"questions":[{{"q":"...","query_type":"factual|scenario|definition|procedural|multi_hop","difficulty":"basic|medium|hard","domain":"GST|Customs|Excise|Income Tax|Service Tax","reasoning":"why this chunk answers it"}},...]}}"""

GEMINI_PROMPT = """You are generating diverse training queries for a CBIC tax-law RAG system. Phrase them differently from typical legal-textbook style — use real client/scenario phrasing.

CHUNK META: {doc_id} | section={section_ref} | {category}/{subcategory} | title={title}
CHUNK TEXT:
{text}

Generate exactly 4 questions, all scenario or multi_hop. Indian client context. Mix difficulty.
Output STRICT JSON only:
{{"questions":[{{"q":"...","query_type":"scenario|multi_hop","difficulty":"basic|medium|hard","domain":"GST|Customs|Excise|Income Tax|Service Tax","reasoning":"..."}},...]}}"""

CLAUDE_PROMPT = """You are generating ADVERSARIAL training queries for a CBIC tax-law RAG system. Your job is to write queries that test whether the system correctly REFUSES or flags out-of-scope content.

CHUNK META: {doc_id} | section={section_ref} | {category}/{subcategory}
CHUNK TEXT (truncated):
{text}

Generate exactly 2 adversarial questions. One must be `refusal_bait` (asks for advice the system should refuse: e.g., specific legal opinion, predictions, personal recommendation). The other must be `out_of_scope` or `near_miss` (sounds related to this chunk but the chunk does NOT contain the answer — would tempt the system to hallucinate).

Output STRICT JSON only, nothing else:
{{"questions":[{{"q":"...","query_type":"refusal_bait|out_of_scope|near_miss","difficulty":"hard","domain":"GST|Customs|Excise|Income Tax|Service Tax","reasoning":"why this is adversarial"}},...]}}"""

def _extract_json(text: str) -> dict | None:
    """Tolerant JSON extract: find first {...} balanced block."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    # find balanced
    start = text.find("{")
    if start < 0: return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try: return json.loads(text[start:i+1])
                except Exception: return None
    return None

def _fmt_meta(meta: dict, text_max: int) -> dict:
    return {
        "doc_id": meta.get("doc_id", ""),
        "section_ref": meta.get("section_ref", "") or "",
        "category": meta.get("category", "") or "",
        "subcategory": meta.get("subcategory", "") or "",
        "title": (meta.get("source") or meta.get("title") or "")[:200],
        "text": (meta.get("text", "") or "")[:text_max],
    }

def call_qwen(meta: dict, timeout=180) -> list[dict]:
    prompt = QWEN_PROMPT.format(**_fmt_meta(meta, 6000))
    r = requests.post(QWEN_URL, timeout=timeout, json={
        "model": "qwen3-14b",
        "messages": [{"role": "user", "content": prompt + "\n\n/no_think"}],
        "temperature": 0.4, "max_tokens": 3000,
    })
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"]
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()
    d = _extract_json(txt)
    return (d or {}).get("questions", [])

def call_gemini(meta: dict, api_key: str, timeout=120) -> list[dict]:
    prompt = GEMINI_PROMPT.format(**_fmt_meta(meta, 5000))
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    r = requests.post(url, timeout=timeout, json={
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1500, "responseMimeType": "application/json"},
    })
    if r.status_code != 200: return []
    txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    d = _extract_json(txt)
    return (d or {}).get("questions", [])

def call_claude_cli(meta: dict, timeout=60) -> list[dict]:
    prompt = CLAUDE_PROMPT.format(**_fmt_meta(meta, 4000))
    try:
        r = subprocess.run(["claude", "-p", prompt], capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0: return []
        d = _extract_json(r.stdout)
        return (d or {}).get("questions", [])
    except subprocess.TimeoutExpired:
        return []

# ---------------- Hard negatives ----------------
def mine_hard_negs(qclient, query_text: str, embedder, doc_id: str, positive_chunk_id: str, k=5):
    """Embed query, search same-doc chunks, return chunks in cosine band [0.60, 0.85] excluding positive."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    try:
        qvec = embedder(query_text)
    except Exception:
        return []
    res = qclient.query_points(
        collection_name=qclient._coll, query=qvec, using="dense", limit=50,
        query_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
        with_payload=False,
    ).points
    out = []
    for hit in res:
        cid = str(hit.id)
        if cid == positive_chunk_id: continue
        if 0.60 <= hit.score <= 0.85:
            out.append({"chunk_id": cid, "score": round(float(hit.score), 4), "rank": len(out)+2, "reason": "cosine_band"})
        if len(out) >= k: break
    return out

# ---------------- Schema-validate + emit ----------------
REQUIRED = ("pair_id","chunk_id","doc_id","text","question","generator","provenance")

def make_pair(chunk_payload: dict, q: dict, generator: str, scope: str, decisions_sha: str,
              hard_negs: list, q_idx: int) -> dict:
    chunk_id = str(chunk_payload["chunk_id"])
    doc_id = chunk_payload["doc_id"]
    pair_id = f"{scope}_{doc_id.split(':')[-1]}_c{chunk_id[:12]}_q{q_idx:02d}_{generator[:3]}"
    return {
        "pair_id": pair_id,
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "category": chunk_payload.get("category"),
        "subcategory": chunk_payload.get("subcategory"),
        "doc_type": chunk_payload.get("subcategory"),
        "doc_number": chunk_payload.get("notification_id"),
        "section_ref": chunk_payload.get("section_ref"),
        "parent_act": chunk_payload.get("parent_hierarchy_text"),
        "title": chunk_payload.get("source"),
        "text": chunk_payload.get("text", ""),
        "question": q.get("q") or q.get("question"),
        "question_id": q_idx,
        "domain": q.get("domain"),
        "difficulty": q.get("difficulty"),
        "query_type": q.get("query_type"),
        "generator": generator,
        "generator_reasoning": q.get("reasoning"),
        "grading": {},
        "positive_chunk_id": chunk_id,
        "hard_negatives": hard_negs,
        "llm_answer": {},
        "judges": {},
        "gate_verdicts": {},
        "provenance": {
            "phase": "phase6_pairs",
            "scope": scope,
            "generated_ts": int(time.time()),
            "gold_source": f"phase6_{generator}",
            "chunker_version": "v2",
            "decisions_yaml_sha": decisions_sha,
        },
    }

def validate(p: dict) -> tuple[bool, str]:
    for f in REQUIRED:
        if not p.get(f): return False, f"missing:{f}"
    if len(p["question"]) < 10: return False, "question_too_short"
    if len(p["text"]) < 30: return False, "text_too_short"
    return True, ""

# ---------------- Resumability ----------------
def already_done(scope: str) -> set:
    """Return set of (chunk_id, generator) tuples already in canonical for this scope."""
    done = set()
    if not os.path.exists(CANONICAL): return done
    for line in open(CANONICAL):
        try:
            d = json.loads(line)
            if d.get("provenance", {}).get("scope") == scope:
                done.add((d["chunk_id"], d["generator"]))
        except Exception: continue
    return done

# ---------------- Main ----------------
def run(scope: str, collection: str, limit: int | None = None, gemini_key: str = ""):
    decisions, sha = load_decisions()
    pg = decisions["pair_generation"]
    print(f"[phase6] DECISIONS sha={sha} scope={scope} collection={collection}")
    print(f"[phase6] generators: qwen3=100% gemini=20% claude=10%")

    from qdrant_client import QdrantClient
    qclient = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
    qclient._coll = collection  # stash for hard-neg miner

    # Embedder via existing direct facade
    sys.path.insert(0, f"{ROOT}/rag/cbic_rag")
    from embedder_direct import get_pool  # type: ignore
    _pool = get_pool()
    embedder = lambda t: _pool.embed([t])[0]

    snapshot = f"{TRAINING_DIR}/cbic_pairs_v2_{scope}_{time.strftime('%Y%m%d')}.jsonl"
    fc = open(CANONICAL, "a"); fs = open(snapshot, "a"); fr = open(REJECTS, "a")

    done = already_done(scope)
    print(f"[phase6] resuming: {len(done)} (chunk,generator) pairs already written for scope={scope}")

    counts = {"qwen3": 0, "gemini": 0, "claude": 0, "rejects": 0, "chunks": 0}
    offset = None
    while True:
        points, offset = qclient.scroll(
            collection_name=collection, limit=64, offset=offset,
            with_payload=True, with_vectors=False,
        )
        if not points: break
        for pt in points:
            counts["chunks"] += 1
            if limit and counts["chunks"] > limit: break
            payload = pt.payload or {}
            if not payload.get("text"): continue
            chunk_id = str(payload.get("chunk_id") or pt.id)
            payload["chunk_id"] = chunk_id
            h = int(hashlib.sha256(chunk_id.encode()).hexdigest()[:8], 16)
            run_gemini = (h % 5 == 0) and bool(gemini_key)
            run_claude = (h % 10 == 0)

            for gen_name, runner in [
                ("qwen3-14b", lambda: call_qwen(payload)),
                ("gemini-flash" if run_gemini else None, lambda: call_gemini(payload, gemini_key)),
                ("claude-cli" if run_claude else None, lambda: call_claude_cli(payload)),
            ]:
                if gen_name is None: continue
                if (chunk_id, gen_name) in done: continue
                try:
                    questions = runner()
                except Exception as e:
                    fr.write(json.dumps({"chunk_id": chunk_id, "generator": gen_name, "error": f"{type(e).__name__}: {e}"})+"\n")
                    counts["rejects"] += 1; continue
                if not questions:
                    fr.write(json.dumps({"chunk_id": chunk_id, "generator": gen_name, "error": "no_questions_parsed"})+"\n")
                    counts["rejects"] += 1; continue
                # 2026-04-25 audit fix: parallelize hard-neg mining across questions of one chunk.
                # mine_hard_negs = embed (pool, thread-safe) + Qdrant search (thread-safe). Bounded 4-wide.
                def _mine_one(args_iq):
                    i, q = args_iq
                    qtxt = q.get("q") or q.get("question") or ""
                    hn = mine_hard_negs(qclient, qtxt, embedder, payload["doc_id"], chunk_id, k=5) if qtxt else []
                    return i, q, hn
                with ThreadPoolExecutor(max_workers=4) as _ex:
                    mined = list(_ex.map(_mine_one, list(enumerate(questions, start=1))))
                for i, q, hard_negs in mined:
                    pair = make_pair(payload, q, gen_name, scope, sha, hard_negs, i)
                    ok, why = validate(pair)
                    if not ok:
                        fr.write(json.dumps({"chunk_id": chunk_id, "generator": gen_name, "error": f"validate:{why}"})+"\n")
                        counts["rejects"] += 1; continue
                    line = json.dumps(pair, ensure_ascii=False) + "\n"
                    fc.write(line); fs.write(line)
                    key = "qwen3" if gen_name.startswith("qwen") else ("gemini" if gen_name.startswith("gemini") else "claude")
                    counts[key] += 1
            if counts["chunks"] % 10 == 0:
                fc.flush(); fs.flush(); fr.flush()
                print(f"[phase6] chunks={counts['chunks']} qwen3={counts['qwen3']} gemini={counts['gemini']} claude={counts['claude']} rej={counts['rejects']}", flush=True)
        if limit and counts["chunks"] >= limit: break
        if offset is None: break

    fc.close(); fs.close(); fr.close()

    # 2026-04-25 user directive: emit per-doc + per-generator + total counts
    # for every run (DECISIONS.yaml: pair_generation.reporting).
    by_doc = {}; by_gen = {"qwen3-14b":0,"gemini-flash":0,"claude-cli":0}
    for line in open(snapshot):
        try: d = json.loads(line)
        except: continue
        did = d.get("doc_id","?"); gen = d.get("generator","?")
        by_doc.setdefault(did, {"qwen3-14b":0,"gemini-flash":0,"claude-cli":0,"total":0})
        if gen in by_doc[did]:
            by_doc[did][gen] += 1; by_doc[did]["total"] += 1
        if gen in by_gen: by_gen[gen] += 1
    summary = {
        "scope": scope, "collection": collection, "ts": int(time.time()),
        "decisions_yaml_sha": sha,
        "totals": {"chunks_processed": counts["chunks"], "pairs_total": sum(by_gen.values()),
                   "by_generator": by_gen, "rejects": counts["rejects"], "docs_covered": len(by_doc)},
        "per_doc": by_doc,
    }
    summary_path = snapshot.replace(".jsonl", ".summary.json")
    open(summary_path, "w").write(json.dumps(summary, indent=2))
    avg = sum(by_gen.values()) / max(1, len(by_doc))
    print(f"[phase6] DONE chunks={counts['chunks']} qwen3={counts['qwen3']} gemini={counts['gemini']} claude={counts['claude']} rej={counts['rejects']}")
    print(f"[phase6] PER-RUN REPORT: docs={len(by_doc)} pairs_total={sum(by_gen.values())} avg_pairs_per_doc={avg:.1f}")
    print(f"[phase6] by_generator: {by_gen}")
    print(f"[phase6] snapshot: {snapshot}")
    print(f"[phase6] summary:  {summary_path}")
    return counts

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gemini-key", default=os.environ.get("GEMINI_API_KEY", ""))
    a = ap.parse_args()
    try:
        run(a.scope, a.collection, a.limit, a.gemini_key)
    except Exception:
        traceback.print_exc(); sys.exit(1)
