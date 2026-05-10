"""Build Set 5: 100 mixed-type doc_ids + filtered gold from existing pairs.

Prefix mix matches Sets 2/3/4 (x2):
  36 cbic-notification-msts
  28 cbic-circular-msts
   8 cbic-allied-act-dtls
   6 cbic-form-msts
   6 cbic-instruction-msts
   6 cbic-rule-msts
   4 cbic-others-document-msts
   2 cbic-regulation-doc-msts
   2 cbic-order-msts
   2 cbic-attachment-dtls
 ---
 100

Excludes any doc_id already in Sets 2/3/4 or in cbic_v2_gst50 (Set 1).
Uses pairs_2000_20260422.jsonl + pairs_opus_highcomplex.jsonl + pairs_sonnet_lowcomplex.jsonl
(the three pair sources that carry doc_id explicitly).
"""
import json, sqlite3, random, collections, os
from pathlib import Path

random.seed(20260425)

ROOT = "/opt/indian-legal-ai"
MANIFEST = f"{ROOT}/data/ingest_manifest_v2.sqlite"
PAIRS_DIR = f"{ROOT}/eval/training_pairs"
SCALE_DIR = f"{ROOT}/reingest_spec/eval/scale_sets"
SET5 = f"{SCALE_DIR}/set5"
os.makedirs(SET5, exist_ok=True)

MIX = {
    "cbic-notification-msts": 36,
    "cbic-circular-msts":     28,
    "cbic-allied-act-dtls":    8,
    "cbic-form-msts":          6,
    "cbic-instruction-msts":   6,
    "cbic-rule-msts":          6,
    "cbic-others-document-msts": 4,
    "cbic-regulation-doc-msts":  2,
    "cbic-order-msts":           2,
    "cbic-attachment-dtls":      2,
}

# 1. exclude doc_ids already used in any prior set or GST50
exclude = set()
for s in ("set2","set3","set4"):
    p = f"{SCALE_DIR}/{s}/doc_ids.csv"
    if os.path.exists(p):
        for line in open(p):
            line=line.strip()
            if line: exclude.add(line)
# Also exclude already phase2_done docs (GST50 etc) to avoid clashes
con = sqlite3.connect(MANIFEST)
done = {r[0] for r in con.execute("SELECT doc_id FROM docs WHERE phase2_done=1").fetchall()}
exclude |= done
print(f"excluding {len(exclude)} doc_ids (prior sets + phase2_done)")

# 2a. build set of doc_ids that have at least one pair (so Set 5 has gold coverage)
pair_docs = set()
sources = ["pairs_2000_20260422.jsonl","pairs_opus_highcomplex.jsonl","pairs_sonnet_lowcomplex.jsonl"]
for src in sources:
    path = f"{PAIRS_DIR}/{src}"
    if not os.path.exists(path): continue
    for line in open(path):
        try: d = json.loads(line)
        except: continue
        did = d.get("doc_id")
        if did: pair_docs.add(did)
print(f"doc_ids with at least one pair: {len(pair_docs)}")

# 2b. sample doc_ids per prefix, restricted to pair_docs ∩ phase2_done=0 ∩ not excluded
sampled = []
for prefix, n in MIX.items():
    cands = [r[0] for r in con.execute(
        f"SELECT doc_id FROM docs WHERE doc_id LIKE ? AND phase2_done=0",
        (prefix+":%",)).fetchall() if r[0] not in exclude and r[0] in pair_docs]
    if len(cands) < n:
        print(f"WARN: only {len(cands)} candidates for {prefix} (pair-covered, unprocessed), requested {n}")
        # top up with non-pair-covered docs of same prefix
        extras = [r[0] for r in con.execute(
            f"SELECT doc_id FROM docs WHERE doc_id LIKE ? AND phase2_done=0",
            (prefix+":%",)).fetchall() if r[0] not in exclude and r[0] not in pair_docs]
        random.shuffle(extras)
        random.shuffle(cands)
        sampled.extend(cands + extras[: max(0, n - len(cands))])
        sampled = sampled[: sum(MIX[p] for p in list(MIX)[:list(MIX).index(prefix)+1])]
    else:
        random.shuffle(cands)
        sampled.extend(cands[:n])
print(f"sampled {len(sampled)} doc_ids")

# 3. write doc_ids.csv
with open(f"{SET5}/doc_ids.csv","w") as f:
    for did in sampled: f.write(did+"\n")
print(f"wrote {SET5}/doc_ids.csv")

# 4. filter pairs
sampled_set = set(sampled)
queries = []
sources = ["pairs_2000_20260422.jsonl","pairs_opus_highcomplex.jsonl","pairs_sonnet_lowcomplex.jsonl"]
for src in sources:
    path = f"{PAIRS_DIR}/{src}"
    if not os.path.exists(path):
        print(f"missing {path}"); continue
    for line in open(path):
        try: d = json.loads(line)
        except: continue
        did = d.get("doc_id")
        if did not in sampled_set: continue
        # pairs_2000 has 'questions' array; opus/sonnet have 'q' directly per pair
        if "questions" in d and isinstance(d["questions"], list):
            for q in d["questions"]:
                qt = q.get("q") or q.get("query")
                if not qt: continue
                queries.append({
                    "query": qt,
                    "expected_doc_id": did,
                    "expected_section": d.get("section_ref") or None,
                    "expected_chunk_id": d.get("chunk_id"),
                    "category": d.get("category"),
                    "_source": src,
                })
        else:
            qt = d.get("q") or d.get("query") or d.get("question")
            if not qt: continue
            queries.append({
                "query": qt,
                "expected_doc_id": did,
                "expected_section": d.get("section_ref") or None,
                "expected_chunk_id": d.get("chunk_id"),
                "category": d.get("category"),
                "_source": src,
            })

print(f"filtered {len(queries)} queries")
out = {"queries": queries}
with open(f"{SET5}/v2_gold_set5.json","w") as f:
    json.dump(out, f, indent=2)
print(f"wrote {SET5}/v2_gold_set5.json")

# Coverage
covered = collections.Counter(q["expected_doc_id"] for q in queries)
miss = [d for d in sampled if d not in covered]
print(f"docs with >=1 query: {len(covered)}/{len(sampled)}, no-query: {len(miss)}")
if miss[:10]: print("sample no-query:", miss[:10])
