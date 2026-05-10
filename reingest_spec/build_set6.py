"""Set 6 build: ~300 docs (244 gold-positive + 50 stratified diversity).

Codified 2026-04-26 (later 4) per DECISIONS.yaml#set6_build.

Run on rig:  python3 reingest_spec/build_set6.py [--ingest]
  - default: writes /tmp/set6_doc_ids.csv + manifest preview
  - --ingest: also kicks off phase-all ingest into cbic_v2_set6
"""
import json, sqlite3, random, os, sys, time, subprocess
from collections import Counter, defaultdict

GOLD_PATH = "/opt/indian-legal-ai/reingest_spec/eval/v2_gold.json"
MANIFEST_V1 = "/opt/indian-legal-ai/data/ingest_manifest_v2.sqlite"
OUT_CSV = "/tmp/set6_doc_ids.csv"
EXTRA_DIVERSITY_DOCS = 50
SEED = 42


def main():
    random.seed(SEED)
    gold = json.load(open(GOLD_PATH))
    queries = gold["queries"] if isinstance(gold, dict) else gold

    # Step 1: gold-positive doc set
    gold_doc_ids = sorted({q.get("expected_doc_id") for q in queries if q.get("expected_doc_id")})
    print(f"[set6] gold-positive docs: {len(gold_doc_ids)} from {len(queries)} queries")

    # Step 2: stratified diversity sample of 50 NON-gold docs from manifest
    conn = sqlite3.connect(MANIFEST_V1)
    cur = conn.cursor()
    # detect schema
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print(f"[set6] manifest tables: {tables}")
    if "docs" in tables:
        cur.execute("PRAGMA table_info(docs)")
        cols = [r[1] for r in cur.fetchall()]
        cat_col = "category" if "category" in cols else None
        cur.execute(f"SELECT doc_id{', '+cat_col if cat_col else ''} FROM docs")
        rows = cur.fetchall()
    else:
        rows = []
    conn.close()

    by_cat = defaultdict(list)
    for r in rows:
        did = r[0]
        cat = r[1] if len(r) > 1 else "unk"
        if did and did not in set(gold_doc_ids):
            by_cat[cat].append(did)

    # Distribute 50 docs across categories proportional to gold distribution
    gold_cat_counts = Counter(q.get("category", "unk") for q in queries)
    total_gold = sum(gold_cat_counts.values())
    diversity = []
    remaining = EXTRA_DIVERSITY_DOCS
    cats_sorted = sorted(gold_cat_counts.items(), key=lambda x: -x[1])
    for i, (cat, n_gold) in enumerate(cats_sorted):
        if i == len(cats_sorted) - 1:
            n_pick = remaining
        else:
            n_pick = min(remaining, max(1, round(EXTRA_DIVERSITY_DOCS * n_gold / total_gold)))
        pool = by_cat.get(cat, [])
        random.shuffle(pool)
        picked = pool[:n_pick]
        diversity.extend(picked)
        remaining -= len(picked)
        print(f"[set6] diversity: {cat} -> picked {len(picked)} of pool {len(pool)}")
        if remaining <= 0:
            break

    # Final doc list
    final_docs = sorted(set(gold_doc_ids) | set(diversity))
    print(f"[set6] FINAL: {len(final_docs)} docs ({len(gold_doc_ids)} gold + {len(diversity)} diversity)")

    open(OUT_CSV, "w").write(",".join(final_docs))
    print(f"[set6] wrote {OUT_CSV}")
    print(f"[set6] first 3: {final_docs[:3]}")

    # Filter gold to set6 docs (every gold-positive doc is in set6 by construction
    # but we still emit a filtered file for the evaluator)
    set6_docset = set(final_docs)
    set6_gold = [q for q in queries if q.get("expected_doc_id") in set6_docset]
    out_gold = "/opt/indian-legal-ai/reingest_spec/eval/v2_gold_set6.json"
    json.dump({"queries": set6_gold}, open(out_gold, "w"), indent=1)
    print(f"[set6] wrote gold subset n={len(set6_gold)} -> {out_gold}")

    if "--ingest" in sys.argv:
        ts = int(time.time())
        log = f"/opt/indian-legal-ai/logs/reingest_set6_{ts}.log"
        env = os.environ.copy()
        env.update({
            "QDRANT_COLL_V2": "cbic_v2_set6",
            "MANIFEST_V2": f"/opt/indian-legal-ai/data/ingest_manifest_set6.sqlite",
            "DENSE_ONLY": "1",
            "EMBED_GPUS": "4,5,6",
            "RADV_DEBUG": "nodcc",
            "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT": "1",
            "PYTHONPATH": "/home/user/.local/lib/python3.10/site-packages:/opt/indian-legal-ai/rag/cbic_rag",
            "PYTHONUNBUFFERED": "1",
        })
        # Delete prior collection if exists, drop manifest
        subprocess.run(["curl", "-s", "-X", "DELETE", "http://127.0.0.1:6343/collections/cbic_v2_set6"])
        try:
            os.remove(env["MANIFEST_V2"])
        except FileNotFoundError:
            pass
        cmd = [
            "/usr/bin/python3", "/opt/indian-legal-ai/reingest_spec/ingest_v2.py",
            "--phase", "all",
            "--doc-ids", ",".join(final_docs),
            "--no-resume",
        ]
        with open(log, "w") as logf:
            p = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                 cwd="/opt/indian-legal-ai")
        print(f"[set6] ingest launched PID={p.pid} log={log}")
        print(f"[set6] expected ~6-7min total (phase2 ~3min, phase3-5 ~3.5min)")


if __name__ == "__main__":
    main()
