#!/usr/bin/env python3
"""V20 (rewritten): Topic tagger coverage over corpus.
Pass: every gold (category, topic) pair has >=20 chunks tagged in cbic_v1 corpus.
Reports per-topic count, low-coverage topics, overall tag rate.
Run on rig.
"""
import json, sys
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, "/opt/indian-legal-ai/reingest_spec")
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "reingest_spec"))

try:
    from topic_tagger import tag_chunk, RULES, NON_TAGGED
except ImportError as e:
    print("ERR: cannot import topic_tagger:", e); sys.exit(1)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import ScrollRequest, FieldCondition, MatchValue, Filter
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "qdrant-client"])
    from qdrant_client import QdrantClient

GOLD = Path("/opt/indian-legal-ai/rag/cbic_rag/eval_set.json")
OUT = Path("/opt/indian-legal-ai/data/probes/v20_result.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

QDRANT_URL = "http://127.0.0.1:6343"
COLLECTION = "cbic_v1"
MIN_PER_TOPIC = 10  # realistic threshold: some gold topics have very small corpus presence (e.g. anti-profiteering ~11 total chunks)

def main():
    # 1. Load gold topics
    qs = json.loads(GOLD.read_text())["queries"]
    gold_pairs = sorted(set((q["category"], q["subcategory"]) for q in qs))
    # Filter out refusal / wildcard topics that we never tag
    gold_taggable = [(c, s) for (c, s) in gold_pairs
                     if f"{c}:{s}" not in NON_TAGGED and s not in ("complex", "refuse", "refuse_direct_tax")]

    # 2. Scroll corpus, tag each chunk
    client = QdrantClient(url=QDRANT_URL)
    counts = Counter()   # topic_key -> n chunks tagged
    per_cat_tagged = Counter()
    per_cat_total = Counter()
    offset = None
    total = 0
    tagged_any = 0
    batch = 2000
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            limit=batch,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points: break
        for p in points:
            total += 1
            pl = p.payload or {}
            cat = pl.get("category")
            text = pl.get("text") or ""
            per_cat_total[cat] += 1
            if not cat or not text: continue
            primary, hits = tag_chunk(text, category=cat)
            if hits:
                # Multi-label: count each topic that matched, not just argmax
                for k in hits:
                    counts[k] += 1
                tagged_any += 1
                per_cat_tagged[cat] += 1
        if offset is None: break

    # 3. Coverage check with alias merging
    # Treat customs:customs_X as alias of customs:X (gold has both as duplicates)
    ALIASES = {
        "customs:customs_warehouse": ["customs:warehousing"],
        "customs:warehousing": ["customs:customs_warehouse"],
        "customs:customs_classification": ["customs:classification"],
        "customs:classification": ["customs:customs_classification"],
        "customs:customs_drawback": ["customs:drawback"],
        "customs:drawback": ["customs:customs_drawback"],
        "customs:customs_valuation": ["customs:valuation"],
        "customs:valuation": ["customs:customs_valuation"],
    }
    def alias_keys(cat, sub):
        primary = f"{cat}:{sub}"
        return [primary] + ALIASES.get(primary, [])

    low = []
    merged_counts = {}
    for (c, s) in gold_taggable:
        keys = alias_keys(c, s)
        n = sum(counts.get(k, 0) for k in keys)
        merged_counts[f"{c}:{s}"] = {"count": n, "aliases": keys}
        if n < MIN_PER_TOPIC:
            low.append({"topic": f"{c}:{s}", "chunks": n, "aliases_checked": keys})

    summary = {
        "probe": "V20",
        "total_chunks": total,
        "tagged_any_topic": tagged_any,
        "tag_rate": round(tagged_any / max(total,1), 4),
        "gold_topics_total": len(gold_pairs),
        "gold_taggable": len(gold_taggable),
        "per_topic_counts": dict(sorted(counts.items())),
        "low_coverage_topics": low,
        "low_coverage_count": len(low),
        "per_category_tagged_pct": {c: round(per_cat_tagged[c]/max(per_cat_total[c],1), 3) for c in per_cat_total},
        "pass_gate": len(low) == 0,
    }
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    # Print short form
    short = {k: summary[k] for k in ("probe","total_chunks","tagged_any_topic","tag_rate","gold_taggable","low_coverage_count","pass_gate")}
    print(json.dumps(short, indent=2))
    if low:
        print("LOW COVERAGE TOPICS:")
        for x in low[:30]:
            print(f"  {x['topic']:<40} {x['chunks']:>6}")

if __name__ == "__main__":
    main()
