#!/usr/bin/env python3
"""Migrate legacy training pairs into unified v2 schema.

Inputs:
  - D:/_gpu_rig_ai/eval/training_pairs/qa_*.jsonl       (Format A: flat)
  - D:/_gpu_rig_ai/eval/training_pairs/pairs_*.jsonl    (Format B: chunk-envelope)
  - D:/_gpu_rig_ai/eval/gold_set_expansion/bucket_*.yaml (hand-authored)

Output:
  - <out>/cbic_pairs_v2.jsonl (canonical schema per memory/pair_schema_cbic_v2.md)

Schema source of truth:
  C:/Users/Rahul Goyanka/.claude/projects/D---gpu-rig-ai/memory/pair_schema_cbic_v2.md

Canonical fields populated on migration:
  pair_id, chunk_id, doc_id, category, subcategory, doc_type, doc_number,
  section_ref, parent_act, title, text, question, question_id, domain,
  difficulty, query_type, generator, generator_reasoning, grading,
  positive_chunk_id, hard_negatives=[], llm_answer=null, judges={},
  gate_verdicts={}, provenance{scope,generated_ts,gold_source,
                               retriever_config,chunker_version,source_file}

Fields populated later by gate runs:
  - positive_chunk_id is set to chunk_id by default (self-positive)
  - hard_negatives[] left empty; G1 mine pass fills it
  - llm_answer/judges/gate_verdicts populated by G2 + θ + G4

This migrator is append-only safe — rerunning appends new rows (dedup by
(chunk_id, question) tuple), never overwrites.

CLI:
  python migrate_pairs_to_v2.py \\
      --pairs-dir D:/_gpu_rig_ai/eval/training_pairs \\
      --buckets-dir D:/_gpu_rig_ai/eval/gold_set_expansion \\
      --out /opt/indian-legal-ai/data/training_corpus/cbic_pairs_v2.jsonl \\
      --manifest /opt/indian-legal-ai/data/scraped/cbic/_manifest.sqlite

Origin: 2026-04-24 GST50 scale test — reconciles Format A + B per user
directive "reconcile, don't fragment".
"""
from __future__ import annotations
import argparse, json, os, sys, time, sqlite3, glob, hashlib
from pathlib import Path

try:
    import yaml  # optional for buckets
except Exception:
    yaml = None


# ---- helpers ---------------------------------------------------------------

def _diff_map(c):
    """Format A grading.complexity → v2 difficulty mapping."""
    if c is None: return None
    c = str(c).lower()
    return {"low": "basic", "medium": "medium", "high": "hard"}.get(c, c)


def _load_chunkid_to_docid(manifest_path: Path | None) -> dict:
    """Best-effort chunk_id → doc_id lookup from v1 ingest manifest.
    Format A rows don't carry doc_id — we need this map for them."""
    if not manifest_path or not manifest_path.exists():
        return {}
    try:
        conn = sqlite3.connect(str(manifest_path))
        # The table name has drifted between versions — try common options
        candidates = ["chunks", "chunk_meta", "v1_chunks"]
        for t in candidates:
            try:
                cols = [r[1] for r in conn.execute(f"PRAGMA table_info({t})")]
                if "chunk_id" in cols and "doc_id" in cols:
                    m = dict(conn.execute(f"SELECT chunk_id, doc_id FROM {t}"))
                    print(f"[migrator] loaded {len(m)} chunk_id→doc_id from {t}", file=sys.stderr)
                    return m
            except sqlite3.OperationalError:
                continue
    except Exception as e:
        print(f"[migrator] manifest load failed: {e}", file=sys.stderr)
    return {}


def _pair_id(scope: str, doc_id: str, chunk_id, qn: int) -> str:
    """Stable pair id: <scope>_<docid>_c<chunk-suffix>_q<qn>."""
    if doc_id:
        docpart = doc_id.replace(":", "_").replace("/", "_")
    else:
        docpart = "noDoc"
    cpart = str(chunk_id)[-12:]
    return f"{scope}_{docpart}_c{cpart}_q{qn:02d}"


def _mk_v2(
    *,
    chunk_id, doc_id, question, question_id,
    category="", subcategory="", doc_type="", doc_number="",
    section_ref="", parent_act="", title="", text="",
    difficulty=None, domain="", query_type=None,
    generator="", generator_reasoning="",
    grading=None, scope="legacy", gold_source="legacy",
    source_file="", chunker_version="v1",
):
    return {
        "pair_id": _pair_id(scope, doc_id or "", chunk_id, question_id),
        "chunk_id": chunk_id,
        "doc_id": doc_id or None,
        "category": category or "",
        "subcategory": subcategory or "",
        "doc_type": doc_type or "",
        "doc_number": doc_number or "",
        "section_ref": section_ref or "",
        "parent_act": parent_act or "",
        "title": title or "",
        "text": text or "",
        "question": question,
        "question_id": question_id,
        "domain": domain or "",
        "difficulty": difficulty,
        "query_type": query_type,
        "generator": generator,
        "generator_reasoning": generator_reasoning or "",
        "grading": grading or {},
        "positive_chunk_id": chunk_id,
        "hard_negatives": [],
        "llm_answer": None,
        "judges": {},
        "gate_verdicts": {},
        "provenance": {
            "scope": scope,
            "generated_ts": int(time.time()),
            "gold_source": gold_source,
            "retriever_config": {"dense_only": True, "theta": None},
            "chunker_version": chunker_version,
            "source_file": source_file,
        },
    }


# ---- Format A loader (qa_*.jsonl) ------------------------------------------

def load_format_a(path: Path, chunk_to_doc: dict, scope: str = "legacy"):
    """Yield v2 rows from a flat qa_*.jsonl file."""
    gen_name = path.stem  # e.g. qa_gemini, qa_sonnet_high, qa_claude_opus
    gen = (gen_name.replace("qa_", "") or "unknown")
    # Align with schema: gemini / claude_sonnet_high / claude_opus
    gen_map = {
        "gemini": "gemini",
        "sonnet_high": "claude_sonnet_high",
        "claude_opus": "claude_opus",
        "claude_smoke": "claude_opus",
    }
    gen_final = gen_map.get(gen, gen)

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line: continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            ch = r.get("chunk_id")
            if ch is None: continue
            grading = r.get("grading") or {}
            yield _mk_v2(
                chunk_id=ch,
                doc_id=chunk_to_doc.get(ch),
                question=r.get("question", ""),
                question_id=1,
                category=r.get("category", ""),
                difficulty=_diff_map(grading.get("complexity")),
                generator=gen_final,
                generator_reasoning=r.get("why_this_chunk", ""),
                grading=grading,
                scope=scope,
                gold_source=f"migrated_{gen_final}",
                source_file=f"{path.name}:{lineno}",
                chunker_version="v1",
            )


# ---- Format B loader (pairs_*.jsonl) ---------------------------------------

def load_format_b(path: Path, scope: str = "legacy"):
    """Yield v2 rows from a chunk-envelope pairs_*.jsonl file."""
    stem = path.stem  # e.g. pairs_2000_20260422, pairs_claude_opus
    if "claude" in stem and "opus" in stem:
        gen_final = "claude_opus"
    elif "sonnet" in stem:
        gen_final = "claude_sonnet_high"
    elif "gemini" in stem or "2000" in stem:
        gen_final = "gemini"
    else:
        gen_final = stem

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line: continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            ch = r.get("chunk_id")
            doc_id = r.get("doc_id")
            questions = r.get("questions") or []
            # envelope-level generator reasoning key varies
            gen_reason = (r.get("gemini_reasoning")
                          or r.get("claude_reasoning")
                          or r.get("opus_reasoning")
                          or r.get("sonnet_reasoning")
                          or r.get("reasoning") or "")
            for i, q in enumerate(questions, start=1):
                if not isinstance(q, dict): continue
                yield _mk_v2(
                    chunk_id=ch,
                    doc_id=doc_id,
                    question=q.get("q") or q.get("question") or "",
                    question_id=i,
                    category=r.get("category", ""),
                    subcategory=r.get("subcategory", ""),
                    doc_type=r.get("doc_type", ""),
                    doc_number=r.get("doc_number", ""),
                    section_ref=r.get("section_ref", ""),
                    parent_act=r.get("parent_act", ""),
                    title=r.get("title", ""),
                    text=r.get("text", ""),
                    difficulty=q.get("difficulty"),
                    domain=q.get("domain", ""),
                    query_type=q.get("query_type"),
                    generator=gen_final,
                    generator_reasoning=q.get("why_this_chunk") or gen_reason,
                    grading=q.get("grading") or {},
                    scope=scope,
                    gold_source=f"migrated_{gen_final}_pairs",
                    source_file=f"{path.name}:{lineno}#q{i}",
                    chunker_version="v1",
                )


# ---- YAML bucket loader ----------------------------------------------------

def load_bucket(path: Path, scope: str = "expansion_bucket"):
    """Yield v2 rows from bucket_*.yaml (hand-authored gold)."""
    if yaml is None: return
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[migrator] bucket parse fail {path}: {e}", file=sys.stderr)
        return
    bucket_type = None
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = (data.get("queries") or data.get("items")
                 or data.get("cases") or data.get("pairs") or [])
        bucket_type = data.get("bucket_type")
    else:
        items = []
    for i, it in enumerate(items, start=1):
        if not isinstance(it, dict): continue
        yield _mk_v2(
            chunk_id=it.get("chunk_id") or it.get("expected_chunk_id") or 0,
            doc_id=it.get("doc_id") or (it.get("expected_doc_ids") or [None])[0],
            question=it.get("query") or it.get("question") or "",
            question_id=1,
            category=it.get("category", ""),
            subcategory=it.get("subcategory", ""),
            section_ref=it.get("section_ref", ""),
            title=it.get("title", ""),
            difficulty=it.get("difficulty"),
            domain=it.get("domain", ""),
            query_type=it.get("query_type") or bucket_type,
            generator="synthetic_handauthored",
            generator_reasoning=it.get("why") or it.get("reason") or "",
            scope=scope,
            gold_source="expansion_bucket",
            source_file=f"{path.name}:#{i}",
            chunker_version="v1",
        )


# ---- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", type=Path, required=True)
    ap.add_argument("--buckets-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="SQLite ingest manifest for chunk_id→doc_id lookup")
    ap.add_argument("--scope", default="legacy_migrated")
    ap.add_argument("--append", action="store_true",
                    help="Append to existing out file (dedup by (chunk_id, question))")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    chunk_to_doc = _load_chunkid_to_docid(args.manifest)

    # existing dedup set
    seen = set()
    existing = 0
    if args.append and args.out.exists():
        with open(args.out, "r", encoding="utf-8") as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                    seen.add((r.get("chunk_id"), r.get("question")))
                    existing += 1
                except Exception: pass
        print(f"[migrator] loaded {existing} existing rows from {args.out}")

    # Format A files
    fa_files = sorted(args.pairs_dir.glob("qa_*.jsonl"))
    # Exclude known-bad / archive variants
    fa_files = [p for p in fa_files if "BAD" not in p.name and ".bak" not in p.name]
    # Format B files
    fb_files = sorted(args.pairs_dir.glob("pairs_*.jsonl"))
    fb_files = [p for p in fb_files if "BAD" not in p.name and ".bak" not in p.name]

    # Bucket YAMLs
    bucket_files = []
    if args.buckets_dir and args.buckets_dir.exists():
        bucket_files = sorted(args.buckets_dir.glob("bucket_*.yaml"))
        # Prefer _v3 / _v2 over base when both present, keep all for append
    print(f"[migrator] Format A files: {len(fa_files)}")
    print(f"[migrator] Format B files: {len(fb_files)}")
    print(f"[migrator] Bucket files:  {len(bucket_files)}")

    counts = {"format_a": 0, "format_b": 0, "buckets": 0, "dup": 0, "written": 0}

    if not args.dry_run:
        args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    fout = open(args.out, mode, encoding="utf-8") if not args.dry_run else None

    def _emit(row):
        k = (row["chunk_id"], row["question"])
        if k in seen:
            counts["dup"] += 1
            return
        seen.add(k)
        if fout is not None:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts["written"] += 1

    for p in fa_files:
        for row in load_format_a(p, chunk_to_doc, scope=args.scope):
            counts["format_a"] += 1
            _emit(row)

    for p in fb_files:
        for row in load_format_b(p, scope=args.scope):
            counts["format_b"] += 1
            _emit(row)

    for p in bucket_files:
        for row in load_bucket(p, scope=args.scope):
            counts["buckets"] += 1
            _emit(row)

    if fout is not None: fout.close()

    print(json.dumps({
        "out": str(args.out), "mode": mode,
        "existing_rows": existing, **counts,
    }, indent=2))


if __name__ == "__main__":
    main()
