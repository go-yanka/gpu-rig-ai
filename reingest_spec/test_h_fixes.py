#!/usr/bin/env python3
"""test_h_fixes.py — T20-T23 for HIGH-severity fixes H1-H4.

T20 H1: retriever.retrieve accepts `collection` kwarg; shadow uses it (no env mutation)
T21 H2: api.py QueryReq has `collection` field; evaluators send it, not fake filter
T22 H3: gate_g1 _is_hit handles singular schema + drops chunk_id match + text fallback
T23 H4: rollback_v1.sh has service-restart logic + health-check
"""
from __future__ import annotations
import sys, re, inspect
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).parent
# Resolve cbic_rag/ across layouts: dev (ROOT/cbic_rag) vs rig (ROOT/rag/cbic_rag)
_CBIC_CANDIDATES = [HERE.parent / "cbic_rag", HERE.parent / "rag" / "cbic_rag"]
CBIC = next((p for p in _CBIC_CANDIDATES if (p / "retriever.py").exists()),
            _CBIC_CANDIDATES[0])
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CBIC))

PASS, FAIL = 0, 0
def _t(name, cond, msg=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"[ ok ] {name}")
    else:
        FAIL += 1; print(f"[FAIL] {name} -- {msg}")


def t20_h1_retriever_collection_kwarg():
    # Static check — retriever imports embedder which may be absent on laptop
    src = (CBIC / "retriever.py").read_text(encoding="utf-8")
    _t("T20.a retrieve signature has `collection` kwarg",
       re.search(r"def retrieve\([^)]*collection[^)]*\)", src, re.DOTALL) is not None)
    _t("T20.b retrieve uses `coll` (not QCOLL) in query_points",
       "collection_name=coll" in src and "collection_name=QCOLL" not in src)

    shadow = (CBIC / "api_v2_shadow.py").read_text(encoding="utf-8")
    _t("T20.c _call_collection does NOT mutate os.environ[QDRANT_COLL]",
       'os.environ["QDRANT_COLL"] = collection' not in shadow)
    _t("T20.d _call_collection passes collection= to retrieve",
       "retrieve(hyde_q, k=k * 3, collection=collection)" in shadow)


def t21_h2_api_collection_field():
    src = (CBIC / "api.py").read_text(encoding="utf-8")
    _t("T21.a QueryReq has `collection` field",
       re.search(r"class QueryReq[\s\S]{0,400}collection:\s*Optional\[str\]", src) is not None)
    _t("T21.b _run_pipeline passes req.collection to retrieve",
       "collection=req.collection" in src)

    # Evaluators use `collection` field, not `_collection` filter
    for name in ["gate_g1_recall.py", "gate_g2_dual_judge.py",
                 "gate_g3_levenshtein.py", "gate_g4_adversarial.py",
                 "probe_v2_runner.py"]:
        p = HERE / "evaluators" / name
        if not p.exists():
            continue
        s = p.read_text(encoding="utf-8")
        _t(f"T21.c {name} no fake _collection filter in body",
           not re.search(r'"filters":\s*\{\s*"_collection"', s))
        _t(f"T21.d {name} sends collection field",
           '"collection": collection' in s or '"collection":collection' in s)

    theta = (HERE / "theta_tune.py").read_text(encoding="utf-8")
    _t("T21.e theta_tune.py uses collection field",
       '"collection": collection' in theta)


def t22_h3_gold_schema_tolerance():
    # Pull _is_hit directly
    sys.path.insert(0, str(HERE / "evaluators"))
    # Reload if already imported from earlier tests
    for m in list(sys.modules):
        if m == "gate_g1_recall":
            del sys.modules[m]
    from gate_g1_recall import _is_hit, _norm_gold

    # Singular-schema gold (what curator actually emits)
    gold = {"expected_doc_id": "cbic-circular:1001866",
            "expected_section": "11(b)",
            "expected_chunk_id": 10841273002496,  # v1 int — MUST be ignored
            "category": "central_excise"}

    # Retrieved hit with v2 SHA256 chunk_id (completely different format)
    hits_doc_match = [{"chunk_id": "abc123" * 10, "doc_id": "cbic-circular:1001866",
                       "section_ref": "Section 99", "text": "unrelated"}]
    _t("T22.a singular doc_id → hit", _is_hit(gold, hits_doc_match))

    hits_sec_match = [{"chunk_id": "xyz", "doc_id": "other-doc",
                       "section_ref": "Section 11(b) — refund", "text": ""}]
    _t("T22.b singular section matches via substring", _is_hit(gold, hits_sec_match))

    hits_none = [{"chunk_id": "xyz", "doc_id": "other-doc",
                  "section_ref": "Section 99", "text": "noise"}]
    _t("T22.c no doc/section match → miss", not _is_hit(gold, hits_none))

    # Chunk_id must NOT alone count as hit (v1 int vs v2 sha256 — never equal)
    hits_chunk_v1 = [{"chunk_id": 10841273002496, "doc_id": "unrelated",
                      "section_ref": "Section 99", "text": ""}]
    _t("T22.d v1 chunk_id alone does NOT trigger hit",
       not _is_hit(gold, hits_chunk_v1))

    # Text fallback works when gold has expected_text
    gold_text = {"expected_doc_id": "",  # no doc_id
                 "expected_text": "duty payment verification with exchange control copy"}
    hits_text = [{"doc_id": "any", "section_ref": "",
                  "text": "The duty payment verification with exchange control copy is permitted per Notification X"}]
    _t("T22.e text substring fallback hits", _is_hit(gold_text, hits_text))

    # Plural schema still works (backward compat with D-H agent's assumption)
    gold_plural = {"expected_doc_ids": ["d1", "d2"],
                   "expected_section_refs": ["Section 16(2)"]}
    hits_plural = [{"doc_id": "d2", "section_ref": "Section 99", "text": ""}]
    _t("T22.f plural schema still works", _is_hit(gold_plural, hits_plural))


def t23_h4_rollback_service_restart():
    src = (HERE / "rollback_v1.sh").read_text(encoding="utf-8")
    _t("T23.a tries systemctl restart",
       "systemctl restart" in src)
    _t("T23.b SERVICE_CMD fallback documented",
       "SERVICE_CMD" in src)
    _t("T23.c exits 2 when auto-restart unavailable",
       "MANUAL step required" in src and "exit 2" in src)
    _t("T23.d post-restart health-check via /v1/stats",
       "/v1/stats" in src and "api responsive" in src)
    _t("T23.e still writes router.config for persistence",
       'printf \'QDRANT_COLL=%s\\n\'' in src)


if __name__ == "__main__":
    t20_h1_retriever_collection_kwarg()
    t21_h2_api_collection_field()
    t22_h3_gold_schema_tolerance()
    t23_h4_rollback_service_restart()
    print(f"\n=== {PASS} passed, {FAIL} failed ===")
    sys.exit(0 if FAIL == 0 else 1)
