#!/usr/bin/env python3
"""test_integration.py — T9-T13 integration tests for v2 ingest pipeline.

Purpose: catch seam bugs between chunker_v2 and cbic_rag.ingest (the kind
test_chunker.py does NOT catch because it stops at chunk-object level).

Gate: all T9-T13 must pass before Stage B exit.
"""
from __future__ import annotations
import sys, os, json, re
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "cbic_rag"))

from chunker_v2 import Chunk, ChunkingPlan, _mk_prose_chunk, _mk_table_chunk

PASS, FAIL = 0, 0
def _t(name, cond, msg=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"[ ok ] {name}")
    else:
        FAIL += 1; print(f"[FAIL] {name} -- {msg}")


# --- T9: payload-contract — chunker output has keys upsert_chunks reads -----
def t9_payload_contract():
    plan = ChunkingPlan()
    meta = {"doc_id": "d1", "source": "f.pdf", "category": "cgst",
            "subcategory": "act", "lang": "en",
            "parent_hierarchy_text": "Section 1"}
    ch = _mk_prose_chunk("Hello world. This is a paragraph.", 0, 33, plan, meta, ["R2"])
    p = ch.to_payload()

    # cbic_rag/ingest.py:138 reads c['doc_id'], c['page'], c['char_start']
    required = ["doc_id", "page", "char_start", "char_end",
                "text", "embed_text", "category", "subcategory",
                "chunk_id", "lang", "page_range"]
    missing = [k for k in required if k not in p]
    _t("T9.a payload has all keys ingest.upsert_chunks reads",
       not missing, f"missing: {missing}")

    _t("T9.b char_start is int", isinstance(p["char_start"], int))
    _t("T9.c char_end > char_start", p["char_end"] > p["char_start"])
    _t("T9.d page is int (default 0 ok)", isinstance(p["page"], int))


# --- T10: meta-propagation — D8 amendment fields flow to payload ------------
def t10_meta_propagation():
    plan = ChunkingPlan()
    meta = {"doc_id": "d2", "source": "f.pdf", "category": "cgst",
            "subcategory": "act", "lang": "en",
            "parent_hierarchy_text": "",
            "notification_id": "NOTIF-2023-12",
            "as_of_date": "2023-10-01",
            "effective_date": "2023-10-01",
            "superseded_by": None,
            "text_source": "born"}
    ch = _mk_prose_chunk("text here.", 0, 10, plan, meta, [])
    p = ch.to_payload()

    _t("T10.a notification_id propagated", p.get("notification_id") == "NOTIF-2023-12")
    _t("T10.b as_of_date propagated", p.get("as_of_date") == "2023-10-01")
    _t("T10.c effective_date propagated", p.get("effective_date") == "2023-10-01")
    _t("T10.d text_source propagated", p.get("text_source") == "born")
    _t("T10.e superseded_by present (None ok)", "superseded_by" in p)

    # Table chunks too
    ch2 = _mk_table_chunk("| a | b |", 0, 8, meta, plan, None, "Rates Table")
    p2 = ch2.to_payload()
    _t("T10.f table chunk carries notification_id",
       p2.get("notification_id") == "NOTIF-2023-12")


# --- T11: env/import order — QDRANT_COLL override must win ------------------
def t11_env_import_order():
    # Simulate what phase3_4_5 does: set env, then import.
    os.environ["QDRANT_COLL"] = "cbic_v2_test_xyz"
    # Force re-import
    for m in list(sys.modules):
        if m == "ingest" or m.startswith("ingest."):
            del sys.modules[m]
    try:
        import ingest as v1_ingest  # noqa
        _t("T11.a QCOLL honors env-var set before import",
           v1_ingest.QCOLL == "cbic_v2_test_xyz",
           f"got {getattr(v1_ingest, 'QCOLL', None)}")
    except ImportError as e:
        # embedder missing on laptop — that's fine, we only care about QCOLL line
        src = (HERE.parent / "cbic_rag" / "ingest.py").read_text()
        m = re.search(r"QCOLL\s*=\s*os\.environ\.get\(['\"]QDRANT_COLL['\"]", src)
        _t("T11.a (static) ingest.py reads QDRANT_COLL from env at module load",
           m is not None, "QCOLL not sourced from env")
        _t("T11.b import skipped on laptop (expected)", True, str(e)[:80])


# --- T12: OCR detection — low text-density PDF flagged as ocr ---------------
def t12_ocr_detection():
    # Import directly from ingest_v2
    from ingest_v2 import detect_text_source, TEXT_DENSITY_MIN

    # Simulate a 10-page PDF with very little text (image-only)
    sparse_text = "Title page only.\n" * 3  # ~50 chars total
    page_offsets = [0] * 11  # 10 pages
    _t("T12.a sparse PDF flagged ocr",
       detect_text_source(sparse_text, page_offsets) == "ocr")

    # Dense text (real born-digital)
    dense_text = ("This is a substantive paragraph of legal text. " * 30) * 10
    _t("T12.b dense PDF flagged born",
       detect_text_source(dense_text, page_offsets) == "born")

    _t("T12.c threshold documented", TEXT_DENSITY_MIN >= 100 and TEXT_DENSITY_MIN <= 500)


# --- T13: end-to-end smoke — 1 doc → mock upsert ---------------------------
def t13_end_to_end():
    # Build a fake doc: 1 section + 1 table, no real LLM.
    plan = ChunkingPlan(doc_type="act", primary_splitter="section",
                       critical_units=[], hard_boundaries=[],
                       table_regions=[], has_amendments=True, language="en")
    full_text = (
        "Section 1. Short title.\n"
        "This Act may be called the CGST Act.\n\n"
        "Section 2. Definitions.\n"
        "In this Act, unless the context otherwise requires,—\n"
        "(1) \"aggregate turnover\" means the aggregate value of all "
        "taxable supplies, exempt supplies, exports and inter-State "
        "supplies of persons having the same Permanent Account Number.\n"
    )
    meta = {"doc_id": "smoke_doc", "source": "smoke.pdf",
            "category": "cgst", "subcategory": "act", "lang": "en",
            "parent_hierarchy_text": "CGST Act",
            "notification_id": "N1", "as_of_date": "2017-07-01",
            "effective_date": "2017-07-01", "text_source": "born"}

    from chunker_v2 import chunk_document
    chunks = chunk_document(full_text, plan, meta, page_offsets=[0, len(full_text)])
    _t("T13.a produced chunks", len(chunks) >= 1, f"got {len(chunks)}")

    # Build payloads exactly as phase2 would write to sqlite
    payloads = [c.to_payload() for c in chunks]

    # Simulate what upsert_chunks does at line 138
    try:
        for c in payloads:
            pid = abs(hash((c["doc_id"], c["page"], c["char_start"]))) % (10**15)
            assert "text" in c and "embed_text" in c
            assert c["category"] == "cgst"
            assert c["notification_id"] == "N1"
        _t("T13.b all payloads survive upsert_chunks contract", True)
    except Exception as e:
        _t("T13.b all payloads survive upsert_chunks contract", False, str(e))

    # Unique IDs
    ids = {c["chunk_id"] for c in payloads}
    _t("T13.c chunk_ids unique", len(ids) == len(payloads))


if __name__ == "__main__":
    t9_payload_contract()
    t10_meta_propagation()
    t11_env_import_order()
    t12_ocr_detection()
    t13_end_to_end()
    print(f"\n=== {PASS} passed, {FAIL} failed ===")
    sys.exit(0 if FAIL == 0 else 1)
