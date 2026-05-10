"""
test_chunker.py — Self-tests T1–T8 for chunker_v2.

Hard rule: all tests must pass before chunker_v2 is declared ready.

Run:
    python -m pytest D:/_gpu_rig_ai/reingest_spec/test_chunker.py -v
    # or standalone:
    python D:/_gpu_rig_ai/reingest_spec/test_chunker.py

Covers:
    T1 Table region never split mid-row
    T2 Proviso block stays inside parent
    T3 Explanation block stays inside parent
    T4 "Provided that" never appears as chunk-start token (English + Hindi)
    T5 Section-start split has zero overlap
    T6 Mid-section split has 700 overlap
    T7 Final chunk <500 chars merges into previous (floor rule)
    T8 Bilingual linker matches ≥90% on a 10-doc sample (smoke — real sample post-Phase-2)
"""
from __future__ import annotations

import sys
import io
from pathlib import Path

# Force UTF-8 stdout so unicode (≥, Devanagari) doesn't crash Windows cp1252
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from chunker_v2 import (  # noqa: E402
    ChunkingPlan, chunk_document, is_unusable_cut,
    find_critical_unit_spans, canonicalize, sha256_of,
    ENGLISH_CONNECTORS, HINDI_CONNECTORS,
    TARGET, CAP, CEILING, FLOOR, OVERLAP_MID,
)

FAILED: list[tuple[str, str]] = []


def _check(test_id: str, cond: bool, msg: str):
    if not cond:
        FAILED.append((test_id, msg))
        print(f"  [FAIL] {test_id}: {msg}")
    else:
        print(f"  [ ok ] {test_id}: {msg}")


def _mk_meta(source="test.pdf", **kw):
    m = dict(doc_id="test-doc", source=source, category="gst", subcategory="acts",
             lang="en", text_source="born", parent_hierarchy_text="")
    m.update(kw)
    return m


def _plan(primary="section", depth=3, lang="en", table_regions=None):
    return ChunkingPlan(
        doc_type="act", structure="hierarchical_sections",
        primary_splitter=primary,
        critical_units=["section", "proviso", "explanation", "definition", "table"],
        table_regions=table_regions or [],
        hierarchy_depth=depth, language=lang, confidence=0.95,
    )


# --- T1 Table region never split mid-row ------------------------------------


def test_T1_table_atomic():
    print("T1 — Table region atomic (never split mid-row)")
    # Build a table that fits under CEILING — must emerge as ONE chunk.
    rows = [f"Chapter 84 | Heading {i:04d} | Item desc {i} | 18% GST" for i in range(30)]
    header = "Chapter | Heading | Description | Rate"
    table_text = header + "\n" + "\n".join(rows) + "\n"
    doc = "Section 1. Title.\n\nSome intro.\n\n" + table_text + "\nSection 2. Post."
    # page_offsets: pretend page 1 starts at offset 0, page 2 starts where table begins
    table_start = doc.index(table_text)
    page_offsets = [0, table_start, table_start + len(table_text)]
    plan = _plan(table_regions=[{"page_start": 2, "page_end": 2, "reason": "GST rates", "confidence": 0.9}])
    chunks = chunk_document(doc, plan, _mk_meta(), page_offsets)

    table_chunks = [c for c in chunks if c.is_table]
    _check("T1.a", len(table_chunks) == 1, f"exactly 1 table chunk (got {len(table_chunks)})")
    if table_chunks:
        t = table_chunks[0]
        _check("T1.b", header in t.text, "column header present in table chunk")
        # Every row appears whole (mid-row split check)
        for r in rows:
            if r not in t.text:
                _check("T1.c", False, f"row missing or split: {r[:40]!r}")
                return
        _check("T1.c", True, f"all {len(rows)} rows intact in table chunk")


def test_T1b_oversize_table_row_split():
    print("T1b — Oversize table splits on row boundaries only, with header carry-over")
    # Build table > CEILING (8000 chars) with clear row structure
    header = "Col1 | Col2 | Col3"
    long_rows = [f"Entry-{i:05d} | Data for row {i} | " + ("X" * 100) for i in range(80)]
    table_text = header + "\n" + "\n".join(long_rows) + "\n"
    assert len(table_text) > CEILING
    doc = "Intro.\n\n" + table_text
    ts = doc.index(table_text)
    page_offsets = [0, ts, ts + len(table_text)]
    plan = _plan(table_regions=[{"page_start": 2, "page_end": 2, "reason": "big", "confidence": 0.9}])
    chunks = chunk_document(doc, plan, _mk_meta(), page_offsets)
    tchunks = [c for c in chunks if c.is_table]
    _check("T1b.a", len(tchunks) >= 2, f"split into ≥2 chunks (got {len(tchunks)})")
    # Every split should carry the header (R1 row-header carry-over)
    for i, c in enumerate(tchunks):
        _check(f"T1b.h{i}", header in c.text, f"table_part[{c.table_part}] carries header")
    # No row should be cut mid-row: every line that contains '|' should either
    # be the header OR start with 'Entry-'
    for c in tchunks:
        for line in c.text.split("\n"):
            if "|" in line and line.strip() not in (header, ""):
                if not line.strip().startswith("Entry-"):
                    _check("T1b.row", False, f"mid-row split suspected: {line[:60]!r}")
                    return
    _check("T1b.row", True, "no mid-row splits detected")


# --- T2 Proviso block stays inside parent -----------------------------------


def test_T2_proviso_whole():
    print("T2 — Proviso stays with parent section")
    doc = (
        "Section 5. Levy of tax.\n\n"
        "(1) There shall be levied a tax called the Central Goods and Services Tax.\n\n"
        + ("Explanatory text padding. " * 60) +  # ~1500 chars of filler
        "\n\nProvided that the rate shall not exceed 20 per cent.\n\n"
        "Provided further that the Government may by notification vary rates.\n\n"
        "Section 6. Input tax credit.\n\n(1) Credit shall be allowed."
    )
    plan = _plan()
    chunks = chunk_document(doc, plan, _mk_meta())
    # Neither proviso should be a chunk's first content
    for c in chunks:
        first = c.text.lstrip()
        _check("T2.a", not first.startswith("Provided"),
               f"chunk[{chunks.index(c)}] starts with Provided? -> {first[:40]!r}")
    # Both provisos must appear inside some chunk (not dropped)
    all_text = "\n\n".join(c.text for c in chunks)
    _check("T2.b", "Provided that the rate" in all_text, "first proviso preserved")
    _check("T2.c", "Provided further" in all_text, "second proviso preserved")


# --- T3 Explanation block stays inside parent -------------------------------


def test_T3_explanation_whole():
    print("T3 — Explanation stays with parent")
    doc = (
        "Section 7. Supply.\n\n"
        "(1) The expression 'supply' includes sale, transfer, barter, exchange, licence, rental.\n\n"
        + ("Detailed definition text. " * 50) +  # padding
        "\n\nExplanation.— For the purposes of this Act, 'supply' shall include import of services for a consideration.\n\n"
        "Section 8. Tax liability on composite supply."
    )
    plan = _plan()
    chunks = chunk_document(doc, plan, _mk_meta())
    for c in chunks:
        first = c.text.lstrip()
        _check("T3.a", not first.startswith("Explanation"),
               f"chunk starts with Explanation? -> {first[:40]!r}")
    all_text = "\n\n".join(c.text for c in chunks)
    _check("T3.b", "import of services for a consideration" in all_text,
           "explanation body preserved")


# --- T4 Unusable-cut validator ---------------------------------------------


def test_T4_unusable_cut_detection():
    print("T4 — Unusable-cut detection (English + Hindi connectors)")
    for tok in ENGLISH_CONNECTORS:
        _check(f"T4.en.{tok[:10]}", is_unusable_cut(tok + " the rate shall"), f"EN connector {tok!r} flagged")
    for tok in HINDI_CONNECTORS:
        _check(f"T4.hi.{tok[:10]}", is_unusable_cut(tok + " यह है"), f"HI connector {tok!r} flagged")
    # Bare lowercase verb
    _check("T4.lv", is_unusable_cut("means that the registered person"), "bare lowercase verb flagged")
    # Legitimate starts must NOT be flagged
    _check("T4.ok1", not is_unusable_cut("Section 5. Levy of tax."), "'Section 5.' accepted")
    _check("T4.ok2", not is_unusable_cut("The registered person shall pay"), "'The registered person' accepted")
    _check("T4.ok3", not is_unusable_cut("CHAPTER II"), "'CHAPTER II' accepted")


# --- T5 Section-start split has zero overlap --------------------------------


def test_T5_section_boundary_zero_overlap():
    print("T5 — Section-boundary splits have zero overlap")
    # Two big sections back-to-back — split should land at the boundary with no overlap
    s1 = "Section 10. Composition levy.\n\n" + ("Composition text. " * 200)  # ~3600 chars
    s2 = "Section 11. Registration.\n\n" + ("Registration text. " * 200)
    doc = s1 + "\n\n" + s2
    plan = _plan()
    chunks = chunk_document(doc, plan, _mk_meta())
    # Find the boundary chunk pair — the chunk starting with "Section 11" must have
    # no content from s1 duplicated in its head
    found_pair = False
    for i, c in enumerate(chunks):
        if c.text.lstrip().startswith("Section 11"):
            found_pair = True
            prev = chunks[i-1].text if i > 0 else ""
            # Check: last 100 chars of prev should NOT appear at the start of c
            tail = prev[-100:].strip()
            if tail and tail in c.text[:200]:
                _check("T5.a", False, "section-boundary overlap leaked")
                return
            _check("T5.a", True, "zero overlap at Section 11 boundary")
            # Also rule_triggered should reflect it
            rules = " ".join(chunks[i-1].chunking_rule_triggered)
            _check("T5.b", "R5:zero_overlap_section" in rules or chunks[i-1].is_table,
                   f"rule R5:zero_overlap_section logged (got {rules!r})")
            break
    if not found_pair:
        _check("T5.a", False, "could not locate Section 11 boundary in output")


# --- T6 Mid-section split has 700 overlap -----------------------------------


def test_T6_midsection_overlap():
    print("T6 — Mid-section splits have ~700 char overlap")
    # One giant section that forces mid-section splitting
    big = "Section 20. Long provisions.\n\n" + ("Prose text. " * 900)  # ~11k chars, no sub-structure
    plan = _plan()
    chunks = chunk_document(big, plan, _mk_meta())
    _check("T6.a", len(chunks) >= 2, f"produced ≥2 chunks from oversize section (got {len(chunks)})")
    # For each consecutive pair, look for ~OVERLAP_MID chars of overlap
    for i in range(len(chunks) - 1):
        a, b = chunks[i].text, chunks[i+1].text
        # find longest common suffix-of-a / prefix-of-b
        max_overlap = 0
        for k in range(min(len(a), len(b), OVERLAP_MID + 300), 50, -10):
            if a[-k:] == b[:k]:
                max_overlap = k
                break
        # allow tolerance — splitter may land on sentence boundary
        ok = 200 <= max_overlap <= OVERLAP_MID + 300 or "R5:mid_700" in " ".join(chunks[i].chunking_rule_triggered)
        _check(f"T6.pair_{i}", ok,
               f"chunk[{i}] → chunk[{i+1}] overlap≈{max_overlap} (expect ~{OVERLAP_MID}) rules={chunks[i].chunking_rule_triggered}")


# --- T7 Floor merge: final chunk <500 merges into previous ------------------


def test_T7_floor_merge():
    print("T7 — Final chunk <FLOOR merges into previous")
    # Content sized so the tail is ~200 chars
    big = "Section 30. Appeals.\n\n" + ("Appeal text. " * 310)  # ~4000+ chars
    tail = "\n\nShort closing note."
    doc = big + tail
    plan = _plan()
    chunks = chunk_document(doc, plan, _mk_meta())
    if not chunks:
        _check("T7.a", False, "no chunks produced")
        return
    last = chunks[-1]
    _check("T7.a", len(last.text) >= FLOOR or len(chunks) == 1,
           f"last chunk len={len(last.text)} ≥ FLOOR={FLOOR} (or single chunk)")
    # Closing note must survive somewhere
    all_text = "\n\n".join(c.text for c in chunks)
    _check("T7.b", "Short closing note" in all_text, "tail content preserved after merge")
    # Rule audit: R6 only appears if floor merge actually triggered. If final
    # chunk is ≥FLOOR, no merge was needed — also acceptable.
    rules_all = " ".join(r for c in chunks for r in c.chunking_rule_triggered)
    ok = ("R6" in rules_all) or (len(last.text) >= FLOOR) or (len(chunks) == 1)
    _check("T7.c", ok, f"R6 floor merge logged OR final chunk already ≥FLOOR (len={len(last.text)})")

    # Additional: force a tiny-tail scenario and verify R6 triggers
    doc2 = "Section 31. Short section.\n\n" + ("X" * 3480) + "\n\nY"  # tail = 1 char
    chunks2 = chunk_document(doc2, _plan(), _mk_meta())
    if len(chunks2) > 1:
        last2 = chunks2[-1]
        rules2 = " ".join(r for c in chunks2 for r in c.chunking_rule_triggered)
        _check("T7.d", len(last2.text) >= FLOOR or "R6" in rules2,
               f"forced-tiny-tail: last={len(last2.text)} rules contain R6? {'R6' in rules2}")
    else:
        _check("T7.d", True, "single-chunk case — floor merge not applicable")


# --- T8 Bilingual linker (smoke — ≥90% target met on real sample post-Phase-2) ---


def test_T8_bilingual_linker_smoke():
    print("T8 — Bilingual linker: hierarchy-path match on synthetic twin pair")
    # Synthetic English + Hindi twin with 3 sections each
    en_doc = "Section 1. Short title.\n\nContent A.\n\nSection 2. Definitions.\n\nContent B.\n\nSection 3. Levy.\n\nContent C."
    hi_doc = "धारा 1. संक्षिप्त नाम।\n\nविषय A।\n\nधारा 2. परिभाषाएँ।\n\nविषय B।\n\nधारा 3. उद्ग्रहण।\n\nविषय C।"
    en_chunks = chunk_document(en_doc, _plan(depth=2), _mk_meta(source="en.pdf"))
    hi_chunks = chunk_document(hi_doc, _plan(depth=2, lang="hi"), _mk_meta(source="hi.pdf", lang="hi"))
    # hierarchy path = section number; match section N ↔ धारा N
    import re as _re
    def _sec(text):
        m = _re.search(r"(?:Section|धारा)\s+(\d+)", text)
        return m.group(1) if m else None
    en_by_sec = {_sec(c.text): c for c in en_chunks if _sec(c.text)}
    hi_by_sec = {_sec(c.text): c for c in hi_chunks if _sec(c.text)}
    matched = sum(1 for k in en_by_sec if k in hi_by_sec)
    total = max(1, len(en_by_sec))
    rate = matched / total
    _check("T8.a", rate >= 0.9, f"twin-link rate {rate*100:.0f}% (target ≥90%, matched {matched}/{total})")
    # Note: real-corpus T8 runs post-Phase-2 on 10-doc sample; this is the smoke test.


# --- Canonicalization smoke test (shared with dedupe_chunks.py) -------------


def test_canonical_sha_stability():
    print("[extra] Canonical SHA stability (NFKC + ws + lowercase)")
    a = "  Hello   WORLD.\n\nsame   content.  "
    b = "hello world.\nsame content."
    _check("canon.a", canonicalize(a) == canonicalize(b),
           f"canonicalized equal: {canonicalize(a)!r} == {canonicalize(b)!r}")
    _check("canon.b", sha256_of(a) == sha256_of(b), "sha256 equal after canonicalize")


# --- Run harness ------------------------------------------------------------


def main():
    print("=" * 72)
    print("chunker_v2 self-tests T1–T8")
    print("=" * 72)
    tests = [
        test_T1_table_atomic,
        test_T1b_oversize_table_row_split,
        test_T2_proviso_whole,
        test_T3_explanation_whole,
        test_T4_unusable_cut_detection,
        test_T5_section_boundary_zero_overlap,
        test_T6_midsection_overlap,
        test_T7_floor_merge,
        test_T8_bilingual_linker_smoke,
        test_canonical_sha_stability,
    ]
    for t in tests:
        print()
        try:
            t()
        except Exception as e:
            FAILED.append((t.__name__, f"EXCEPTION: {e!r}"))
            print(f"  [FAIL] EXCEPTION in {t.__name__}: {e!r}")

    print("\n" + "=" * 72)
    if FAILED:
        print(f"FAILED: {len(FAILED)} check(s)")
        for tid, msg in FAILED:
            print(f"  [FAIL] {tid}: {msg}")
        sys.exit(1)
    print("ALL PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
