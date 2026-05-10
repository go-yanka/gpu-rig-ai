"""Patch chunker_v2.py:
  1. Add tail-dedup pass: drop any chunk whose char range is 100% inside another's.
  2. Add section_ref backfill: if section_ref is empty, fall back to parent_hierarchy_text.
Codified 2026-04-26 after CP-2 G1=0.8132/G3=0.8242 retrieval-merit failures.
"""
p = '/opt/indian-legal-ai/reingest_spec/chunker_v2.py'
s = open(p).read()

# ---------- Patch 1: tail-dedup pass after _merge_floor ----------
old1 = """    # -- R6 floor merge: final chunk < FLOOR merges into previous ------------
    all_chunks = _merge_floor(all_chunks)

    return all_chunks"""
new1 = """    # -- R6 floor merge: final chunk < FLOOR merges into previous ------------
    all_chunks = _merge_floor(all_chunks)

    # -- R7 tail-dedup pass: drop any chunk whose char range is 100% inside another's
    # Codified 2026-04-26 after CP-2 lint flagged 622 tail-dup chunks corpus-wide,
    # despite Defect F2 fix at line 1003. Other code paths (section_bounded_split,
    # table chunks crossing prose spans) can still emit overlapping ranges.
    all_chunks = _dedup_contained(all_chunks)

    # -- R8 section_ref backfill: if section_ref empty, use parent_hierarchy_text
    # Codified 2026-04-26 after CP-1 G3 root-cause showed 3/6 misses had empty section_ref.
    for c in all_chunks:
        if not (c.section_ref or "").strip() and (c.parent_hierarchy_text or "").strip():
            c.section_ref = c.parent_hierarchy_text.strip().split("\\n")[0][:120]

    return all_chunks


def _dedup_contained(chunks: list) -> list:
    \"\"\"Drop any chunk whose [start, end] is fully inside another's range.
    Keeps the larger (containing) chunk. O(n log n) via sort by start asc, end desc.
    \"\"\"
    if not chunks:
        return chunks
    # Sort by start asc, then end desc so contained chunks come right after their containers
    sortable = sorted(enumerate(chunks),
                      key=lambda ic: (ic[1].page_range[0], -ic[1].page_range[1]))
    keep = [True] * len(chunks)
    for i in range(len(sortable)):
        if not keep[sortable[i][0]]: continue
        ai_idx, ai = sortable[i]
        a_start, a_end = ai.page_range
        for j in range(i + 1, len(sortable)):
            bj_idx, bj = sortable[j]
            if not keep[bj_idx]: continue
            b_start, b_end = bj.page_range
            if b_start > a_end:
                break  # past the container's reach
            # b is fully inside a (strict containment, not identity)
            if b_start >= a_start and b_end <= a_end and (b_start, b_end) != (a_start, a_end):
                keep[bj_idx] = False
    out = [c for i, c in enumerate(chunks) if keep[i]]
    if len(out) < len(chunks):
        # surface to logs
        try:
            import sys
            sys.stderr.write(f\"[chunker R7 dedup] dropped {len(chunks) - len(out)} contained chunks (kept {len(out)}/{len(chunks)})\\n\")
        except Exception:
            pass
    return out"""
if old1 in s:
    s = s.replace(old1, new1)
    print("patch 1 (tail-dedup + section_ref backfill) applied")
else:
    print("ERROR: patch 1 anchor not found")
    raise SystemExit(2)

open(p, 'w').write(s)
print("written, total bytes:", len(s))

# Smoke compile
import py_compile
try:
    py_compile.compile(p, doraise=True)
    print("syntax OK")
except py_compile.PyCompileError as e:
    print("SYNTAX ERROR:", e)
    raise SystemExit(3)
