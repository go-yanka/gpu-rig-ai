#!/usr/bin/env python3
"""B22 B19 apply: insert augment_section_aware before _reranker = None in retriever.py."""
import io, sys
PATH = "/opt/indian-legal-ai/rag/cbic_rag/retriever.py"

BLOCK = '''# b22_v1 B19: section-aware post-retrieval augmentation.
# Runs targeted queries for Section/Rule references and common GST legal
# phrases, unions into the hit set. Dedupes on doc_id+chunk_index.
import re as _b19_re
_B19_SEC_RE = _b19_re.compile(r'\\b(?:[Ss]ection|[Rr]ule)\\s+(\\d+[A-Z]?(?:\\(\\d+\\))?(?:\\([a-z]\\))?)\\b')
_B19_PHRASES = [
    'bill-to-ship-to', 'bill to ship to', 'composite supply', 'mixed supply',
    'place of supply', 'time of supply', 'input tax credit',
    'reverse charge',
]

def _b19_chunk_key(c):
    return (c.get('doc_id') or '', c.get('chunk_index', c.get('char_start', '')))

def augment_section_aware(question: str, hits: list, filters=None,
                          k_per_ref: int = 3, max_total: int = 40,
                          timings=None) -> list:
    """Additive retrieval boost. Runs targeted queries for each Section/Rule
    reference and common GST phrases found in the question, unions them into
    `hits` deduped by (doc_id, chunk_index). Returns augmented list capped at
    `max_total`. Never breaks existing retrieval."""
    if not question:
        if timings is not None:
            timings['section_aware_added'] = 0
        return hits
    added = 0
    existing_keys = {_b19_chunk_key(c) for c in hits}
    out = list(hits)
    targets = []
    for m in _B19_SEC_RE.finditer(question):
        targets.append(question[m.start():m.end()])
    ql = question.lower()
    for ph in _B19_PHRASES:
        if ph in ql:
            targets.append(ph)
    seen = set(); uniq_targets = []
    for t in targets:
        tl = t.lower().strip()
        if tl in seen: continue
        seen.add(tl); uniq_targets.append(t)
    if not uniq_targets:
        if timings is not None:
            timings['section_aware_added'] = 0
        return out
    for tgt in uniq_targets:
        if len(out) >= max_total:
            break
        try:
            extra = retrieve(tgt, k=k_per_ref, filters=filters, timings=None)
        except Exception:
            continue
        for c in extra:
            key = _b19_chunk_key(c)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            c['_b19_added_via'] = tgt
            out.append(c)
            added += 1
            if len(out) >= max_total:
                break
    if timings is not None:
        timings['section_aware_added'] = added
    return out


'''

with io.open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

if "def augment_section_aware" in src:
    print("ERROR: augment_section_aware already present; aborting to avoid dup")
    sys.exit(2)

anchor = "_reranker = None"
idx = src.find(anchor)
if idx < 0:
    print("ERROR: anchor '_reranker = None' not found")
    sys.exit(3)

# Find start of that line
line_start = src.rfind("\n", 0, idx) + 1

new_src = src[:line_start] + BLOCK + src[line_start:]

if "b22_v1 B19" not in new_src or "augment_section_aware" not in new_src:
    print("ERROR: sentinel or function missing after patch")
    sys.exit(4)

with io.open(PATH, "w", encoding="utf-8") as f:
    f.write(new_src)

# Quick syntax check
import py_compile
py_compile.compile(PATH, doraise=True)
print("OK: retriever.py patched (b22_v1 B19)")
