#!/usr/bin/env python3
"""B22 apply: patch api.py — import line, append B17 helpers, replace retrieve block."""
import io, sys, re
PATH = "/opt/indian-legal-ai/rag/cbic_rag/api.py"

with io.open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

if "b22_v1" in src:
    print("ERROR: b22_v1 already present in api.py; aborting")
    sys.exit(2)

# ---- Edit A: import line ----
old_imp = "from retriever import retrieve, rerank"
new_imp = "# b22_v1 sentinel\nfrom retriever import retrieve, rerank, augment_section_aware"
if old_imp not in src:
    print("ERROR: import anchor not found")
    sys.exit(3)
src = src.replace(old_imp, new_imp, 1)

# ---- Edit B: append B17 helpers after _ms_since() body ----
HELPERS = '''


# b22_v1 B17: multi-sub-query decomposition for complex multi-part questions.
import functools as _b17_ft
import re as _b17_re

_B17_DECOMP_SYS = (
    "You split a complex legal question into focused sub-queries for retrieval. "
    "Output ONE sub-query per line, no numbering, no commentary, no preamble. "
    "Each sub-query focuses on a single legal concept, section, rule, or factual aspect."
)
_B17_DECOMP_USER = (
    "Split this complex legal question into 3-6 focused sub-queries for retrieval. "
    "Output ONE sub-query per line, no numbering, no commentary. Each sub-query "
    "should focus on a single legal concept, section, rule, or factual aspect.\\n\\n"
    "Question: {q}\\n\\nSub-queries:"
)

def _b17_is_multipart(question: str) -> bool:
    if not question:
        return False
    qmarks = question.count('?')
    if qmarks > 2:
        return True
    if len(question) > 500:
        return True
    numbered = len(_b17_re.findall(r'(?:^|\\s)\\(?\\d+[\\.\\)]', question))
    if numbered > 1:
        return True
    return False

@_b17_ft.lru_cache(maxsize=256)
def _b17_decompose_cached(question: str) -> tuple:
    try:
        raw = _call_llm(_B17_DECOMP_SYS, _B17_DECOMP_USER.format(q=question),
                        temperature=0.1)
    except Exception as _e:
        print(f'[b17] decompose failed: {_e}')
        return tuple()
    lines = []
    for ln in (raw or '').splitlines():
        s = ln.strip()
        if not s:
            continue
        s = _b17_re.sub(r'^[\\-\\*\\u2022]\\s*', '', s)
        s = _b17_re.sub(r'^\\(?\\d+[\\.\\)]\\s*', '', s)
        s = _b17_re.sub(r'^sub-?quer(?:y|ies)\\s*[:\\-]\\s*', '', s, flags=_b17_re.I)
        if len(s) < 8:
            continue
        if s.lower().startswith(('here are', 'sub-queries', 'sub queries', 'output:')):
            continue
        lines.append(s)
        if len(lines) >= 6:
            break
    return tuple(lines)

def _b17_multi_retrieve(question: str, filters, k_sub: int = 6,
                       max_union: int = 40, timings=None) -> tuple:
    if not _b17_is_multipart(question):
        return [], 0, 0
    subs = _b17_decompose_cached(question)
    if not subs:
        return [], 0, 0
    seen = set()
    out = []
    for s in subs:
        try:
            extra = retrieve(s, k=k_sub, filters=filters, timings=None)
        except Exception:
            continue
        for c in extra:
            key = ((c.get('doc_id') or ''), c.get('chunk_index', c.get('char_start', '')))
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
            if len(out) >= max_union:
                break
        if len(out) >= max_union:
            break
    if timings is not None:
        timings['multi_subquery_count'] = len(subs)
        timings['subquery_union_size'] = len(out)
    return out, len(subs), len(out)
'''

# Insert right after the _ms_since function. Locate its body end.
m = re.search(r'(def _ms_since\(t0: float\) -> float:\s*\n\s*return[^\n]*\n)', src)
if not m:
    print("ERROR: could not find _ms_since body")
    sys.exit(4)
insert_at = m.end()
src = src[:insert_at] + HELPERS + src[insert_at:]

# ---- Edit C: replace the retrieve block (# 3) retrieve ... timings['retrieve_ms']=_ms_since(t0)) ----
NEW_BLOCK = '''    # 3) retrieve
    t0 = time.perf_counter()
    _sub = {}  # b2_subtimings

    hits = retrieve(search_text, k=TOP_K_RETR, filters=filters, timings=_sub)

    timings['embed_query_ms'] = _sub.get('embed_query_ms')

    timings['qdrant_ms']      = _sub.get('qdrant_ms')
    # b11b12a8_v1: surface embed cache stats
    if 'embed_cache_stats' in _sub:
        timings['embed_cache_stats'] = _sub['embed_cache_stats']
    if 'embed_cache_hit' in _sub:
        timings['embed_cache_hit'] = _sub['embed_cache_hit']
    if not hits and filters:
        hits = retrieve(search_text, k=TOP_K_RETR, filters=None)
    # b22_v1 B17: multi-sub-query union for complex questions
    timings['multi_subquery_count'] = 0
    timings['subquery_union_size'] = 0
    try:
        multi_hits, n_subs, union_sz = _b17_multi_retrieve(
            req.question, filters, k_sub=6, max_union=40, timings=timings)
        if multi_hits:
            seen = {((c.get('doc_id') or ''), c.get('chunk_index', c.get('char_start',''))) for c in hits}
            for c in multi_hits:
                k = ((c.get('doc_id') or ''), c.get('chunk_index', c.get('char_start','')))
                if k in seen:
                    continue
                seen.add(k); hits.append(c)
                if len(hits) >= 40:
                    break
    except Exception as _e:
        print(f'[b17] multi-retrieve err: {_e}')
    # b22_v1 B19: section-aware augmentation (additive, capped at 40)
    try:
        hits = augment_section_aware(req.question, hits, filters=filters,
                                     k_per_ref=3, max_total=40, timings=timings)
    except Exception as _e:
        print(f'[b19] section-aware err: {_e}')
        timings.setdefault('section_aware_added', 0)
    timings['retrieve_ms'] = _ms_since(t0)
'''

# Replace starting "    # 3) retrieve" up to the line containing "timings['retrieve_ms'] = _ms_since(t0)" inclusive.
pat = re.compile(
    r"    # 3\) retrieve\n[\s\S]*?    timings\[\'retrieve_ms\'\] = _ms_since\(t0\)\n",
    re.M,
)
m2 = pat.search(src)
if not m2:
    print("ERROR: could not match retrieve block for replacement")
    sys.exit(5)
src = src[:m2.start()] + NEW_BLOCK + src[m2.end():]

if src.count("b22_v1") < 3:
    print(f"ERROR: sentinel count too low ({src.count('b22_v1')})")
    sys.exit(6)

with io.open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

import py_compile
py_compile.compile(PATH, doraise=True)
print("OK: api.py patched (b22_v1)")
