"""A3 two-pass span validator (sentinel: two_pass_v1).

Validates that an LLM-emitted `verbatim_span` is actually a verbatim (or
near-verbatim) substring of the cited chunk text.

Validation ladder (fail-fast):
  1. length in [80, 450] + at least one clause terminator (. ? ! ;)
  2. canonical substring (NFKC + whitespace collapse + lowercase)
  3. is_table bypass: exact canonical substring only, no fuzzy fallback
  4. 6-gram character-Jaccard >= 0.80
  5. BGE-M3 cosine fallback >= 0.92 (lazy import from embedder)

Returns (ok, reason).  `reason` is a short tag suitable for logging and for
surfacing in `suspicious_quotes[]` on the response.
"""
from __future__ import annotations
import re
import unicodedata
from typing import Tuple


def _canon(s: str) -> str:
    return unicodedata.normalize("NFKC", re.sub(r"\s+", " ", s)).strip().lower()


def _grams(s: str, n: int = 6):
    if len(s) < n:
        return set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def validate_span(span: str, chunk_text: str, is_table: bool = False) -> Tuple[bool, str]:
    """Returns (ok, reason). Fails fast on obvious violations."""
    if span is None or chunk_text is None:
        return (False, "null_input")

    # 1. length + clause terminator
    if not (80 <= len(span) <= 450):
        return (False, f"length_{len(span)}")
    if not re.search(r"[.?!;]", span):
        return (False, "no_clause_terminator")

    # 2. canonicalize
    cs = _canon(span)
    cc = _canon(chunk_text)
    if not cs:
        return (False, "empty_span")

    # 3. exact substring after canon
    if cs in cc:
        return (True, "exact")

    # 4. tables: exact only, never fuzzy (numbers matter)
    if is_table:
        return (False, "table_exact_miss")

    # 5. 6-gram Jaccard
    g1 = _grams(cs, 6)
    g2 = _grams(cc, 6)
    if not g1:
        return (False, "too_short_for_grams")
    union = g1 | g2
    if not union:
        return (False, "empty_union")
    jac = len(g1 & g2) / len(union)
    if jac >= 0.80:
        return (True, f"jaccard_{jac:.2f}")

    # 6. BGE-M3 cosine fallback (lazy)
    try:
        import numpy as np  # local
        try:
            # Prefer retriever's cached embed
            from retriever import _cached_embed_query as _emb  # noqa
            v1 = _emb(span.strip().lower())["dense"]
            v2 = _emb(chunk_text.strip().lower())["dense"]
        except Exception:
            # Fall back to embedder.embed_query directly
            import embedder  # type: ignore
            v1 = embedder.embed_query(span.strip().lower())["dense"]
            v2 = embedder.embed_query(chunk_text.strip().lower())["dense"]
        v1 = np.asarray(v1, dtype="float32")
        v2 = np.asarray(v2, dtype="float32")
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 == 0.0 or n2 == 0.0:
            return (False, f"jaccard_{jac:.2f}_zero_norm")
        cos = float(np.dot(v1, v2) / (n1 * n2))
        if cos >= 0.92:
            return (True, f"cos_{cos:.2f}")
        return (False, f"cos_{cos:.2f}_jac_{jac:.2f}")
    except Exception as e:
        return (False, f"embed_err_{type(e).__name__}_jac_{jac:.2f}")
