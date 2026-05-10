"""Groundedness check — corpus-intrinsic refusal signal.

DESIGN (2026-04-25): The CBIC corpus is intentionally cross-statute (TDS-under-GST,
transfer pricing JWG, PAN cross-refs, allied_acts: FTP/FEMA/NDPS, etc.). A topic
classifier ("is this CBIC?") would false-positive-refuse legitimate cross-statute
queries. The right primitive is GROUNDEDNESS: do the retrieved chunks actually
contain enough material to answer the question?

This module asks qwen3-14b a single yes/no/partial question with the top-K reranked
chunks as evidence. Designed to be fast (~300-500ms): short prompt, max 30 output
tokens, temperature 0, structured JSON output.

USAGE:
    from groundedness import check_groundedness
    verdict = check_groundedness(question, top_chunks, llm_caller=_call_llm)
    # verdict = {"grounded": "yes"|"partial"|"no", "reason": "..."}
    if verdict["grounded"] == "no":
        # refuse with corpus-honest message

Calling pattern: pass a `llm_caller(system, user, temperature, max_tokens)` callable
so this module is decoupled from api.py / litellm. The caller wires it to whatever
LLM is in front (qwen3-14b on :9082 via LiteLLM gateway in production).
"""
from __future__ import annotations
import json
import re
from typing import List, Dict, Any, Callable

# Per-chunk character budget for the evidence block. Reranker uses 6000 chars
# upstream; here we pass a smaller window to qwen3 to keep prompt under ~4K tokens.
_EVIDENCE_CHARS_PER_CHUNK = 500
_EVIDENCE_TOP_K = 3  # top-3 reranked chunks; lower = faster (~2s vs ~6s)

_SYS = (
    "You are a corpus-grounded retrieval validator. Given a USER QUESTION and "
    "EVIDENCE PASSAGES retrieved from an Indian indirect-tax corpus (CBIC: GST, "
    "Customs, Central Excise, Service Tax + cross-statute allied acts), decide "
    "whether the evidence is sufficient to answer the question.\n\n"
    "Rules:\n"
    "1. \"yes\" — the passages directly contain the facts/rules/values needed.\n"
    "2. \"partial\" — passages mention the topic but a key fact is missing; an "
    "honest partial answer is possible with a caveat.\n"
    "3. \"no\" — passages are about a related-but-different statute, or only share "
    "vocabulary (e.g. 'TDS' appears but the question asks about income-tax TDS while "
    "evidence is about GST TDS), or are unrelated.\n\n"
    "Reply with EXACTLY ONE LINE of JSON: "
    "{\"grounded\": \"yes\"|\"partial\"|\"no\", \"reason\": \"<<=20 words>\"}\n"
    "/no_think"  # qwen3 directive: skip internal monologue for speed
)


def _build_user_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    parts = [f"USER QUESTION: {question}\n\nEVIDENCE PASSAGES:"]
    for i, c in enumerate(chunks[:_EVIDENCE_TOP_K], 1):
        text = (c.get("text") or "")[:_EVIDENCE_CHARS_PER_CHUNK]
        section = c.get("section_ref") or c.get("title") or c.get("doc_id") or "?"
        parts.append(f"\n[{i}] ({section}) {text}")
    parts.append("\n\nRespond with one JSON line only.")
    return "".join(parts)


_JSON_RE = re.compile(r'\{[^{}]*"grounded"[^{}]*\}', re.DOTALL)


def _extract_verdict(raw: str) -> Dict[str, str]:
    raw = (raw or "").strip()
    # qwen3 sometimes wraps in code fences or adds preamble; pull the JSON object.
    m = _JSON_RE.search(raw)
    if not m:
        # fall back: try the whole string
        try:
            v = json.loads(raw)
        except Exception:
            return {"grounded": "unknown", "reason": f"unparseable:{raw[:80]}"}
    else:
        try:
            v = json.loads(m.group(0))
        except Exception:
            return {"grounded": "unknown", "reason": f"jsonerr:{m.group(0)[:80]}"}
    g = str(v.get("grounded", "unknown")).lower().strip()
    if g not in ("yes", "no", "partial"):
        g = "unknown"
    return {"grounded": g, "reason": str(v.get("reason", ""))[:200]}


def check_groundedness(
    question: str,
    top_chunks: List[Dict[str, Any]],
    llm_caller: Callable[..., str],
    *,
    max_tokens: int = 60,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Run groundedness check on retrieved chunks.

    Args:
        question: user query
        top_chunks: list of reranked chunk dicts (with .text, .section_ref, etc.)
        llm_caller: function (system, user, temperature, max_tokens) -> str
        max_tokens: cap output length (default 60 — verdict is short)
        temperature: 0.0 for deterministic verdicts

    Returns:
        {"grounded": "yes"|"partial"|"no"|"unknown", "reason": "...",
         "evidence_count": int, "raw": "<llm raw>"}
    """
    if not top_chunks:
        return {"grounded": "no", "reason": "no_evidence_retrieved",
                "evidence_count": 0, "raw": ""}
    user = _build_user_prompt(question, top_chunks)
    try:
        raw = llm_caller(_SYS, user, temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        return {"grounded": "unknown", "reason": f"llm_error:{type(e).__name__}",
                "evidence_count": len(top_chunks), "raw": str(e)[:200]}
    verdict = _extract_verdict(raw)
    verdict["evidence_count"] = len(top_chunks[:_EVIDENCE_TOP_K])
    verdict["raw"] = raw[:300]
    return verdict


def is_refusal(verdict: Dict[str, Any]) -> bool:
    """Convenience: True iff this verdict should trigger a refusal."""
    return verdict.get("grounded") == "no"
