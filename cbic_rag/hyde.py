"""HyDE — Hypothetical Document Embeddings.

Instead of embedding the query directly, we ask a small LLM to write a
hypothetical answer/passage in the style of CBIC tax documents. That
hypothetical document tends to embed closer to the real retrieved chunks.

For CBIC queries, typical user queries are short ('rate on cement?') while
source docs are formal and contain legal phrasing. HyDE bridges this gap.

Fast path: skip HyDE for queries that already look like document passages
(long, formal, contain statute refs) — they don't benefit.
"""
from __future__ import annotations
import os, re, requests

LITELLM_URL   = os.environ.get('LITELLM_URL', 'http://127.0.0.1:4444')
HYDE_MODEL    = os.environ.get('HYDE_MODEL', 'gemma4')
HYDE_MAX_TOK  = int(os.environ.get('HYDE_MAX_TOK', '180'))

HYDE_SYS = ("Write a short passage (4-6 sentences) in the style of an Indian "
            "CBIC tax circular or notification, as if it directly answers the "
            "question. Use formal legal/tax language. Include plausible section "
            "numbers, rates, or dates. Do not say 'I don't know' — invent "
            "plausible details; this is for retrieval, not for the final answer.")

_STATUTE_HINT = re.compile(r'\b(section|rule|notification|circular|act)\b', re.I)


def looks_like_passage(q: str) -> bool:
    return len(q) >= 180 or bool(_STATUTE_HINT.search(q))


def hyde(query: str, timeout: float = 15.0) -> str:
    """Return a hypothetical passage. On failure, return the original query."""
    if looks_like_passage(query):
        return query
    try:
        r = requests.post(
            f'{LITELLM_URL}/v1/chat/completions',
            json={
                'model': HYDE_MODEL,
                'messages': [
                    {'role': 'system', 'content': HYDE_SYS},
                    {'role': 'user',   'content': query},
                ],
                'max_tokens': HYDE_MAX_TOK,
                'temperature': 0.4,
            },
            timeout=timeout,
        )
        r.raise_for_status()
        text = r.json()['choices'][0]['message']['content'].strip()
        if not text:
            return query
        # combine: query + hypothetical → better recall on both semantic and lexical sides
        return f'{query}\n\n{text}'
    except Exception:
        return query
