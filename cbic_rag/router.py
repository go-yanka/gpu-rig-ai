"""Query-intent router.

Before retrieval, classify the query into one of the top-level CBIC categories
(GST, Customs, Central Excise, Service Tax, etc.) so we can bias retrieval via
payload filters. If ambiguous, no filter is applied.

Two tiers:
1. Cheap keyword routing (0 ms, most queries).
2. LLM fallback for ambiguous queries (optional, adds ~500 ms).
"""
from __future__ import annotations
import os, re, requests

LITELLM_URL   = os.environ.get('LITELLM_URL', 'http://127.0.0.1:4444')
ROUTER_MODEL  = os.environ.get('ROUTER_MODEL', 'gemma4')

CATEGORIES = ['gst', 'customs', 'central-excise', 'service-tax', 'ntrp', 'general']

_KW = {
    'gst': [r'\bgst\b', r'\bhsn\b', r'\bsac\b', r'goods and services tax',
            r'cgst', r'sgst', r'igst', r'utgst', r'compensation cess',
            r'itc\b', r'input tax credit', r'eway bill', r'e-way', r'gstin',
            r'tds under gst', r'reverse charge'],
    'customs': [r'\bcustoms\b', r'import\s*duty', r'export\s*duty',
                r'basic customs duty', r'\bbcd\b', r'\bsws\b', r'anti-dumping',
                r'countervailing', r'drawback', r'customs tariff', r'\bicegate\b'],
    'central-excise': [r'central excise', r'excise duty', r'cenvat', r'\bcetsh\b'],
    'service-tax': [r'service tax', r'\bstc\b'],
    'ntrp':    [r'\bntrp\b', r'non-tax revenue'],
}
_KW_C = {k: [re.compile(p, re.I) for p in v] for k, v in _KW.items()}


def route_keyword(query: str) -> str | None:
    """Cheap regex routing. Returns category or None."""
    scores = {k: 0 for k in _KW_C}
    for cat, pats in _KW_C.items():
        for p in pats:
            if p.search(query):
                scores[cat] += 1
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return None
    # if two categories tied and both have hits, don't filter
    top = sorted(scores.values(), reverse=True)
    if len(top) >= 2 and top[0] == top[1] and top[1] > 0:
        return None
    return best


def route_llm(query: str, timeout: float = 8.0) -> str | None:
    """Fallback: ask small LLM. Returns one of CATEGORIES or None."""
    sys_msg = ("Classify the query into exactly one of: "
               + ', '.join(CATEGORIES) + ". Respond with only the label.")
    try:
        r = requests.post(
            f'{LITELLM_URL}/v1/chat/completions',
            json={'model': ROUTER_MODEL,
                  'messages': [{'role': 'system', 'content': sys_msg},
                               {'role': 'user',   'content': query}],
                  'max_tokens': 8, 'temperature': 0.0},
            timeout=timeout,
        )
        r.raise_for_status()
        label = r.json()['choices'][0]['message']['content'].strip().lower()
        label = re.sub(r'[^a-z-]', '', label)
        if label in CATEGORIES and label != 'general':
            return label
    except Exception:
        pass
    return None


def route(query: str, use_llm_fallback: bool = False) -> str | None:
    c = route_keyword(query)
    if c is not None:
        return c
    if use_llm_fallback:
        return route_llm(query)
    return None


def filters_for(category: str | None) -> dict | None:
    if not category:
        return None
    return {'category': category}
