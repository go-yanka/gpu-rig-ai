"""HyDE / Query Abstraction — 2026-05-08 redesign.

OLD (deprecated): generated hypothetical legal passages, concatenated with query
-> this HURT recall for our long-scenario gold queries because qwen3's hypothetical
text drifted off-topic, pulling retrieval embeddings away from the actual notification.

NEW: extracts the essential legal ask from a long business-scenario query.
Strips narrative noise (company names, locations, amounts) and produces a
formal-statute-style query that aligns with the corpus's amendment/notification language.

Reference: 2026-05-08 consultation brief — two independent consultants agreed
the asymmetric query->chunk mismatch is the bottleneck, and extraction-style
query rewriting is the highest-leverage intervention before fine-tuning.

Pattern: gold query 'Prakash & Associates, a Cost Accounting firm in Chennai,
representing Suvarna Steels Pvt Ltd, has received SCN under 11A(4) for ₹4.5cr
CENVAT willful suppression. Wants to know personal hearing timeline.'
becomes: 'Timeline for granting personal hearing under Section 11A(4) Central
Excise Act for Show Cause Notice on irregular CENVAT credit availment.'

The /no_think prefix on the system message disables qwen3-14b's chain-of-thought
emission so the model returns the answer directly (otherwise it consumes the
max_tokens budget on reasoning_content and emits empty content).
"""
from __future__ import annotations
import os, re, json, urllib.request

LLM_URL = os.environ.get('HYDE_LLM_URL', 'http://127.0.0.1:9082/v1/chat/completions')
LLM_MODEL = os.environ.get('HYDE_MODEL', 'qwen3-14b-q4_k_m.gguf')
HYDE_MAX_TOK = int(os.environ.get('HYDE_MAX_TOK', '200'))
HYDE_TIMEOUT = float(os.environ.get('HYDE_TIMEOUT_S', '20'))

EXTRACT_SYS = (
    '/no_think\n'
    'You are an expert Indian indirect-tax consultant familiar with CBIC notifications, '
    'GST/CGST/IGST Acts, Customs Act, Central Excise Act, and circulars. Given a long '
    'client scenario or business query, extract the CORE LEGAL QUESTION suitable for '
    'retrieval against formal statutory text.\n\n'
    'Rules:\n'
    '- Strip away company names, fictional party names, locations, monetary amounts, '
    'and narrative filler unless they are essential legal anchors (e.g. "SEZ unit", '
    '"100% EOU" matter; "Hindalco-Bharat Copper Works LLP in Gujarat" does not).\n'
    '- Preserve and emphasize: section numbers, rule numbers, notification numbers, '
    'tax types (CVD, CENVAT, IGST, BCD), procedures (SCN, personal hearing, refund, '
    'audit), specific concepts (input tax credit, exemption, drawback, anti-dumping).\n'
    '- Output 1-2 concise sentences (max 50 words), in formal statutory language as if '
    'searching for the relevant notification/circular/section.\n'
    '- If the query already looks like a formal statutory query (short, legal-style), '
    'return it unchanged.\n\n'
    'Output ONLY the extracted query text. No JSON, no preamble, no thinking.'
)

_NARRATIVE_HINT = re.compile(
    r'(Pvt Ltd|LLP|Inc|Ltd\.|company|firm|client|business|importer|manufacturer|'
    r'exporter|trader|consultant|wishes? to|wants? to|representing|received|imports|'
    r'exports|operates|located in)',
    re.I
)
_STATUTE_HINT = re.compile(r'\b(section|rule|notification|circular|act)\b', re.I)


def looks_like_long_scenario(q: str) -> bool:
    """True if query is long narrative — rewrite via extraction."""
    return len(q) >= 200 and bool(_NARRATIVE_HINT.search(q))


def looks_like_formal_query(q: str) -> bool:
    """True if query is already short + formal — skip rewriting."""
    return len(q) < 150 and bool(_STATUTE_HINT.search(q))


def hyde(query: str, timeout: float = HYDE_TIMEOUT) -> str:
    """Return a retrieval-optimized version of the query.

    For long narrative scenarios: extracts the core legal ask via LLM.
    For short formal queries: returns the query unchanged.
    On any failure: returns the original query (safe fallback).

    The returned string is what gets embedded for dense retrieval.
    """
    if looks_like_formal_query(query):
        return query
    if not looks_like_long_scenario(query):
        return query
    try:
        body = {
            'model': LLM_MODEL,
            'messages': [
                {'role': 'system', 'content': EXTRACT_SYS},
                {'role': 'user', 'content': query},
            ],
            'max_tokens': HYDE_MAX_TOK,
            'temperature': 0.0,
        }
        req = urllib.request.Request(
            LLM_URL, method='POST',
            data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read())
        msg = resp['choices'][0]['message']
        # Prefer 'content' (model's actual answer); if empty due to thinking-mode quirk,
        # fall back to 'reasoning_content' tail.
        extracted = (msg.get('content') or '').strip()
        if not extracted:
            rc = (msg.get('reasoning_content') or '').strip()
            # take the last plausible sentence as the extracted query
            if rc:
                # split by period, take last non-empty fragment
                parts = [p.strip() for p in rc.split('.') if p.strip()]
                if parts:
                    extracted = parts[-1]
        if not extracted or len(extracted) < 10:
            return query
        # Concatenate original + extracted: dense embedding averages both, sparse picks
        # up keywords from both. Best of both worlds — preserves any specific terms
        # the LLM might have stripped, while adding the formalized version that aligns
        # with statutory text.
        return query + '\n\n[FORMAL ASK] ' + extracted
    except Exception:
        return query
