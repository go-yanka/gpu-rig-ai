"""Issue-spotting query rewrite + reciprocal-rank fusion.

Long business scenarios ("Hindalco-Bharat Copper Works LLP, a manufacturer of
copper wires in Gujarat, wishes to claim ...") share almost no words with the
notification that answers them. Legal-retrieval work in 2025-26 (Stanford
issue-spotting rewrites, LegalMALR) gets +6-10 pts recall@10 by first asking an
LLM to name the legal issue and restate it in statute language, then fusing the
results of several such queries. This differs from HyDE (L2 in
LESSONS_2026-05-08), which wrote a generic hypothetical answer and hurt recall.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Dict, Iterable, List

REWRITE_SYSTEM = """You are a research librarian for Indian indirect tax law published by CBIC:
GST (CGST/IGST/UTGST Acts, rules, rate notifications, circulars), Customs (Customs Act,
Customs Tariff Act, exemption / anti-dumping / tariff-value notifications, drawback,
warehousing), Central Excise, Service Tax (legacy), and allied acts CBIC administers.

Given a user's question (it may be a long business scenario, and may be in Hindi or
Hinglish), do NOT answer it. Instead:
1. Name the legal issue(s) it raises, in one short line each.
2. Write up to 3 search queries (English) phrased the way the governing statute,
   rule, notification or circular would phrase it. Include likely section / rule
   numbers, notification series (e.g. "Customs", "Central Tax (Rate)"), tariff or
   HSN terms. Drop company and person names, GSTINs, amounts and cities unless they
   matter legally (a state matters for place of supply; a port can matter for customs).
3. List any notification / circular numbers the user mentions, exactly as written.
4. in_scope = false ONLY if the question is clearly outside CBIC law (income tax,
   company law, RBI/FEMA banking, labour law, general knowledge). If unsure, true.

Reply with JSON only:
{"issues": ["..."], "queries": ["..."], "notification_refs": ["..."],
 "in_scope": true, "language": "en|hi|other"}"""


def _parse(raw: str) -> Dict:
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.S)
    m = re.search(r'\{.*\}', raw, flags=re.S)
    if not m:
        raise ValueError('no JSON object in rewrite output')
    d = json.loads(m.group(0))
    return {
        'issues': [str(x) for x in d.get('issues') or []][:4],
        'queries': [str(x).strip() for x in d.get('queries') or [] if str(x).strip()][:3],
        'notification_refs': [str(x) for x in d.get('notification_refs') or []][:5],
        'in_scope': d.get('in_scope') is not False,
        'language': str(d.get('language') or 'en'),
    }


def rewrite(question: str, call_llm: Callable[[str, str], str]) -> Dict:
    """call_llm(system, user) -> text. Returns issues, queries, refs, in_scope, language.

    If the model's output can't be parsed, the original question is used alone
    and `rewrite_error` says why — retrieval still runs, just without rewrites.
    """
    try:
        out = _parse(call_llm(REWRITE_SYSTEM, question))
    except (ValueError, json.JSONDecodeError) as e:
        return {'issues': [], 'queries': [], 'notification_refs': [], 'in_scope': True,
                'language': 'en', 'rewrite_error': str(e)}
    return out


def rrf(ranked_lists: Iterable[List[Dict]], key: Callable[[Dict], str], k: int = 60) -> List[Dict]:
    """Reciprocal-rank fusion. Items keep the fields of their best-ranked copy,
    plus `rrf` (fused score) and `hit_by` (how many query lists found them)."""
    scores: Dict[str, float] = {}
    best: Dict[str, Dict] = {}
    hits: Dict[str, int] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst):
            kk = key(item)
            scores[kk] = scores.get(kk, 0.0) + 1.0 / (k + rank + 1)
            hits[kk] = hits.get(kk, 0) + 1
            if kk not in best:
                best[kk] = item
    fused = []
    for kk in sorted(scores, key=scores.get, reverse=True):
        item = dict(best[kk])
        item['rrf'] = round(scores[kk], 5)
        item['hit_by'] = hits[kk]
        fused.append(item)
    return fused
