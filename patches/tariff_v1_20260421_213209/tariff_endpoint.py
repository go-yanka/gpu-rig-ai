"""
tariff_endpoint.py — /v1/rate-query endpoint. Bypasses RAG for HSN/SAC/rate queries.

Sentinel: tariff_v1
Mount: `app.include_router(tariff_router)` in api.py (added by apply.py).
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

import tariff_query

router = APIRouter(prefix='/v1', tags=['tariff'])


class RateQueryReq(BaseModel):
    question: str
    asof: Optional[str] = None   # ISO date override; defaults to today


@router.get('/rate-query/health')
def health():
    """Sentinel endpoint — confirms the tariff_v1 router is mounted."""
    return {'sentinel': 'tariff_v1', 'status': 'ok'}


@router.post('/rate-query')
def rate_query(req: RateQueryReq):
    if not tariff_query.is_rate_query(req.question):
        raise HTTPException(
            status_code=400,
            detail='Not a rate/HSN/SAC query. Use /query for general RAG.',
        )
    rows = tariff_query.lookup(req.question, asof=req.asof)
    if not rows:
        return {
            'sentinel': 'tariff_v1',
            'rate_table_hit': None,
            'all_matches': [],
            'interpretive_answer': None,
            'citations': [],
            'note': 'No matching tariff rows found. Consider RAG fallback at /query.',
        }
    best = rows[0]
    # TODO (follow-up deploy): call existing RAG with the query scoped to
    # best['notification_id'] to produce an interpretive paragraph.
    return {
        'sentinel': 'tariff_v1',
        'rate_table_hit': best,
        'all_matches': rows,
        'interpretive_answer': None,
        'citations': [{
            'pdf_path': best.get('pdf_path'),
            'page': best.get('doc_page'),
            'notification_id': best.get('notification_id'),
        }],
    }
