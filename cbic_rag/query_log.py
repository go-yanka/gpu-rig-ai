"""Tiny SQLite-backed query-quality tracking layer for CBIC RAG.

Logs every /query call with per-stage timings + citation summary, and lets
the user rate answers. Exposed via api.py as /v1/stats,
/v1/queries/recent, /v1/queries/{id}, /v1/queries/{id}/rate.

WAL-mode SQLite so writes don't block concurrent reads. Safe to import at
module load -- init_db() is idempotent.
"""
from __future__ import annotations
import os, json, sqlite3, threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

DB_PATH = os.environ.get(
    'CBIC_QUERY_LOG_DB',
    '/opt/indian-legal-ai/rag/cbic_rag/query_log.sqlite',
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    question TEXT NOT NULL,
    k INTEGER,
    filters_json TEXT,
    router_category TEXT,
    retrieved_count INTEGER,
    reranked_count INTEGER,
    citations_json TEXT,
    verified_count INTEGER,
    suspicious_count INTEGER,
    answer_markdown TEXT,
    t_total_ms REAL,
    t_router_ms REAL,
    t_hyde_ms REAL,
    t_retrieve_ms REAL,
    t_rerank_ms REAL,
    t_llm_ms REAL,
    user_rating INTEGER,
    user_feedback TEXT,
    client_ip TEXT
);
CREATE INDEX IF NOT EXISTS ix_queries_ts ON queries(ts);
CREATE INDEX IF NOT EXISTS ix_queries_rating ON queries(user_rating);
CREATE INDEX IF NOT EXISTS ix_queries_category ON queries(router_category);
"""

# Body caps (char-based approximates bytes for mostly-ASCII text).
MAX_ANSWER_CHARS = 16 * 1024
MAX_ROW_CHARS = 32 * 1024

_lock = threading.Lock()
_inited = False


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    global _inited
    if _inited:
        return
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _lock:
        conn = _connect()
        try:
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=NORMAL')
            conn.executescript(_SCHEMA)
        finally:
            conn.close()
        _inited = True


def _truncate(s: Optional[str], cap: int) -> Optional[str]:
    if s is None:
        return None
    if len(s) <= cap:
        return s
    return s[: cap - 20] + '\n...[truncated]'


def _slim_citations(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Store minimal fields; drop full excerpts."""
    out = []
    for c in citations or []:
        title = (c.get('title') or '')[:120]
        out.append({
            'index': c.get('index'),
            'doc_id': c.get('doc_id'),
            'page': c.get('page'),
            'score': c.get('score'),
            'category': c.get('category'),
            'subcategory': c.get('subcategory'),
            'title': title,
        })
    return out


def log_query(row: Dict[str, Any]) -> int:
    """Insert a query record; returns new row id."""
    init_db()
    answer = _truncate(row.get('answer_markdown'), MAX_ANSWER_CHARS)
    filters_json = json.dumps(row.get('filters') or {}, ensure_ascii=False)
    citations_json = json.dumps(
        _slim_citations(row.get('citations') or []), ensure_ascii=False
    )
    filters_json = _truncate(filters_json, 4096) or '{}'
    citations_json = _truncate(citations_json, MAX_ROW_CHARS) or '[]'

    ts = datetime.now(timezone.utc).isoformat(timespec='seconds')
    params = (
        ts,
        (row.get('question') or '')[:8000],
        row.get('k'),
        filters_json,
        row.get('router_category'),
        row.get('retrieved_count'),
        row.get('reranked_count'),
        citations_json,
        row.get('verified_count'),
        row.get('suspicious_count'),
        answer,
        row.get('t_total_ms'),
        row.get('t_router_ms'),
        row.get('t_hyde_ms'),
        row.get('t_retrieve_ms'),
        row.get('t_rerank_ms'),
        row.get('t_llm_ms'),
        None,
        None,
        row.get('client_ip'),
    )
    sql = (
        'INSERT INTO queries (ts, question, k, filters_json, router_category,'
        ' retrieved_count, reranked_count, citations_json, verified_count,'
        ' suspicious_count, answer_markdown, t_total_ms, t_router_ms,'
        ' t_hyde_ms, t_retrieve_ms, t_rerank_ms, t_llm_ms, user_rating,'
        ' user_feedback, client_ip) VALUES'
        ' (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'
    )
    with _lock:
        conn = _connect()
        try:
            cur = conn.execute(sql, params)
            return int(cur.lastrowid)
        finally:
            conn.close()


def rate_query(qid: int, rating: int, feedback: Optional[str] = None) -> bool:
    init_db()
    rating = int(rating)
    if rating < 1 or rating > 5:
        raise ValueError('rating must be 1..5')
    with _lock:
        conn = _connect()
        try:
            cur = conn.execute(
                'UPDATE queries SET user_rating=?, user_feedback=? WHERE id=?',
                (rating, (feedback or '')[:4000], qid),
            )
            return cur.rowcount > 0
        finally:
            conn.close()


def _row_summary(r: sqlite3.Row) -> Dict[str, Any]:
    return {
        'id': r['id'],
        'ts': r['ts'],
        'question': r['question'],
        'k': r['k'],
        'router_category': r['router_category'],
        'retrieved_count': r['retrieved_count'],
        'reranked_count': r['reranked_count'],
        'verified_count': r['verified_count'],
        'suspicious_count': r['suspicious_count'],
        't_total_ms': r['t_total_ms'],
        't_retrieve_ms': r['t_retrieve_ms'],
        't_rerank_ms': r['t_rerank_ms'],
        't_llm_ms': r['t_llm_ms'],
        'user_rating': r['user_rating'],
    }


def recent(n: int = 20) -> List[Dict[str, Any]]:
    init_db()
    n = max(1, min(int(n), 500))
    conn = _connect()
    try:
        rows = conn.execute(
            'SELECT * FROM queries ORDER BY id DESC LIMIT ?', (n,)
        ).fetchall()
        return [_row_summary(r) for r in rows]
    finally:
        conn.close()


def by_id(qid: int) -> Optional[Dict[str, Any]]:
    init_db()
    conn = _connect()
    try:
        r = conn.execute(
            'SELECT * FROM queries WHERE id=?', (int(qid),)
        ).fetchone()
        if r is None:
            return None
        d = dict(r)
        try:
            d['filters'] = json.loads(d.pop('filters_json') or '{}')
        except Exception:
            d['filters'] = {}
        try:
            d['citations'] = json.loads(d.pop('citations_json') or '[]')
        except Exception:
            d['citations'] = []
        return d
    finally:
        conn.close()


def stats() -> Dict[str, Any]:
    init_db()
    conn = _connect()
    try:
        total = conn.execute('SELECT COUNT(*) AS c FROM queries').fetchone()['c']
        agg = conn.execute(
            'SELECT AVG(t_total_ms) AS avg_ms,'
            ' AVG(verified_count) AS avg_verified,'
            ' AVG(suspicious_count) AS avg_suspicious,'
            ' SUM(CASE WHEN user_rating IS NOT NULL THEN 1 ELSE 0 END) AS rated,'
            ' SUM(CASE WHEN user_rating >= 4 THEN 1 ELSE 0 END) AS good,'
            ' AVG(CASE WHEN user_rating IS NOT NULL THEN user_rating END) AS avg_rating'
            ' FROM queries'
        ).fetchone()
        cat_rows = conn.execute(
            'SELECT COALESCE(router_category, "(none)") AS cat, COUNT(*) AS c'
            ' FROM queries GROUP BY cat ORDER BY c DESC'
        ).fetchall()
        last = conn.execute(
            'SELECT * FROM queries ORDER BY id DESC LIMIT 20'
        ).fetchall()
        return {
            'total': total,
            'avg_total_ms': round(agg['avg_ms'] or 0.0, 1),
            'avg_verified_per_query': round(agg['avg_verified'] or 0.0, 2),
            'avg_suspicious_per_query': round(agg['avg_suspicious'] or 0.0, 2),
            'rated_count': agg['rated'] or 0,
            'good_count': agg['good'] or 0,
            'pct_good_of_rated': (
                round(100.0 * (agg['good'] or 0) / (agg['rated'] or 1), 1)
                if (agg['rated'] or 0) > 0 else 0.0
            ),
            'avg_rating': round(agg['avg_rating'] or 0.0, 2),
            'by_category': {r['cat']: r['c'] for r in cat_rows},
            'last_20': [_row_summary(r) for r in last],
        }
    finally:
        conn.close()


if __name__ == '__main__':
    init_db()
    print(json.dumps(stats(), indent=2))
