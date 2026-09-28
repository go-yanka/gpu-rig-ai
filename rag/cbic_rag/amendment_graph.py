#!/usr/bin/env python3
"""Notification amendment graph for the CBIC corpus.

Extracts "X amends / rescinds / supersedes Y" relations from notification text
with deterministic regexes (no model), stores them in SQLite, and answers
"what is the amendment history / current status of notification Y?" with the
verbatim sentence that proves each link.

Build (on the rig, from the v2 ingest manifest):
    python3 amendment_graph.py build --manifest /opt/indian-legal-ai/data/ingest_manifest_v2.sqlite \
        --out /opt/indian-legal-ai/data/amendment_graph.sqlite
Build (anywhere, from a JSONL of {doc_id, title, doc_number, text}):
    python3 amendment_graph.py build --jsonl chunks.jsonl --out graph.sqlite
Query:
    python3 amendment_graph.py chain "35/2020-Central Tax" --db graph.sqlite
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Iterator, List, Optional

# "Notification No. 31/86-Customs", "notfn. 20/2004-Cus (N.T.)", "No. 55/2020 – Central Tax"
_REF = re.compile(
    r'(?:notification|notfn\.?|\bno\.)\s*(?:no\.?|number)?\s*'
    r'(?P<num>\d{1,4})\s*/\s*(?P<yr>\d{2,4})\s*[-–—]?\s*'
    r'(?P<series>[A-Za-z][A-Za-z .()&]{0,45})',
    re.I)

_SERIES_STOP = re.compile(
    r'\b(dated|dt|of|which|and|to|is|was|in|the|as|has|vide|published|with|by|for|'
    r'so|namely|ibid|shall|are|hereby|further|on|date|sl|gsr|refers|may)\b', re.I)

_MONTHS = {m: i for i, m in enumerate(
    ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
     'september', 'october', 'november', 'december'], start=1)}
_DATE_WORDS = re.compile(
    r'(?:dated|dt\.?)\s*(?:the\s*)?(\d{1,2})\s*(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?([A-Za-z]+),?\s*(\d{4})', re.I)
_DATE_NUM = re.compile(r'(?:dated|dt\.?)\s*(?:the\s*)?(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})', re.I)


def normalize_series(raw: str) -> Optional[str]:
    """Map a free-text series ('Customs (N.T.)', 'CT(R)', 'Cus.') to a code."""
    s = _SERIES_STOP.split(raw)[0]
    low = re.sub(r'[\s.]', '', s.lower())
    if not low:
        return None
    nt = '(nt' in low or low.endswith('nt)') or low.endswith('nt')
    rate = '(rate' in low or '(r)' in low
    if low.startswith(('cus', 'custom')):
        if '(add' in low:
            return 'CUS-ADD'
        if '(cvd' in low:
            return 'CUS-CVD'
        if '(sg' in low:
            return 'CUS-SG'
        return 'CUS-NT' if nt else 'CUS'
    if low.startswith('centra'):
        if low.startswith('centralexcise'):
            return 'CE-NT' if nt else 'CE'
        if low.startswith('centraltax'):
            return 'CT-RATE' if rate else 'CT'
        return None  # truncated ("65/2020-Centra") — ambiguous
    if low.startswith(('ce', 'cx', 'c.e')):
        return 'CE-NT' if nt else 'CE'
    if low.startswith(('servicetax', 'st')):
        return 'ST'
    if low.startswith('ct'):
        return 'CT-RATE' if rate else 'CT'
    if low.startswith(('integratedtax', 'it(', 'it')):
        return 'IT-RATE' if rate else 'IT'
    if low.startswith(('unionterritory', 'utt')):
        return 'UTT-RATE' if rate else 'UTT'
    if low.startswith('compensationcess'):
        return 'CESS-RATE' if rate else 'CESS'
    return None


def normalize_year(yr: str) -> int:
    y = int(yr)
    if y < 100:
        y += 1900 if y >= 50 else 2000
    return y


@dataclass(frozen=True)
class Ref:
    key: str          # canonical, e.g. "35/2020-CT"
    start: int
    end: int
    date: Optional[str]  # ISO date if stated right after the reference


def _date_after(text: str, pos: int) -> Optional[str]:
    window = text[pos:pos + 60]
    m = _DATE_WORDS.match(window.lstrip(' ,')) or _DATE_WORDS.search(window[:40])
    if m and m.group(2).lower() in _MONTHS:
        try:
            return date(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1))).isoformat()
        except ValueError:
            return None
    m = _DATE_NUM.search(window[:40])
    if m:
        try:
            return date(normalize_year(m.group(3)), int(m.group(2)), int(m.group(1))).isoformat()
        except ValueError:
            return None
    return None


def parse_key(text: str) -> Optional[str]:
    """Parse a user-typed reference like '35/2020-Central Tax' or '50/2017 Customs'."""
    m = re.search(r'(\d{1,4})\s*/\s*(\d{2,4})\s*[-–—]?\s*([A-Za-z][A-Za-z .()&]*)', text)
    if not m:
        return None
    series = normalize_series(m.group(3))
    if not series:
        return None
    return f'{int(m.group(1))}/{normalize_year(m.group(2))}-{series}'


def find_refs(text: str) -> List[Ref]:
    out = []
    for m in _REF.finditer(text):
        series = normalize_series(m.group('series'))
        if not series:
            continue
        key = f"{int(m.group('num'))}/{normalize_year(m.group('yr'))}-{series}"
        end = m.start('series') + len(_SERIES_STOP.split(m.group('series'))[0].rstrip(' ,'))
        out.append(Ref(key, m.start(), end, _date_after(text, end)))
    return out


# relation patterns: verb BEFORE the target reference ...
_FORWARD = [
    ('amends', re.compile(r'(?:amendments?\s+(?:shall\s+be\s+made\s+)?in\s+the\s+'
                          r'(?:principal\s+)?notification|seeks\s+to\s+amend|\bamends?\b\s+(?:the\s+)?'
                          r'(?:principal\s+)?notification|further\s+to\s+amend)', re.I)),
    ('rescinds', re.compile(r'(?:seeks\s+to\s+rescind|hereby\s+rescinds?|\brescinds?\b\s+(?:the\s+)?notification)', re.I)),
    ('supersedes', re.compile(r'in\s+supersession\s+of', re.I)),
]
# ... or AFTER it ("notification No. X ... is hereby rescinded")
_BACKWARD = [
    ('rescinds', re.compile(r'(?:is|are|shall\s+stand)\s+(?:hereby\s+)?rescinded', re.I)),
]
# passive, about THIS document: "Rescinded vide Notification No. 7/2001-CE"
_PASSIVE_SELF = re.compile(r'\b(rescinded|superseded|amended)\s+(?:vide|by)\b', re.I)
_PASSIVE_REL = {'rescinded': 'rescinds', 'superseded': 'supersedes', 'amended': 'amends'}
_HEADER_DATE = re.compile(
    r'New\s+Delhi,?\s*(?:the\s*)?(\d{1,2})\s*(?:st|nd|rd|th)?\s*(?:day\s+of\s+)?([A-Za-z]+),?\s*(\d{4})', re.I)


def header_date(text: str) -> Optional[str]:
    m = _HEADER_DATE.search(text[:800])
    if m and m.group(2).lower() in _MONTHS:
        try:
            return date(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1))).isoformat()
        except ValueError:
            return None
    return None
_PRINCIPAL = re.compile(r'principal\s+notification', re.I)

_WINDOW = 220
_VERB = re.compile(r'amend|rescind|supersed|supersess|substitut', re.I)


def _evidence(text: str, a: int, b: int) -> str:
    s = max(0, a - 60)
    e = min(len(text), b + 60)
    return re.sub(r'\s+', ' ', text[s:e]).strip()


@dataclass(frozen=True)
class Edge:
    src: str
    rel: str          # amends | rescinds | supersedes
    dst: str
    src_date: Optional[str]
    dst_date: Optional[str]
    evidence: str


def self_key(doc_number: str, title: str, text: str) -> Optional[str]:
    """Which notification does this document itself carry?

    The manifest's doc_number is often truncated ("65/2020–Centra"), so its
    number/year is trusted but the series may have to come from the text, or
    — for a truncated "Centra…" — from the family of the notification the
    title says it amends (amendments stay within their series family).
    """
    m = re.match(r'\s*(\d{1,4})\s*/\s*(\d{2,4})', doc_number or '')
    if m:
        k = parse_key(doc_number)
        if k:
            return k
        prefix = f'{int(m.group(1))}/{normalize_year(m.group(2))}-'
        for r in find_refs(text):
            if r.key.startswith(prefix):
                return r.key
        if re.search(r'[-–—]\s*centra', doc_number, re.I):
            for r in find_refs(title):
                base = r.key.split('-', 1)[1]
                if base.startswith(('CT', 'CE')):
                    return prefix + base
        return None
    for r in find_refs(text[:400]):
        if not re.search(r'(principal|amend|rescind|supersess|of)\s*$', text[max(0, r.start - 40):r.start], re.I):
            return r.key
    return None


def extract_edges(text: str, self_k: Optional[str], title: str = '') -> List[Edge]:
    """All relations stated in one chunk. `self_k` is the notification the chunk belongs to."""
    edges: List[Edge] = []
    refs = find_refs(text)
    self_date = next((r.date for r in refs if r.key == self_k and r.date), None) or header_date(text)

    def refs_after(pos: int, pool: Optional[List[Ref]] = None, src: str = text) -> List[Ref]:
        """The reference(s) right after a verb: 'amends notifications No. A, No. B and No. C'."""
        out: List[Ref] = []
        for r in (refs if pool is None else pool):
            if r.start < pos:
                continue
            if not out and r.start > pos + _WINDOW:
                break
            if out and (r.start - out[-1].end > 90 or _VERB.search(src, out[-1].end, r.start)):
                break
            out.append(r)
        return out

    def add(src, rel, dst, sd, dd, ev):
        if src and dst and src != dst:
            edges.append(Edge(src, rel, dst, sd, dd, ev))

    def forward(txt: str, pool: List[Ref], evidence_of) -> None:
        for rel, rx in _FORWARD:
            for m in rx.finditer(txt):
                for r in refs_after(m.end(), pool, txt):
                    add(self_k, rel, r.key, self_date, r.date, evidence_of(m.start(), r.end))

    if self_k:
        forward(text, refs, lambda a, b: _evidence(text, a, b))
        for rel, rx in _BACKWARD:
            for m in rx.finditer(text):
                before = [r for r in refs if m.start() - _WINDOW <= r.end <= m.start()]
                if before:
                    r = before[-1]
                    add(self_k, rel, r.key, self_date, r.date, _evidence(text, r.start, m.end()))

    # "X ... (was last / subsequently) amended by Y", "Rescinded vide Y" (subject = X, or this doc)
    for m in _PASSIVE_SELF.finditer(text):
        ys = refs_after(m.end())
        if not ys:
            continue
        rel = _PASSIVE_REL[m.group(1).lower()]
        prev = [r for r in refs if r.end <= m.start() and m.start() - r.end <= 250]
        subject = prev[-1].key if prev else None
        subject_date = prev[-1].date if prev else None
        if subject is None:
            for p in _PRINCIPAL.finditer(text, max(0, m.start() - 600), m.start()):
                pr = refs_after(p.end())
                if pr:
                    subject, subject_date = pr[0].key, pr[0].date
        if subject is None:
            subject, subject_date = self_k, self_date
        for y in ys:
            if y.key != subject:
                add(y.key, rel, subject, y.date, subject_date, _evidence(text, m.start(), y.end))

    if self_k and title:
        tl = find_refs(title)
        passive = _PASSIVE_SELF.search(title)
        if tl and passive:
            rel = _PASSIVE_REL[passive.group(1).lower()]
            for r in tl:
                add(r.key, rel, self_k, r.date, self_date, title.strip())
        elif tl:
            forward(title, tl, lambda a, b: title.strip())
    return edges


# ---------------------------------------------------------------- storage

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notif (key TEXT, doc_id TEXT, date TEXT, title TEXT,
                                  PRIMARY KEY (key, doc_id));
CREATE TABLE IF NOT EXISTS edge (src TEXT, rel TEXT, dst TEXT, src_date TEXT, dst_date TEXT,
                                 evidence TEXT, from_doc TEXT,
                                 PRIMARY KEY (src, rel, dst));
CREATE INDEX IF NOT EXISTS idx_edge_dst ON edge(dst);
CREATE INDEX IF NOT EXISTS idx_notif_doc ON notif(doc_id);
"""


def iter_manifest(path: str) -> Iterator[dict]:
    con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    q = ("SELECT c.doc_id, d.title, c.payload_json FROM chunks c "
         "LEFT JOIN docs d ON d.doc_id = c.doc_id WHERE c.is_canonical = 1")
    for doc_id, title, pj in con.execute(q):
        p = json.loads(pj or '{}')
        yield {'doc_id': doc_id, 'title': title or p.get('title') or '',
               'doc_number': p.get('doc_number') or '', 'text': p.get('text') or ''}


def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield {'doc_id': d.get('doc_id'), 'title': d.get('title') or '',
                   'doc_number': d.get('doc_number') or '', 'text': d.get('text') or ''}


def build(chunks: Iterable[dict], out: str) -> dict:
    con = sqlite3.connect(out)
    con.executescript(_SCHEMA)
    doc_key: dict = {}
    n_chunks = n_edges = 0
    for c in chunks:
        n_chunks += 1
        did, title, text = c['doc_id'], c['title'], c['text']
        if not text or not did:
            continue
        if did not in doc_key:
            doc_key[did] = self_key(c['doc_number'], title, text)
        k = doc_key[did]
        if k:
            d = next((r.date for r in find_refs(text[:1500]) if r.key == k and r.date),
                     None) or header_date(text)
            con.execute('INSERT OR IGNORE INTO notif VALUES (?,?,?,?)', (k, did, d, title))
            if d:
                con.execute('UPDATE notif SET date=? WHERE key=? AND doc_id=? AND date IS NULL',
                            (d, k, did))
        for e in extract_edges(text, k, title):
            n_edges += 1
            con.execute('INSERT OR IGNORE INTO edge VALUES (?,?,?,?,?,?,?)',
                        (e.src, e.rel, e.dst, e.src_date, e.dst_date, e.evidence, did))
    con.commit()
    stats = {'chunks': n_chunks, 'docs_with_key': sum(1 for v in doc_key.values() if v),
             'notifications': con.execute('SELECT COUNT(DISTINCT key) FROM notif').fetchone()[0],
             'edges': con.execute('SELECT COUNT(*) FROM edge').fetchone()[0]}
    con.close()
    return stats


class Graph:
    def __init__(self, path: str):
        self.con = sqlite3.connect(f'file:{path}?mode=ro', uri=True, check_same_thread=False)

    def _docs(self, key: str) -> List[str]:
        return [r[0] for r in self.con.execute('SELECT doc_id FROM notif WHERE key=?', (key,))]

    def _date(self, key: str) -> Optional[str]:
        r = self.con.execute('SELECT MIN(date) FROM notif WHERE key=?', (key,)).fetchone()
        if r and r[0]:
            return r[0]
        r = self.con.execute('SELECT MIN(dst_date) FROM edge WHERE dst=?', (key,)).fetchone()
        if r and r[0]:
            return r[0]
        r = self.con.execute('SELECT MIN(src_date) FROM edge WHERE src=?', (key,)).fetchone()
        return r[0] if r else None

    def key_for_doc(self, doc_id: str) -> Optional[str]:
        r = self.con.execute('SELECT key FROM notif WHERE doc_id=?', (doc_id,)).fetchone()
        return r[0] if r else None

    def chain(self, key: str) -> dict:
        incoming = self.con.execute(
            'SELECT src, rel, src_date, evidence, from_doc FROM edge WHERE dst=?', (key,)).fetchall()
        outgoing = self.con.execute(
            'SELECT dst, rel, dst_date, evidence, from_doc FROM edge WHERE src=?', (key,)).fetchall()

        def item(k, rel, d, ev, frm):
            return {'key': k, 'relation': rel, 'date': d or self._date(k),
                    'doc_ids': self._docs(k) or ([frm] if frm else []), 'evidence': ev}

        def when(x):  # undated: fall back to the year in the notification number
            return x['date'] or x['key'].split('/')[1].split('-')[0] + '-99'

        amended_by = sorted((item(*r) for r in incoming if r[1] == 'amends'), key=when)
        ended_by = sorted((item(*r) for r in incoming if r[1] in ('rescinds', 'supersedes')),
                          key=when)
        acts_on = [item(*r) for r in outgoing]
        if ended_by:
            e = ended_by[-1]
            status = f"{'Rescinded' if e['relation'] == 'rescinds' else 'Superseded'} by {e['key']}"
        elif amended_by:
            status = f"In force as amended; latest amendment found: {amended_by[-1]['key']}"
        else:
            status = 'No amendment or rescission found in the corpus'
        return {'key': key, 'date': self._date(key), 'doc_ids': self._docs(key),
                'status': status, 'amended_by': amended_by, 'ended_by': ended_by,
                'acts_on': acts_on,
                'known': bool(self._docs(key) or incoming or outgoing)}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    src = b.add_mutually_exclusive_group(required=True)
    src.add_argument('--manifest')
    src.add_argument('--jsonl')
    b.add_argument('--out', required=True)
    c = sub.add_parser('chain')
    c.add_argument('ref')
    c.add_argument('--db', required=True)
    a = ap.parse_args()
    if a.cmd == 'build':
        it = iter_manifest(a.manifest) if a.manifest else iter_jsonl(a.jsonl)
        print(json.dumps(build(it, a.out), indent=2))
    else:
        key = parse_key(a.ref)
        if not key:
            raise SystemExit(f'could not parse a notification reference from {a.ref!r}')
        print(json.dumps(Graph(a.db).chain(key), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
