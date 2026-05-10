"""Story-with-quotes answer builder.

The system prompt forces the LLM to produce a narrative that:
  1. States the thesis answer in one line.
  2. Walks through the supporting evidence, each step with an **exact verbatim
     quote** pulled from a retrieved chunk, plus a bracketed citation.
  3. Ends with a one-line conclusion.

We then post-process: any quote the LLM emitted that doesn't appear verbatim in
the chunks is flagged (to prevent hallucinated citations). Citations are
normalized to [Doc: <title>, Cat: <subcategory>, p.<page>] with a hidden
`chunk_id` so the frontend can deep-link.

SENTINEL (A3 two-pass): two_pass_v1
"""
from __future__ import annotations
import re, json, hashlib, textwrap, time  # esuite_v1
from typing import List, Dict, Any

SYS_PROMPT = """You are a CBIC/GST legal research assistant. Every answer you
give MUST be a short narrative ("story") that walks the reader through how you
arrived at the conclusion, citing the source documents.

SENTINEL: b11b12a8_v1

STRICT RULES:
1. Start with one line: **Answer:** <your direct answer>.
2. Then a section **How we got here:** followed by 2-5 paragraphs.
3. When a factual claim is supported by a verbatim phrase visible in the
   SOURCES, quote that phrase inline in italics+double-quotes followed by its
   source tag, for example:
   *"he is in possession of a tax invoice or debit note"* [S2]
   The text inside the quotes MUST be copied verbatim from one of the SOURCES
   (any contiguous span of >=5 words is acceptable). If no such verbatim phrase
   exists for a given claim, DO NOT write a quote for that claim - just state
   the claim with its [S<n>] citation and move on.
4. NEVER invent quotes. NEVER paraphrase inside the quote marks.
5. Quotes must contain ONLY text that literally appears in a SOURCE block.
   Do not write any kind of angle-bracket or square-bracket placeholder text
   (such as template markers for missing content) inside a quote. If you
   cannot find a matching verbatim phrase, omit the quote line entirely and
   just cite with [S<n>].
6. If a claim cannot be supported by the SOURCES, say so explicitly:
   "The sources do not directly address this."
7. End with one line: **Conclusion:** <one-sentence restatement>.
8. Keep total length under 400 words.
"""

USER_TMPL = """QUESTION: {question}

SOURCES (you may quote verbatim from these; each is tagged [S<n>]):

{sources}

Write the story now, following the rules strictly.
"""


# =====================================================================
# A3 two-pass prompts (sentinel: two_pass_v1)
# =====================================================================

EXTRACTION_SYS = """You are a CBIC legal-text extractor. For each sub-question, find the single best supporting chunk from the CONTEXT and copy its most relevant verbatim span EXACTLY (character-for-character). Do not paraphrase, do not abbreviate, do not add punctuation.

Return strict JSON matching the schema. Span length 80-450 chars. Span must contain at least one complete clause (ending in . ? ! or ;). Do not copy fragments like "Provided that" alone.

If no chunk contains a defensible verbatim span for a sub-question, omit that sub-question from sub_answers."""


SYNTHESIS_SYS = """You are rendering verified legal facts into a practitioner advisory. Rules:
- Copy each verbatim_span EXACTLY. Do not alter wording.
- Cite [S#] immediately after each quote, where # matches the cited_chunk_id's S-index assigned in VERIFIED_FACTS.
- Do NOT introduce any legal claim not present in VERIFIED_FACTS.
- Format: one paragraph per sub_question. Final paragraph = overall conclusion stitched from per-sub conclusions. No new conclusions."""


DECOMP_VERIFY_SYS = """Does this list of sub-questions fully cover the original question without adding facets not in the original or dropping facets from the original? Answer exactly YES or NO followed by one-sentence reason."""


DECOMP_SYS = """You are a query planner for a CBIC legal RAG. Break the user's
question into 1-6 atomic sub-questions, each answerable from a single statutory
provision, rule, or circular. Return strict JSON:
{"sub_questions": ["...", "..."]}
Do not invent facets not implied by the original question. Do not drop facets.
If the question is already atomic, return a single-element list."""


# JSON schema for Pass 1 extraction output.
EXTRACTION_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "sub_answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sub_question":    {"type": "string"},
                    "cited_chunk_id":  {"type": "string"},
                    "verbatim_span":   {"type": "string", "minLength": 80, "maxLength": 450},
                    "conclusion":      {"type": "string"},
                },
                "required": ["sub_question", "cited_chunk_id", "verbatim_span", "conclusion"],
            },
        },
    },
    "required": ["sub_answers"],
}


DECOMP_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "sub_questions": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
    },
    "required": ["sub_questions"],
}


def build_extraction_user(sub_questions: List[str], chunks: List[dict]) -> str:
    """Build the user message for Pass 1 (extraction).

    Each chunk is tagged [S<n>] AND carries a stable `cited_chunk_id` the LLM
    must echo back in its JSON.  The `cited_chunk_id` is the S-index as a
    string (e.g. "S3") so Pass 2 rendering can map back trivially.
    """
    src_blocks = []
    for i, c in enumerate(chunks, start=1):
        title = (c.get('title') or '').strip()
        sub = (c.get('subcategory') or '').strip()
        page = c.get('page', '?')
        text = c.get('text', '').strip()
        if len(text) > 1800:
            text = text[:1800] + ' …'
        sid = f"S{i}"
        src_blocks.append(
            f"[{sid}] cited_chunk_id={sid}  |  Doc: {title}  |  Cat: {sub}  |  p.{page}\n"
            f"---\n{text}\n---"
        )
    sources = "\n\n".join(src_blocks)
    sq_json = json.dumps(sub_questions, ensure_ascii=False)
    return (
        f"SUB_QUESTIONS (JSON array): {sq_json}\n\n"
        f"CONTEXT (each chunk tagged with a cited_chunk_id you must echo back):\n\n"
        f"{sources}\n\n"
        f"Emit ONLY the JSON object matching the schema."
    )


def build_synthesis_user(original_question: str, verified_facts: List[dict]) -> str:
    """Build the user message for Pass 2 (synthesis).

    `verified_facts` is the list of validated sub_answer dicts, each already
    carrying `sub_question`, `verbatim_span`, `conclusion`, and the S-index
    to cite (as `sid`).  Chunks are NOT re-fed here, by design.
    """
    blob = json.dumps({
        "original_question": original_question,
        "verified_facts": verified_facts,
    }, ensure_ascii=False, indent=2)
    return (
        "VERIFIED_FACTS (JSON; copy verbatim_span values EXACTLY, cite using sid):\n\n"
        f"{blob}\n\n"
        "Write the practitioner advisory now following the rules strictly."
    )


def build_decomp_user(question: str) -> str:
    return f"ORIGINAL QUESTION:\n{question}\n\nReturn JSON with sub_questions."


# =====================================================================
# Legacy single-shot helpers (kept intact — fallback path when
# TWO_PASS_ENABLED=0).
# =====================================================================

def build_prompt(question: str, chunks: List[dict]) -> tuple:
    """Return (system, user) prompt strings + the source index used."""
    src_blocks = []
    for i, c in enumerate(chunks, start=1):
        title = (c.get('title') or '').strip()
        sub = (c.get('subcategory') or '').strip()
        page = c.get('page', '?')
        text = c.get('text', '').strip()
        # keep a reasonable cap per source so context fits
        if len(text) > 1800:
            text = text[:1800] + ' …'
        src_blocks.append(
            f"[S{i}] Doc: {title}  |  Cat: {sub}  |  p.{page}\n"
            f"---\n{text}\n---"
        )
    sources = "\n\n".join(src_blocks)
    return SYS_PROMPT, USER_TMPL.format(question=question, sources=sources)


QUOTE_RE = re.compile(r'\*?["\u201c\u201d]([^"\u201c\u201d]{15,400})["\u201c\u201d]\*?\s*\[S(\d+)[^\]]*\]\*?')


_WS_RE = re.compile(r"[\s\u00a0\u200b\u2028\u2029]+")
_DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212]")
_SMART_Q_RE = re.compile(r"[\u2018\u2019\u201a\u201b\u201c\u201d\u201e\u201f\u2032\u2033]")
_SOFTHYPHEN_RE = re.compile(r"[\u00ad]")
_PUNCT_EDGE_RE = re.compile(r"\s+([,;:.)\]])")


def _canon(s: str) -> str:
    """Canonical form for quote/source comparison.

    Normalises away common PDF/LLM artefacts: smart quotes -> straight,
    em/en-dashes -> hyphen, NBSP / zero-width / line-sep -> space,
    soft-hyphen removed, section sign expanded, Rs./INR/rupee sign unified,
    lowercase, collapsed whitespace, and whitespace before closing punctuation
    stripped.
    """
    if not s:
        return ""
    s = s.replace("\u00a0", " ").replace("\u200b", " ")
    s = s.replace("\u2028", " ").replace("\u2029", " ")
    s = _SOFTHYPHEN_RE.sub("", s)
    s = _SMART_Q_RE.sub("'", s)
    s = _DASH_RE.sub("-", s)
    s = s.replace("\u00a7", "section ")
    s = s.replace("\u20b9", "rs.").replace("Rs.", "rs.").replace("INR", "rs.")
    s = s.lower()
    s = _WS_RE.sub(" ", s)
    s = _PUNCT_EDGE_RE.sub(r"\1", s)
    return s.strip()


def _ngrams(s: str, n: int = 5):
    toks = s.split()
    if len(toks) < n:
        return [tuple(toks)] if toks else []
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def _fuzzy_contains(quote: str, src: str, threshold: float = 0.85) -> bool:
    """Approximate containment via 5-gram coverage of quote inside src.
    Tolerates small LLM-inserted prefixes / suffixes (e.g. clause labels
    like '(a) ') and minor mid-quote edits."""
    qg = _ngrams(quote, 5)
    if not qg:
        qg = _ngrams(quote, 3)
        if not qg:
            return False
    sg = set(_ngrams(src, 5)) | set(_ngrams(src, 3))
    hits = sum(1 for g in qg if g in sg)
    return (hits / len(qg)) >= threshold


def verify_quotes(answer: str, chunks: List[dict]) -> Dict[str, Any]:
    """b6_fuzzy_verify: For every [S<n>] quote the model emitted, check whether
    it is a (near-)verbatim substring of that chunk. Uses progressive fallbacks:
    (1) exact, (2) canonicalized substring, (3) label-prefix strip, (4) 6-gram
    coverage >= 0.80. Returns dict with `verified`, `suspicious` lists and an
    annotated answer (suspicious quotes tagged inline)."""
    import unicodedata as _ud
    verified = []
    suspicious = []
    annotated = answer

    _WS = re.compile(r"\s+")
    _SMART_Q = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})
    _DASHES = str.maketrans({"\u2013": "-", "\u2014": "-", "\u2212": "-"})
    _SOFT = str.maketrans({"\u00ad": "", "\u200b": "", "\ufeff": "", "\u2028": " ", "\u2029": " ", "\u00a0": " "})
    _LABEL_PREFIX = re.compile(r"^\s*(?:\([a-z0-9]{1,4}\)|[0-9]{1,3}\.[0-9]{0,3}|[a-z]\.)\s*", re.I)

    def _canon(s: str) -> str:
        s = _ud.normalize("NFKC", s)
        s = s.translate(_SOFT).translate(_SMART_Q).translate(_DASHES)
        s = s.lower()
        s = _WS.sub(" ", s).strip()
        return s

    def _ngrams(s: str, n: int = 6):
        s = re.sub(r"[^a-z0-9]+", " ", s).strip()
        s = _WS.sub(" ", s)
        if len(s) < n:
            return set()
        return {s[i:i + n] for i in range(len(s) - n + 1)}

    def _fuzzy_ok(quote: str, src_text: str) -> tuple[bool, str]:
        cq = _canon(quote)
        cs = _canon(src_text)
        if not cq:
            return False, "empty"
        if cq in cs:
            return True, "canon_exact"
        # Strip leading label-glue, retry
        cq2 = _LABEL_PREFIX.sub("", cq).strip()
        if cq2 and cq2 != cq and cq2 in cs:
            return True, "prefix_strip"
        # 6-gram coverage
        qg = _ngrams(cq)
        if not qg:
            return False, "short"
        sg = _ngrams(cs)
        coverage = len(qg & sg) / len(qg)
        if coverage >= 0.80:
            return True, f"ngram={coverage:.2f}"
        # Try stripped quote ngrams too
        if cq2:
            qg2 = _ngrams(cq2)
            if qg2:
                c2 = len(qg2 & sg) / len(qg2)
                if c2 >= 0.80:
                    return True, f"ngram_strip={c2:.2f}"
        return False, f"ngram={coverage:.2f}"

    for m in QUOTE_RE.finditer(answer):
        quote = m.group(1).strip()
        idx = int(m.group(2)) - 1
        if idx < 0 or idx >= len(chunks):
            suspicious.append({"quote": quote, "reason": "source index out of range"})
            continue
        src_text = chunks[idx].get("text", "")
        ok, how = _fuzzy_ok(quote, src_text)
        if ok:
            verified.append({
                "quote": quote,
                "source_index": idx + 1,
                "doc_id": chunks[idx].get("doc_id"),
                "page": chunks[idx].get("page", chunks[idx].get("page_number", 0)),
                "match": how,
            })
        else:
            suspicious.append({
                "quote": quote,
                "source_index": idx + 1,
                "reason": f"not found verbatim in source ({how})",
            })
            annotated = annotated.replace(m.group(0), m.group(0) + " \u26a0\ufe0f[unverified]")

    return {
        "verified": verified,
        "suspicious": suspicious,
        "annotated_answer": annotated,
    }

def build_response_payload(question: str, chunks: List[dict], answer: str) -> dict:
    """Final response for the frontend: story + structured citations."""
    v = verify_quotes(answer, chunks)
    citations = []
    for i, c in enumerate(chunks, start=1):
        cid = hashlib.md5(
            (c.get('doc_id','') + str(c.get('page','')) + str(c.get('char_start',''))).encode()
        ).hexdigest()[:12]
        citations.append({
            'index': i,
            'chunk_id': cid,
            'doc_id': c.get('doc_id'),
            'title': c.get('title'),
            'category': c.get('category'),
            'subcategory': c.get('subcategory'),
            'page': c.get('page'),
            'char_start': c.get('char_start'),
            'char_end': c.get('char_end'),
            'file_path': c.get('file_path'),
            'download_source': c.get('download_source'),
            'source_url': c.get('source_url'),
            'score': c.get('rerank_score', c.get('score')),
            'excerpt': c.get('text','')[:300],
            # esuite_v1 — enriched fields
            'text_full': c.get('text', ''),
            'date': c.get('date') or c.get('doc_date') or c.get('issued_date'),
            'number': c.get('number') or c.get('circular_no') or c.get('notification_no'),
        })
    query_id = hashlib.blake2b(f"{question}|{time.time()}".encode(), digest_size=8).hexdigest()  # esuite_v1
    return {
        'query_id': query_id,
        'question': question,
        'answer_markdown': v['annotated_answer'],
        'verified_quotes': v['verified'],
        'suspicious_quotes': v['suspicious'],
        'citations': citations,
    }


def build_citations(chunks: List[dict]) -> List[dict]:
    """Return the same `citations` list that build_response_payload builds —
    exposed so the two-pass path can reuse it without going through the
    single-shot verify_quotes pipeline."""
    citations = []
    for i, c in enumerate(chunks, start=1):
        cid = hashlib.md5(
            (c.get('doc_id','') + str(c.get('page','')) + str(c.get('char_start',''))).encode()
        ).hexdigest()[:12]
        citations.append({
            'index': i,
            'chunk_id': cid,
            'doc_id': c.get('doc_id'),
            'title': c.get('title'),
            'category': c.get('category'),
            'subcategory': c.get('subcategory'),
            'page': c.get('page'),
            'char_start': c.get('char_start'),
            'char_end': c.get('char_end'),
            'file_path': c.get('file_path'),
            'download_source': c.get('download_source'),
            'source_url': c.get('source_url'),
            'score': c.get('rerank_score', c.get('score')),
            'excerpt': c.get('text','')[:300],
            'text_full': c.get('text', ''),
            'date': c.get('date') or c.get('doc_date') or c.get('issued_date'),
            'number': c.get('number') or c.get('circular_no') or c.get('notification_no'),
        })
    return citations
