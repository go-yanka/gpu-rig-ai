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
"""
from __future__ import annotations
import re, json, hashlib, textwrap
from typing import List, Dict, Any

SYS_PROMPT = """You are a CBIC/GST legal research assistant. Every answer you
give MUST be a short narrative ("story") that walks the reader through how you
arrived at the conclusion, citing the source documents.

STRICT RULES:
1. Start with one line: **Answer:** <your direct answer>.
2. Then a section **How we got here:** followed by 2-5 paragraphs.
3. Every factual claim MUST be immediately followed by an exact verbatim quote
   from the SOURCES below, in double-quotes and italics, like:
   *"<exact text>"* [S<n>]
   where <n> is the 1-based source index.
4. NEVER invent quotes. NEVER paraphrase inside the quote marks. If a claim
   cannot be supported by the SOURCES, say so explicitly:
   "The sources do not directly address this."
5. End with one line: **Conclusion:** <one-sentence restatement>.
6. Keep total length under 400 words.
"""

USER_TMPL = """QUESTION: {question}

SOURCES (you may quote verbatim from these; each is tagged [S<n>]):

{sources}

Write the story now, following the rules strictly.
"""


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


QUOTE_RE = re.compile(r'\*"([^"]{15,400})"\*\s*\[S(\d+)\]')


def verify_quotes(answer: str, chunks: List[dict]) -> Dict[str, Any]:
    """For every [S<n>] quote the model emitted, check it is (near-)verbatim in
    that chunk. Returns dict with `verified`, `suspicious` lists and an
    annotated answer where bad quotes are flagged."""
    verified = []
    suspicious = []
    annotated = answer
    for m in QUOTE_RE.finditer(answer):
        quote = m.group(1).strip()
        idx = int(m.group(2)) - 1
        if idx < 0 or idx >= len(chunks):
            suspicious.append({'quote': quote, 'reason': 'source index out of range'})
            continue
        src = chunks[idx]['text']
        # exact, then normalized match
        ok = quote in src
        if not ok:
            # try whitespace-normalized comparison
            nq = re.sub(r'\s+', ' ', quote).lower()
            ns = re.sub(r'\s+', ' ', src).lower()
            ok = nq in ns
        if ok:
            verified.append({'quote': quote, 'source_index': idx+1,
                             'doc_id': chunks[idx]['doc_id'],
                             'page': chunks[idx]['page']})
        else:
            suspicious.append({'quote': quote, 'source_index': idx+1,
                               'reason': 'not found verbatim in source'})
            annotated = annotated.replace(m.group(0),
                m.group(0) + ' ⚠️[unverified]')
    return {'verified': verified, 'suspicious': suspicious,
            'annotated_answer': annotated}


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
        })
    return {
        'question': question,
        'answer_markdown': v['annotated_answer'],
        'verified_quotes': v['verified'],
        'suspicious_quotes': v['suspicious'],
        'citations': citations,
    }
