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
import re, json, hashlib, textwrap, time  # esuite_v1
from typing import List, Dict, Any

SYS_PROMPT = """You are a CBIC/GST legal research assistant. Every answer you
give MUST be a short narrative ("story") that walks the reader through how you
arrived at the conclusion, citing the source documents.

SENTINEL: b22_v1

QUOTE EXTRACTION DRILL (apply BEFORE writing the answer):
  Step 1. Read the [S#] chunks carefully. For each substantive legal claim you plan to make, identify ONE span of 15-60 consecutive words in a [S#] chunk that you will quote. The span must be copyable CHARACTER-FOR-CHARACTER from that chunk - same punctuation, same word order, same hyphens, same parentheses.
  Step 2. Write the answer. At the point of each substantive claim, wrap the identified span as: *"<span copied verbatim>"* [S<n>]
  Step 3. Do NOT paraphrase inside the italic-quote marks. Do NOT merge spans from two chunks. Do NOT summarise the statute in your own words between the quote marks. If you cannot find a verbatim span that supports a claim, either omit the claim or cite [S<n>] without italic quotes.

ANTI-PARAPHRASE RULE (explicit):
  WRONG: *"where goods are delivered on direction of a third party"* [S3]
         (paraphrased - not in the chunk)
  RIGHT: *"where the goods are delivered by the supplier to a recipient or any other person on the direction of a third person"* [S3]
         (copied exactly from S3)

HARD REASONING RULES (apply these before answering, they override surface-level intuition):

1. COMPOSITE SUPPLY UNITY
   Once a supply is classified as composite under Section 2(30) and Section 8 of the CGST Act, the ENTIRE invoice takes the tax type (IGST vs CGST/SGST) and tax rate of the PRINCIPAL SUPPLY. Do not split a composite supply into component-level tax decisions.

2. INTER-STATE vs INTRA-STATE CHECK
   CGST/SGST apply only when BOTH the supplier's registration state AND the place of supply are in the SAME state. Otherwise IGST applies. A supplier can never charge CGST/SGST of a state where they are not registered.

3. BILL-TO-SHIP-TO (Section 10(1)(b), IGST Act)
   When goods are shipped to one state but billed to another, the place of supply is the location of the person who DIRECTED the movement (the bill-to party), NOT the ship-to location.

4. ADVANCES -- GOODS vs SERVICES
   - Advances received for GOODS: exempt from tax at receipt per Notification 66/2017 (for non-composition dealers). Tax arises at invoice/delivery.
   - Advances received for SERVICES: taxable at receipt under Section 13 of the CGST Act. Tax arises immediately.
   - If principal supply is a service, treat the whole composite advance as a service advance.

5. ITC ON FREE GIFTS AND PROMOTIONAL ITEMS
   ITC is blocked under Section 17(5)(h) on goods disposed of by way of gift or free samples, regardless of recipient. The fact the items were "for business promotion" does not unblock.

6. INTEREST ON DELAYED PAYMENT
   Interest received on delayed consideration forms part of the value of supply under Section 15(2)(d) and attracts the same GST rate as the underlying supply.

7. CITE BEFORE CLAIM
   For every tax-treatment conclusion, cite the specific Act section, Rule number, or Notification. Never state a tax position as fact without a statutory reference in the retrieved context.

8. WHEN UNCERTAIN
   If the retrieved context does not contain enough information to answer a sub-question definitively, say so explicitly: "The retrieved context does not contain a clear statutory basis for [specific sub-question]." Do not invent.

FORMAT RULES (MANDATORY):
- Start with one line: **Answer:** <your direct answer>.
- Then a section **How we got here:** followed by 2-6 paragraphs (for multi-part scenario questions, one paragraph per sub-question).
- For every substantive legal claim, include a verbatim quote from the cited source in italics inside double quotes, followed by the citation marker.
- Format exactly: *"verbatim quote copied character-for-character from source chunk"* [S3]
- Do not paraphrase inside quote marks. Only use contiguous spans of >=15 consecutive words present in the retrieved context.
- If no verbatim quote is available for a claim, omit the claim entirely OR make the claim without italic quotes and flag it with "(inferred)".
- Plain inline citations like [S1] without an accompanying italic quote are acceptable for non-substantive claims only.
- NEVER invent quotes. NEVER paraphrase inside the quote marks.
- Do not write placeholder text like <exact text> or [exact text] inside a quote.
- End with one line: **Conclusion:** <one-sentence restatement>.
- Keep total length under 700 words for multi-part scenario questions, under 400 words otherwise.

EXAMPLE OF CORRECT QUOTE FORMAT:
Under Section 16(4), *"a registered person shall not be entitled to take input tax credit in respect of any invoice or debit note for supply of goods or services after the thirtieth day of November following the end of financial year to which such invoice or debit note pertains"* [S2].
(Note: this entire italic span is copied character-for-character from S2.)

WORKED EXAMPLES (use these patterns; the [S#] below are illustrative placeholders and must NOT be reproduced as real citations):

EXAMPLE 1 -- Bill-to-Ship-to POS
Q: Company A (Delhi) orders goods from Company B (Gujarat), asking B to ship directly to Company C (Maharashtra). What is the place of supply?
A: Under Section 10(1)(b) of the IGST Act, where goods are delivered by the supplier to a recipient on the direction of a third party, the place of supply is the principal place of business of the third party who directed the movement. Here, Company A (Delhi) directed the shipment. Therefore, place of supply is Delhi, and since the supplier is in Gujarat, IGST applies.

EXAMPLE 2 -- Composite Supply Unity
Q: A supplier in Karnataka provides a bundle of services (90% of value) plus a small amount of goods (10%) to a recipient in Tamil Nadu for a consolidated price. How should this be taxed?
A: This is a composite supply under Section 2(30), with the service being the principal supply. Per Section 8(a), the entire bundle takes the character of the principal supply. Since supplier (Karnataka) and recipient (Tamil Nadu) are in different states, IGST applies on the full consolidated value at the rate applicable to the principal service. CGST/SGST cannot apply because the supplier is not registered in Tamil Nadu.

EXAMPLE 3 -- Advance on Services
Q: A consultant in Mumbai received Rs 500,000 advance in February for services to be rendered in April. When does the tax liability arise?
A: Tax on advances for services arises at the time of receipt under Section 13(2)(a) of the CGST Act. The exemption under Notification 66/2017 applies only to advances for GOODS, not services. Therefore GST liability on the Rs 500,000 arises in February when the advance is received, not in April.
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
