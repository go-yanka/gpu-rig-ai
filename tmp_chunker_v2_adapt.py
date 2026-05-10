"""
chunker_v2.py — CBIC two-pass structure-aware chunker.

Pass 1: LLM (Claude CLI primary, qwen3-14b fallback) classifies each doc into
        a `chunking_plan` JSON blob.
Pass 2: rule-driven splits (R1 tables atomic, R2 critical units whole,
        R3 unusable-cut validator, R4 hierarchy splits, R5 overlap,
        R6 size targets, R7 payload audit).

Single-file, no heavy deps. External calls:
  - Claude CLI at /home/user/.local/bin/claude (rig)
  - qwen3-14b at http://192.168.1.107:9082/v1/completions
  - hashlib / unicodedata / json / re / subprocess / requests

Reference: reingest_spec/SPEC.md §2 Phase 2, chunking_strategy_cbic_v2.md.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# --- Config (override via env or constructor) -------------------------------

CLAUDE_CLI = "/home/user/.local/bin/claude"
QWEN3_URL = "http://192.168.1.107:9082/v1/completions"
PROMPT_PATH = Path(__file__).parent / "chunking_plan_prompt.md"

TARGET = 3500
CAP = 5500
CEILING = 8000
FLOOR = 500
OVERLAP_MID = 700
OVERLAP_BOUNDARY = 0

# --- R3 unusable-cut connector tokens ---------------------------------------

ENGLISH_CONNECTORS = (
    "Provided", "Except", "However", "For the purposes of this",
    "Explanation", "Illustration", "Notwithstanding",
    "Subject to", "Where", "In case",
)

HINDI_CONNECTORS = (
    "बशर्ते कि", "परंतु", "किंतु", "तथापि",
    "स्पष्टीकरण", "व्याख्या", "परिभाषा",
    "उदाहरणार्थ", "दृष्टांत", "अपवाद",
    "इसके बावजूद", "परंतुक",
)

# Narrow: only orphaning verbs that indicate a mid-clause cut. Broad regex
# (any lowercase word) was too aggressive — matched normal prose like "text. text.".
ORPHAN_VERBS = (
    "means", "includes", "applies", "refers", "denotes",
    "shall", "may", "must", "applies to", "refers to",
)
LOWERCASE_VERB_RE = re.compile(
    r"^(?:" + "|".join(re.escape(v) for v in ORPHAN_VERBS) + r")\b"
)


# --- Data classes -----------------------------------------------------------


@dataclass
class ChunkingPlan:
    doc_type: str = "mixed"
    structure: str = "flat_paragraphs"
    primary_splitter: str = "paragraph"
    critical_units: list = field(default_factory=list)
    hard_boundaries: list = field(default_factory=list)
    table_regions: list = field(default_factory=list)
    has_amendments: bool = False
    hierarchy_depth: int = 1
    language: str = "en"
    confidence: float = 0.0
    notes: str = ""

    @classmethod
    def from_json(cls, raw: str) -> "ChunkingPlan":
        # 2026-04-24 ROBUSTNESS: qwen3-14b sometimes emits non-strict JSON —
        # trailing commas, unquoted keys, single quotes, JS-style comments.
        # Old behaviour: json.JSONDecodeError → phase2 silent drop on the doc.
        # cbic-allied-act-dtls:1000221 was the 2026-04-24 trigger case ("Expecting
        # ',' delimiter: line 7 column 150"). The fix is tolerant parsing with
        # an escalating ladder of recovery attempts, not swallow+skip.
        obj = _tolerant_json_loads(raw)
        # defensive coercion — LLMs sometimes emit strings for bools/ints
        if isinstance(obj.get("has_amendments"), str):
            obj["has_amendments"] = obj["has_amendments"].lower() == "true"
        if isinstance(obj.get("hierarchy_depth"), str):
            obj["hierarchy_depth"] = int(obj["hierarchy_depth"])
        return cls(**{k: obj.get(k, getattr(cls, k, None)) for k in cls.__annotations__})


def _tolerant_json_loads(raw: str) -> dict:
    """Parse LLM-emitted JSON with graceful recovery. Ladder:
      L0  strict json.loads
      L1  strip // and /* */ comments + trailing commas
      L2  balance braces/brackets + strip trailing non-JSON
      L3  extract the largest balanced {...} substring
    Raises the L0 JSONDecodeError if the whole ladder fails, so the caller's
    RuntimeError carries the most informative diagnostic, not a generic 'parse
    failed' message.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e0:
        original_err = e0
    # L0.5: strip markdown fences. Qwen3 sometimes wraps JSON in ```json ... ```
    # AND prepends conversational text like 'First character must be `{`'.
    # That prior text contains spurious { } which broke L2 balance logic.
    # Added 2026-04-24 after GST50 classify failed 10/50 docs on this pattern.
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', raw, re.DOTALL)
    if m:
        fenced = m.group(1).strip()
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            # fall through to L1 ladder but use fenced content as source
            raw = fenced
    else:
        # L0.6: UNCLOSED opening fence (qwen3 hit max_tokens mid-JSON and never
        # emitted the closing ```). Without this, L1/L2 operate on raw which
        # still has the instruction-echo prefix (stray { }) and L2 balance fails.
        # Strip from first ```json (or ```) onwards. Added 2026-04-24 after
        # GST50 docs 1000998 and 1001015 failed this exact pattern.
        m2 = re.search(r'```(?:json)?\s*\n?', raw)
        if m2:
            raw = raw[m2.end():]
    # L1: comments + trailing commas
    s = re.sub(r"//[^\n]*", "", raw)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # L2: balance brackets from first { to matching close
    start = s.find("{")
    if start != -1:
        depth = 0
        end = -1
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if esc:
                esc = False
                continue
            if ch == "\\" and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end != -1:
            candidate = s[start:end]
            candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
    # L3: give up with the ORIGINAL error so the caller's log names the real
    # line/column in the raw LLM output, not a mutated derivative.
    raise original_err


@dataclass
class Chunk:
    chunk_id: str            # sha256 of canonical text
    doc_id: str
    sha256: str              # same as chunk_id (redundant, for compat)
    source: str
    category: str
    subcategory: str
    lang: str
    text: str
    embed_text: str          # parent_hierarchy breadcrumb + text
    section_ref: str
    parent_hierarchy_text: str
    chunk_type: str          # narrative | table | form_field_block | footnote
    is_table: bool
    table_part: Optional[str]  # "1/3" etc or None
    page_range: tuple
    effective_date: Optional[str]
    text_source: str         # born | ocr
    hindi_twin_chunk_ids: list = field(default_factory=list)
    topic_tags: list = field(default_factory=list)
    also_appears_in: list = field(default_factory=list)
    dup_of_chunk_id: Optional[str] = None
    # R7 audit fields
    chunking_plan_used: bool = True
    chunking_rule_triggered: list = field(default_factory=list)
    # D8 amendment/version metadata (G2 fix)
    notification_id: Optional[str] = None
    as_of_date: Optional[str] = None
    superseded_by: Optional[str] = None

    def to_payload(self) -> dict:
        d = asdict(self)
        d["page_range"] = list(self.page_range)
        # G1 fix: emit keys required by cbic_rag.ingest.upsert_chunks (line 138).
        # page_range carries (char_start, char_end); `page` is derived or 0 if unknown.
        d["char_start"] = int(self.page_range[0])
        d["char_end"] = int(self.page_range[1])
        d["page"] = int(getattr(self, "_page", 0))
        return d


# --- Pass 1: LLM classification --------------------------------------------


def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _build_pass1_input(meta: dict, head: str, tail: str, toc: str, page_map: list) -> str:
    """Assemble the concrete user message appended to the system prompt."""
    return (
        f"\n\nDOCUMENT METADATA:\n"
        f"title: {meta.get('title','')}\n"
        f"source: {meta.get('source','')}\n"
        f"category: {meta.get('category','')}\n"
        f"subcategory: {meta.get('subcategory','')}\n"
        f"pages: {meta.get('pages',0)}\n"
        f"is_ocr: {meta.get('is_ocr', False)}\n"
        f"language_hint: {meta.get('lang', 'en')}\n\n"
        f"TABLE OF CONTENTS:\n{toc or 'none'}\n\n"
        f"PAGE MAP (page:char_count): {json.dumps(page_map)[:500]}\n\n"
        f"DOCUMENT HEAD (first 2000 chars):\n{head[:2000]}\n\n"
        f"DOCUMENT TAIL (last 1500 chars):\n{tail[-1500:]}\n\n"
        "Respond with STRICT JSON only."
    )


def classify_doc_claude(meta: dict, head: str, tail: str, toc: str = "", page_map: list = None,
                        timeout: int = 30) -> ChunkingPlan:
    """Pass-1 via Claude CLI. Falls through to qwen3 on failure."""
    page_map = page_map or []
    prompt = _load_prompt() + _build_pass1_input(meta, head, tail, toc, page_map)
    try:
        proc = subprocess.run(
            [CLAUDE_CLI, "-p", prompt, "--output-format", "json"],
            capture_output=True, text=True, timeout=timeout,
        )
        raw = proc.stdout.strip()
        # Claude CLI --output-format json wraps in {"result": "..."}
        try:
            wrap = json.loads(raw)
            inner = wrap.get("result", raw) if isinstance(wrap, dict) else raw
        except json.JSONDecodeError:
            inner = raw
        # strip markdown fences if model slipped
        inner = re.sub(r"^```(?:json)?\s*|\s*```$", "", inner.strip(), flags=re.MULTILINE)
        return ChunkingPlan.from_json(inner)
    except Exception as e:
        return classify_doc_qwen(meta, head, tail, toc, page_map, _claude_err=str(e))


def classify_doc_qwen(meta: dict, head: str, tail: str, toc: str = "", page_map: list = None,
                      _claude_err: str = "", timeout: int = 60, _retries: int = 3) -> ChunkingPlan:
    """Pass-1 via qwen3-14b fallback (V2b proven 30/30 parse).

    2026-04-24: added retry loop + stricter prompting on parse failure. Was
    single-shot and silently returned a default ChunkingPlan when qwen emitted
    malformed JSON — which is how cbic-allied-act-dtls:1000221 slipped into
    phase2_done=0 state. Now: retry up to _retries times with escalating
    strictness ('respond with valid JSON ONLY, no prose'), and RAISE on
    exhaustion so the caller's phase2 guard records a real error.
    """
    import requests
    page_map = page_map or []
    base_body = _load_prompt() + _build_pass1_input(meta, head, tail, toc, page_map)
    last_err: Exception | None = None
    last_raw: str = ""
    for attempt in range(_retries):
        if attempt == 0:
            body = "/no_think " + base_body
        else:
            # Reprompt more explicitly on retry
            body = ("/no_think " + base_body
                    + "\n\nYOUR PRIOR RESPONSE WAS NOT VALID JSON. "
                    "Respond with ONLY a JSON object. "
                    "No prose, no markdown fences, no trailing commas.")
        # 2026-04-24: max_tokens bumped 200→1024. Root cause of
        # cbic-allied-act-dtls:1000221 silent failure was truncation at 200
        # tokens, not JSON malformation — the response was a valid-start
        # object cut off mid-field, so no closing brace, so regex extraction
        # failed and fallback path returned default plan. 1024 is comfortably
        # above any observed ChunkingPlan size (median ~350 tokens).
        resp = requests.post(QWEN3_URL, json={
            "prompt": body, "max_tokens": 1024, "temperature": 0.0, "stop": ["\n\n\n"],
        }, timeout=timeout)
        text = resp.json()["choices"][0]["text"].strip()
        last_raw = text
        # Route directly through tolerant parser (it handles prose preamble
        # via balanced-brace extraction at L2). The old \{.*\} regex required
        # a closing brace in the raw text, which hid truncation errors as
        # "no JSON" rather than "truncated JSON".
        try:
            return ChunkingPlan.from_json(text)
        except Exception as e:
            last_err = e
            continue
    # Exhausted retries. RAISE with full diagnostic so phase2 guard records it.
    raise RuntimeError(
        f"qwen3 classifier failed after {_retries} attempts for doc_id={meta.get('doc_id')}: "
        f"last_err={type(last_err).__name__}: {last_err}; "
        f"claude_err={_claude_err}; last_raw={last_raw[:300]!r}"
    )


# --- Canonical form / hashing (shared with dedupe_chunks.py) ----------------


def canonicalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def sha256_of(text: str) -> str:
    return hashlib.sha256(canonicalize(text).encode("utf-8")).hexdigest()


# --- R3 unusable-cut validator ----------------------------------------------


def is_unusable_cut(text: str) -> bool:
    """Return True if first non-whitespace token is an orphaned connector."""
    stripped = text.lstrip()
    if not stripped:
        return True
    for tok in ENGLISH_CONNECTORS:
        if stripped.startswith(tok):
            return True
    for tok in HINDI_CONNECTORS:
        if stripped.startswith(tok):
            return True
    if LOWERCASE_VERB_RE.match(stripped) and not stripped.startswith(("i ", "a ")):
        return True
    return False


# --- R2 critical-unit detectors ---------------------------------------------

PROVISO_RE = re.compile(r"(?m)^(\s*)(Provided (that|further|also)\b)")
EXPLANATION_RE = re.compile(r"(?m)^(\s*)(Explanation(\s*\d*)?[\.\-—:])")
DEFINITION_RE = re.compile(r"(?m)^(\s*)(\(\d+\)|\d+\.)\s+\"[^\"]+\"\s+means\b")


def find_critical_unit_spans(text: str) -> list[tuple[int, int, str]]:
    """Return [(start, end, kind)] of spans that must stay whole."""
    spans = []
    for m in PROVISO_RE.finditer(text):
        end = _end_of_sentence_group(text, m.start())
        spans.append((m.start(), end, "proviso"))
    for m in EXPLANATION_RE.finditer(text):
        end = _end_of_sentence_group(text, m.start())
        spans.append((m.start(), end, "explanation"))
    for m in DEFINITION_RE.finditer(text):
        end = _end_of_sentence_group(text, m.start())
        spans.append((m.start(), end, "definition"))
    # merge overlaps, sort by start
    spans.sort()
    merged = []
    for s, e, k in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e), merged[-1][2])
        else:
            merged.append((s, e, k))
    return merged


def _end_of_sentence_group(text: str, start: int) -> int:
    """End of a proviso/explanation group — look for next blank line or section-start."""
    # Find next double newline OR next top-level section header
    m_dbl = re.search(r"\n\s*\n", text[start:])
    m_sec = re.search(r"(?m)^(Section|Rule|Chapter|Part|SECTION|RULE|CHAPTER|PART)\s+[0-9IVXLC]+", text[start:])
    end = len(text)
    if m_dbl:
        end = min(end, start + m_dbl.start())
    if m_sec and m_sec.start() > 0:
        end = min(end, start + m_sec.start())
    return end


# --- R4 hierarchy splitters -------------------------------------------------

HIERARCHY_PATTERNS = [
    (r"(?m)^(CHAPTER|Chapter|PART|Part|SCHEDULE|Schedule)\s+[0-9IVXLC]+", "chapter"),
    (r"(?m)^(SECTION|Section|RULE|Rule|Clause)\s+\d+[A-Z]?\b", "section"),
    (r"(?m)^\(\d+\)\s+", "subsection"),
    (r"(?m)^\([a-z]+\)\s+", "subclause"),
    (r"(?m)^\d+\.\s+", "paragraph"),
]

SENTENCE_RE = re.compile(r"(?<=[.?!])\s+(?=[A-Z\"'(])")


def hierarchy_split_points(text: str, primary: str = "section") -> list[int]:
    """Return sorted unique split offsets preferring primary splitter."""
    points = set([0])
    # primary splitter first (higher priority)
    for pat, name in HIERARCHY_PATTERNS:
        if name == primary or name.startswith(primary):
            for m in re.finditer(pat, text):
                points.add(m.start())
    # then chapter/section as boundaries
    for pat, _ in HIERARCHY_PATTERNS[:2]:
        for m in re.finditer(pat, text):
            points.add(m.start())
    return sorted(points)


# --- Main Pass-2 chunker ----------------------------------------------------


def chunk_document(full_text: str, plan: ChunkingPlan, meta: dict,
                   page_offsets: list[int] = None,
                   is_table_region: callable = None) -> list[Chunk]:
    """
    Pass 2: rule-driven chunking.

    page_offsets: [char_offset_start_of_page_i for i=0..n_pages-1] (used for page_range).
    is_table_region: optional callable(char_offset) -> (bool, region_meta) — if None,
                     derived from plan.table_regions + page_offsets.
    """
    page_offsets = page_offsets or [0]
    doc_id = meta.get("doc_id") or meta.get("source", "unknown")

    # -- Step 1: extract table regions as atomic chunks (R1) -----------------
    table_chunks, non_table_spans = _extract_table_chunks(full_text, plan, meta, page_offsets)

    # -- Step 2: chunk non-table spans -- ADAPTIVE DISPATCH (2026-04-24) ------
    # Pass 1 emits plan.primary_splitter; Pass 2 now dispatches on it.
    # Defect fix: previously _chunk_prose_span ignored primary_splitter and
    # plan.hard_boundaries, sliced acts mid-section, dropped section_ref on
    # 99/109 chunks of cbic-act-msts:1000006 → dense G1 recall = 0.51.
    use_section_bounded = (plan.primary_splitter or "").lower() in ("section", "rule")
    prose_chunks = []
    for span_start, span_end in non_table_spans:
        span = full_text[span_start:span_end]
        if use_section_bounded:
            prose_chunks.extend(
                _section_bounded_split(span, span_start, plan, meta, page_offsets)
            )
        else:
            prose_chunks.extend(
                _chunk_prose_span(span, span_start, plan, meta, page_offsets)
            )

    # -- Step 3: merge, sort by char offset, assign chunk_ids ----------------
    all_chunks = sorted(prose_chunks + table_chunks, key=lambda c: c.page_range[0])

    # -- R6 floor merge: final chunk < FLOOR merges into previous ------------
    all_chunks = _merge_floor(all_chunks)

    return all_chunks


# --- Internal: table extraction (R1) ----------------------------------------


def _extract_table_chunks(text: str, plan: ChunkingPlan, meta: dict,
                          page_offsets: list[int]) -> tuple[list[Chunk], list[tuple[int, int]]]:
    """Pull table regions out as atomic chunks. Return (table_chunks, remaining_spans)."""
    table_chunks = []
    taken = []  # [(start, end)]

    # Adaptive filter (added 2026-04-24): for hierarchical-section docs (Acts,
    # Rules), ignore low-confidence table_regions. Pass 1 over-flags whole
    # pages as tables on Acts (confidence=0.0); honoring those regions
    # consumes the entire document and starves the section splitter, which
    # is exactly the cbic-act-msts:1000006 failure mode.
    _is_section_doc = (plan.primary_splitter or "").lower() in ("section", "rule")
    for tr in plan.table_regions:
        tr_conf = float(tr.get("confidence", 1.0) or 0.0)
        # For Acts/Rules (section/rule splitters), skip ALL table_regions:
        # Pass 1 was run against pdfminer page_offsets=[0,total] and tagged
        # table pages that collapse to the whole document. The section-
        # bounded splitter handles inline tables (schedules, repeals) as
        # subsections of their parent Section — no separate carve-out needed.
        if _is_section_doc:
            continue
        ps, pe = tr.get("page_start", 1) - 1, tr.get("page_end", 1) - 1
        if ps < 0 or ps >= len(page_offsets):
            continue
        start = page_offsets[ps]
        end = page_offsets[pe + 1] if pe + 1 < len(page_offsets) else len(text)
        region = text[start:end]
        if len(region) <= CEILING:
            table_chunks.append(_mk_table_chunk(
                region, start, end, meta, plan, part=None, reason=tr.get("reason", "")
            ))
        else:
            # R1 oversize: sub-table detection, else row-boundary split
            sub_tables = _detect_subtables(region)
            if sub_tables and all(len(s) <= CEILING for s in sub_tables):
                for i, sub in enumerate(sub_tables):
                    sub_start = start + region.index(sub)
                    table_chunks.append(_mk_table_chunk(
                        sub, sub_start, sub_start + len(sub), meta, plan,
                        part=f"{i+1}/{len(sub_tables)}", reason=tr.get("reason", "") + " [subtable]"
                    ))
            else:
                parts = _row_boundary_split(region)
                header_row = _extract_header_row(region)
                last_parent = _extract_parent_row(region)
                for i, part in enumerate(parts):
                    prefixed = _prepend_context(part, header_row, last_parent)
                    psub = start + region.index(part) if part in region else start
                    table_chunks.append(_mk_table_chunk(
                        prefixed, psub, psub + len(part), meta, plan,
                        part=f"{i+1}/{len(parts)}", reason=tr.get("reason", "") + " [rowsplit]"
                    ))
        taken.append((start, end))

    # remaining spans = complement of taken
    taken.sort()
    remaining = []
    cur = 0
    for s, e in taken:
        if cur < s:
            remaining.append((cur, s))
        cur = max(cur, e)
    if cur < len(text):
        remaining.append((cur, len(text)))
    if not remaining:
        remaining = [(0, len(text))] if not taken else []
    return table_chunks, remaining


def _detect_subtables(region: str) -> list[str]:
    """Heuristic: repeated header patterns (e.g. 'Chapter 84', 'Heading XXXX') signal sub-tables."""
    markers = list(re.finditer(r"(?m)^\s*(Chapter|Heading|Sub-heading|CHAPTER|HEADING)\s+\d+", region))
    if len(markers) < 2:
        return []
    starts = [m.start() for m in markers] + [len(region)]
    return [region[starts[i]:starts[i+1]].strip() for i in range(len(markers))]


def _row_boundary_split(region: str) -> list[str]:
    """Split on newline boundaries as a proxy for row boundaries in OCR'd tables."""
    lines = region.split("\n")
    chunks, buf = [], []
    size = 0
    for ln in lines:
        if size + len(ln) + 1 > CAP and buf:
            chunks.append("\n".join(buf))
            buf, size = [], 0
        buf.append(ln)
        size += len(ln) + 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def _extract_header_row(region: str) -> str:
    """Heuristic first non-empty line as header."""
    for ln in region.split("\n"):
        if ln.strip():
            return ln.strip()
    return ""


def _extract_parent_row(region: str) -> str:
    """Last 'Chapter X' or 'Heading X' marker."""
    m = list(re.finditer(r"(?m)^\s*(Chapter|Heading|CHAPTER|HEADING)\s+\d+[^\n]*", region))
    return m[-1].group(0).strip() if m else ""


def _prepend_context(part: str, header: str, parent: str) -> str:
    pieces = []
    if header and header not in part:
        pieces.append(header)
    if parent and parent not in part:
        pieces.append(parent)
    pieces.append(part)
    return "\n".join(pieces)


def _mk_table_chunk(text: str, start: int, end: int, meta: dict, plan: ChunkingPlan,
                    part: Optional[str], reason: str) -> Chunk:
    parent = meta.get("parent_hierarchy_text", "")
    return Chunk(
        chunk_id=sha256_of(text),
        doc_id=meta.get("doc_id", meta.get("source", "unknown")),
        sha256=sha256_of(text),
        source=meta.get("source", ""),
        category=meta.get("category", ""),
        subcategory=meta.get("subcategory", ""),
        lang=plan.language if plan.language in ("en", "hi") else meta.get("lang", "en"),
        text=text,
        embed_text=(parent + "\n\n" + text) if parent else text,
        section_ref=reason or "table",
        parent_hierarchy_text=parent,
        chunk_type="table",
        is_table=True,
        table_part=part,
        page_range=(start, end),
        effective_date=meta.get("effective_date"),
        text_source=meta.get("text_source", "born"),
        chunking_rule_triggered=["R1:table_atomic" + ("" if part is None else f":part_{part}")],
        notification_id=meta.get("notification_id"),
        as_of_date=meta.get("as_of_date") or meta.get("effective_date"),
        superseded_by=meta.get("superseded_by"),
    )


# --- Internal: prose chunking (R2/R3/R4/R5/R6) ------------------------------


def _chunk_prose_span(span: str, span_offset: int, plan: ChunkingPlan, meta: dict,
                      page_offsets: list[int]) -> list[Chunk]:
    if not span.strip():
        return []

    crit_spans = find_critical_unit_spans(span)
    hier_pts = hierarchy_split_points(span, primary=plan.primary_splitter)

    chunks = []
    cur = 0
    N = len(span)

    while cur < N:
        target_end = min(cur + TARGET, N)
        cap_end = min(cur + CAP, N)

        # R2: if target cut lands inside a critical unit, extend to cap
        adj_end = target_end
        for cs, ce, kind in crit_spans:
            if cs < adj_end < ce:
                adj_end = min(ce, cap_end)
                break

        # Prefer hierarchy boundary at or before adj_end (R4)
        best_h = None
        for p in hier_pts:
            if cur + FLOOR <= p <= adj_end:
                best_h = p
        if best_h is not None:
            adj_end = best_h

        # Sentence boundary fallback (R4 step 5)
        if adj_end == cur + TARGET and cur + TARGET < N:
            tail_region = span[cur:adj_end]
            m = list(SENTENCE_RE.finditer(tail_region))
            if m:
                adj_end = cur + m[-1].end()

        piece = span[cur:adj_end].strip()

        # R3 unusable-cut retry: pull back 200 chars up to 3×
        retries, rule_triggered = 0, []
        while piece and is_unusable_cut(piece) and retries < 3 and adj_end - 200 > cur + FLOOR:
            adj_end -= 200
            piece = span[cur:adj_end].strip()
            retries += 1
            rule_triggered.append(f"R3:unusable_retry_{retries}")

        if piece and is_unusable_cut(piece):
            # merge into previous if possible (R3 final)
            if chunks:
                prev = chunks[-1]
                merged_text = prev.text + "\n\n" + piece
                if len(merged_text) <= CEILING:
                    prev.text = merged_text
                    prev.embed_text = (prev.parent_hierarchy_text + "\n\n" + merged_text) \
                                       if prev.parent_hierarchy_text else merged_text
                    prev.page_range = (prev.page_range[0], span_offset + adj_end)
                    prev.chunking_rule_triggered.append("R3:merged_into_prev")
                    cur = adj_end
                    continue
            rule_triggered.append("R3:kept_despite_warn")

        if len(piece) < FLOOR and chunks:
            # R6 floor: merge tiny tail into previous
            prev = chunks[-1]
            prev.text = prev.text + "\n\n" + piece
            prev.embed_text = (prev.parent_hierarchy_text + "\n\n" + prev.text) \
                               if prev.parent_hierarchy_text else prev.text
            prev.page_range = (prev.page_range[0], span_offset + adj_end)
            prev.chunking_rule_triggered.append("R6:floor_merge")
            cur = adj_end
            continue

        # R5 overlap decision: section-boundary split → 0; mid-section → OVERLAP_MID
        is_boundary = adj_end in hier_pts
        rule_triggered.append("R5:zero_overlap_section" if is_boundary else "R5:mid_700")

        chunks.append(_mk_prose_chunk(piece, span_offset + cur, span_offset + adj_end,
                                      plan, meta, rule_triggered))

        # advance with overlap
        overlap = OVERLAP_BOUNDARY if is_boundary else OVERLAP_MID
        next_cur = adj_end - overlap
        if next_cur <= cur:
            next_cur = adj_end  # safety — never go backwards
        cur = next_cur

    return chunks


# --- ADAPTIVE: section-bounded splitter (added 2026-04-24) -----------------
# Used when Pass 1 plan says primary_splitter in {"section","rule"}.
# Hard-enforces section boundaries (never crosses a Section/Rule header),
# carries section header into chunk text + section_ref payload, and
# sub-splits sections only when they exceed CEILING.

# Section header detector — tolerates "Section 17.", "Rule 36A.", "1." numbered
# variants common in CBIC corpus. Returns (offset, header_line, header_label).
_SECTION_START_RE = re.compile(
    # Allow leading whitespace and footnote/amendment markers like '*', '**',
    # numeric superscripts '1', '2' (PyMuPDF flattens them inline) before the
    # keyword. Without this, body section headers in CBIC Acts (which are
    # heavily annotated with amendment footnotes) are missed and only ToC
    # entries match — producing only ~6 unique section_refs on the CGST Act.
    r"(?m)^[ \t]*[\*\d]{0,4}[ \t]*(Section|SECTION|Rule|RULE|Chapter|CHAPTER)\s+"
    r"([0-9]+[A-Z]?)\b[\.\:\-\s]*([^\n]*)"
)


def _find_section_starts(text: str, plan: ChunkingPlan) -> list[tuple[int, str, str]]:
    """Return [(offset, full_header_line, label_short)] sorted by offset.

    Sources combined:
      - Built-in _SECTION_START_RE (Section/Rule/Chapter + number)
      - plan.hard_boundaries regexes (per-doc, from Pass 1)
    """
    points: dict[int, tuple[str, str]] = {}
    for m in _SECTION_START_RE.finditer(text):
        kind, num, rest = m.group(1), m.group(2), (m.group(3) or "").strip()
        # canonical capitalization for retrieval (Section 17 / Rule 36)
        ck = kind.capitalize() if kind.isupper() else kind
        label = f"{ck} {num}".strip()
        # full header line (may include the title after the number)
        header_line = m.group(0).strip()
        points[m.start()] = (header_line, label)
    # also honor per-doc hard_boundaries from Pass 1
    for hb in (plan.hard_boundaries or []):
        pat = hb.get("regex_or_marker") if isinstance(hb, dict) else None
        if not pat:
            continue
        try:
            for m in re.finditer(pat, text, flags=re.MULTILINE):
                if m.start() not in points:
                    line = m.group(0).strip()
                    points[m.start()] = (line, line[:60])
        except re.error:
            continue
    # ToC dedup: a Table of Contents lists every section header densely at
    # the top, then the actual body re-mentions each header. If the same
    # label appears multiple times, keep the LAST occurrence (the body).
    ordered = sorted(points.items())
    last_by_label: dict[str, int] = {}
    for off, (_h, l) in ordered:
        last_by_label[l] = off
    deduped = [(off, h, l) for off, (h, l) in ordered if last_by_label.get(l) == off]
    return deduped


def _section_bounded_split(span: str, span_offset: int, plan: ChunkingPlan,
                           meta: dict, page_offsets: list[int]) -> list[Chunk]:
    """Split span on section/rule boundaries — never cross a header.

    Each section becomes one chunk if <= CEILING. Larger sections are
    sub-split on subsection markers '^\\(\\d+\\)' or on sentence boundaries
    near TARGET, but never cross the next Section/Rule header.

    Every chunk:
      - has section_ref populated with the active section label
      - has section header prepended to embed_text so dense vector includes it
      - has chunking_rule_triggered = ['ADAPT:section_bounded']
    """
    if not span.strip():
        return []
    starts = _find_section_starts(span, plan)
    chunks: list[Chunk] = []
    crit_spans = find_critical_unit_spans(span)

    if not starts:
        # No section markers found → fall back to existing prose splitter
        return _chunk_prose_span(span, span_offset, plan, meta, page_offsets)

    # Add a sentinel "end of span" boundary
    boundaries = [s[0] for s in starts] + [len(span)]
    headers = [(s[1], s[2]) for s in starts]

    # If first boundary > 0, the prefix (preamble) becomes its own chunk
    if starts[0][0] > 0:
        prefix_text = span[:starts[0][0]].strip()
        if len(prefix_text) >= FLOOR:
            chunks.append(_mk_section_chunk(
                body_text=prefix_text,
                start=span_offset, end=span_offset + starts[0][0],
                section_label="", section_header="",
                plan=plan, meta=meta,
                rules=["ADAPT:section_bounded:preamble"],
            ))

    for i, (sec_off, header_line, label) in enumerate(starts):
        sec_end = boundaries[i + 1]
        body = span[sec_off:sec_end]
        body_stripped = body.strip()
        if len(body_stripped) < FLOOR:
            continue  # skip empty/tiny section stub

        # -- single-chunk section (fits in CEILING) --
        if len(body) <= CEILING:
            chunks.append(_mk_section_chunk(
                body_text=body_stripped,
                start=span_offset + sec_off, end=span_offset + sec_end,
                section_label=label, section_header=header_line,
                plan=plan, meta=meta,
                rules=["ADAPT:section_bounded:single"],
            ))
            continue

        # -- oversize section → sub-split, never cross next section header --
        sub_pts = sorted({sec_off} | {
            sec_off + m.start()
            for m in re.finditer(r"(?m)^\(\d+[A-Z]?\)\s+", body)
        } | {
            sec_off + m.start()
            for m in re.finditer(r"(?m)^\([a-z]+\)\s+", body)
        } | {sec_end})
        # walk pairs, packing into [TARGET..CAP] windows aligned to sub_pts
        sub_chunks_for_section: list[tuple[int, int]] = []
        cur_sub = sec_off
        while cur_sub < sec_end:
            target = min(cur_sub + TARGET, sec_end)
            cap = min(cur_sub + CAP, sec_end)
            # find latest sub_pt within [cur_sub+FLOOR, target] preferred,
            # else within [cur_sub+FLOOR, cap]
            best = None
            for p in sub_pts:
                if cur_sub + FLOOR <= p <= target:
                    best = p
            if best is None:
                for p in sub_pts:
                    if cur_sub + FLOOR <= p <= cap:
                        best = p
            if best is None:
                # sentence boundary fallback within [target..cap]
                tail = span[cur_sub:cap]
                m_iter = list(SENTENCE_RE.finditer(tail))
                if m_iter:
                    best = cur_sub + m_iter[-1].end()
                else:
                    best = cap
            sub_chunks_for_section.append((cur_sub, best))
            cur_sub = best
        # emit as chunks, all carrying same section_ref + header
        for j, (a, b) in enumerate(sub_chunks_for_section):
            piece = span[a:b].strip()
            if len(piece) < FLOOR and chunks and chunks[-1].section_ref == label:
                # floor-merge into previous sub-chunk in this section
                prev = chunks[-1]
                prev.text = prev.text + "\n\n" + piece
                prev.embed_text = prev.embed_text + "\n\n" + piece
                prev.page_range = (prev.page_range[0], span_offset + b)
                prev.chunking_rule_triggered.append("ADAPT:section_bounded:floor_merge")
                continue
            chunks.append(_mk_section_chunk(
                body_text=piece,
                start=span_offset + a, end=span_offset + b,
                section_label=label, section_header=header_line,
                plan=plan, meta=meta,
                rules=[f"ADAPT:section_bounded:part_{j+1}_of_{len(sub_chunks_for_section)}"],
            ))

    return chunks


def _mk_section_chunk(body_text: str, start: int, end: int,
                      section_label: str, section_header: str,
                      plan: ChunkingPlan, meta: dict, rules: list) -> Chunk:
    """Make a chunk with section_ref + header prepended to embed_text."""
    parent = meta.get("parent_hierarchy_text", "")
    # Prepend the section header into the body so the chunk text itself
    # carries it (visible to readers AND embedded for dense retrieval).
    if section_header and not body_text.lstrip().startswith(section_header):
        text = section_header + "\n" + body_text
    else:
        text = body_text
    embed_text = (parent + "\n\n" + text) if parent else text
    return Chunk(
        chunk_id=sha256_of(text),
        doc_id=meta.get("doc_id", meta.get("source", "unknown")),
        sha256=sha256_of(text),
        source=meta.get("source", ""),
        category=meta.get("category", ""),
        subcategory=meta.get("subcategory", ""),
        lang=plan.language if plan.language in ("en", "hi") else meta.get("lang", "en"),
        text=text,
        embed_text=embed_text,
        section_ref=section_label,
        parent_hierarchy_text=parent,
        chunk_type="narrative",
        is_table=False,
        table_part=None,
        page_range=(start, end),
        effective_date=meta.get("effective_date"),
        text_source=meta.get("text_source", "born"),
        chunking_rule_triggered=rules,
        notification_id=meta.get("notification_id"),
        as_of_date=meta.get("as_of_date") or meta.get("effective_date"),
        superseded_by=meta.get("superseded_by"),
    )


def _mk_prose_chunk(text: str, start: int, end: int, plan: ChunkingPlan, meta: dict,
                    rules: list) -> Chunk:
    parent = meta.get("parent_hierarchy_text", "")
    section_ref = _detect_section_ref(text)
    return Chunk(
        chunk_id=sha256_of(text),
        doc_id=meta.get("doc_id", meta.get("source", "unknown")),
        sha256=sha256_of(text),
        source=meta.get("source", ""),
        category=meta.get("category", ""),
        subcategory=meta.get("subcategory", ""),
        lang=plan.language if plan.language in ("en", "hi") else meta.get("lang", "en"),
        text=text,
        embed_text=(parent + "\n\n" + text) if parent else text,
        section_ref=section_ref,
        parent_hierarchy_text=parent,
        chunk_type="narrative",
        is_table=False,
        table_part=None,
        page_range=(start, end),
        effective_date=meta.get("effective_date"),
        text_source=meta.get("text_source", "born"),
        chunking_rule_triggered=rules,
        notification_id=meta.get("notification_id"),
        as_of_date=meta.get("as_of_date") or meta.get("effective_date"),
        superseded_by=meta.get("superseded_by"),
    )


def _detect_section_ref(text: str) -> str:
    m = re.search(r"(?m)^(Section|Rule|Chapter|Clause|Article)\s+[0-9IVXLCA-Z\.\-]+", text)
    return m.group(0).strip() if m else ""


# --- R6 floor-merge pass ----------------------------------------------------


def _merge_floor(chunks: list[Chunk]) -> list[Chunk]:
    if len(chunks) < 2:
        return chunks
    if len(chunks[-1].text) < FLOOR:
        prev, tail = chunks[-2], chunks[-1]
        if prev.is_table:
            return chunks  # never merge a non-table tail into a table chunk
        merged_text = prev.text + "\n\n" + tail.text
        if len(merged_text) <= CEILING:
            prev.text = merged_text
            prev.embed_text = (prev.parent_hierarchy_text + "\n\n" + merged_text) \
                               if prev.parent_hierarchy_text else merged_text
            prev.page_range = (prev.page_range[0], tail.page_range[1])
            prev.chunking_rule_triggered.append("R6:final_floor_merge")
            return chunks[:-1]
    return chunks


# --- Public convenience -----------------------------------------------------


def classify_and_chunk(full_text: str, meta: dict,
                       head: str = "", tail: str = "", toc: str = "",
                       page_offsets: list[int] = None,
                       page_map: list = None,
                       use_qwen_first: bool = False) -> tuple[ChunkingPlan, list[Chunk]]:
    """End-to-end: Pass 1 LLM + Pass 2 rules."""
    head = head or full_text[:2000]
    tail = tail or full_text[-1500:]
    if use_qwen_first:
        plan = classify_doc_qwen(meta, head, tail, toc, page_map)
    else:
        plan = classify_doc_claude(meta, head, tail, toc, page_map)
    chunks = chunk_document(full_text, plan, meta, page_offsets)
    return plan, chunks


if __name__ == "__main__":
    # Smoke check on a tiny synthetic doc
    sample = (
        "CHAPTER I\n\nSection 1. Short title.\n\n"
        "(1) This Act may be called the Central Goods and Services Tax Act, 2017.\n\n"
        "(2) It extends to the whole of India.\n\n"
        "Provided that nothing in this Act shall apply to Jammu and Kashmir.\n\n"
        "Explanation.— For the purposes of this section, 'India' means the territory of India.\n\n"
        "Section 2. Definitions.\n\n(1) \"actionable claim\" means a claim to any debt.\n"
    )
    plan = ChunkingPlan(
        doc_type="act", structure="hierarchical_sections",
        primary_splitter="section", critical_units=["section", "proviso", "explanation"],
        language="en", confidence=0.95, hierarchy_depth=3,
    )
    chunks = chunk_document(sample, plan, {"source": "smoke.pdf", "doc_id": "smoke"})
    print(f"[smoke] plan={plan}")
    print(f"[smoke] produced {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        print(f"  chunk[{i}] len={len(c.text)} rules={c.chunking_rule_triggered}")
        print(f"    head: {c.text[:80]!r}")
