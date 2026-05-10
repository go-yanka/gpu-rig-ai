"""Patch script: inject Defect C fix into chunker_v2.py on rig.

Adds:
  1. _DEFAULT_PLANS_BY_PREFIX  — hardcoded ChunkingPlan templates per known
     doc_id prefix, used to bypass qwen3 entirely (for forms, where the
     classifier is overkill and prone to regex-repetition trap)
  2. _detect_repetition() — checks last_raw for >=4x same 30-char substring,
     signals classifier entered the loop
  3. classify_doc_qwen() — on parse failure OR repetition detection, fall
     back to prefix-based default plan if available; only RAISE when no
     fallback exists.
"""
import re
from pathlib import Path

P = Path("/opt/indian-legal-ai/reingest_spec/chunker_v2.py")
src = P.read_text()

INSERT_AFTER = '''def classify_doc_claude(meta: dict, head: str, tail: str, toc: str = "", page_map: list = None,'''

# Sanity: anchor must exist
assert INSERT_AFTER in src, "anchor not found"

NEW_BLOCK = '''# ---------------------------------------------------------------------------
# 2026-04-25 Defect C — qwen3 classifier repetition trap on form/instruction docs
#
# qwen3-14b enters a regex-repetition loop on certain doc types (forms have
# rigid field layouts the model tries to capture as a regex). It emits the
# same `\\\\s*\\\\d+\\\\.\\\\s*\\\\w+\\\\s*\\\\d*\\\\.` fragment hundreds of
# times until max_tokens, never closes the JSON, no comma to truncate at.
# L4 brace recovery cannot fix; retries reproduce the same trap.
#
# Two-layer fix:
#   (a) Prefix-based bypass: known-rigid doc types use a hardcoded plan and
#       skip qwen3 entirely. Cheaper, deterministic, no regex hazard.
#   (b) Repetition detector + fallback: any other doc that hits the trap
#       falls back to a generic prefix-default plan instead of raising.
#
# Set 2 evidence: 4/50 docs failed (3 forms + 1 instruction) due to this
# trap, dropping raw recall from 0.9736 (adjusted) to 0.835. At full-rig
# scale this would lose ~1190/14925 docs.
# ---------------------------------------------------------------------------

_DEFAULT_PLANS_BY_PREFIX = {
    "cbic-form-msts": dict(
        doc_type="form",
        structure="form_fields",
        primary_splitter="heading",
        critical_units=["form_field_block", "footnote"],
        hard_boundaries=[],
        table_regions=[],
        has_amendments=False,
        hierarchy_depth=1,
        notes="default plan: forms (Defect C bypass)",
    ),
    "cbic-instruction-msts": dict(
        doc_type="instruction",
        structure="flat_paragraphs",
        primary_splitter="paragraph",
        critical_units=["proviso", "explanation", "definition"],
        hard_boundaries=[],
        table_regions=[],
        has_amendments=False,
        hierarchy_depth=1,
        notes="default plan: instruction (Defect C fallback)",
    ),
    "cbic-attachment-dtls": dict(
        doc_type="attachment",
        structure="flat_paragraphs",
        primary_splitter="paragraph",
        critical_units=[],
        hard_boundaries=[],
        table_regions=[],
        has_amendments=False,
        hierarchy_depth=1,
        notes="default plan: attachment (Defect C fallback)",
    ),
}

# Prefixes that ALWAYS bypass qwen3 (deterministic, no classifier needed)
_BYPASS_QWEN_PREFIXES = ("cbic-form-msts",)


def _doc_id_prefix(meta: dict) -> str:
    did = str(meta.get("doc_id") or "")
    return did.split(":", 1)[0] if ":" in did else ""


def _default_plan_for(meta: dict) -> "ChunkingPlan | None":
    prefix = _doc_id_prefix(meta)
    tmpl = _DEFAULT_PLANS_BY_PREFIX.get(prefix)
    if tmpl is None:
        return None
    return ChunkingPlan(**tmpl)


def _detect_repetition(text: str, win: int = 30, threshold: int = 4) -> bool:
    """True if the same `win`-char substring appears `threshold`+ times.

    Catches qwen3's regex-loop output where the same 30-char fragment
    (e.g. '\\\\s*\\\\d*\\\\.\\\\s*\\\\w+\\\\s*\\\\d*\\\\.') repeats hundreds of
    times. Sampling middle 4000 chars to bound runtime.
    """
    if not text or len(text) < win * threshold:
        return False
    sample = text[len(text)//2 - 2000: len(text)//2 + 2000] if len(text) > 4000 else text
    counts: dict[str, int] = {}
    step = 5
    for i in range(0, len(sample) - win, step):
        sub = sample[i:i + win]
        c = counts.get(sub, 0) + 1
        counts[sub] = c
        if c >= threshold:
            return True
    return False


def classify_doc_claude(meta: dict, head: str, tail: str, toc: str = "", page_map: list = None,'''

# Replace the classify_doc_claude header line + everything before it stays untouched;
# inject NEW_BLOCK to replace just the anchor line.
new_src = src.replace(INSERT_AFTER, NEW_BLOCK, 1)
assert new_src != src, "no replacement happened"

# Now patch classify_doc_qwen to use prefix bypass + repetition fallback.
# Find the function start and insert bypass logic before the loop, and
# wrap the final `raise` with fallback logic.

OLD_QWEN_HEADER = '''def classify_doc_qwen(meta: dict, head: str, tail: str, toc: str = "", page_map: list = None,
                      _claude_err: str = "", timeout: int = 180, _retries: int = 3) -> ChunkingPlan:
    """Pass-1 via qwen3-14b fallback (V2b proven 30/30 parse).

    2026-04-24: added retry loop + stricter prompting on parse failure. Was
    single-shot and silently returned a default ChunkingPlan when qwen emitted
    malformed JSON — which is how cbic-allied-act-dtls:1000221 slipped into
    phase2_done=0 state. Now: retry up to _retries times with escalating
    strictness ('respond with valid JSON ONLY, no prose'), and RAISE on
    exhaustion so the caller's phase2 guard records a real error.
    """
    import requests
    page_map = page_map or []'''

NEW_QWEN_HEADER = '''def classify_doc_qwen(meta: dict, head: str, tail: str, toc: str = "", page_map: list = None,
                      _claude_err: str = "", timeout: int = 180, _retries: int = 3) -> ChunkingPlan:
    """Pass-1 via qwen3-14b fallback (V2b proven 30/30 parse).

    2026-04-24: added retry loop + stricter prompting on parse failure.
    2026-04-25 Defect C: prefix-based bypass for known-rigid doc types
    (forms) + repetition-trap detection with default-plan fallback.
    """
    import requests
    # Defect C(a): bypass qwen3 entirely for prefixes with deterministic
    # structure (forms). Avoids the regex-repetition trap and saves an
    # LLM call. Logs to stdout so phase2 summary shows the bypass.
    if _doc_id_prefix(meta) in _BYPASS_QWEN_PREFIXES:
        plan = _default_plan_for(meta)
        if plan is not None:
            print(f"[classify] BYPASS qwen3 for {meta.get('doc_id')} -> default plan ({plan.doc_type})", flush=True)
            return plan
    page_map = page_map or []'''

assert OLD_QWEN_HEADER in new_src, "qwen header anchor not found"
new_src = new_src.replace(OLD_QWEN_HEADER, NEW_QWEN_HEADER, 1)

# Patch the inner retry loop: add repetition detection between API call and parse
OLD_INNER = '''        text = resp.json()["choices"][0]["text"].strip()
        last_raw = text
        # Route directly through tolerant parser (it handles prose preamble
        # via balanced-brace extraction at L2). The old \\{.*\\} regex required
        # a closing brace in the raw text, which hid truncation errors as
        # "no JSON" rather than "truncated JSON".
        try:
            return ChunkingPlan.from_json(text)
        except Exception as e:
            last_err = e
            continue'''

NEW_INNER = '''        text = resp.json()["choices"][0]["text"].strip()
        last_raw = text
        # Defect C(b): early-detect regex-repetition trap. If detected,
        # don't waste retries — break out and use prefix-default plan.
        if _detect_repetition(text):
            last_err = RuntimeError("qwen3 regex-repetition trap detected in last_raw")
            print(f"[classify] REPETITION-TRAP detected for {meta.get('doc_id')} attempt={attempt}", flush=True)
            break
        try:
            return ChunkingPlan.from_json(text)
        except Exception as e:
            last_err = e
            continue'''

assert OLD_INNER in new_src, "inner retry anchor not found"
new_src = new_src.replace(OLD_INNER, NEW_INNER, 1)

# Patch the final RAISE: try fallback plan first
OLD_RAISE = '''    # Exhausted retries. RAISE with full diagnostic so phase2 guard records it.
    raise RuntimeError(
        f"qwen3 classifier failed after {_retries} attempts for doc_id={meta.get('doc_id')}: "
        f"last_err={type(last_err).__name__}: {last_err}; "
        f"claude_err={_claude_err}; last_raw_len={len(last_raw)}; last_raw={last_raw[:2000]!r}"
    )'''

NEW_RAISE = '''    # Defect C(b): exhausted retries — try prefix-based default plan as
    # a last resort before raising. For known prefixes (instruction,
    # attachment) this rescues the doc with a sensible-but-generic plan.
    fallback = _default_plan_for(meta)
    if fallback is not None:
        print(f"[classify] FALLBACK to default plan for {meta.get('doc_id')} ({fallback.doc_type}) after {_retries} attempts", flush=True)
        return fallback
    # No fallback available — raise with full diagnostic.
    raise RuntimeError(
        f"qwen3 classifier failed after {_retries} attempts for doc_id={meta.get('doc_id')}: "
        f"last_err={type(last_err).__name__}: {last_err}; "
        f"claude_err={_claude_err}; last_raw_len={len(last_raw)}; last_raw={last_raw[:2000]!r}"
    )'''

assert OLD_RAISE in new_src, "raise anchor not found"
new_src = new_src.replace(OLD_RAISE, NEW_RAISE, 1)

P.write_text(new_src)
print(f"PATCHED {P} — {len(new_src) - len(src)} bytes added")
