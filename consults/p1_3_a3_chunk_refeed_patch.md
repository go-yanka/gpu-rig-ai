# P1.3 A3 pass-2 chunk re-feed patch (template draft, NOT deployed)

**Note:** local mirror of `api.py` lacks two-pass code; this is a template.
Confirm actual function names on rig before applying (`run_pass2`, `build_pass2_prompt` assumed).

## Proposed pass-2 system prompt

```
You are a CBIC/GST legal-research assistant. Answer the user's
question using ONLY the VALIDATED JSON SPANS as citation sources.

Rules:
1. Every factual claim in your answer MUST be backed by a span in
   VALIDATED JSON. Cite as [doc_id:section].
2. CONTEXT CHUNKS are provided for disambiguation of pronouns,
   abbreviations, and cross-references ONLY. Do NOT quote them
   verbatim. Do NOT cite them. Do NOT introduce facts that appear
   only in CONTEXT CHUNKS and not in VALIDATED JSON.
3. If VALIDATED JSON is empty or insufficient to answer, reply
   exactly: "Insufficient validated evidence to answer. Retrieved
   context suggests <one-line hint>, but no span passed validation."
   Do NOT fall back on prior knowledge.
4. No preamble. No hedging boilerplate. Cite inline.
```

## Patch (template)

```diff
-def run_pass2(question, validated_spans):
-    prompt = build_pass2_prompt(question, validated_spans)
-    return llm_chat(prompt, system=PASS2_SYSTEM_PROMPT)
+def run_pass2(question, validated_spans, retrieved_chunks=None):
+    prompt = build_pass2_prompt(question, validated_spans, retrieved_chunks or [])
+    system = PASS2_SYSTEM_PROMPT
+    if not validated_spans:
+        system += ("\n\nGUARD: VALIDATED JSON is EMPTY. Answer conservatively. "
+                   "If CONTEXT CHUNKS do not clearly answer, refuse per Rule 3. "
+                   "Do NOT synthesize from weights.")
+    return llm_chat(prompt, system=system)

-def build_pass2_prompt(question, validated_spans):
-    spans_blob = json.dumps(validated_spans, ensure_ascii=False, indent=2)
-    return f"QUESTION:\n{question}\n\nVALIDATED JSON SPANS:\n{spans_blob}\n\nANSWER:"
+def build_pass2_prompt(question, validated_spans, retrieved_chunks):
+    spans_blob = json.dumps(validated_spans, ensure_ascii=False, indent=2)
+    chunks_blob = "\n---\n".join(
+        f"[{c.get('doc_id','?')}:{c.get('section','?')}] {c.get('text','')[:400]}"
+        for c in retrieved_chunks[:5]   # cap at 5, truncate to 400 chars each
+    ) or "(none)"
+    return (f"QUESTION:\n{question}\n\n"
+            f"VALIDATED JSON SPANS (authoritative citation source):\n{spans_blob}\n\n"
+            f"CONTEXT CHUNKS (for disambiguation only, do NOT quote or cite):\n{chunks_blob}\n\n"
+            f"ANSWER:")

-    answer = run_pass2(question, validated_spans)
+    answer = run_pass2(question, validated_spans, retrieved_chunks=chunks)
```

## Risks & mitigations

1. **Context leakage** — model quotes chunks despite instructions. → post-hoc citation checker: reject `[doc:sec]` not in `validated_spans`
2. **Token budget blowout** — cap to top-5 chunks × 400 chars
3. **Refusal regression** — A/B behind `TWO_PASS_CHUNK_REFEED=1` flag, compare vs 137-char baseline before default-on

## Deploy flag

```
# /etc/systemd/system/cbic-rag-api.service.d/two_pass.conf
Environment=TWO_PASS_ENABLED=1
Environment=TWO_PASS_CHUNK_REFEED=1
```
