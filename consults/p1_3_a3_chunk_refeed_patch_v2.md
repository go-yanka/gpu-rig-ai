# P1.3 A3 — Pass-2 Chunk Re-feed Patch (v2, REAL against live rig)

Target: `/opt/indian-legal-ai/rag/cbic_rag/api.py` (live, rig 192.168.1.107)
Backup present: `/opt/indian-legal-ai/rag/cbic_rag/api.py.bak.two_pass_v1.20260421_213026`
Sentinel in file: `two_pass_v1`

---

## 1. LIVE GREPS (line numbers)

```
17:  SENTINEL (A3 two-pass): two_pass_v1  (feature flag TWO_PASS_ENABLED, default 0)
78:  TWO_PASS_ENABLED = os.environ.get('TWO_PASS_ENABLED', '0') == '1'
390: def _synthesize_pass2(question: str, verified: List[dict]) -> str:
407: print(f"[two_pass_v1] pass2 failed, emitting fallback: {e}")
441: prose = _synthesize_pass2(question, verified)
575: # 5) LLM (single-shot legacy OR two-pass A3, gated on TWO_PASS_ENABLED)
578: if TWO_PASS_ENABLED:
```

## 2. Pass-2 function (verbatim, api.py L390–408)

```python
def _synthesize_pass2(question: str, verified: List[dict]) -> str:
    """Pass 2: LLM renders prose from VALIDATED JSON only. Chunks NOT re-fed."""
    if not verified:
        return ("**Answer:** The retrieved sources did not yield a defensible "
                "verbatim span for this question.\n\n"
                "**Conclusion:** Cannot answer from corpus.")
    facts = [{
        "sid": v["sid"],
        "sub_question": v["sub_question"],
        "verbatim_span": v["verbatim_span"],
        "conclusion": v["conclusion"],
    } for v in verified]
    user = build_synthesis_user(question, facts)
    try:
        return _call_llm(SYNTHESIS_SYS, user, temperature=0.0, max_tokens=3072)
    except Exception as e:
        ...
```

Pass-2 invocation site (api.py L441):
```python
prose = _synthesize_pass2(question, verified)
```
The enclosing `two_pass_generate(question, chunks, timings)` already has `chunks` in scope.

## 3. Current SYNTHESIS_SYS prompt (verbatim, storyformat.py L70–74)

```
You are rendering verified legal facts into a practitioner advisory. Rules:
- Copy each verbatim_span EXACTLY. Do not alter wording.
- Cite [S#] immediately after each quote, where # matches the cited_chunk_id's S-index assigned in VERIFIED_FACTS.
- Do NOT introduce any legal claim not present in VERIFIED_FACTS.
- Format: one paragraph per sub_question. Final paragraph = overall conclusion stitched from per-sub conclusions. No new conclusions.
```

`build_synthesis_user()` (storyformat.py L153–166) today feeds ONLY `{original_question, verified_facts}` JSON — no chunks.

## 4. Chunks variable shape

- Variable name: `chunks` inside `two_pass_generate` (from caller's `top` at api.py L580).
- Type: `List[dict]`.
- Keys observed in live code: `doc_id`, `title`, `subcategory`, `page`, `text`, `text_full` (optional), `char_start`, `is_table` (bool), `type`, `score`, `source_url` (optional), plus S-index assigned positionally as `S{i}` in `build_extraction_user`.

## 5. Pass-1 → Pass-2 handoff

`two_pass_generate()` (api.py L421–441) computes `verified` via `_validate_sub_answers(raw_sas, chunks)` then calls `_synthesize_pass2(question, verified)`. `chunks` is LIVE in that scope — the patch only has to forward it.

---

## 6. REAL unified diff (applies with `patch -p0` from `/`)

```diff
--- /opt/indian-legal-ai/rag/cbic_rag/api.py
+++ /opt/indian-legal-ai/rag/cbic_rag/api.py
@@ -76,6 +76,7 @@
 TWO_PASS_ENABLED = os.environ.get('TWO_PASS_ENABLED', '0') == '1'
+TWO_PASS_CHUNK_REFEED = os.environ.get('TWO_PASS_CHUNK_REFEED', '0') == '1'
@@ -387,7 +388,7 @@
     return verified, suspicious
 
 
-def _synthesize_pass2(question: str, verified: List[dict]) -> str:
+def _synthesize_pass2(question: str, verified: List[dict], chunks: Optional[List[dict]] = None) -> str:
     """Pass 2: LLM renders prose from VALIDATED JSON only. Chunks NOT re-fed."""
     if not verified:
         return ("**Answer:** The retrieved sources did not yield a defensible "
@@ -399,9 +400,26 @@
         "verbatim_span": v["verbatim_span"],
         "conclusion": v["conclusion"],
     } for v in verified]
     user = build_synthesis_user(question, facts)
+    # P1.3 A3: optional chunk re-feed for disambiguation (NOT for quoting).
+    # Guarded by TWO_PASS_CHUNK_REFEED env var (default 0).
+    if TWO_PASS_CHUNK_REFEED and chunks:
+        ctx_blocks = []
+        for i, c in enumerate(chunks[:5], start=1):
+            t = (c.get('text_full') or c.get('text') or '').strip()
+            if len(t) > 400:
+                t = t[:400] + ' …'
+            ctx_blocks.append(f"[S{i}] {t}")
+        if ctx_blocks:
+            user = (
+                user
+                + "\n\nCONTEXT CHUNKS (for disambiguation only, do NOT quote or cite):\n\n"
+                + "\n\n".join(ctx_blocks)
+            )
     try:
         return _call_llm(SYNTHESIS_SYS, user, temperature=0.0, max_tokens=3072)
     except Exception as e:
@@ -438,7 +456,7 @@
     timings["two_pass_n_suspicious"] = len(suspicious)
 
     t_p2 = time.perf_counter()
-    prose = _synthesize_pass2(question, verified)
+    prose = _synthesize_pass2(question, verified, chunks)
     timings["two_pass_synth_ms"] = _ms_since(t_p2)
```

Notes:
- Empty-spans guard is already present at L392–395 (`if not verified: return ...`). Patch does not weaken it; refeed block runs only when `verified` is non-empty (we reach past that early return).
- The `ctx_blocks` list is itself guarded (`if ctx_blocks`), so an all-empty-text chunks list falls back to legacy behavior.
- Flag defaults to 0 → zero behavior change until explicitly enabled.

## 7. Deploy steps (rig, as root/service user)

```bash
# (a) Fresh backup
sudo cp /opt/indian-legal-ai/rag/cbic_rag/api.py \
        /opt/indian-legal-ai/rag/cbic_rag/api.py.bak.chunk_refeed.$(date +%Y%m%d_%H%M%S)

# (b) Stage patch file on rig
scp D:/_gpu_rig_ai/consults/p1_3_a3_chunk_refeed_v2.patch rig:/tmp/
# (extract the ```diff block above into that .patch file first)

# (c) Dry-run then apply
sudo patch --dry-run -p0 -d / < /tmp/p1_3_a3_chunk_refeed_v2.patch
sudo patch          -p0 -d / < /tmp/p1_3_a3_chunk_refeed_v2.patch

# (d) Byte-compile sanity
sudo python3 -m py_compile /opt/indian-legal-ai/rag/cbic_rag/api.py

# (e) Enable flag in the systemd unit (or env-file) — keep TWO_PASS_ENABLED=1 too
sudo systemctl edit cbic-rag  # or wherever the unit lives; add:
#   [Service]
#   Environment=TWO_PASS_ENABLED=1
#   Environment=TWO_PASS_CHUNK_REFEED=1

# (f) Restart + smoke test
sudo systemctl restart cbic-rag
curl -s http://127.0.0.1:9500/healthz | jq .
curl -s -X POST http://127.0.0.1:9500/query \
  -H 'content-type: application/json' \
  -d '{"question":"What is the time limit for filing GSTR-9?","k":8}' | jq '.timings'
# Expect: two_pass_enabled=true, two_pass_synth_ms present; answer unchanged in shape.

# Rollback:
sudo cp /opt/indian-legal-ai/rag/cbic_rag/api.py.bak.chunk_refeed.<ts> \
        /opt/indian-legal-ai/rag/cbic_rag/api.py
sudo systemctl restart cbic-rag
# or simply: unset TWO_PASS_CHUNK_REFEED and restart (code path is gated).
```
