# P1.2 Validator simplification patch (draft, NOT deployed)

**File:** `/opt/indian-legal-ai/rag/cbic_rag/validator.py` + one banner string in `api.py`
**Function:** `validate_span(span, chunk_text, is_table=False)`
**Change:** drop 6-gram Jaccard rung; loosen cosine 0.92 → 0.85; keep length+clause / NFKC substring / is_table bypass untouched.

## Patch

```diff
--- a/opt/indian-legal-ai/rag/cbic_rag/validator.py
+++ b/opt/indian-legal-ai/rag/cbic_rag/validator.py
@@ -6,8 +6,7 @@
 Validation ladder (fail-fast):
   1. length in [80, 450] + at least one clause terminator (. ? ! ;)
   2. canonical substring (NFKC + whitespace collapse + lowercase)
   3. is_table bypass: exact canonical substring only, no fuzzy fallback
-  4. 6-gram character-Jaccard >= 0.80
-  5. BGE-M3 cosine fallback >= 0.92 (lazy import from embedder)
+  4. BGE-M3 cosine fallback >= 0.85 (lazy import from embedder)
@@ -54,18 +53,7 @@ def validate_span(span, chunk_text, is_table=False):
     if is_table:
         return (False, "table_exact_miss")

-    # 5. 6-gram Jaccard
-    g1 = _grams(cs, 6); g2 = _grams(cc, 6)
-    if not g1: return (False, "too_short_for_grams")
-    union = g1 | g2
-    if not union: return (False, "empty_union")
-    jac = len(g1 & g2) / len(union)
-    if jac >= 0.80:
-        return (True, f"jaccard_{jac:.2f}")
-
-    # 6. BGE-M3 cosine fallback (lazy)
+    # 4. BGE-M3 cosine fallback (lazy) — loosened 0.92 -> 0.85
     try:
         ...
-        if cos >= 0.92:
+        if cos >= 0.85:
             return (True, f"cos_{cos:.2f}")
-        return (False, f"cos_{cos:.2f}_jac_{jac:.2f}")
+        return (False, f"cos_{cos:.2f}")
     except Exception as e:
-        return (False, f"embed_err_{type(e).__name__}_jac_{jac:.2f}")
+        return (False, f"embed_err_{type(e).__name__}")

--- a/opt/indian-legal-ai/rag/cbic_rag/api.py
+++ b/opt/indian-legal-ai/rag/cbic_rag/api.py
@@ -1143,7 +1143,7 @@
-        'verifier': 'fuzzy (canon + label-strip + 6-gram 0.80)',
+        'verifier': '3-rung (len+clause -> NFKC substr/is_table -> BGE cos >= 0.85)',
```

## Deploy

```bash
ssh rig
cd /opt/indian-legal-ai/rag/cbic_rag
cp validator.py validator.py.bak.ladder_3rung.$(date +%s)
# apply patch
sudo systemctl restart cbic-rag-api
```

## Rollback

```bash
ssh rig "cp /opt/indian-legal-ai/rag/cbic_rag/validator.py.bak.ladder_3rung.* /opt/indian-legal-ai/rag/cbic_rag/validator.py && sudo systemctl restart cbic-rag-api"
```
