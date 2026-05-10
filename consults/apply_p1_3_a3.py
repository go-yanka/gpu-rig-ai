#!/usr/bin/env python3
"""P1.3 A3 chunk-refeed patch applier (idempotent, no-op if already applied)."""
import sys, re, pathlib, shutil, time

TARGET = pathlib.Path("/opt/indian-legal-ai/rag/cbic_rag/api.py")

src = TARGET.read_text()
orig = src

# Edit 1: add TWO_PASS_CHUNK_REFEED env flag after TWO_PASS_ENABLED definition
old1 = "TWO_PASS_ENABLED = os.environ.get('TWO_PASS_ENABLED', '0') == '1'\n"
new1 = ("TWO_PASS_ENABLED = os.environ.get('TWO_PASS_ENABLED', '0') == '1'\n"
        "TWO_PASS_CHUNK_REFEED = os.environ.get('TWO_PASS_CHUNK_REFEED', '0') == '1'\n")
if "TWO_PASS_CHUNK_REFEED" not in src:
    assert src.count(old1) == 1, f"edit1 anchor count={src.count(old1)}"
    src = src.replace(old1, new1, 1)

# Edit 2: change _synthesize_pass2 signature
old2 = "def _synthesize_pass2(question: str, verified: List[dict]) -> str:\n"
new2 = "def _synthesize_pass2(question: str, verified: List[dict], chunks: Optional[List[dict]] = None) -> str:\n"
if old2 in src:
    src = src.replace(old2, new2, 1)
elif new2 not in src:
    sys.exit("edit2 anchor not found")

# Edit 3: insert chunk-refeed block between `user = build_synthesis_user(question, facts)` and `    try:`
anchor3 = "    user = build_synthesis_user(question, facts)\n    try:\n"
inject = (
"    user = build_synthesis_user(question, facts)\n"
"    # P1.3 A3: optional chunk re-feed for disambiguation (NOT for quoting).\n"
"    # Guarded by TWO_PASS_CHUNK_REFEED env var (default 0).\n"
"    if TWO_PASS_CHUNK_REFEED and chunks:\n"
"        ctx_blocks = []\n"
"        for i, c in enumerate(chunks[:5], start=1):\n"
"            t = (c.get('text_full') or c.get('text') or '').strip()\n"
"            if len(t) > 400:\n"
"                t = t[:400] + ' ...'\n"
"            ctx_blocks.append(f\"[S{i}] {t}\")\n"
"        if ctx_blocks:\n"
"            user = (\n"
"                user\n"
"                + \"\\n\\nCONTEXT CHUNKS (for disambiguation only, do NOT quote or cite):\\n\\n\"\n"
"                + \"\\n\\n\".join(ctx_blocks)\n"
"            )\n"
"    try:\n"
)
if "P1.3 A3: optional chunk re-feed" not in src:
    assert src.count(anchor3) == 1, f"edit3 anchor count={src.count(anchor3)}"
    src = src.replace(anchor3, inject, 1)

# Edit 4: caller site — forward chunks
old4 = "    prose = _synthesize_pass2(question, verified)\n"
new4 = "    prose = _synthesize_pass2(question, verified, chunks)\n"
if old4 in src:
    src = src.replace(old4, new4, 1)
elif new4 not in src:
    sys.exit("edit4 anchor not found")

if src == orig:
    print("NO-OP: patch already applied")
    sys.exit(0)

# Write with backup
ts = time.strftime("%Y%m%d_%H%M%S")
bak = TARGET.with_suffix(TARGET.suffix + f".bak.night_p1_3_{ts}")
shutil.copy2(TARGET, bak)
TARGET.write_text(src)
print(f"APPLIED. backup={bak}")
