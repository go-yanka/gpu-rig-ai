#!/usr/bin/env python3
"""PreToolUse hook — surface DECISIONS.yaml + memory canon matches before
Bash/Write/Edit on protected paths.

Fires on Claude Code's `PreToolUse` event. Reads the tool input, extracts
keywords, greps DECISIONS.yaml + memory canon files, and emits a hook
context block to stderr (which Claude Code surfaces to the agent).

Exits 0 always — this is advisory, never blocking.

Wired in ~/.claude/settings.json:
  {
    "hooks": {
      "PreToolUse": [
        {"matcher": "Bash|Write|Edit",
         "hooks": [{"type":"command","command":"python3 ~/.claude/scripts/hook_decision_check.py"}]}
      ]
    }
  }
"""
from __future__ import annotations
import json, os, re, sys, hashlib

PROTECTED = (
    "/opt/indian-legal-ai/reingest_spec",
    "/opt/indian-legal-ai/data/training_corpus",
    "/opt/indian-legal-ai/eval",
    "eval/scale_sets",
    "training_pairs",
    "cbic_pairs_v2",
    "phase6",
    "DECISIONS.yaml",
    "ingest_v2",
    "chunker_v2",
    "embedder",
)

# Files to grep for matches (rig paths checked first, then local fallback)
SOURCES = [
    ("DECISIONS",        ["/opt/indian-legal-ai/reingest_spec/DECISIONS.yaml"]),
    ("PAIR_GEN_SPEC",    ["/opt/indian-legal-ai/reingest_spec/PAIR_GEN_SPEC.md"]),
    ("RUNBOOK",          ["/opt/indian-legal-ai/reingest_spec/RUNBOOK.md"]),
    ("MEMORY",           [os.path.expanduser("~/.claude/projects/D---gpu-rig-ai/memory/MEMORY.md"),
                          "C:/Users/Rahul Goyanka/.claude/projects/D---gpu-rig-ai/memory/MEMORY.md"]),
    ("pair_schema",      [os.path.expanduser("~/.claude/projects/D---gpu-rig-ai/memory/pair_schema_cbic_v2.md"),
                          "C:/Users/Rahul Goyanka/.claude/projects/D---gpu-rig-ai/memory/pair_schema_cbic_v2.md"]),
    ("ingest_playbook",  [os.path.expanduser("~/.claude/projects/D---gpu-rig-ai/memory/ingest_playbook_cbic.md"),
                          "C:/Users/Rahul Goyanka/.claude/projects/D---gpu-rig-ai/memory/ingest_playbook_cbic.md"]),
]

# Words to ignore as keywords
STOP = set("the a an and or but for to in on at of with from by is are was were be been "
           "this that these those it its as if then so do does did doing have has had "
           "i you we they he she them us our your ssh root rig run cd cat ls echo grep "
           "python python3 bash sh tail head wc which file path command tool".split())

def extract_keywords(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{3,}", text)
    seen = []; out = []
    for w in words:
        lw = w.lower()
        if lw in STOP: continue
        if lw in seen: continue
        seen.append(lw); out.append(w)
    return out[:30]

def grep_file(path: str, keywords: list[str]) -> list[str]:
    try: lines = open(path, encoding="utf-8", errors="ignore").read().splitlines()
    except Exception: return []
    hits = []
    pattern = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE) if keywords else None
    if not pattern: return []
    for i, line in enumerate(lines, 1):
        if pattern.search(line):
            hits.append(f"  L{i}: {line.strip()[:160]}")
            if len(hits) >= 4: break
    return hits

def main():
    try: ev = json.loads(sys.stdin.read() or "{}")
    except Exception: ev = {}
    tin = ev.get("tool_input", {}) or {}
    tname = ev.get("tool_name", "")
    blob = " ".join(str(v) for v in tin.values() if isinstance(v, (str, int, float)))[:4000]

    # Only trigger on protected-path mentions
    if not any(p.lower() in blob.lower() for p in PROTECTED):
        return 0

    keywords = extract_keywords(blob)
    if not keywords:
        return 0

    out = ["[hook_decision_check] protected-path tool use detected — relevant canon excerpts:"]
    for label, paths in SOURCES:
        for path in paths:
            if not os.path.exists(path): continue
            hits = grep_file(path, keywords)
            if hits:
                out.append(f"\n--- {label} ({path}) ---")
                out.extend(hits)
            break  # first existing path per source

    if len(out) > 1:
        out.append(f"\n[hook] keywords: {', '.join(keywords[:10])}")
        out.append("[hook] If your action contradicts the above, STOP and reconcile before proceeding.")
        sys.stderr.write("\n".join(out) + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
