#!/usr/bin/env bash
# rule_check.sh — keyword-indexed rule lookup
# Usage: bash rule_check.sh "<keyword>" [<keyword2> ...]
# Greps RULES_INDEX.md for matching [TRIGGER: ...] blocks and prints the full block.

set -u
RULES_FILE="${RULES_FILE:-D:/_gpu_rig_ai/RULES_INDEX.md}"
# Auto-detect on POSIX (rig)
if [ ! -f "$RULES_FILE" ] && [ -f "/opt/indian-legal-ai/RULES_INDEX.md" ]; then
  RULES_FILE="/opt/indian-legal-ai/RULES_INDEX.md"
fi
if [ ! -f "$RULES_FILE" ]; then
  echo "ERROR: RULES_INDEX.md not found at $RULES_FILE" >&2
  exit 2
fi

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <keyword> [keyword2 ...]" >&2
  echo "" >&2
  echo "Available triggers:" >&2
  grep -oE '\[TRIGGER:[^]]+\]' "$RULES_FILE" >&2
  exit 1
fi

# Log the consultation (used by task_preflight.sh to verify rule_check was run)
LOG="${RULE_CHECK_LOG:-/tmp/rule_check.log}"
mkdir -p "$(dirname "$LOG")" 2>/dev/null
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) keywords=$*" >> "$LOG"

found_any=0
for kw in "$@"; do
  # Find [TRIGGER: ...keyword...] blocks (case-insensitive). Print from [TRIGGER line through next blank line.
  awk -v kw="$(echo "$kw" | tr '[:upper:]' '[:lower:]')" '
    BEGIN{ inblk=0 }
    /^\[TRIGGER:/ {
      lc = tolower($0)
      if (index(lc, kw) > 0) { inblk=1; print "---"; print; next }
      else { inblk=0; next }
    }
    /^$/ { if (inblk) { inblk=0; print "" }; next }
    { if (inblk) print }
  ' "$RULES_FILE" > /tmp/rule_check_out.$$
  if [ -s /tmp/rule_check_out.$$ ]; then
    echo "===== keyword: $kw ====="
    cat /tmp/rule_check_out.$$
    found_any=1
  else
    echo "===== keyword: $kw → NO MATCH ====="
  fi
  rm -f /tmp/rule_check_out.$$
done

if [ "$found_any" -eq 0 ]; then
  echo "" >&2
  echo "No matching rule blocks. Available triggers:" >&2
  grep -oE '\[TRIGGER:[^]]+\]' "$RULES_FILE" | sed 's/^/  /' >&2
  exit 3
fi
exit 0
