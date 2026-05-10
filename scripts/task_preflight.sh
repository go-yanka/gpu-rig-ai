#!/usr/bin/env bash
# task_preflight.sh — hard-fail unless rule consultation has happened.
# Usage: bash task_preflight.sh "<keyword>" [<keyword2> ...]
#
# What this script does (and what it ENFORCES):
#   1. Asserts inventory.sh has been run in the last 30 minutes (timestamp check on its output).
#   2. Runs rule_check.sh with the provided keywords; FAILS if no rule block matches.
#   3. Prints the matched rule blocks so they MUST appear in the conversation.
#   4. Logs the preflight invocation; downstream scripts (e.g. ingest launchers) can verify
#      a fresh preflight ran before they accept arguments.
#
# Exit codes:
#   0 = preflight passed; you may proceed (and you MUST quote the printed rule blocks).
#   1 = inventory.sh stale or missing.
#   2 = rule_check.sh found no matching block — your keywords are wrong, refine them.
#   3 = bad usage.

set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
RULE_CHECK="$HERE/rule_check.sh"

# POSIX vs Windows-shell path detection
if [ -f "/opt/indian-legal-ai/scripts/rule_check.sh" ]; then
  RULE_CHECK="/opt/indian-legal-ai/scripts/rule_check.sh"
fi

# Default state dir is project-rooted so it works on both Git Bash (Windows) and Linux (rig).
STATE_DIR="${PREFLIGHT_STATE_DIR:-}"
if [ -z "$STATE_DIR" ]; then
  if [ -d "D:/_gpu_rig_ai" ]; then STATE_DIR="D:/_gpu_rig_ai/.preflight"
  elif [ -d "/opt/indian-legal-ai" ]; then STATE_DIR="/opt/indian-legal-ai/.preflight"
  else STATE_DIR="/tmp"
  fi
fi
mkdir -p "$STATE_DIR" 2>/dev/null
INVENTORY_OUT="${INVENTORY_OUT:-$STATE_DIR/inventory_last.txt}"
INVENTORY_FRESH_MIN="${INVENTORY_FRESH_MIN:-30}"
PREFLIGHT_LOG="${PREFLIGHT_LOG:-$STATE_DIR/task_preflight.log}"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <keyword> [keyword2 ...]" >&2
  echo "  Pass at least one keyword describing the action you're about to take." >&2
  exit 3
fi

echo "================================================================"
echo "  TASK PREFLIGHT — keywords: $*"
echo "================================================================"

# --- 1. inventory.sh freshness ---
echo ""
echo "[1/2] inventory.sh freshness check"
if [ ! -f "$INVENTORY_OUT" ]; then
  echo "  FAIL: $INVENTORY_OUT not found."
  echo "        Run:  bash D:/_gpu_rig_ai/inventory.sh > $INVENTORY_OUT 2>&1"
  echo "        (or  bash /opt/indian-legal-ai/inventory.sh > $INVENTORY_OUT 2>&1  on the rig)"
  exit 1
fi
# stat -c on linux, mtime via python fallback for cross-platform
age_sec=$(python3 -c "import os,time; print(int(time.time() - os.path.getmtime('$INVENTORY_OUT')))" 2>/dev/null || echo 999999)
age_min=$((age_sec / 60))
if [ "$age_min" -ge "$INVENTORY_FRESH_MIN" ]; then
  echo "  FAIL: inventory output is $age_min min old (limit $INVENTORY_FRESH_MIN min)."
  echo "        Re-run:  bash inventory.sh > $INVENTORY_OUT"
  exit 1
fi
echo "  OK: inventory output age = $age_min min (limit $INVENTORY_FRESH_MIN)."

# --- 2. rule_check on each keyword ---
echo ""
echo "[2/2] rule_check.sh on keywords"
echo ""
if [ ! -x "$RULE_CHECK" ] && [ ! -f "$RULE_CHECK" ]; then
  echo "  FAIL: rule_check.sh not found at $RULE_CHECK"
  exit 2
fi
bash "$RULE_CHECK" "$@"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo ""
  echo "  FAIL: rule_check.sh did not match any block (rc=$rc)."
  echo "        Refine your keywords or add a rule to RULES_INDEX.md."
  exit 2
fi

# --- log success ---
mkdir -p "$(dirname "$PREFLIGHT_LOG")" 2>/dev/null
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) keywords=$*" >> "$PREFLIGHT_LOG"

echo ""
echo "================================================================"
echo "  PREFLIGHT PASSED — you MUST now:"
echo "    1) Quote the rule blocks above in a > [CODIFIED RULE] block."
echo "    2) Fill the Self-Audit template (RULES_INDEX.md → bottom)."
echo "    3) Only then propose / execute the action."
echo "================================================================"
exit 0
