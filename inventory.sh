#!/usr/bin/env bash
# inventory.sh — truth-from-disk dump for planning.
# Lists every data/code/doc file under D:/_gpu_rig_ai/ with line count + first line.
# Run this FIRST before proposing any plan that involves drafting new eval data,
# new queries, new labels, new prompts, or new code that might already exist.

ROOT="${1:-D:/_gpu_rig_ai}"

cd "$ROOT" || { echo "ERROR: cannot cd to $ROOT"; exit 1; }

echo "# Inventory of $ROOT — generated $(date)"
echo ""

for pat in "*.jsonl" "*.json" "*.yaml" "*.yml" "*.md" "*.py" "*.sh"; do
  echo "## $pat"
  # find files, skip heavy/noise dirs
  find . -type f -name "$pat" \
      -not -path "./.git/*" \
      -not -path "./.venv/*" \
      -not -path "./venv/*" \
      -not -path "./node_modules/*" \
      -not -path "./__pycache__/*" \
      -not -path "*/\.*" \
      2>/dev/null | sort | while read -r f; do
    # line count
    lc=$(wc -l <"$f" 2>/dev/null | tr -d ' ')
    # first non-empty line, truncated to 120 chars
    first=$(grep -v '^[[:space:]]*$' "$f" 2>/dev/null | head -1 | cut -c1-120)
    printf "  %-80s %6s lines  | %s\n" "$f" "$lc" "$first"
  done
  echo ""
done

echo "## Directories (top-level under $ROOT)"
ls -la "$ROOT" 2>/dev/null | awk 'NR>1 {print "  " $NF}' | grep -v '^\s*\.$\|^\s*\.\.$'
