#!/usr/bin/env bash
# test_archive.sh — static shellcheck-lite for archive_v1.sh + rollback_v1.sh.
# Grep-based checks; no execution. Exits 0 pass, 1 on any failure.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
A="$ROOT/archive_v1.sh"
R="$ROOT/rollback_v1.sh"

fail=0
chk() { # name, file, pattern, (optional) "neg" to require absence
  local name="$1" file="$2" pat="$3" mode="${4:-pos}"
  if [ "$mode" = "neg" ]; then
    if grep -qE "$pat" "$file"; then
      echo "FAIL [$name] $file: forbidden pattern /$pat/ present"; fail=1
    else
      echo "ok   [$name] $file"
    fi
  else
    if grep -qE "$pat" "$file"; then
      echo "ok   [$name] $file"
    else
      echo "FAIL [$name] $file: missing /$pat/"; fail=1
    fi
  fi
}

for f in "$A" "$R"; do
  [ -f "$f" ] || { echo "FAIL missing: $f"; fail=1; continue; }
  chk "set-euo"       "$f" '^set -euo pipefail'
  chk "shebang-bash"  "$f" '^#!/usr/bin/env bash'
  chk "no-rm-rf-raw"  "$f" 'rm -rf [^"$]*(\*|/)' neg
  chk "exit-codes"    "$f" 'exit (0|2|3)'
done

chk "archive-manifest-write" "$A" 'manifest\.jsonl'
chk "archive-sha256"         "$A" 'sha256sum'
chk "archive-tar-gz"         "$A" 'tar -czf'
chk "archive-restore-hint"   "$A" 'rollback_v1\.sh'

chk "rollback-upload-put"    "$R" 'snapshots/upload'
chk "rollback-recover"       "$R" 'snapshots/recover'
chk "rollback-router-flip"   "$R" 'QDRANT_COLL'
chk "rollback-points-verify" "$R" 'points_count'
chk "rollback-dashboard"     "$R" 'DASH_URL'

if [ "$fail" -eq 0 ]; then
  echo "ALL PASS"; exit 0
else
  echo "FAILURES above"; exit 1
fi
