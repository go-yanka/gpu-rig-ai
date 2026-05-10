#!/usr/bin/env bash
# snapshot_v2.sh (B7) — Qdrant snapshot + rotation for cbic_v2.
# Wraps probe_v6 pattern, adds: size-sanity check, timestamped copy, 14d rotation.
# Needed for D10 rollback to v1 or to a prior v2 snapshot at any gate.
#
# Run (on rig):
#   bash snapshot_v2.sh                  # default: cbic_v2 → /opt/snapshots/cbic_v2/
#   bash snapshot_v2.sh cbic_v2_shadow /mnt/extra/snaps
#
# Exit codes:
#   0 ok
#   2 snapshot API error
#   3 size sanity failed (<50% of prior snapshot → corruption suspected)

set -euo pipefail

COLL="${1:-cbic_v2}"
DEST="${2:-/opt/snapshots/$COLL}"
QDRANT="${QDRANT_URL:-http://127.0.0.1:6343}"
RETAIN_DAYS="${SNAPSHOT_RETAIN_DAYS:-14}"
MIN_BYTES="${SNAPSHOT_MIN_BYTES:-1048576}"   # 1 MiB floor

mkdir -p "$DEST"
TS="$(date +%Y%m%d_%H%M%S)"

echo "[snapshot_v2] coll=$COLL dest=$DEST qdrant=$QDRANT"

# 1. trigger snapshot
resp="$(curl -sf -X POST "$QDRANT/collections/$COLL/snapshots")" || {
  echo "[snapshot_v2] POST snapshots failed"; exit 2; }
name="$(echo "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin)["result"]["name"])')"
size="$(echo "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin)["result"].get("size",0))')"
echo "[snapshot_v2] created name=$name size=$size"

# 2. size sanity
if [ "$size" -lt "$MIN_BYTES" ]; then
  echo "[snapshot_v2] FAIL size=$size < min=$MIN_BYTES — corruption suspected"
  exit 3
fi

# 3. compare vs last snapshot on disk (corruption detection)
prev="$(ls -1t "$DEST"/*.snapshot 2>/dev/null | head -1 || true)"
if [ -n "$prev" ]; then
  prev_size=$(stat -c %s "$prev")
  half=$(( prev_size / 2 ))
  if [ "$size" -lt "$half" ]; then
    echo "[snapshot_v2] FAIL size=$size < 50% of prev=$prev_size"
    exit 3
  fi
fi

# 4. download to durable storage
url="$QDRANT/collections/$COLL/snapshots/$name"
target="$DEST/${COLL}_${TS}.snapshot"
curl -sf -o "$target" "$url" || { echo "[snapshot_v2] download failed"; exit 2; }
got_size=$(stat -c %s "$target")
echo "[snapshot_v2] downloaded $target size=$got_size"

# 5. rotation — delete snapshots older than RETAIN_DAYS
find "$DEST" -name "${COLL}_*.snapshot" -type f -mtime +"$RETAIN_DAYS" -print -delete

# 6. write manifest row
manifest="$DEST/manifest.jsonl"
python3 - <<PY >>"$manifest"
import json, os
print(json.dumps({
  "ts": "$TS", "collection": "$COLL",
  "name": "$name", "path": "$target",
  "size_server": $size, "size_local": $got_size,
}))
PY

echo "[snapshot_v2] OK  path=$target"
exit 0
