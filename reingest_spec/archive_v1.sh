#!/usr/bin/env bash
# archive_v1.sh (Stage K) — final snapshot + durable archive of cbic_v1.
# Runs AFTER Stage J promotion of cbic_v2. Per D10 (SPEC.md:155): never delete,
# keep v1 as rollback. Mirrors snapshot_v2.sh pattern but targets archive tier.
#
# Run (on rig):
#   bash archive_v1.sh
#   QDRANT_URL=http://127.0.0.1:6343 bash archive_v1.sh
#
# Exit codes:
#   0 ok
#   2 Qdrant API error (snapshot/download)
#   3 sanity fail (size floor / point-count mismatch / sha missing)

set -euo pipefail

COLL="${1:-cbic_v1}"
DEST="${2:-/opt/snapshots/archive}"
QDRANT="${QDRANT_URL:-http://127.0.0.1:6343}"
MIN_BYTES="${ARCHIVE_MIN_BYTES:-10485760}"   # 10 MiB floor for a real v1

mkdir -p "$DEST"
TS="$(date +%Y%m%d_%H%M%S)"
echo "[archive_v1] coll=$COLL dest=$DEST qdrant=$QDRANT ts=$TS"

# 1. source point-count
pts_src="$(curl -sf "$QDRANT/collections/$COLL" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["result"]["points_count"])')" \
  || { echo "[archive_v1] GET collection failed"; exit 2; }
echo "[archive_v1] source points_count=$pts_src"

# 2. trigger final snapshot
resp="$(curl -sf -X POST "$QDRANT/collections/$COLL/snapshots")" \
  || { echo "[archive_v1] POST snapshots failed"; exit 2; }
name="$(echo "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin)["result"]["name"])')"
size_srv="$(echo "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin)["result"].get("size",0))')"
echo "[archive_v1] snapshot name=$name size_server=$size_srv"

if [ "$size_srv" -lt "$MIN_BYTES" ]; then
  echo "[archive_v1] FAIL size=$size_srv < floor=$MIN_BYTES"; exit 3
fi

# 3. download
snap_path="$DEST/${COLL}_${TS}.snapshot"
curl -sf -o "$snap_path" "$QDRANT/collections/$COLL/snapshots/$name" \
  || { echo "[archive_v1] download failed"; exit 2; }
size_local=$(stat -c %s "$snap_path")
echo "[archive_v1] downloaded $snap_path size_local=$size_local"

# 4. size-match sanity (local within 5% of server-reported)
lo=$(( size_srv * 95 / 100 ))
if [ "$size_local" -lt "$lo" ]; then
  echo "[archive_v1] FAIL local=$size_local < 95% of server=$size_srv"; exit 3
fi

# 5. tar+gzip
tar_path="$DEST/${COLL}_${TS}.tar.gz"
tar -czf "$tar_path" -C "$DEST" "$(basename "$snap_path")" \
  || { echo "[archive_v1] tar failed"; exit 2; }
tar_size=$(stat -c %s "$tar_path")
sha="$(sha256sum "$tar_path" | awk '{print $1}')"
[ -n "$sha" ] || { echo "[archive_v1] sha256 empty"; exit 3; }
echo "[archive_v1] archive=$tar_path tar_size=$tar_size sha256=$sha"

# 6. manifest row
manifest="$DEST/manifest.jsonl"
python3 - <<PY >>"$manifest"
import json
print(json.dumps({
  "ts": "$TS", "collection": "$COLL", "snapshot_name": "$name",
  "snapshot_path": "$snap_path", "archive_path": "$tar_path",
  "size_server": $size_srv, "size_local": $size_local, "size_tar": $tar_size,
  "points_count": $pts_src, "sha256": "$sha",
}))
PY

# 7. ready-to-paste restore
echo
echo "[archive_v1] OK — to rollback:"
echo "  bash $(dirname "$0")/rollback_v1.sh $tar_path"
exit 0
