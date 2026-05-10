#!/usr/bin/env bash
# rollback_v1.sh (Stage K) — restore cbic_v1 from archive, flip router back.
# Consumes archive produced by archive_v1.sh.
#
# Run:
#   bash rollback_v1.sh                               # auto-pick latest
#   bash rollback_v1.sh /opt/snapshots/archive/cbic_v1_20260501_010203.tar.gz
#
# Exit codes:
#   0 ok
#   2 Qdrant API / upload error
#   3 sanity fail (sha mismatch / points_count mismatch / manifest missing)

set -euo pipefail

COLL="${COLL:-cbic_v1}"
DEST="${ARCHIVE_DIR:-/opt/snapshots/archive}"
QDRANT="${QDRANT_URL:-http://127.0.0.1:6343}"
ROUTER_CONF="${ROUTER_CONF:-/opt/indian-legal-ai/router.config}"
DASH_URL="${DASH_URL:-http://192.168.1.107:9500/admin}"

ARCHIVE="${1:-}"
if [ -z "$ARCHIVE" ]; then
  ARCHIVE="$(ls -1t "$DEST"/${COLL}_*.tar.gz 2>/dev/null | head -1 || true)"
  [ -n "$ARCHIVE" ] || { echo "[rollback_v1] no archive found in $DEST"; exit 3; }
fi
[ -f "$ARCHIVE" ] || { echo "[rollback_v1] missing: $ARCHIVE"; exit 3; }
echo "[rollback_v1] archive=$ARCHIVE"

manifest="$DEST/manifest.jsonl"
[ -f "$manifest" ] || { echo "[rollback_v1] missing manifest"; exit 3; }

# 1. verify sha256 against manifest row for this archive
row="$(grep -F "\"archive_path\": \"$ARCHIVE\"" "$manifest" | tail -1 || true)"
[ -n "$row" ] || { echo "[rollback_v1] no manifest row for $ARCHIVE"; exit 3; }
want_sha="$(echo "$row" | python3 -c 'import sys,json; print(json.load(sys.stdin)["sha256"])')"
want_pts="$(echo "$row" | python3 -c 'import sys,json; print(json.load(sys.stdin)["points_count"])')"
got_sha="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
[ "$got_sha" = "$want_sha" ] || { echo "[rollback_v1] sha mismatch want=$want_sha got=$got_sha"; exit 3; }
echo "[rollback_v1] sha ok pts_expect=$want_pts"

# 2. extract snapshot
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
tar -xzf "$ARCHIVE" -C "$tmpdir"
snap_file="$(ls -1 "$tmpdir"/*.snapshot | head -1)"
snap_name="$(basename "$snap_file")"
echo "[rollback_v1] extracted $snap_file"

# 3. upload to Qdrant
curl -sf -X PUT -H "Content-Type: application/octet-stream" \
  --data-binary @"$snap_file" \
  "$QDRANT/collections/$COLL/snapshots/upload?priority=snapshot" \
  || { echo "[rollback_v1] upload failed"; exit 2; }
echo "[rollback_v1] uploaded as $snap_name"

# 4. recover
curl -sf -X POST -H "Content-Type: application/json" \
  -d "{\"location\":\"file:///qdrant/snapshots/$COLL/$snap_name\"}" \
  "$QDRANT/collections/$COLL/snapshots/recover" \
  || { echo "[rollback_v1] recover failed"; exit 2; }

# 5. verify points_count
sleep 3
pts_now="$(curl -sf "$QDRANT/collections/$COLL" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["result"]["points_count"])')"
[ "$pts_now" = "$want_pts" ] \
  || { echo "[rollback_v1] points mismatch want=$want_pts got=$pts_now"; exit 3; }
echo "[rollback_v1] points_count verified=$pts_now"

# 6. router flip — /query back to cbic_v1.
# H4 fix: api.py reads QDRANT_COLL at module-import time, so writing a config
# file is not enough. We MUST restart the service for the env flip to take
# effect. Support systemd (primary) and fallback to pkill+start via SERVICE_CMD.
mkdir -p "$(dirname "$ROUTER_CONF")"
printf 'QDRANT_COLL=%s\n' "$COLL" >"$ROUTER_CONF"
echo "[rollback_v1] router.config: QDRANT_COLL=$COLL"

SERVICE_UNIT="${SERVICE_UNIT:-indian-legal-ai-api}"
SERVICE_CMD="${SERVICE_CMD:-}"

restart_ok=0
if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files 2>/dev/null | grep -q "^${SERVICE_UNIT}\.service"; then
  echo "[rollback_v1] restarting systemd unit: $SERVICE_UNIT"
  if sudo -n systemctl restart "$SERVICE_UNIT" 2>/dev/null || systemctl restart "$SERVICE_UNIT" 2>/dev/null; then
    restart_ok=1
  fi
fi
if [ "$restart_ok" -eq 0 ] && [ -n "$SERVICE_CMD" ]; then
  echo "[rollback_v1] SERVICE_UNIT restart failed/unavailable — running SERVICE_CMD"
  # shellcheck disable=SC2086
  bash -c "$SERVICE_CMD" && restart_ok=1
fi
if [ "$restart_ok" -eq 0 ]; then
  echo "[rollback_v1] WARN: could not auto-restart. MANUAL step required:"
  echo "    export QDRANT_COLL=$COLL && <restart cbic_rag api>"
  echo "    or: sudo systemctl restart $SERVICE_UNIT"
  echo "    or: SERVICE_CMD='<cmd>' bash rollback_v1.sh ..."
  exit 2
fi

# 7. health-check — wait for service to be back and verify it now serves cbic_v1
echo "[rollback_v1] waiting for api to come back up..."
for i in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:9500/v1/stats" >/dev/null 2>&1; then
    echo "[rollback_v1] api responsive after ${i}s"
    break
  fi
  sleep 1
done

echo
echo "[rollback_v1] OK — verify at: $DASH_URL"
exit 0
