#!/usr/bin/env bash
# Deploy dashboard to rig from this Windows machine (via scp)
# Run this from Windows Git Bash or WSL

RIG="user@192.168.1.107"
RIG_DIR="/opt/dashboard"

echo "Deploying dashboard to ${RIG}:${RIG_DIR}..."

# Create dir on rig
ssh "${RIG}" "sudo mkdir -p ${RIG_DIR} && sudo chown user:user ${RIG_DIR}"

# Copy files
scp -r "$(dirname "$0")"/* "${RIG}:${RIG_DIR}/"

# Install deps and start
ssh "${RIG}" "
  pip3 install --break-system-packages fastapi 'uvicorn[standard]' 2>/dev/null || pip3 install fastapi 'uvicorn[standard]'
  chmod +x ${RIG_DIR}/start.sh
  echo ''
  echo 'Deploy done. Run on rig:'
  echo '  ${RIG_DIR}/start.sh'
  echo 'Dashboard: http://192.168.1.107:8080'
"
