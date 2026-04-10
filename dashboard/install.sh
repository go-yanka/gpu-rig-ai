#!/usr/bin/env bash
# Install dashboard dependencies on the rig
set -e

echo "Installing AI Rig Dashboard dependencies..."

# Create target dir
sudo mkdir -p /opt/dashboard
sudo cp -r "$(dirname "$0")"/* /opt/dashboard/
sudo chmod +x /opt/dashboard/start.sh

# Install Python deps
pip3 install --break-system-packages fastapi "uvicorn[standard]" 2>/dev/null \
  || pip3 install fastapi "uvicorn[standard]"

echo ""
echo "Done! Run: /opt/dashboard/start.sh"
echo "Dashboard: http://192.168.1.107:8080"
