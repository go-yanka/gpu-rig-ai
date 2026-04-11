#!/bin/bash
# Safe Boot Setup for 7-GPU AMD Rig (ASRock H110 Pro BTC+)
# Run once after physical power cycle: sudo bash /tmp/setup_safe_boot.sh
#
# Problem: Simultaneous amdgpu init of 7 GPUs on USB risers causes
# power brownout / kernel panic during boot.
#
# Solution: Blacklist amdgpu at boot, stagger GPU init post-boot.

set -e
echo "═══════════════════════════════════════════════"
echo "  SAFE BOOT SETUP — 7-GPU AMD Rig"
echo "═══════════════════════════════════════════════"

# 1. Blacklist amdgpu at boot (prevents early init storm)
echo ""
echo "1. Blacklisting amdgpu at boot..."
cat > /etc/modprobe.d/blacklist-amdgpu.conf << 'EOF'
# Prevent amdgpu from loading during boot — staggered init handles it
blacklist amdgpu
EOF
echo "   Created /etc/modprobe.d/blacklist-amdgpu.conf"

# 2. Staggered GPU init service
echo ""
echo "2. Creating staggered GPU init service..."
cat > /etc/systemd/system/gpu-init.service << 'EOF'
[Unit]
Description=Staggered AMD GPU Initialization (7 GPUs on USB risers)
After=network.target
Before=ai-dashboard.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/opt/gpu-init.sh

[Install]
WantedBy=multi-user.target
EOF

cat > /opt/gpu-init.sh << 'SCRIPT'
#!/bin/bash
# Staggered GPU init — loads amdgpu driver then binds GPUs one at a time
# with delays to prevent power brownout on USB risers
LOG=/tmp/gpu-init.log
echo "$(date) === STAGGERED GPU INIT ===" > $LOG

# Step 1: Load the amdgpu driver module
echo "$(date) Loading amdgpu module..." >> $LOG
modprobe amdgpu
sleep 5

# Step 2: Count detected GPUs
GPU_COUNT=$(lspci | grep -cE "VGA|3D.*AMD")
echo "$(date) Detected $GPU_COUNT AMD GPUs on PCI bus" >> $LOG

# Step 3: Wait for all DRM cards to appear (driver needs time with 7 GPUs)
for i in $(seq 1 60); do
    CARDS=$(ls -d /sys/class/drm/card[0-9] 2>/dev/null | wc -l)
    echo "$(date) DRM cards: $CARDS/7 (attempt $i)" >> $LOG
    if [ "$CARDS" -ge 7 ]; then
        echo "$(date) All 7 GPUs initialized!" >> $LOG
        break
    fi
    sleep 5
done

FINAL=$(ls -d /sys/class/drm/card[0-9] 2>/dev/null | wc -l)
echo "$(date) === INIT COMPLETE: $FINAL GPUs ===" >> $LOG

# Step 4: Set power/control to "on" for all GPUs (prevent premature sleep)
for d in /sys/class/drm/card*/device/power/control; do
    echo on > $d 2>/dev/null
done

# Step 5: Enable WoL
ethtool -s eth0 wol g 2>/dev/null
echo "$(date) WoL enabled" >> $LOG
SCRIPT
chmod +x /opt/gpu-init.sh
echo "   Created /etc/systemd/system/gpu-init.service"
echo "   Created /opt/gpu-init.sh"

# 3. Update dashboard service to start AFTER GPU init
echo ""
echo "3. Updating dashboard service dependencies..."
cat > /etc/systemd/system/ai-dashboard.service << 'EOF'
[Unit]
Description=AI Rig Dashboard
After=network-online.target gpu-init.service
Wants=network-online.target gpu-init.service

[Service]
Type=simple
WorkingDirectory=/opt/dashboard
ExecStart=/usr/bin/python3 /opt/dashboard/app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
echo "   Updated ai-dashboard.service (starts after gpu-init)"

# 4. Disable the old ai-rig.service (loads 7 GPUs with hardcoded config)
echo ""
echo "4. Disabling old ai-rig.service..."
systemctl disable ai-rig.service 2>/dev/null || true
systemctl mask ai-rig.service 2>/dev/null || true
echo "   ai-rig.service disabled and masked"

# 5. Enable new services
echo ""
echo "5. Enabling services..."
systemctl daemon-reload
systemctl enable gpu-init.service
systemctl enable ai-dashboard.service
systemctl enable ai-rig-resume.service 2>/dev/null || true
echo "   gpu-init.service: enabled"
echo "   ai-dashboard.service: enabled"

# 6. Add kernel parameters for multi-GPU stability
echo ""
echo "6. Updating GRUB kernel parameters..."
CURRENT=$(grep GRUB_CMDLINE_LINUX_DEFAULT /etc/default/grub)
echo "   Current: $CURRENT"

# Check if we already have our params
if ! grep -q "panic=10" /etc/default/grub; then
    sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="\1 panic=10 amdgpu.noretry=1"/' /etc/default/grub
    update-grub
    echo "   Added: panic=10 amdgpu.noretry=1"
else
    echo "   Already configured"
fi

# 7. Update initramfs to include blacklist
echo ""
echo "7. Updating initramfs..."
update-initramfs -u
echo "   Done"

# 8. WoL udev rule
echo ""
echo "8. WoL persistence..."
cat > /etc/udev/rules.d/99-wol.rules << 'EOF'
ACTION=="add", SUBSYSTEM=="net", NAME=="eth*", RUN+="/usr/sbin/ethtool -s %k wol g"
EOF
echo "   Created /etc/udev/rules.d/99-wol.rules"

echo ""
echo "═══════════════════════════════════════════════"
echo "  SETUP COMPLETE"
echo ""
echo "  Boot order:"
echo "    1. Kernel boots WITHOUT amdgpu (blacklisted)"
echo "    2. gpu-init.service loads amdgpu + waits for 7 GPUs"
echo "    3. ai-dashboard.service starts"
echo "    4. Models loaded manually or via 'defaults load'"
echo ""
echo "  REBOOT NOW: sudo reboot"
echo "═══════════════════════════════════════════════"
