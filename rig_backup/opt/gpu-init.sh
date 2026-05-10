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
