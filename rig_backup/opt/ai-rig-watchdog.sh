#!/bin/bash
# AI Rig Watchdog — runs every 5 minutes
# 1. GPU health check (reboot if GPUs lost)
# 2. Sync DB to USB (protect against data loss)
# 3. OOM prevention (kill bloated processes)
LOG=/tmp/ai-rig-watchdog.log

# 1. Check GPU count
CARDS=$(ls -d /sys/class/drm/card[0-9] 2>/dev/null | wc -l)
if [ "$CARDS" -lt 7 ]; then
    echo "$(date) CRITICAL: Only $CARDS/7 GPUs detected. Rebooting..." >> $LOG
    sync
    reboot
    exit 1
fi

# 2. Sync runtime DB to USB (every 5 min)
if [ -f /dev/shm/ai-rig.db ]; then
    cp /dev/shm/ai-rig.db /opt/ai-rig.db.tmp 2>/dev/null && mv /opt/ai-rig.db.tmp /opt/ai-rig.db 2>/dev/null
fi

# 3. OOM prevention — kill biggest llama-server if RAM < 1GB
AVAIL=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
if [ "$AVAIL" -lt 1024 ]; then
    echo "$(date) WARNING: Low RAM (${AVAIL}MB). Killing biggest llama-server..." >> $LOG
    PID=$(ps aux --sort=-%mem | grep llama-server | grep -v grep | head -1 | awk '{print $2}')
    if [ -n "$PID" ]; then
        kill -9 $PID
        echo "$(date) Killed PID $PID" >> $LOG
    fi
fi

# 4. Log health (keep last 100 lines)
echo "$(date) OK: $CARDS GPUs, ${AVAIL}MB RAM free" >> $LOG
tail -100 $LOG > $LOG.tmp && mv $LOG.tmp $LOG
