#!/bin/bash
# Post-S3-resume: health check → reload crashed models → reboot if stuck
LOG=/tmp/ai-rig-resume.log
echo "$(date) === POST-RESUME HEALTH CHECK ===" >> $LOG

# Wait for system to settle after S3 wake
sleep 10

# Step 1: Check if GPUs are even visible
GPU_COUNT=$(ls -d /sys/class/drm/card[0-9] 2>/dev/null | wc -l)
echo "$(date) DRM cards found: $GPU_COUNT" >> $LOG

if [ "$GPU_COUNT" -lt 7 ]; then
    echo "$(date) CRITICAL: Only $GPU_COUNT/7 GPUs detected. S3 resume failed." >> $LOG
    echo "$(date) Forcing reboot in 10 seconds..." >> $LOG
    sleep 10
    reboot
    exit 1
fi

# Step 2: Check each GPU that had a running model
RELOADED=0
for GPU in 0 1 2 3 4 5 6; do
    PORT=$((9080 + GPU))
    PID=$(lsof -ti :$PORT 2>/dev/null)
    
    if [ -z "$PID" ]; then
        echo "$(date) GPU $GPU: no process" >> $LOG
        continue
    fi
    
    # Process exists — give it 30s to respond
    HEALTHY=0
    for CHECK in 1 2 3 4 5 6; do
        HEALTH=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://localhost:$PORT/health 2>/dev/null)
        if [ "$HEALTH" = "200" ]; then
            echo "$(date) GPU $GPU: healthy" >> $LOG
            HEALTHY=1
            break
        fi
        echo "$(date) GPU $GPU: waiting... (health=$HEALTH, attempt $CHECK/6)" >> $LOG
        sleep 5
    done
    
    if [ "$HEALTHY" = "0" ]; then
        echo "$(date) GPU $GPU: FAILED after 30s — killing and reloading" >> $LOG
        kill -9 $PID 2>/dev/null
        sleep 2
        
        # Reload from SQLite defaults
        python3 << PYEOF >> $LOG 2>&1
import sys; sys.path.insert(0, '/opt')
import importlib.util, sqlite3
spec = importlib.util.spec_from_file_location('shell', '/opt/ai-rig-shell.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.init_db()
c = sqlite3.connect('/opt/ai-rig.db')
c.row_factory = sqlite3.Row
r = c.execute('SELECT default_model, default_mode, default_temp FROM gpu_state WHERE gpu_id=?', (,)).fetchone()
c.close()
if r and r['default_model']:
    cfg = m.MODEL_CONFIGS.get(r['default_model'])
    if cfg:
        mode = r['default_mode'] or cfg['default_mode']
        mode_cfg = cfg['modes'].get(mode, {})
        flags = mode_cfg.get('flags_12gb', mode_cfg.get('flags')) if  == 2 else mode_cfg.get('flags')
        if r['default_temp'] is not None: flags += ' --temp ' + str(r['default_temp'])
        m.start_gpu(, str(m.MODEL_DIR / cfg['file']), flags)
        print(f'Reloaded GPU : {r[default_model]} {mode}')
PYEOF
        RELOADED=$((RELOADED + 1))
    fi
done

echo "$(date) === RESUME COMPLETE: $RELOADED GPU(s) reloaded ===" >> $LOG
