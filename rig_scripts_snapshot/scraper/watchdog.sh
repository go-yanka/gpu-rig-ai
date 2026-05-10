#!/bin/bash
# watchdog for the CBIC scraper pipeline.
# Responsibilities:
#   1. If the downloader is running but its log has been idle > STALL seconds, SIGKILL it.
#   2. If neither the downloader nor pipeline_all.sh is running and the pipeline
#      has not logged "ALL DONE" yet, relaunch pipeline_all.sh.
#   3. If pipeline_all.sh has logged "ALL DONE", exit cleanly.

LOGDIR=/opt/indian-legal-ai/scraper/logs
SCRIPTDIR=/opt/indian-legal-ai/scraper
STALL=300      # per-downloader log-inactivity threshold
DEAD=180       # tolerate this much "nothing running" before relaunching
CHECK=60
dead_for=0

mkdir -p "$LOGDIR"
echo "[$(date)] watchdog v2 started stall=${STALL}s dead=${DEAD}s check=${CHECK}s" >> "$LOGDIR/watchdog.log"

relaunch_pipeline () {
  echo "[$(date)] relaunching pipeline_all.sh" >> "$LOGDIR/watchdog.log"
  ( cd "$SCRIPTDIR" && nohup ./pipeline_all.sh >> "$LOGDIR/pipeline_all.out" 2>&1 & )
  sleep 5
}

while true; do
  sleep "$CHECK"
  DL_PID=$(pgrep -f 'run_scrape.py cbic --stage download' | head -1)
  PIPE_PID=$(pgrep -f 'pipeline_all.sh' | grep -v $$ | head -1)

  # Case A: downloader running — do classic stall check.
  if [ -n "$DL_PID" ]; then
    dead_for=0
    LOG=$(ls -t "$LOGDIR"/cbic_download_*.log 2>/dev/null | head -1)
    [ -z "$LOG" ] && continue
    LAST=$(stat -c %Y "$LOG")
    NOW=$(date +%s)
    AGE=$((NOW-LAST))
    if [ "$AGE" -gt "$STALL" ]; then
      echo "[$(date)] STALL: $LOG inactive ${AGE}s; killing pid $DL_PID" >> "$LOGDIR/watchdog.log"
      kill -9 "$DL_PID" 2>/dev/null
      sleep 5
    fi
    continue
  fi

  # Case B: downloader not running but pipeline_all.sh is — between scopes, fine.
  if [ -n "$PIPE_PID" ]; then
    dead_for=0
    continue
  fi

  # Case C: nothing is running. Finished?
  if tail -n 5 "$LOGDIR/cbic_pipeline.log" 2>/dev/null | grep -q "ALL DONE"; then
    echo "[$(date)] pipeline reports ALL DONE; watchdog exiting" >> "$LOGDIR/watchdog.log"
    exit 0
  fi

  # Case D: nothing is running and we're not done. Count idle time, then relaunch.
  dead_for=$((dead_for + CHECK))
  echo "[$(date)] idle ${dead_for}s with no downloader and no pipeline" >> "$LOGDIR/watchdog.log"
  if [ "$dead_for" -ge "$DEAD" ]; then
    relaunch_pipeline
    dead_for=0
  fi
done
