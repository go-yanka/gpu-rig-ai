#!/bin/bash
# qa_watchdog.sh — continuous quality audit of downloaded PDFs.
#
# Runs qa_pdfs.py --only-new in a loop, auditing any new files as they arrive.
# Results go to /opt/indian-legal-ai/data/scraped/cbic/_qa.sqlite.
# The monitor dashboard reads that DB to show quality status.
#
# Per cycle: check up to 500 new files, then sleep 120s.
# Safe to kill/restart at any time — --only-new skips already-checked files.

set -u
cd /opt/indian-legal-ai/scraper

LOGDIR=/opt/indian-legal-ai/scraper/logs
mkdir -p "$LOGDIR"
LOG="$LOGDIR/qa_watchdog.log"
CYCLE_LIMIT=500
SLEEP_SECS=120

echo "[$(date)] qa_watchdog started (cycle_limit=$CYCLE_LIMIT sleep=${SLEEP_SECS}s)" >> "$LOG"

while true; do
  echo "[$(date)] starting QA cycle" >> "$LOG"
  python3 qa_pdfs.py --only-new --limit "$CYCLE_LIMIT" >> "$LOG" 2>&1
  rc=$?
  echo "[$(date)] QA cycle rc=$rc" >> "$LOG"

  # Short pause so we don't hog IO while downloads are writing files
  sleep "$SLEEP_SECS"
done
