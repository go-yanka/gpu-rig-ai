#!/bin/bash
cd /opt/indian-legal-ai/scraper
LOG=logs/cbic_pipeline.log

run_scope () {
  local scope=$1
  local logf=logs/cbic_download_${scope}.log
  local tries=0
  while [ $tries -lt 8 ]; do
    tries=$((tries+1))
    echo "[$(date)] scope=$scope attempt=$tries" >> $LOG
    python3 run_scrape.py cbic --stage download --scope $scope --root /opt/indian-legal-ai/data/scraped >> $logf 2>&1
    rc=$?
    echo "[$(date)] scope=$scope attempt=$tries rc=$rc" >> $LOG
    if tail -n 20 $logf | grep -qE 'elapsed_s|ALL DONE|"total"'; then
      echo "[$(date)] scope=$scope looks complete; moving on" >> $LOG
      break
    fi
    sleep 10
  done
}

echo "[$(date)] waiting for any in-flight GST run to finish..." >> $LOG
while pgrep -f "run_scrape.py cbic --stage download --scope gst" > /dev/null; do sleep 20; done

run_scope customs
run_scope central_excise
run_scope service_tax
run_scope others
run_scope hsn_cess

echo "[$(date)] ALL DONE" >> $LOG
