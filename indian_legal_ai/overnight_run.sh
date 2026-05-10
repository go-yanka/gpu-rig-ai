#!/bin/bash
# overnight_run.sh - wraps dispatcher and auto-swaps RAG API on completion.
#
# nohup bash /opt/indian-legal-ai/scripts/overnight_run.sh \
#       > /opt/indian-legal-ai/logs/overnight.log 2>&1 &

set -u
LOG=/opt/indian-legal-ai/logs/overnight.log
WORK=/opt/indian-legal-ai

log() { echo "[$(date +%H:%M:%S)][OVER] $*"; }

log "starting dispatcher"
python3 $WORK/scripts/dispatcher.py
RC=$?
log "dispatcher exited rc=$RC"

# Only swap if dispatcher actually ran to completion (exit 0).
if [ $RC -ne 0 ]; then
    log "dispatcher did not exit cleanly; leaving rag_api (TF-IDF) in place"
    exit $RC
fi

log "reporting final Qdrant point count"
curl -sS http://localhost:6333/collections/indian_legal_full \
  | python3 -c "import sys,json; r=json.load(sys.stdin)['result']; print('  points=%d status=%s' % (r['points_count'], r['status']))"

log "stopping existing rag_api (TF-IDF) on port 7000"
pkill -f "rag_api.py" || true
sleep 3

log "starting rag_api_qdrant.py on port 7000"
cd $WORK
nohup python3 $WORK/scripts/rag_api_qdrant.py > $WORK/logs/rag_api_qdrant.log 2>&1 &
NEW_PID=$!
log "rag_api_qdrant launched pid=$NEW_PID"
sleep 8

log "probing :7000"
curl -sS -m 5 http://localhost:7000/ 2>&1 | head -c 400
echo
log "overnight wrapper done"
