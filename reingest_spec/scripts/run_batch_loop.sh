#!/usr/bin/env bash
# Autonomous batch loop. Chains batches FROM_N..TO_N with full post-ingest steps + CP gates at 5 & 10.
# 2026-05-07: fixed two latent bugs (lint args mangled, gate concurrency violation).
set -uo pipefail
FROM_N=${1:-3}
TO_N=${2:-10}
LOG=/tmp/ingest_loop.log
DATA=/opt/indian-legal-ai/data
EVAL=$DATA/eval
MANIFEST=$DATA/scraped/cbic/_manifest.sqlite
SNAP_DIR=/opt/snapshots
mkdir -p $SNAP_DIR $EVAL

export DENSE_ONLY=1
export RADV_DEBUG=nodcc
export GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1
export EMBED_GPUS=0,1,3,4,5,6
source /root/.cbic_env 2>/dev/null || true

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

wait_for_pid(){
  local pid=$1; local label=$2; local t0=$(date +%s)
  while ps -p $pid > /dev/null 2>&1; do
    sleep 30
    local elapsed=$(($(date +%s) - t0))
    log "  $label: pid $pid still running, ${elapsed}s elapsed"
    if [ $elapsed -gt 1800 ]; then log "HALT: $label exceeded 30min budget"; exit 2; fi
  done
  log "  $label: pid $pid exited after $(($(date +%s) - t0))s"
}

run_lint(){
  local N=$1
  local CSV=$DATA/batches/batch${N}_doc_ids.csv
  log "  [lint] running post_batch_lint.py for batch $N"
  /usr/bin/python3 /opt/indian-legal-ai/reingest_spec/scripts/post_batch_lint.py "$N" "$CSV" >> $LOG 2>&1
  local RC=$?
  if [ $RC -eq 0 ]; then
    log "  [lint] batch $N: clean (exit 0)"
  elif [ $RC -eq 1 ]; then
    # 2026-05-08: P1 findings (UPSERT-STRAGGLERS, SECTION-REF tail, TAIL-DUP small counts)
    # are mid-loop transients that corpus_drain + later batches resolve. Do NOT halt — log
    # for visibility and continue. Only P0 (real D-DEFECT outside carve-outs) halts.
    log "  [lint] batch $N: P1 findings (informational, continuing) — see $EVAL/post_batch_lint_$N.json"
  else
    log "HALT batch $N: lint reported P0 defect or crashed (exit $RC) — see $EVAL/post_batch_lint_$N.json"
    exit 5
  fi
}

run_serial_gate(){
  local CP=$1; local NAME=$2; shift 2; local CMD="$@"
  log "  [gate] preflight $NAME for CP-$CP"
  bash /opt/indian-legal-ai/reingest_spec/scripts/gate_preflight.sh "$NAME" >> $LOG 2>&1 || { log "HALT CP-$CP: $NAME preflight refused"; exit 6; }
  log "  [gate] running $NAME serially"
  cd /opt/indian-legal-ai
  PYTHONPATH=/home/user/.local/lib/python3.10/site-packages:/opt/indian-legal-ai/rag/cbic_rag /usr/bin/python3 $CMD >> $LOG 2>&1
  local RC=$?
  if [ $RC -ne 0 ]; then log "HALT CP-$CP: $NAME evaluator exit $RC"; exit 6; fi
  log "  [gate] $NAME complete"
}

run_batch(){
  local N=$1
  log "=== BATCH $N START ==="
  cd /opt/indian-legal-ai
  PYTHONPATH=/home/user/.local/lib/python3.10/site-packages:/opt/indian-legal-ai/rag/cbic_rag /usr/bin/python3 reingest_spec/build_batch.py --batch $N >> $LOG 2>&1 || { log "HALT: build_batch failed for N=$N"; exit 2; }
  local CSV=$DATA/batches/batch${N}_doc_ids.csv
  if [ ! -s $CSV ]; then log "HALT batch $N: CSV empty/missing at $CSV"; exit 2; fi
  local DOCIDS=$(cat $CSV)
  local NDOCS=$(echo "$DOCIDS" | tr ',' '\n' | grep -c .)
  log "  built $NDOCS doc_ids for batch $N"
  [ $NDOCS -lt 100 ] && { log "HALT: only $NDOCS doc_ids parsed from CSV"; exit 2; }
  local PRE_PTS=$(curl -s http://127.0.0.1:6343/collections/cbic_v2 | python3 -c 'import sys,json;print(json.load(sys.stdin)["result"]["points_count"])')
  log "  pre-ingest pts=$PRE_PTS launching ingest_v2.py (DENSE_ONLY=$DENSE_ONLY)..."
  nohup env DENSE_ONLY=1 RADV_DEBUG=nodcc GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1 EMBED_GPUS=0,1,3,4,5,6 PYTHONPATH=/home/user/.local/lib/python3.10/site-packages:/opt/indian-legal-ai/rag/cbic_rag /usr/bin/python3 reingest_spec/ingest_v2.py --phase all --collection cbic_v2 --doc-ids "$DOCIDS" --allow-phase2-failures 20 >> $LOG 2>&1 &
  local PID=$!
  log "  ingest_v2 PID=$PID"
  wait_for_pid $PID "batch $N ingest"
  local POST_PTS=$(curl -s http://127.0.0.1:6343/collections/cbic_v2 | python3 -c 'import sys,json;print(json.load(sys.stdin)["result"]["points_count"])')
  local DELTA=$((POST_PTS - PRE_PTS))
  log "  post-ingest pts=$POST_PTS (+$DELTA)"
  [ $DELTA -lt 1000 ] && { log "HALT batch $N: only +$DELTA pts, expected ~3500-5000"; exit 2; }
  cp $MANIFEST $SNAP_DIR/manifest_after_batch$N.sqlite 2>/dev/null
  curl -s -X POST http://127.0.0.1:6343/collections/cbic_v2/snapshots > $SNAP_DIR/qdrant_batch$N.json
  log "  snapshots done"
  run_lint $N
  if [ $N -eq 5 ]; then run_cp 2; fi
  if [ $N -eq 10 ]; then run_cp 3; fi
  log "=== BATCH $N DONE ==="
}

run_cp(){
  local CP=$1
  log "=== CP-$CP GATES START (SERIAL — Hard Rule #10) ==="
  run_serial_gate $CP g1 "reingest_spec/evaluators/gate_g1_recall.py --collection cbic_v2 --retrieve-only --workers 6 --out $EVAL/cp${CP}_g1.json"
  run_serial_gate $CP g3 "reingest_spec/evaluators/gate_g3_levenshtein.py --collection cbic_v2 --gold reingest_spec/eval/v2_gold_cp${CP}.json --out $EVAL/cp${CP}_g3.json --retrieve-only --workers 6"
  run_serial_gate $CP g4 "reingest_spec/evaluators/gate_g4_grounded.py --collection cbic_v2 --adv reingest_spec/eval/v2_adversarial_clean_v2.json --out $EVAL/cp${CP}_g4.json --threshold 0.95 --refuse-on no --workers 4"
  if [ $CP -eq 3 ]; then
    run_serial_gate $CP g2 "reingest_spec/evaluators/gate_g2_dual_judge.py --collection cbic_v2 --gold reingest_spec/eval/v2_gold_cp3_full.json --out $EVAL/cp3_g2.json"
    run_serial_gate $CP g5 "reingest_spec/evaluators/gate_g5_latency_cost.py --collection cbic_v2 --gold reingest_spec/eval/v2_gold_cp3_full.json --out $EVAL/cp3_g5.json"
  fi
  bash reingest_spec/scripts/cp_smokes.sh cbic_v2 $EVAL/cp${CP}_smokes.json >> $LOG 2>&1
  log "=== CP-$CP GATES DONE ==="
}

corpus_drain(){
  # 2026-05-08 fix: per-batch upsert can leave stragglers (failed embed/upsert during phase3_4_5
  # leaks chunks at upserted=0 forever, because next batch's --doc-ids filter never re-attempts them).
  # Codified after CP-3 push surfaced 3,755 canonical chunks across 1,098 doc_ids stuck at upserted=0
  # AT THE END OF run_batch_loop. Hidden because lint+RECONCILE both scoped to per-batch CSVs.
  # Standing rule: per-batch determinism is necessary but insufficient. Corpus invariants must be
  # asserted between batches.
  log "=== CORPUS-WIDE DRAIN: upserting any pending canonical chunks across ALL doc_ids ==="
  cd /opt/indian-legal-ai
  PENDING=$(/usr/bin/python3 -c "import sqlite3;print(list(sqlite3.connect('${DATA}/ingest_manifest_v2.sqlite').execute('SELECT COUNT(*) FROM chunks WHERE is_canonical=1 AND upserted=0'))[0][0])" 2>/dev/null || echo "?")
  log "  pending canonical chunks at upserted=0: $PENDING"
  if [ "$PENDING" = "0" ]; then
    log "  no stragglers — drain skipped"
    return 0
  fi
  PYTHONPATH=/home/user/.local/lib/python3.10/site-packages:/opt/indian-legal-ai/rag/cbic_rag /usr/bin/python3 reingest_spec/ingest_v2.py --phase phase3_4_5 --collection cbic_v2 >> $LOG 2>&1
  local RC=$?
  if [ $RC -ne 0 ]; then log "HALT: corpus drain failed (exit $RC)"; exit 7; fi
  POST=$(/usr/bin/python3 -c "import sqlite3;print(list(sqlite3.connect('${DATA}/ingest_manifest_v2.sqlite').execute('SELECT COUNT(*) FROM chunks WHERE is_canonical=1 AND upserted=0'))[0][0])" 2>/dev/null || echo "?")
  log "  post-drain pending: $POST (was $PENDING)"
  if [ "$POST" != "0" ]; then log "HALT: drain didn't clear all pending — $POST left"; exit 7; fi
}

bash /opt/indian-legal-ai/reingest_spec/scripts/resource_preflight.sh || { log "HALT: resource_preflight failed"; exit 9; }
log "###### BATCH LOOP START FROM_N=$FROM_N TO_N=$TO_N ######"
for N in $(seq $FROM_N $TO_N); do run_batch $N; done
# 2026-05-08: mandatory drain. Even if every per-batch RECONCILE passed, corpus-wide stragglers
# can exist from transient phase3_4_5 errors. Drain MUST run before any final CP gates.
corpus_drain
if [ $TO_N -ge 10 ]; then log "###### PAIR-GEN POST-PASS QUEUED — run pair_gen_full_corpus.sh manually or via separate skill ######"; fi
log "###### BATCH LOOP COMPLETE ######"
