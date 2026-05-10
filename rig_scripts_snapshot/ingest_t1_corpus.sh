#!/usr/bin/env bash
# Orchestrate T1 corpus ingestion via existing parallel_ingest.py
# One category -> one dataset label. Tier 1 = foundational statutes.
# Writes per-category log + summary log.
set -u
STAGE="/opt/indian-legal-ai/gst_stage"
SCRIPT="/opt/indian-legal-ai/scripts/parallel_ingest.py"
LOGDIR="/opt/indian-legal-ai/logs/t1_ingest_$(date +%Y%m%d_%H%M%S)"
SUMMARY="$LOGDIR/summary.log"
mkdir -p "$LOGDIR"

echo "T1 INGESTION START $(date)" | tee "$SUMMARY"
echo "logdir=$LOGDIR" | tee -a "$SUMMARY"

# Category label map (dataset key = label shown in payload)
# Iterate any t1_* dir containing PDFs
total_dirs=0
total_pdfs=0
ok_dirs=0
fail_dirs=()

for d in "$STAGE"/t1_*/; do
    [[ -d "$d" ]] || continue
    pdfs=$(ls "$d"*.pdf 2>/dev/null | wc -l)
    if [[ "$pdfs" -eq 0 ]]; then
        echo "SKIP (no pdfs) $(basename $d)" | tee -a "$SUMMARY"
        continue
    fi
    name=$(basename "$d")
    total_dirs=$((total_dirs+1))
    total_pdfs=$((total_pdfs+pdfs))
    log="$LOGDIR/${name}.log"
    echo "---- $(date +%H:%M:%S) START $name ($pdfs pdfs) ----" | tee -a "$SUMMARY"
    # Label prefix: human-readable category name derived from dir name
    label_prefix="${name#t1_}"
    # Run foreground per-category; parallel_ingest itself parallelises batches
    if python3 "$SCRIPT" --dir "$d" --label "$label_prefix" --dataset "$name" --tier 1 >"$log" 2>&1; then
        chunks=$(grep -oE 'DONE [0-9]+/' "$log" | tail -1 | grep -oE '[0-9]+' | head -1)
        echo "  [OK] $name chunks=$chunks log=$log" | tee -a "$SUMMARY"
        ok_dirs=$((ok_dirs+1))
    else
        echo "  [FAIL] $name (see $log)" | tee -a "$SUMMARY"
        fail_dirs+=("$name")
    fi
done

echo "" | tee -a "$SUMMARY"
echo "T1 INGESTION COMPLETE $(date)" | tee -a "$SUMMARY"
echo "dirs_ok=$ok_dirs / $total_dirs  pdfs=$total_pdfs" | tee -a "$SUMMARY"
if (( ${#fail_dirs[@]} )); then
    echo "FAILED: ${fail_dirs[*]}" | tee -a "$SUMMARY"
fi

# Post-run Qdrant counts
echo "" | tee -a "$SUMMARY"
echo "=== Qdrant point counts ===" | tee -a "$SUMMARY"
curl -s -X POST http://localhost:6333/collections/indian_legal_full/points/count \
    -H 'content-type: application/json' -d '{"exact":true}' | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
for d in "$STAGE"/t1_*/; do
    [[ -d "$d" ]] || continue
    name=$(basename "$d")
    cnt=$(curl -s -X POST http://localhost:6333/collections/indian_legal_full/points/count \
        -H 'content-type: application/json' \
        -d "{\"exact\":true,\"filter\":{\"must\":[{\"key\":\"dataset\",\"match\":{\"value\":\"$name\"}}]}}" \
        | python3 -c 'import sys,json; print(json.load(sys.stdin)["result"]["count"])' 2>/dev/null)
    printf "  %-36s %s\n" "$name" "${cnt:-?}" | tee -a "$SUMMARY"
done
