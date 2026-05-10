#!/bin/bash
# preflight.sh — sanity checks before cbic-ingest starts.
#
# Runs as ExecStartPre in cbic-ingest.service AND manually before any ad-hoc
# ingest_v2.py run. Any non-zero exit aborts start.
#
# === 2026-04-23 REWRITE ===
# Previous version ran COMPONENT probes only (swap, Qdrant, VRAM, model file).
# That missed the 2026-04-23 Ollama-CPU-fallback trap because no probe exercised
# embed-on-real-hardware. This version adds an END-TO-END DRY-RUN that runs a
# single doc through Pass-1 classify + embed + sparse + Qdrant upsert into a
# throwaway collection (`cbic_v2_preflight`) and drops it at the end.
#
# If E2E passes, the real --doc-ids run is safe.
# If E2E fails, the real run would fail the same way — abort now.
#
# Also fixed: check 6 was "no llama-server running" which FALSE-positives because
# qwen3-14b llama-server IS required on GPU 2 port 9082. Now checks that only
# embed-pool GPUs (0,1,4,5,6) are clean.

set -eu

fail() { echo "[preflight] FAIL: $*" >&2; exit 1; }
ok()   { echo "[preflight] ok: $*"; }

STATE_DIR=/opt/indian-legal-ai/state
PAUSE_FILE="$STATE_DIR/ingest.paused"
DISABLED_FILE="$STATE_DIR/ingest.disabled"

# 1. disabled flag (operator brake)
if [ -f "$DISABLED_FILE" ]; then
    fail "$DISABLED_FILE exists — remove it to allow start"
fi
ok "no disabled flag"

# 2. state dir writable
mkdir -p "$STATE_DIR" 2>/dev/null || true
[ -w "$STATE_DIR" ] || fail "state dir not writable: $STATE_DIR"
ok "state dir writable"

# 3. clear any stale pause file from previous run
if [ -f "$PAUSE_FILE" ]; then
    echo "[preflight] removing stale pause file: $PAUSE_FILE"
    rm -f "$PAUSE_FILE"
fi

# 4. swap active (16 GB expected)
swap_total_kb=$(awk '/SwapTotal:/ {print $2}' /proc/meminfo)
if [ "${swap_total_kb:-0}" -lt $((8 * 1024 * 1024)) ]; then
    fail "swap < 8 GB (got ${swap_total_kb} KB). Enable /dev/sda1 swap."
fi
ok "swap active (${swap_total_kb} KB)"

# 5. Qdrant reachable
QURL="${QDRANT_URL:-http://127.0.0.1:6343}"
if ! curl -sf --max-time 5 "$QURL/collections" >/dev/null 2>&1; then
    fail "Qdrant not reachable at $QURL"
fi
ok "Qdrant reachable at $QURL"

# 6. qwen3-14b at LLM_URL (Phase-2 Pass-1 classifier). NOT "no llama-server";
#    qwen3 IS llama-server on GPU 2. We require it UP.
LLM_URL="${LLM_URL:-http://127.0.0.1:9082}"
if ! curl -sf --max-time 5 "$LLM_URL/health" >/dev/null 2>&1 \
  && ! curl -sf --max-time 5 "$LLM_URL/v1/models" >/dev/null 2>&1; then
    fail "qwen3-14b not reachable at $LLM_URL (need it for Pass-1 classify, ~14925 calls in prod run)"
fi
ok "qwen3-14b reachable at $LLM_URL"

# 7. Ollama is NOT running (would fight for VRAM and has been the regression
#    vector — see 2026-04-23 incident).
if pgrep -f 'ollama serve|ollama-embed' >/dev/null 2>&1; then
    fail "ollama is running — stop it. Embedder MUST be llama-cpp Vulkan in-process, not Ollama."
fi
ok "no ollama running"

# 8. GPU VRAM baseline on EMBED pool GPUs only (0,1,4,5,6). Skip GPU 2 (qwen3
#    lives there by design) and GPU 3 (SMU-faulted; never use).
ROCM_SMI=/opt/rocm/bin/rocm-smi
EMBED_GPUS_LIST="${EMBED_GPUS:-0,1,4,5,6}"
if [ -x "$ROCM_SMI" ]; then
    smi_out=$("$ROCM_SMI" --showmeminfo vram 2>/dev/null || true)
    viol=""
    IFS=',' read -ra GPU_ARR <<< "$EMBED_GPUS_LIST"
    for gid in "${GPU_ARR[@]}"; do
        used_bytes=$(echo "$smi_out" | awk -v g="GPU\\[$gid\\]" '
            $0 ~ g && /Used Memory/ { for(i=1;i<=NF;i++) if ($i ~ /^[0-9]+$/) print $i }
        ' | head -1)
        used_mb=$(( ${used_bytes:-0} / 1024 / 1024 ))
        # Threshold: BGE-M3 loaded = ~2.5GB per GPU. Anything >1500MB means
        # another model is ACTIVELY resident. <=1500MB = allocator leftover
        # from a recent teardown (harmless; VRAM will release on next load).
        if [ "$used_mb" -gt 1500 ]; then
            viol="$viol GPU$gid=${used_mb}MB"
        fi
    done
    if [ -n "$viol" ]; then
        fail "VRAM baseline exceeded 500 MB on embed-pool GPUs:$viol — stop the offender"
    fi
    ok "embed-pool GPUs ($EMBED_GPUS_LIST) all under 1500 MB (no foreign model resident)"
else
    echo "[preflight] warn: rocm-smi not found, skipping VRAM check"
fi

# 9. BGE-M3 model blob
MODEL="${EMBED_MODEL_PATH:-/usr/share/ollama/.ollama/models/blobs/sha256-daec91ffb5dd0c27411bd71f29932917c49cf529a641d0168496c3a501e3062c}"
[ -r "$MODEL" ] || fail "BGE-M3 GGUF missing: $MODEL"
ok "BGE-M3 model file readable"

# 10. embedder facade version sentinel (catches Ollama regression of embedder.py)
EMBEDDER_PATH="${EMBEDDER_PATH:-/opt/indian-legal-ai/rag/cbic_rag/embedder.py}"
if ! grep -q '_FACADE_VERSION = "direct-v1"' "$EMBEDDER_PATH"; then
    fail "embedder.py at $EMBEDDER_PATH missing _FACADE_VERSION sentinel — probably regressed to Ollama. Restore from /opt/indian-legal-ai/scripts/cbic_ingest/embedder.py.direct.ref"
fi
ok "embedder.py is the llama-cpp Vulkan facade (direct-v1)"

# 11. llama-cpp-python importable with Vulkan (USER-LOCAL install)
PYTHONPATH_USER="${PYTHONPATH_USER:-/home/user/.local/lib/python3.10/site-packages}"
RAG_DIR="${RAG_DIR:-/opt/indian-legal-ai/rag/cbic_rag}"
PYBIN="${PYBIN:-/usr/bin/python3}"
if ! PYTHONPATH="$PYTHONPATH_USER:$RAG_DIR" "$PYBIN" -c "import llama_cpp; assert llama_cpp.llama_supports_gpu_offload(), 'no GPU offload'" 2>/dev/null; then
    fail "llama-cpp-python not importable or no GPU offload. Check PYTHONPATH=$PYTHONPATH_USER"
fi
ok "llama-cpp-python imports with GPU offload"

# 11b. FORBIDDEN Vulkan BGE-M3 embedder flags audit (gap 2 fix).
#      Per known_good_configs.md these flags crash the BGE-M3 Vulkan pool:
#        --flash-attn              (crashes on Navi 10 Vulkan)
#        --mlock                   (OOMs 16GB host under ingest pressure)
#        --cache-type-k q4_/q5_    (KV-quant crashes Vulkan; q8_0 is fine)
#        --cache-type-v q4_/q5_    (same)
#      Scope: BGE-M3 embedder launch scripts ONLY. qwen3-14b runs on ROCm and
#      uses q8_0 KV quant safely — excluded from this scan.
#      Exclusions: preflight.sh itself (self-references these strings in docs),
#      .bak/.bak.*/archive files.
FORBIDDEN_FA='--flash-attn'
FORBIDDEN_MLOCK='--mlock'
FORBIDDEN_KV_RE='--cache-type-[kv][[:space:]]+q[45]'
# Embedder-specific scan paths (qwen3 ROCm scripts excluded by path).
EMB_SCAN_PATHS=(
    /opt/indian-legal-ai/scripts/cbic_ingest/build-llamacpp-vulkan.sh
    /opt/indian-legal-ai/scripts/cbic_ingest/run-ingest-direct.sh
    /opt/indian-legal-ai/rag/cbic_rag/embedder_direct.py
)
viol=""
for path in "${EMB_SCAN_PATHS[@]}"; do
    [ -f "$path" ] || continue
    # Skip comment lines (^# or ^//) — these are warnings about the flags, not uses.
    active_lines=$(grep -v -E '^\s*(#|//)' "$path" 2>/dev/null || true)
    for pat in "$FORBIDDEN_FA" "$FORBIDDEN_MLOCK"; do
        if echo "$active_lines" | grep -qF -- "$pat"; then
            viol="$viol $path:$pat"
        fi
    done
    if echo "$active_lines" | grep -qE -- "$FORBIDDEN_KV_RE"; then
        viol="$viol $path:KV-quant-q4_or_q5"
    fi
done
if [ -n "$viol" ]; then
    fail "forbidden Vulkan-BGE-M3 flag found in active launch config:$viol (see known_good_configs.md)"
fi
ok "no forbidden Vulkan-BGE-M3 flags in embedder launch configs"

# 11c. Chunker R1–R7 self-tests (gap 3 fix). If chunker_v2 regresses on
#      proviso-split or table-atomic rules, preflight must catch it BEFORE
#      a 25h production run wastes time on bad chunks.
if [ "${SKIP_CHUNKER_TESTS:-0}" != "1" ]; then
    INGEST_DIR="${INGEST_DIR:-/opt/indian-legal-ai/reingest_spec}"
    if [ -f "$INGEST_DIR/test_chunker.py" ]; then
        if ! PYTHONPATH="$PYTHONPATH_USER:$RAG_DIR:$INGEST_DIR" "$PYBIN" \
                -m pytest -x -q "$INGEST_DIR/test_chunker.py" \
                >/tmp/preflight_chunker.log 2>&1; then
            # Fallback: try running as a plain script if pytest not installed.
            if ! PYTHONPATH="$PYTHONPATH_USER:$RAG_DIR:$INGEST_DIR" "$PYBIN" \
                    "$INGEST_DIR/test_chunker.py" >/tmp/preflight_chunker.log 2>&1; then
                echo "=== chunker test log (last 30 lines) ==="
                tail -30 /tmp/preflight_chunker.log >&2 || true
                fail "chunker_v2 self-tests (T1–T8) FAILED — do not ingest with a broken chunker"
            fi
        fi
        ok "chunker_v2 self-tests (T1–T8) passed"
    else
        echo "[preflight] warn: $INGEST_DIR/test_chunker.py missing — skipping T1–T8"
    fi
else
    echo "[preflight] warn: SKIP_CHUNKER_TESTS=1 — skipping T1–T8"
fi

# 12. END-TO-END DRY RUN (added 2026-04-23 — the thing that was missing)
#     Run exactly one tiny real doc through Pass-1 classify + embed + upsert to
#     a throwaway collection. If this works, the 14,925-doc run will too.
if [ "${SKIP_E2E:-0}" != "1" ]; then
    ok "starting E2E dry-run (1 real doc → throwaway Qdrant collection + temp manifest)"
    TS=$(date +%s)
    PREFLIGHT_COLL="cbic_v2_preflight_$TS"
    PREFLIGHT_MANIFEST="/tmp/ingest_manifest_preflight_$TS.sqlite"
    SMOKE_DOC="${PREFLIGHT_DOC:-cbic-notification-msts:1008308}"  # smallest proven doc
    export QDRANT_COLL_V2="$PREFLIGHT_COLL"
    export MANIFEST_V2="$PREFLIGHT_MANIFEST"
    export PYTHONPATH="$PYTHONPATH_USER:$RAG_DIR"
    export EMBED_GPUS="$EMBED_GPUS_LIST"
    export RADV_DEBUG=nodcc GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    INGEST_DIR="${INGEST_DIR:-/opt/indian-legal-ai/reingest_spec}"

    if ! "$PYBIN" -u "$INGEST_DIR/ingest_v2.py" --phase all \
            --doc-ids "$SMOKE_DOC" --no-preflight >/tmp/preflight_e2e.log 2>&1; then
        echo "=== preflight E2E log (last 40 lines) ==="
        tail -40 /tmp/preflight_e2e.log >&2 || true
        rm -f "$PREFLIGHT_MANIFEST"
        curl -sf -X DELETE "$QURL/collections/$PREFLIGHT_COLL" >/dev/null 2>&1 || true
        fail "E2E dry-run FAILED — do NOT launch the real ingest. See /tmp/preflight_e2e.log"
    fi

    # Verify the throwaway collection actually got points
    pts=$(curl -sf --max-time 5 "$QURL/collections/$PREFLIGHT_COLL" \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['result']['points_count'])" 2>/dev/null || echo 0)
    if [ "${pts:-0}" -lt 1 ]; then
        fail "E2E dry-run left 0 points in $PREFLIGHT_COLL — silent-success failure mode"
    fi
    ok "E2E dry-run wrote $pts points to $PREFLIGHT_COLL"

    # Drop the throwaway collection + temp manifest
    curl -sf -X DELETE "$QURL/collections/$PREFLIGHT_COLL" >/dev/null 2>&1 || true
    rm -f "$PREFLIGHT_MANIFEST"
    ok "dropped throwaway collection $PREFLIGHT_COLL + temp manifest"
else
    echo "[preflight] warn: SKIP_E2E=1 — skipping end-to-end dry-run (component-only mode)"
fi

# 13. state log writable
touch "$STATE_DIR/cbic-files.log" 2>/dev/null || fail "cannot write $STATE_DIR/cbic-files.log"
ok "state log writable"

echo "[preflight] all checks passed"
