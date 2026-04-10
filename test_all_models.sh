#!/usr/bin/env bash
# Systematic model load/verify/unload test for all configured models
# Run on the rig: sudo bash /tmp/test_all_models.sh

BINARY="/opt/llama-server/llama-server-rocm"
MODEL_DIR="/opt/ai-models"
RESULTS_FILE="/tmp/model_test_results.txt"
> "$RESULTS_FILE"

# Test prompt — factual, verifiable
TEST_PROMPT='{"messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":20,"temperature":0}'

log() { echo "$(date '+%H:%M:%S') $*"; echo "$(date '+%H:%M:%S') $*" >> "$RESULTS_FILE"; }

test_model() {
    local NAME="$1"
    local FILE="$2"
    local PORT="$3"
    local GPU="$4"
    local FLAGS="$5"
    local MODE="$6"
    local EXPECTED_CTX="$7"

    log "═══════════════════════════════════════════════"
    log "TEST: $NAME mode=$MODE on GPU $GPU (port $PORT)"
    log "FLAGS: $FLAGS"
    log "═══════════════════════════════════════════════"

    # Kill anything on this port
    PID=$(lsof -ti :$PORT 2>/dev/null)
    if [ -n "$PID" ]; then kill $PID 2>/dev/null; sleep 2; fi

    # Launch
    log "LAUNCHING..."
    GGML_VK_VISIBLE_DEVICES=$GPU RADV_DEBUG=nodcc LD_LIBRARY_PATH=/opt/llama-server $BINARY --model "$MODEL_DIR/$FILE" --host 0.0.0.0 --port $PORT $FLAGS > /tmp/gpu${GPU}_out.log 2>/tmp/gpu${GPU}_err.log &
    LAUNCH_PID=$!
    log "PID: $LAUNCH_PID"

    # Wait for ready (up to 240s for slow BAR GPUs)
    READY=0
    for i in $(seq 1 48); do
        sleep 5
        ELAPSED=$((i * 5))
        # Check if process died
        if ! kill -0 $LAUNCH_PID 2>/dev/null; then
            log "FAIL: Process died after ${ELAPSED}s"
            log "STDERR: $(tail -5 /tmp/gpu${GPU}_err.log)"
            echo "$NAME|$MODE|GPU$GPU|CRASH|${ELAPSED}s|0|Process died" >> "$RESULTS_FILE"
            return 1
        fi
        # Check health
        HTTP=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/health 2>/dev/null)
        PSTATE=$(cat /proc/$LAUNCH_PID/status 2>/dev/null | grep '^State:' | awk '{print $2}')
        if [ "$HTTP" = "200" ]; then
            log "READY after ${ELAPSED}s (state=$PSTATE)"
            READY=1
            break
        fi
        # Progress every 15s
        if [ $((i % 3)) -eq 0 ]; then
            log "  waiting... ${ELAPSED}s health=$HTTP state=$PSTATE"
        fi
    done

    if [ $READY -eq 0 ]; then
        log "FAIL: Not ready after 240s"
        log "STDERR: $(tail -5 /tmp/gpu${GPU}_err.log)"
        kill $LAUNCH_PID 2>/dev/null
        echo "$NAME|$MODE|GPU$GPU|TIMEOUT|240s|0|Never reached ready" >> "$RESULTS_FILE"
        return 1
    fi

    LOAD_TIME=$ELAPSED

    # Send test prompt
    log "TESTING inference..."
    RESP=$(curl -s --max-time 30 -X POST http://localhost:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "$TEST_PROMPT" 2>/dev/null)

    if [ -z "$RESP" ]; then
        log "FAIL: No response from inference"
        kill $LAUNCH_PID 2>/dev/null
        echo "$NAME|$MODE|GPU$GPU|NO_RESPONSE|${LOAD_TIME}s|0|Inference returned empty" >> "$RESULTS_FILE"
        return 1
    fi

    # Extract content and speed
    CONTENT=$(echo "$RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])" 2>/dev/null)
    SPEED=$(echo "$RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('timings',{}).get('predicted_per_second',0))" 2>/dev/null)
    TOKENS=$(echo "$RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('usage',{}).get('completion_tokens',0))" 2>/dev/null)

    # Check if answer contains "Paris"
    if echo "$CONTENT" | grep -qi "paris"; then
        log "PASS: Response='$CONTENT' Speed=${SPEED} tok/s Tokens=$TOKENS"
        VERDICT="PASS"
    else
        log "WARN: Response='$CONTENT' (expected Paris) Speed=${SPEED} tok/s"
        VERDICT="WRONG_ANSWER"
    fi

    # Unload
    log "UNLOADING..."
    kill $LAUNCH_PID 2>/dev/null
    sleep 2
    # Verify unloaded
    if lsof -ti :$PORT > /dev/null 2>&1; then
        kill -9 $(lsof -ti :$PORT) 2>/dev/null
        log "WARN: Had to force-kill"
        sleep 1
    fi
    log "UNLOADED"

    echo "$NAME|$MODE|GPU$GPU|$VERDICT|${LOAD_TIME}s|${SPEED}|$CONTENT" >> "$RESULTS_FILE"
    log ""
    return 0
}

log "Starting systematic model tests at $(date)"
log ""

# ═══ GEMMA4 FAST MODE — GPU 0 (8GB) ═══
test_model "gemma4" "gemma-4-E4B-it-Q4_K_M.gguf" 9080 0 \
    "-ngl 99 -c 8192 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 --mmap -b 512 -ub 256 -t 4" \
    "fast" 8192

# ═══ GEMMA4 16K MODE — GPU 0 (8GB) ═══
test_model "gemma4" "gemma-4-E4B-it-Q4_K_M.gguf" 9080 0 \
    "-ngl 99 -c 16384 --parallel 1 --cache-ram 0 --mmap -b 2048 -ub 512 -t 4 --flash-attn off" \
    "16k" 16384

# ═══ QWEN 3.5 4B — GPU 0 (8GB) ═══
test_model "qwen35-4b" "qwen3.5-4b-q4_k_m.gguf" 9080 0 \
    "-ngl 99 -c 16384 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 --mmap -b 2048 -ub 512 -t 4" \
    "default" 16384

# ═══ QWEN 3.5 9B — GPU 0 (8GB, 4K context) ═══
test_model "qwen35-9b" "qwen3.5-9b-q4_k_m.gguf" 9080 0 \
    "-ngl 99 -c 4096 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 --mmap -b 2048 -ub 512 -t 4" \
    "default-8gb" 4096

# ═══ QWEN 3.5 9B — GPU 2 (12GB, 16K context) ═══
test_model "qwen35-9b" "qwen3.5-9b-q4_k_m.gguf" 9082 2 \
    "-ngl 99 -c 16384 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 --mmap -b 2048 -ub 512 -t 4" \
    "default-12gb" 16384

# ═══ MISTRAL NEMO 12B — GPU 2 (12GB only) ═══
test_model "mistral-nemo" "mistral-nemo-12b-instruct-q4_k_m.gguf" 9082 2 \
    "-ngl 99 -c 8192 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 --mmap -b 2048 -ub 512 -t 4" \
    "default" 8192

# ═══ QWEN 2.5 CODER 7B — GPU 0 (8GB) ═══
test_model "qwen25-coder" "qwen2.5-coder-7b-instruct-q4_k_m.gguf" 9080 0 \
    "-ngl 99 -c 16384 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 --mmap -b 2048 -ub 512 -t 4" \
    "default" 16384

# ═══ SUMMARY ═══
log ""
log "═══════════════════════════════════════════════"
log "SUMMARY"
log "═══════════════════════════════════════════════"
grep '|' "$RESULTS_FILE" | while IFS='|' read NAME MODE GPU RESULT LOADTIME SPEED CONTENT; do
    printf "%-15s %-12s %-6s %-12s %-8s %8s tok/s  %s\n" "$NAME" "$MODE" "$GPU" "$RESULT" "$LOADTIME" "$SPEED" "$CONTENT"
done

log ""
log "Done at $(date)"
