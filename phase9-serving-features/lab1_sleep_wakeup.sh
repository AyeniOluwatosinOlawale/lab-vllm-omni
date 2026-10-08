#!/usr/bin/env bash
# =============================================================================
# Phase 9 — Lab 1: Sleep / Wakeup API (new in 0.30.0)
#
# WHAT YOU LEARN:
#   vllm-omni 0.30.0 added a sleep mode that frees the GPU memory pool without
#   killing the server process. The model weights are offloaded to CPU/disk;
#   the server stays alive and accepts wakeup calls.
#
# USE CASE:
#   Multi-tenant systems where multiple models share a GPU fleet. When model A
#   is idle, it sleeps → frees memory → model B can load. When A is needed
#   again, it wakes up (weights reload) without a full server restart.
#
# FIVE-STEP EXPERIMENT:
#   1. Baseline: send 5 requests, measure warm latency
#   2. POST /v1/omni/sleep → GPU memory freed (verify with nvidia-smi)
#   3. Request while sleeping → expect 503 or queued response
#   4. POST /v1/omni/wakeup → weights reload, measure cold-start latency
#   5. Send 5 requests again → compare warm vs cold start latency
#
# BOTTLENECK:
#   Wakeup latency = weight transfer bandwidth (PCIe for CPU offload ≈ 32 GB/s,
#   NVLink for multi-GPU ≈ 600 GB/s). A 30B model at BF16 = 60 GB to transfer.
#
# Outputs: results/p9_lab1_sleep_wakeup/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"
PORT=8091
RESULTS_DIR="$(dirname "$0")/../results/p9_lab1_sleep_wakeup"
mkdir -p "$RESULTS_DIR"

SAMPLE_REQUEST='{
  "model": "'"$MODEL"'",
  "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
  "modalities": ["text"],
  "max_tokens": 5
}'

measure_latency() {
  local label=$1
  local start end latency
  start=$(date +%s%3N)
  curl -s "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$SAMPLE_REQUEST" > /dev/null
  end=$(date +%s%3N)
  latency=$((end - start))
  echo "  [$label] latency: ${latency} ms"
  echo "$label,$latency" >> "$RESULTS_DIR/latency_log.csv"
}

echo "======================================================="
echo " Phase 9 | Lab 1: Sleep / Wakeup API"
echo "======================================================="

echo "[1/6] Starting server with --enable-sleep-mode ..."
vllm serve "$MODEL" --omni --enable-sleep-mode --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

echo "request_label,latency_ms" > "$RESULTS_DIR/latency_log.csv"

# Step 1 — warm baseline
echo ""
echo "[2/6] Step 1: Warm baseline (5 requests) ..."
for i in 1 2 3 4 5; do
  measure_latency "warm_$i"
done

# Step 2 — put to sleep
echo ""
echo "[3/6] Step 2: POST /v1/omni/sleep ..."
curl -s -X POST "http://localhost:$PORT/v1/omni/sleep" | python3 -m json.tool
echo "  Checking GPU memory after sleep (expect drop):"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
  | awk '{print "  GPU memory used: "$1" MiB"}' || echo "  (nvidia-smi not available)"

# Step 3 — request while sleeping
echo ""
echo "[4/6] Step 3: Request while sleeping (expect 503 or queued) ..."
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
  "http://localhost:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$SAMPLE_REQUEST")
echo "  HTTP status while sleeping: $HTTP_STATUS"

# Step 4 — wakeup
echo ""
echo "[5/6] Step 4: POST /v1/omni/wakeup (measure cold-start) ..."
WAKE_START=$(date +%s%3N)
curl -s -X POST "http://localhost:$PORT/v1/omni/wakeup" | python3 -m json.tool
WAKE_END=$(date +%s%3N)
echo "  Wakeup API call latency: $((WAKE_END - WAKE_START)) ms"
echo "wakeup_api,$((WAKE_END - WAKE_START))" >> "$RESULTS_DIR/latency_log.csv"

# Step 5 — post-wakeup
echo ""
echo "[6/6] Step 5: Post-wakeup requests (warm vs cold comparison) ..."
for i in 1 2 3 4 5; do
  measure_latency "post_wakeup_$i"
done

echo ""
echo "Lab 1 complete. Results in $RESULTS_DIR"
echo "  Latency log: $RESULTS_DIR/latency_log.csv"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. Wakeup latency (request 1 post-wakeup) vs warm latency?"
echo "     Large gap → weight reload from CPU (PCIe bound ~32 GB/s)"
echo "     Small gap → weights stayed in GPU memory (sleep was a no-op; check --enable-sleep-mode)"
echo "  2. GPU memory drop during sleep?"
echo "     No drop → sleep mode not freeing KV cache; model is too small to observe"
echo "  3. HTTP 503 during sleep?"
echo "     200 → server queued the request and served it after wakeup (correct)"
echo "     503 → server rejected the request (less ideal for seamless hand-off)"
