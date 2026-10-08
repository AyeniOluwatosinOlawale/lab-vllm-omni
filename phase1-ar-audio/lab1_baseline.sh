#!/usr/bin/env bash
# =============================================================================
# Phase 1 — Lab 1: Baseline Unified Sync Serving
#
# What you learn:
#   Reference TTFT, E2E latency, and RTF with async_chunk OFF.
#   This is the comparison point for Lab 2 (async chunk gain).
#
# Outputs: results/p1_lab1_sync_text/  results/p1_lab1_sync_audio/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"
PORT=8091
RESULTS_DIR="$(dirname "$0")/../results"
mkdir -p "$RESULTS_DIR"

echo "======================================================="
echo " Phase 1 | Lab 1: Baseline — Unified Sync Serving"
echo "======================================================="

# Start server with async_chunk disabled
echo "[1/3] Starting server (sync, no pipeline overlap) on port $PORT ..."
vllm serve "$MODEL" --omni --no-async-chunk --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

echo "  Waiting for health ..."
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

# Benchmark A: text-only output
echo ""
echo "[2/3] Benchmarking text-only output ..."
vllm bench serve --omni \
  --host localhost --port $PORT \
  --model "$MODEL" \
  --backend openai-chat-omni \
  --dataset-name random \
  --num-prompts 20 \
  --max-concurrency 4 \
  --random-input-len 2500 \
  --random-output-len 900 \
  --ignore-eos \
  --extra-body '{"modalities":["text"]}' \
  --percentile-metrics ttft,tpot,itl,e2el \
  --save-result \
  --result-dir "$RESULTS_DIR/p1_lab1_sync_text"

# Benchmark B: text + audio output
echo ""
echo "[3/3] Benchmarking text + audio output ..."
vllm bench serve --omni \
  --host localhost --port $PORT \
  --model "$MODEL" \
  --backend openai-chat-omni \
  --dataset-name random \
  --num-prompts 20 \
  --max-concurrency 4 \
  --random-input-len 2500 \
  --random-output-len 900 \
  --ignore-eos \
  --extra-body '{"modalities":["text","audio"]}' \
  --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf \
  --save-result \
  --result-dir "$RESULTS_DIR/p1_lab1_sync_audio"

echo ""
echo "Lab 1 complete. Results saved to $RESULTS_DIR/p1_lab1_*"
echo "Next: bash phase1-ar-audio/lab2_async_chunk.sh"
