#!/usr/bin/env bash
# =============================================================================
# Phase 1 — Lab 2: Async Chunk — Stage-Level Pipeline Concurrency
#
# What you learn:
#   async_chunk=true (default) lets Talker + Code2Wav start before the
#   Thinker finishes. Compare TTFP and E2E against Lab 1 to measure the
#   stage-pipelining gain.
#
# Outputs: results/p1_lab2_async_text_audio/
# =============================================================================
set -euo pipefail
MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"
PORT=8091
RESULTS_DIR="$(dirname "$0")/../results"
mkdir -p "$RESULTS_DIR"

echo "======================================================="
echo " Phase 1 | Lab 2: Async Chunk Stage Pipelining"
echo "======================================================="

echo "[1/2] Starting server (async_chunk ON — default MoE config) on port $PORT ..."
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

echo ""
echo "[2/2] Benchmarking text + audio (identical workload to Lab 1 Benchmark B) ..."
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
  --result-dir "$RESULTS_DIR/p1_lab2_async_text_audio"

echo ""
echo "Lab 2 complete."
echo "Compare results/p1_lab1_sync_audio vs results/p1_lab2_async_text_audio"
echo "  Key delta: audio_ttfp_p50 and e2el_p50 should be lower with async chunk."
echo "Next: bash phase1-ar-audio/lab3_all_modalities.sh"
