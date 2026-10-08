#!/usr/bin/env bash
# =============================================================================
# Phase 13 — Lab 1: Daily-Omni Video QA Benchmark
#
# WHAT YOU LEARN:
#   Daily-Omni is a video question-answering benchmark with both visual and
#   audio questions about real-world daily activities. It tests QUALITY (answer
#   correctness) AND LATENCY simultaneously — unlike random synthetic datasets
#   which only measure serving speed.
#
# KEY DIFFERENCE FROM SYNTHETIC BENCHMARKS:
#   random / random-mm datasets: random tokens, no ground truth, latency only
#   daily-omni dataset:          real video QA, ground truth answers, accuracy+latency
#
# TWO EXPERIMENTS:
#   A. Text output on video QA prompts
#   B. Audio output on the same prompts — reveals if audio output degrades accuracy
#      (it shouldn't, but codec latency affects when answers arrive)
#
# Outputs: results/p13_lab1_daily_omni/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"
PORT=8091
RESULTS_DIR="$(dirname "$0")/../results/p13_lab1_daily_omni"
mkdir -p "$RESULTS_DIR"

echo "======================================================="
echo " Phase 13 | Lab 1: Daily-Omni Video QA Benchmark"
echo "======================================================="

echo "[1/3] Starting Qwen3-Omni server on port $PORT ..."
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

# ── A. Text output ─────────────────────────────────────────────────────────
echo ""
echo "[2/3] A. Daily-Omni text output ..."
mkdir -p "$RESULTS_DIR/text"
vllm bench serve --omni \
  --host localhost --port $PORT \
  --model "$MODEL" \
  --backend daily-omni \
  --dataset-name daily-omni \
  --num-prompts 50 \
  --max-concurrency 4 \
  --extra-body '{"modalities":["text"]}' \
  --percentile-metrics ttft,tpot,e2el \
  --save-result \
  --result-dir "$RESULTS_DIR/text"

# ── B. Audio output ────────────────────────────────────────────────────────
echo ""
echo "[3/3] B. Daily-Omni audio output ..."
mkdir -p "$RESULTS_DIR/audio"
vllm bench serve --omni \
  --host localhost --port $PORT \
  --model "$MODEL" \
  --backend openai-chat-omni \
  --dataset-name daily-omni \
  --num-prompts 50 \
  --max-concurrency 4 \
  --extra-body '{"modalities":["text","audio"]}' \
  --percentile-metrics ttft,tpot,e2el,audio_ttfp,audio_rtf \
  --save-result \
  --result-dir "$RESULTS_DIR/audio"

echo ""
echo "Lab 1 complete. Results in $RESULTS_DIR"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. Text vs audio E2E latency on same prompts?"
echo "     Audio E2E overhead = audio_e2e - text_e2e → codec + streaming cost"
echo "  2. Does accuracy differ between text and audio output?"
echo "     YES → audio codec is affecting semantic content (unexpected/bug)"
echo "     NO  → audio output is a lossless channel for the answer"
echo "  3. E2E latency on Daily-Omni vs random dataset (Phase 1 lab1)?"
echo "     Daily-Omni may have longer outputs (real answers vs random tokens)"
echo "     Compare output_token_count to understand the length distribution"
