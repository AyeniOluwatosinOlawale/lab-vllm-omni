#!/usr/bin/env bash
# =============================================================================
# Phase 13 — Lab 2: OmniInteract Multi-Turn Realtime Duplex Evaluation
#
# WHAT YOU LEARN:
#   OmniInteract evaluates FUNCTIONAL CORRECTNESS of multi-turn duplex
#   conversations. A "turn" is correct if:
#     1. The model responds to the user's audio input appropriately
#     2. The response arrives within the VAD silence boundary
#     3. The model correctly handles barge-in (user interrupting mid-response)
#
# KEY METRICS:
#   turn_accuracy    — % of turns where the model responded correctly
#   audio_rtf        — real-time factor of audio generation
#   session_e2el     — full session end-to-end latency
#   barge_in_rate    — % of barge-ins correctly handled
#
# TWO EXPERIMENTS:
#   A. vllm bench serve with openai-realtime-duplex backend
#   B. Standalone omniinteract evaluator (deeper accuracy metrics)
#
# Outputs: results/p13_lab2_omniinteract/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"
PORT=8091
RESULTS_DIR="$(dirname "$0")/../results/p13_lab2_omniinteract"
mkdir -p "$RESULTS_DIR"

echo "======================================================="
echo " Phase 13 | Lab 2: OmniInteract (Multi-Turn Duplex Eval)"
echo "======================================================="

echo "[1/3] Starting Qwen3-Omni server on port $PORT ..."
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

# ── A. vllm bench serve (throughput + basic metrics) ──────────────────────
echo ""
echo "[2/3] A. vllm bench serve with omniinteract dataset ..."
mkdir -p "$RESULTS_DIR/bench"
vllm bench serve --omni \
  --host localhost --port $PORT \
  --model "$MODEL" \
  --backend openai-realtime-duplex \
  --dataset-name omniinteract \
  --num-prompts 20 \
  --max-concurrency 2 \
  --percentile-metrics audio_ttfp,audio_rtf,e2el \
  --save-result \
  --result-dir "$RESULTS_DIR/bench"

# ── B. Standalone OmniInteract evaluator ──────────────────────────────────
echo ""
echo "[3/3] B. Standalone OmniInteract evaluator (accuracy + barge-in) ..."
mkdir -p "$RESULTS_DIR/accuracy"
python3 "$VLLM_OMNI_DIR/vllm_omni/benchmarks/omniinteract.py" \
  --host localhost \
  --port $PORT \
  --model "$MODEL" \
  --num-sessions 5 \
  --output "$RESULTS_DIR/accuracy/omniinteract_results.json" \
  2>/dev/null || echo "  (standalone evaluator not found; check VLLM_OMNI_DIR)"

echo ""
echo "Lab 2 complete. Results in $RESULTS_DIR"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. turn_accuracy < 80%?"
echo "     YES → VAD boundary detection is cutting off responses too early"
echo "     Fix: adjust VAD silence threshold in --deploy-config"
echo "  2. barge_in_rate < 90%?"
echo "     YES → model is not stopping generation when user speaks"
echo "     This is a server-side VAD configuration issue, not GPU performance"
echo "  3. audio_rtf > 0.8 at c=2?"
echo "     YES → codec is approaching the realtime cliff at this concurrency"
echo "     Reduce to c=1 for guaranteed barge-in responsiveness"
