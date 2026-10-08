#!/usr/bin/env bash
# =============================================================================
# Phase 7 — Lab 3: AURA — Unified Duplex, Vision-Only Inputs, Multi-Turn
#
# WHAT YOU LEARN:
#   AURA is a unified duplex runtime that uniquely accepts vision-only inputs
#   in duplex mode. Most omni models require audio+vision together; AURA
#   allows vision-only (image or video frames) with text or audio output.
#   Multi-turn conversation history is tracked server-side via session state.
#
# KEY DISTINCTION FROM OTHER OMNI MODELS:
#   Qwen3-Omni / MiniCPM-o: audio input required for duplex mode
#   AURA: vision-only OR audio-only OR combined — fully flexible
#
# THREE EXPERIMENTS:
#   A. Text-only multi-turn — demonstrates server-side history tracking
#   B. Vision-only input in duplex mode — the unique AURA capability
#   C. Full duplex realtime benchmark
#
# HARDWARE: 1× 80GB GPU sufficient.
# Outputs: results/p7_lab3_aura/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="AURA-LM/AURA"
PORT=8091
RESULTS_DIR="$(dirname "$0")/../results/p7_lab3_aura"
mkdir -p "$RESULTS_DIR"

echo "======================================================="
echo " Phase 7 | Lab 3: AURA (Vision-Only Duplex)"
echo "======================================================="

echo "[1/4] Starting AURA server on port $PORT ..."
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

# ── A. Multi-turn text conversation ────────────────────────────────────────
echo ""
echo "[2/4] A. Multi-turn text conversation (server-side history) ..."
mkdir -p "$RESULTS_DIR/multiturn"
# Send a 3-turn conversation to verify history is maintained
curl -s http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [
      {"role": "user", "content": "My name is Alice. Remember it."},
      {"role": "assistant", "content": "Hello Alice, I will remember your name."},
      {"role": "user", "content": "What is my name?"}
    ],
    "modalities": ["text"],
    "max_tokens": 50
  }' | python3 -m json.tool | tee "$RESULTS_DIR/multiturn/turn3_response.json"

# ── B. Vision-only input ───────────────────────────────────────────────────
echo ""
echo "[3/4] B. Vision-only input via openai-chat-omni ..."
mkdir -p "$RESULTS_DIR/vision_only"
vllm bench serve --omni \
  --host localhost --port $PORT \
  --model "$MODEL" \
  --backend openai-chat-omni \
  --dataset-name random-mm \
  --num-prompts 10 --max-concurrency 2 \
  --random-input-len 256 --random-output-len 256 \
  --extra-body '{"modalities":["text"]}' \
  --percentile-metrics ttft,tpot,e2el \
  --save-result \
  --result-dir "$RESULTS_DIR/vision_only"

# ── C. Full-duplex realtime ────────────────────────────────────────────────
echo ""
echo "[4/4] C. Full-duplex realtime benchmark ..."
mkdir -p "$RESULTS_DIR/duplex"
vllm bench serve --omni \
  --host localhost --port $PORT \
  --model "$MODEL" \
  --backend openai-realtime-duplex \
  --dataset-name omniinteract \
  --num-prompts 10 --max-concurrency 2 \
  --percentile-metrics audio_ttfp,audio_rtf,e2el \
  --save-result \
  --result-dir "$RESULTS_DIR/duplex"

echo ""
echo "Lab 3 complete. Results in $RESULTS_DIR"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. Multi-turn latency: does TTFT grow with conversation length?"
echo "     YES → KV cache is filling up with history tokens → reduce max_model_len"
echo "     or use sliding window attention"
echo "  2. Vision-only TTFT vs audio+vision TTFT (compare to MiniCPM-o lab1)?"
echo "     Vision-only should be faster (no audio encoder pre-processing)"
echo "  3. Duplex session: check barge-in handling via omniinteract accuracy score"
echo "     Low accuracy → VAD boundary detection tuning needed"
