#!/usr/bin/env bash
# =============================================================================
# Phase 7 — Lab 1: MiniCPM-o 4.5 (graduated from experimental in 0.30.0)
#
# WHAT YOU LEARN:
#   MiniCPM-o 4.5 is a unified omni model that graduated from experimental
#   status in vllm-omni 0.30.0, meaning its pipeline is production-stable.
#   It supports text, image, and audio input with text+audio output — the same
#   any-to-any capability as Qwen3-Omni but with a different architecture.
#
# WHAT IS NEW IN 0.30.0:
#   - Stable production pipeline (no more experimental caveats)
#   - Shared realtime web UI support (--profile minicpmo)
#   - Full duplex session support alongside standard Chat Completions
#
# THREE EXPERIMENTS:
#   A. Text-only chat — baseline TTFT, TPOT
#   B. Image+text input — multimodal input latency
#   C. Audio output — audio_ttfp, audio_rtf (same metrics as Qwen3-Omni)
#
# HARDWARE: 2× 80GB GPU recommended; MiniCPM-o 4.5 fits on 1× 80GB at
#   reduced max_model_len.
#
# Outputs: results/p7_lab1_minicpmo/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="openbmb/MiniCPM-o-4_5"
PORT=8091
RESULTS_DIR="$(dirname "$0")/../results/p7_lab1_minicpmo"
mkdir -p "$RESULTS_DIR"

echo "======================================================="
echo " Phase 7 | Lab 1: MiniCPM-o 4.5"
echo "======================================================="

echo "[1/4] Starting MiniCPM-o 4.5 server on port $PORT ..."
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

# ── A. Text-only chat ──────────────────────────────────────────────────────
echo ""
echo "[2/4] A. Text-only chat (baseline TTFT/TPOT) ..."
mkdir -p "$RESULTS_DIR/text"
vllm bench serve --omni \
  --host localhost --port $PORT \
  --model "$MODEL" \
  --backend openai-chat-omni \
  --dataset-name random \
  --num-prompts 20 --max-concurrency 4 \
  --random-input-len 512 --random-output-len 256 \
  --extra-body '{"modalities":["text"]}' \
  --percentile-metrics ttft,tpot,e2el \
  --save-result \
  --result-dir "$RESULTS_DIR/text"

# ── B. Image+text input ────────────────────────────────────────────────────
echo ""
echo "[3/4] B. Image+text multimodal input ..."
mkdir -p "$RESULTS_DIR/image_input"
vllm bench serve --omni \
  --host localhost --port $PORT \
  --model "$MODEL" \
  --backend openai-chat-omni \
  --dataset-name random-mm \
  --num-prompts 10 --max-concurrency 2 \
  --random-input-len 512 --random-output-len 256 \
  --extra-body '{"modalities":["text"]}' \
  --percentile-metrics ttft,tpot,e2el \
  --save-result \
  --result-dir "$RESULTS_DIR/image_input"

# ── C. Audio output ────────────────────────────────────────────────────────
echo ""
echo "[4/4] C. Text+audio output (audio_ttfp, audio_rtf) ..."
mkdir -p "$RESULTS_DIR/audio_output"
vllm bench serve --omni \
  --host localhost --port $PORT \
  --model "$MODEL" \
  --backend openai-chat-omni \
  --dataset-name random \
  --num-prompts 20 --max-concurrency 4 \
  --random-input-len 512 --random-output-len 512 \
  --extra-body '{"modalities":["text","audio"]}' \
  --percentile-metrics ttft,tpot,e2el,audio_ttfp,audio_rtf \
  --save-result \
  --result-dir "$RESULTS_DIR/audio_output"

echo ""
echo "Lab 1 complete. Results in $RESULTS_DIR"
echo ""
echo "REALTIME UI MODE:"
echo "  python -m examples.online_serving.realtime_web --profile minicpmo --vad"
echo "  (runs in a separate terminal after server is up)"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. Compare text-only TTFT vs Qwen3-Omni from Phase 1."
echo "     MiniCPM-o 4.5 uses a different architecture; TTFT difference"
echo "     reveals which encoder is faster at the same input length."
echo "  2. Does image_input TTFT grow linearly with image resolution?"
echo "     YES → ViT encoder is the bottleneck → use --tensor-parallel-size 2"
echo "  3. Is audio_rtf < 1.0 at c=4?"
echo "     RTF > 1.0 → codec cannot keep up; reduce concurrency or shard codec"
