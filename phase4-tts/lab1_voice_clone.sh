#!/usr/bin/env bash
# =============================================================================
# Phase 4 — Lab 1: TTS Voice Clone Baseline
#
# What you learn:
#   Pure audio generation pipeline. The model takes text + a reference
#   audio clip and synthesises speech matching that voice.
#   Key metrics: TTFP (time to first audio packet), RTF (real-time factor),
#   audio_underrun (streaming continuity under a simulated realtime player).
#
#   RTF < 1.0  means the server generates audio faster than realtime.
#   audio_underrun > 0 means at least one gap would be audible to a listener.
#
# Outputs: results/p4_lab1_voice_clone/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
PORT=8000
RESULTS_DIR="$(dirname "$0")/../results/p4_lab1_voice_clone"
mkdir -p "$RESULTS_DIR"

echo "======================================================="
echo " Phase 4 | Lab 1: TTS Voice Clone Baseline"
echo "======================================================="

echo "[1/2] Starting TTS server on port $PORT ..."
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

echo ""
echo "[2/2] Benchmarking voice clone (concurrency=1, 20 prompts) ..."
vllm bench serve --omni \
  --host 127.0.0.1 --port $PORT \
  --model "$MODEL" \
  --backend openai-audio-speech \
  --endpoint /v1/audio/speech \
  --dataset-name seed-tts-text \
  --dataset-path "$VLLM_OMNI_DIR/benchmarks/build_dataset/seed_tts_smoke" \
  --seed-tts-locale en \
  --num-prompts 20 \
  --num-warmups 2 \
  --extra-body '{"task_type":"Base"}' \
  --max-concurrency 1 \
  --request-rate inf \
  --percentile-metrics ttft,e2el,audio_rtf,audio_ttfp,audio_duration,audio_underrun \
  --save-result \
  --result-dir "$RESULTS_DIR"

echo ""
echo "Lab 1 complete. Results in $RESULTS_DIR"
echo "  Check: Median AUDIO_RTF should be < 1.0 (faster than realtime)"
echo "  Check: AUDIO_UNDERRUN should be close to 0 at c=1"
echo "Next: bash phase4-tts/lab2_voice_design.sh"
