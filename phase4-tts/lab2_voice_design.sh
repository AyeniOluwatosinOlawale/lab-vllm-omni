#!/usr/bin/env bash
# =============================================================================
# Phase 4 — Lab 2: Voice Design
#
# What you learn:
#   Instead of cloning a reference voice, the model generates a novel voice
#   described in natural language ("warm, slightly husky female voice with
#   a calm pace"). Compare TTFP and RTF against voice clone from Lab 1 —
#   voice design uses a different checkpoint (VoiceDesign vs Base).
#
# Outputs: results/p4_lab2_voice_design/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
PORT=8000
RESULTS_DIR="$(dirname "$0")/../results/p4_lab2_voice_design"
mkdir -p "$RESULTS_DIR"

echo "======================================================="
echo " Phase 4 | Lab 2: TTS Voice Design"
echo "======================================================="

echo "[1/2] Starting TTS VoiceDesign server on port $PORT ..."
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

echo ""
echo "[2/2] Benchmarking voice design (concurrency=1, 20 prompts) ..."
vllm bench serve --omni \
  --host 127.0.0.1 --port $PORT \
  --model "$MODEL" \
  --backend openai-audio-speech \
  --endpoint /v1/audio/speech \
  --dataset-name seed-tts-design \
  --dataset-path "$VLLM_OMNI_DIR/benchmarks/build_dataset/seed_tts_design" \
  --seed-tts-locale en \
  --num-prompts 20 \
  --num-warmups 2 \
  --extra-body '{"task_type":"VoiceDesign","language":"English"}' \
  --max-concurrency 1 \
  --request-rate inf \
  --percentile-metrics ttft,e2el,audio_rtf,audio_ttfp,audio_duration,audio_underrun \
  --save-result \
  --result-dir "$RESULTS_DIR"

echo ""
echo "Lab 2 complete. Results in $RESULTS_DIR"
echo "Next: bash phase4-tts/lab3_concurrency_cliff.sh"
