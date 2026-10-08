#!/usr/bin/env bash
# =============================================================================
# Phase 4 — Lab 4: TTS Quality Evaluation
#
# What you learn:
#   Three objective quality metrics over 200 prompts at c=4:
#     WER  — Whisper-large-v3 ASR → word error rate (lower is better)
#     SIM  — WavLM speaker similarity cosine score (higher is better)
#     UTMOS — audio naturalness MOS predictor (higher is better)
#
#   NOTE: Requires the full Seed-TTS eval dataset (~1.2GB).
#         Download: follow benchmarks/build_dataset/download_process_data_seedtts.md
#         Then set SEED_TTS_PATH to the extracted seedtts_testset directory.
#
# Outputs: results/p4_lab4_quality/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
PORT=8000
SEED_TTS_PATH="${SEED_TTS_PATH:-/path/to/seedtts_testset}"   # <-- UPDATE THIS
RESULTS_DIR="$(dirname "$0")/../results/p4_lab4_quality"
mkdir -p "$RESULTS_DIR"

echo "======================================================="
echo " Phase 4 | Lab 4: TTS Quality Evaluation"
echo "======================================================="
echo "  Seed-TTS path: $SEED_TTS_PATH"

if [ ! -d "$SEED_TTS_PATH" ]; then
  echo ""
  echo "  ERROR: Seed-TTS dataset not found at $SEED_TTS_PATH"
  echo "  Set SEED_TTS_PATH or follow:"
  echo "  $VLLM_OMNI_DIR/benchmarks/build_dataset/download_process_data_seedtts.md"
  exit 1
fi

echo "[1/2] Starting TTS server on port $PORT ..."
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

echo ""
echo "[2/2] Running quality eval (200 prompts, c=4, WER+SIM+UTMOS) ..."
python3 "$VLLM_OMNI_DIR/benchmarks/tts/bench_tts.py" \
  --model "$MODEL" \
  --task voice_clone \
  --dataset-path "$SEED_TTS_PATH" \
  --host 127.0.0.1 \
  --port $PORT \
  --wer-eval \
  --concurrency 4 \
  --num-prompts 200 \
  --num-warmups 5 \
  --output-dir "$RESULTS_DIR"

echo ""
echo "Lab 4 complete. Results in $RESULTS_DIR"
echo ""
echo "Phase 4 complete. Proceed to: bash phase5-actions/lab1_internvla.sh"
