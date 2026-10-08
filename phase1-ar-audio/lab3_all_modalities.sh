#!/usr/bin/env bash
# =============================================================================
# Phase 1 — Lab 3: All Input Modalities
#
# What you learn:
#   Each --query-type exercises a different input path through the Thinker.
#   Video/audio inputs add encode time before generation — observe how
#   TTFT and E2E shift across modality types.
#
#   Query types:
#     text              — text prompt only
#     use_audio         — audio file input
#     use_image         — image file input
#     use_video         — video frames input
#     use_multi_audios  — multiple audio files
#     use_mixed_modalities  — video + image + audio simultaneously
#     use_audio_in_video    — video with embedded audio track
#
# Outputs: results/p1_lab3_modalities/<query_type>/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"
PORT=8091
RESULTS_DIR="$(dirname "$0")/../results"
mkdir -p "$RESULTS_DIR"

QUERY_TYPES=(
  text
  use_audio
  use_image
  use_video
  use_multi_audios
  use_mixed_modalities
  use_audio_in_video
)

echo "======================================================="
echo " Phase 1 | Lab 3: All Input Modalities"
echo "======================================================="

echo "[1/2] Starting server (async_chunk ON) on port $PORT ..."
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

echo ""
echo "[2/2] Running each input modality type (3 prompts each) ..."
for QTYPE in "${QUERY_TYPES[@]}"; do
  echo ""
  echo "  --- Query type: $QTYPE ---"
  OUT_DIR="$RESULTS_DIR/p1_lab3_modalities/$QTYPE"
  mkdir -p "$OUT_DIR"

  python3 "$VLLM_OMNI_DIR/examples/offline_inference/qwen3_omni/end2end.py" \
    --model "$MODEL" \
    --query-type "$QTYPE" \
    --modalities text,audio \
    --num-prompts 3 \
    --output-dir "$OUT_DIR" \
    2>&1 | tee "$OUT_DIR/run.log"

  echo "  Saved to $OUT_DIR"
done

echo ""
echo "Lab 3 complete."
echo "Check results/p1_lab3_modalities/<type>/run.log for per-modality latency."
echo "Next: bash phase1-ar-audio/lab4_concurrency_sweep.sh"
