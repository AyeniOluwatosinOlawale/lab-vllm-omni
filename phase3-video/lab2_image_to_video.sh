#!/usr/bin/env bash
# =============================================================================
# Phase 3 — Lab 2: Image-to-Video
#
# What you learn:
#   Conditioning on a reference image constrains the first frame and motion
#   direction. Compare I2V output against the T2V output from Lab 1 to see
#   how image conditioning tightens the output distribution.
#
#   Default model: Wan2.2-TI2V-5B (~20-25GB — smallest I2V model)
#   Heavy alternative: Wan2.2-I2V-A14B (~60GB)
#
# Outputs: results/p3_lab2_i2v/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="${I2V_MODEL:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
PORT=8099
RESULTS_DIR="$(dirname "$0")/../results/p3_lab2_i2v"
mkdir -p "$RESULTS_DIR"

echo "======================================================="
echo " Phase 3 | Lab 2: Image-to-Video"
echo "======================================================="
echo "  Model: $MODEL"
echo ""

# Download reference image
INPUT_IMG="$RESULTS_DIR/cherry_blossom.jpg"
if [ ! -f "$INPUT_IMG" ]; then
  echo "[0/3] Downloading reference image ..."
  curl -L -o "$INPUT_IMG" \
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"
fi

PROMPTS=(
  "cherry blossoms falling gently in the breeze"
  "wind picks up, petals swirl wildly around the tree"
  "a cat suddenly runs through frame under the tree"
)

for i in "${!PROMPTS[@]}"; do
  echo "[Prompt $((i+1))/${#PROMPTS[@]}] ${PROMPTS[$i]}"
  python3 "$VLLM_OMNI_DIR/examples/offline_inference/image_to_video/image_to_video.py" \
    --model "$MODEL" \
    --image "$INPUT_IMG" \
    --prompt "${PROMPTS[$i]}" \
    --height 480 --width 832 \
    --num-frames 81 --fps 24 \
    --num-inference-steps 50 \
    --output "$RESULTS_DIR/i2v_$i.mp4"
  echo "  Saved: $RESULTS_DIR/i2v_$i.mp4"
  echo ""
done

echo "Lab 2 complete."
echo "Next: bash phase3-video/lab3_speech_to_video.sh"
