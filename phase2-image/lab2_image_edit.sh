#!/usr/bin/env bash
# =============================================================================
# Phase 2 — Lab 2: Image-to-Image Editing
#
# What you learn:
#   Image editing uses a conditioning image alongside a text instruction.
#   The model modifies specific aspects while preserving structure.
#   Compare output quality against different edit instructions.
#
# Outputs: results/p2_lab2_edits/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="${I2I_MODEL:-Qwen/Qwen-Image-Edit}"
PORT=8099
RESULTS_DIR="$(dirname "$0")/../results/p2_lab2_edits"
mkdir -p "$RESULTS_DIR"

echo "======================================================="
echo " Phase 2 | Lab 2: Image-to-Image Editing"
echo "======================================================="

# Download reference image
INPUT_IMG="$RESULTS_DIR/cherry_blossom.jpg"
if [ ! -f "$INPUT_IMG" ]; then
  echo "[0/3] Downloading reference image ..."
  curl -L -o "$INPUT_IMG" \
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"
fi

echo "[1/3] Starting image edit server on port $PORT ..."
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

EDITS=(
  "make it snowing heavily"
  "change to autumn with orange and red leaves"
  "make it a night scene with moonlight"
)

echo ""
echo "[2/3] Running ${#EDITS[@]} edit instructions ..."
for i in "${!EDITS[@]}"; do
  echo "  Edit $((i+1)): ${EDITS[$i]}"
  python3 "$VLLM_OMNI_DIR/examples/offline_inference/image_to_image/image_edit.py" \
    --model "$MODEL" \
    --image "$INPUT_IMG" \
    --prompt "${EDITS[$i]}" \
    --num-inference-steps 50 \
    --output "$RESULTS_DIR/edit_$i.png"
done

echo ""
echo "[3/3] Results saved to $RESULTS_DIR"
ls -lh "$RESULTS_DIR"
echo ""
echo "Lab 2 complete."
echo "Next: bash phase2-image/lab3_concurrency_resolution.sh"
