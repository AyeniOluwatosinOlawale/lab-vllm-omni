#!/usr/bin/env bash
# =============================================================================
# Phase 2 — Lab 1: Text-to-Image Baseline
#
# What you learn:
#   DiT (non-AR) image generation pipeline. Unlike Phase 1 which uses
#   autoregressive token generation, image models run a fixed number of
#   diffusion denoising steps. Measure throughput in images/sec.
#
#   Default model: black-forest-labs/FLUX.2-klein-4B (~15GB, fits 1× GPU)
#   Heavy alternative: Qwen/Qwen-Image (~54GB, needs 80GB GPU)
#
# Outputs: results/p2_lab1_images/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="${T2I_MODEL:-black-forest-labs/FLUX.2-klein-4B}"
PORT=8099
RESULTS_DIR="$(dirname "$0")/../results/p2_lab1_images"
mkdir -p "$RESULTS_DIR"

echo "======================================================="
echo " Phase 2 | Lab 1: Text-to-Image Baseline"
echo "======================================================="
echo "  Model: $MODEL"
echo "  (Set T2I_MODEL env var to override)"
echo ""

echo "[1/3] Starting image server on port $PORT ..."
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

echo ""
echo "[2/3] Single image — 'a red fox standing in fresh snow, cinematic lighting' ..."
python3 "$VLLM_OMNI_DIR/examples/offline_inference/text_to_image/text_to_image.py" \
  --model "$MODEL" \
  --prompt "a red fox standing in fresh snow, cinematic lighting" \
  --num-inference-steps 20 \
  --guidance-scale 0.0 \
  --height 1024 --width 1024 \
  --output "$RESULTS_DIR/fox.png"

echo ""
echo "[3/3] Batch — 4 diverse prompts ..."
PROMPTS=(
  "a futuristic city at night, neon lights, rain"
  "a detailed oil painting of a mountain lake at sunrise"
  "a robot chef cooking in a modern kitchen, photorealistic"
  "abstract geometric patterns, vibrant colors, 4K"
)
for i in "${!PROMPTS[@]}"; do
  echo "  Prompt $((i+1))/4: ${PROMPTS[$i]}"
  python3 "$VLLM_OMNI_DIR/examples/offline_inference/text_to_image/text_to_image.py" \
    --model "$MODEL" \
    --prompt "${PROMPTS[$i]}" \
    --num-inference-steps 20 \
    --guidance-scale 0.0 \
    --height 1024 --width 1024 \
    --output "$RESULTS_DIR/batch_$i.png"
done

echo ""
echo "Lab 1 complete. Images saved to $RESULTS_DIR"
echo "Next: bash phase2-image/lab2_image_edit.sh"
