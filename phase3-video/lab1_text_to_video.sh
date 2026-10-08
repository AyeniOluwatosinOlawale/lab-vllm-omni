#!/usr/bin/env bash
# =============================================================================
# Phase 3 — Lab 1: Text-to-Video
#
# What you learn:
#   Video generation runs a full diffusion denoising process over a
#   sequence of frames. The token count is much larger than images
#   (81 frames × spatial tokens), so attention cost dominates.
#   Built-in presets handle resolution/fps/steps per model family.
#
#   Preset options: wan | hunyuan | cosmos | ltx2 | helios | lingbot
#   Default: wan (Wan2.2-T2V-A14B, 720p, 81 frames)
#
# Outputs: results/p3_lab1_t2v/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="${T2V_MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
PRESET="${T2V_PRESET:-wan}"
PORT=8099
RESULTS_DIR="$(dirname "$0")/../results/p3_lab1_t2v"
mkdir -p "$RESULTS_DIR"

echo "======================================================="
echo " Phase 3 | Lab 1: Text-to-Video"
echo "======================================================="
echo "  Model:  $MODEL"
echo "  Preset: $PRESET"
echo ""

PROMPTS=(
  "a lone astronaut walking across a red desert, wide shot, cinematic"
  "ocean waves crashing on a rocky coastline at sunset, slow motion"
  "a time-lapse of a city street from dawn to dusk, people and cars moving"
)

for i in "${!PROMPTS[@]}"; do
  echo "[Prompt $((i+1))/${#PROMPTS[@]}] ${PROMPTS[$i]}"
  python3 "$VLLM_OMNI_DIR/examples/offline_inference/text_to_video/text_to_video.py" \
    --model "$MODEL" \
    --preset "$PRESET" \
    --prompt "${PROMPTS[$i]}" \
    --output "$RESULTS_DIR/t2v_$i.mp4"
  echo "  Saved: $RESULTS_DIR/t2v_$i.mp4"
  echo ""
done

echo "Lab 1 complete. Videos saved to $RESULTS_DIR"
echo "Next: bash phase3-video/lab2_image_to_video.sh"
