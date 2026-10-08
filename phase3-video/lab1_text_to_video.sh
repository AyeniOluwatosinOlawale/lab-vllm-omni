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
# MEMORY GUIDE (2× A100/H100 80GB):
#   wan preset = 1280×720, 81 frames ≈ 80K spatio-temporal tokens.
#   Wan2.2-T2V-A14B weights ≈ 28GB BF16; activation memory at 720p
#   can spike close to 80GB on one GPU.
#
#   GPU_SAFE=1 (default on single-GPU systems) drops to 480p and adds
#   --vae-use-tiling so activations stay well under 80GB.
#   GPU_SAFE=0 keeps the full 720p wan preset (2× GPU recommended).
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

# Auto-detect single GPU; override with GPU_SAFE=0 to force 720p
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
GPU_SAFE="${GPU_SAFE:-$([ "$GPU_COUNT" -ge 2 ] && echo 0 || echo 1)}"

# Build extra flags for the safe path
EXTRA_FLAGS=""
RESOLUTION_NOTE="720p (wan preset)"
if [ "$GPU_SAFE" = "1" ]; then
  # 480p, 81 frames ≈ 40K tokens — fits comfortably in 80GB
  EXTRA_FLAGS="--height 480 --width 832 --num-frames 81 --vae-use-tiling"
  RESOLUTION_NOTE="480p + --vae-use-tiling (GPU_SAFE=1)"
  PRESET="wan"  # still uses wan model family but overrides resolution
fi

echo "======================================================="
echo " Phase 3 | Lab 1: Text-to-Video"
echo "======================================================="
echo "  Model:      $MODEL"
echo "  Preset:     $PRESET"
echo "  Resolution: $RESOLUTION_NOTE"
echo "  GPUs found: $GPU_COUNT  (GPU_SAFE=$GPU_SAFE)"
echo ""
if [ "$GPU_SAFE" = "1" ]; then
  echo "  NOTE: Running at 480p for single-GPU safety."
  echo "  To run the full 720p preset on 2× GPUs: GPU_SAFE=0 bash $0"
  echo ""
fi

PROMPTS=(
  "a lone astronaut walking across a red desert, wide shot, cinematic"
  "ocean waves crashing on a rocky coastline at sunset, slow motion"
  "a time-lapse of a city street from dawn to dusk, people and cars moving"
)

for i in "${!PROMPTS[@]}"; do
  echo "[Prompt $((i+1))/${#PROMPTS[@]}] ${PROMPTS[$i]}"
  # shellcheck disable=SC2086
  python3 "$VLLM_OMNI_DIR/examples/offline_inference/text_to_video/text_to_video.py" \
    --model "$MODEL" \
    --preset "$PRESET" \
    --prompt "${PROMPTS[$i]}" \
    --output "$RESULTS_DIR/t2v_$i.mp4" \
    $EXTRA_FLAGS
  echo "  Saved: $RESULTS_DIR/t2v_$i.mp4"
  echo ""
done

echo "Lab 1 complete. Videos saved to $RESULTS_DIR"
echo "Next: bash phase3-video/lab2_image_to_video.sh"
