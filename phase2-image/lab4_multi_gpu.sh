#!/usr/bin/env bash
# =============================================================================
# Phase 2 — Lab 4: Multi-GPU — CFG Parallel + Ulysses Sequence Parallel
#
# What you learn:
#   CFG Parallel splits the conditional + unconditional DiT forward passes
#   across 2 GPUs — near-linear 2× speedup on the diffusion compute.
#   Ulysses SP shards the attention sequence dimension across GPUs.
#
# Requires: 2× GPUs
#
# Outputs: results/p2_lab4_multi_gpu/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="${T2I_MODEL:-Tongyi-MAI/Z-Image-Turbo}"
RESULTS_DIR="$(dirname "$0")/../results/p2_lab4_multi_gpu"
mkdir -p "$RESULTS_DIR"
PROMPT="a breathtaking landscape photograph of misty mountains at golden hour"

echo "======================================================="
echo " Phase 2 | Lab 4: Multi-GPU CFG Parallel"
echo "======================================================="
echo "  Requires 2× GPUs. Model: $MODEL"
echo ""

echo "[1/2] Single-GPU baseline ..."
python3 "$VLLM_OMNI_DIR/examples/offline_inference/text_to_image/text_to_image.py" \
  --model "$MODEL" \
  --prompt "$PROMPT" \
  --num-inference-steps 9 \
  --guidance-scale 0.0 \
  --height 1024 --width 1024 \
  --output "$RESULTS_DIR/single_gpu.png"

echo ""
echo "[2/2] 2× GPU with CFG Parallel + Ulysses SP ..."
python3 "$VLLM_OMNI_DIR/examples/offline_inference/text_to_image/text_to_image.py" \
  --model "$MODEL" \
  --prompt "$PROMPT" \
  --cfg-parallel-size 2 \
  --ulysses-degree 2 \
  --ulysses-mode advanced_uaa \
  --num-inference-steps 9 \
  --guidance-scale 0.0 \
  --height 1024 --width 1024 \
  --output "$RESULTS_DIR/cfg_parallel.png"

echo ""
echo "Lab 4 complete."
echo "  Compare generation time: single_gpu.png vs cfg_parallel.png"
echo "  Images should be visually identical; wall-clock should be ~2× faster."
echo ""
echo "Phase 2 complete. Proceed to: bash phase3-video/lab1_text_to_video.sh"
