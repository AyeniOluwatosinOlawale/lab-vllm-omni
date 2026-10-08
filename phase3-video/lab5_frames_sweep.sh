#!/usr/bin/env bash
# =============================================================================
# Phase 3 — Lab 5: Frame Count + Concurrency Sweep
#
# CONTEXT WINDOW EQUIVALENT FOR VIDEO DiT:
#   Video generation has two sequence-length axes:
#     1. num_frames   — temporal depth (more frames = longer video)
#     2. spatial res  — per-frame token count (resolution²)
#   At 720p, 81 frames = ~80K total tokens fed to self-attention per step.
#   This is the direct analogue of context length for AR models.
#
# BOTTLENECK THIS EXPOSES:
#   Video DiT attention is O((frames × spatial_tokens)²) per step.
#   The dominant bottleneck is attention compute — specifically the QK^T
#   matrix multiply across the full spatio-temporal token sequence.
#
#   Known scaling (Wan2.2-T2V on single H100):
#     17 frames 480p  (~8K tokens)   → ~20s per video
#     49 frames 480p  (~24K tokens)  → ~55s per video
#     81 frames 480p  (~40K tokens)  → ~90s per video
#     81 frames 720p  (~80K tokens)  → ~240s per video  (3× slower than 480p)
#
# OPTIMIZATION SIGNALS:
#   If latency scales as frames²  → attention-bound → use Tensor Parallel
#   If latency scales linearly with frames → memory-bound → use VAE slicing
#   At c>2: requests queue behind the long video job → use async polling
#   If OOM at high frames: use --vae-use-tiling + --enable-cpu-offload
#
# Outputs: results/p3_lab5_frames/frames<N>_c<M>/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="${T2V_MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
PORT=8099
RESULTS_DIR="$(dirname "$0")/../results"
FRAME_COUNTS=(17 49 81)
CONCURRENCY_LEVELS=(1 2 4)

echo "======================================================="
echo " Phase 3 | Lab 5: Frame Count (Context Equivalent) + Concurrency Sweep"
echo "======================================================="

echo "[1/2] Starting video server on port $PORT ..."
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

echo ""
echo "[2/2] Sweeping frame counts × concurrency ..."
for F in "${FRAME_COUNTS[@]}"; do
  for C in "${CONCURRENCY_LEVELS[@]}"; do
    OUT="$RESULTS_DIR/p3_lab5_frames/frames${F}_c${C}"
    mkdir -p "$OUT"
    echo ""
    echo "  -- frames=$F  concurrency=$C --"

    python3 "$VLLM_OMNI_DIR/benchmarks/diffusion/diffusion_benchmark_serving.py" \
      --base-url "http://localhost:$PORT" \
      --model "$MODEL" \
      --task t2v \
      --dataset random \
      --num-prompts 6 \
      --max-concurrency "$C" \
      --width 832 --height 480 \
      --num-frames "$F" \
      --fps 16 \
      --num-inference-steps 20 \
      --video-job-timeout 1800 \
      --output-file "$OUT/result.json"
  done
done

echo ""
echo "Lab 5 complete. Results in results/p3_lab5_frames/"
echo ""
echo "BOTTLENECK ANALYSIS — check these patterns:"
echo "  1. Plot E2E vs num_frames."
echo "     Linear growth   → memory-bound (VAE decode); use --vae-use-slicing"
echo "     Quadratic growth → attention-bound; use --tensor-parallel-size 2"
echo "  2. Does throughput (vid/s) drop sharply at c>2?"
echo "     YES → GPU fully occupied by one video at a time (expected for 14B model)"
echo "     Mitigation: use smaller model (LingBot-Video 1.3B for c>2)"
echo "  3. OOM errors at high frames?"
echo "     Add: --vae-use-tiling --vae-use-slicing --enable-layerwise-offload"
echo ""
echo "Run phase6-benchmark/bottleneck_analysis.py for automated findings."
