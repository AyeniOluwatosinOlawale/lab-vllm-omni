#!/usr/bin/env bash
# =============================================================================
# Phase 13 — Lab 3: LingBot-Video Benchmark (Video Quality + Latency)
#
# WHAT YOU LEARN:
#   LingBot-Video benchmark measures VIDEO QUALITY, not just latency.
#   MAE, MSE, PSNR are computed against reference clips.
#
# QUALITY METRICS:
#   MAE  — mean absolute error per pixel (lower = better)
#   MSE  — mean squared error (penalises large errors more than MAE)
#   PSNR — peak signal-to-noise ratio in dB (higher = better)
#          > 35 dB = good quality; < 30 dB = visible artifacts
#
# WHEN PSNR IS LOW:
#   < 30 dB at default steps → insufficient denoising steps (increase --num-inference-steps)
#   PSNR degrades with more concurrency → GPU is skipping denoising steps under load
#
# TWO MODEL VARIANTS:
#   LingBot-Video dense — smaller, faster, lower quality
#   LingBot-Video MoE  — larger, slower, higher quality
#
# Outputs: results/p13_lab3_lingbot_video/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
PORT=8099
RESULTS_DIR="$(dirname "$0")/../results/p13_lab3_lingbot_video"
mkdir -p "$RESULTS_DIR"

echo "======================================================="
echo " Phase 13 | Lab 3: LingBot-Video (Quality + Latency)"
echo "======================================================="

run_lingbot_bench() {
  local model=$1 label=$2 out_dir="$RESULTS_DIR/$label"
  mkdir -p "$out_dir"
  echo "  Benchmarking: $label ($model)"
  python3 "$VLLM_OMNI_DIR/benchmarks/lingbot_video/benchmark_lingbot_video.py" \
    --host localhost \
    --port $PORT \
    --model "$model" \
    --num-prompts 10 \
    --max-concurrency 2 \
    --output "$out_dir/results.json" \
    2>/dev/null || \
  python3 "$VLLM_OMNI_DIR/benchmarks/diffusion/diffusion_benchmark_serving.py" \
    --base-url "http://localhost:$PORT" \
    --model "$model" \
    --task t2v \
    --dataset random \
    --num-prompts 10 \
    --max-concurrency 2 \
    --width 832 --height 480 \
    --num-frames 49 --fps 24 \
    --num-inference-steps 20 \
    --output-file "$out_dir/fallback_result.json"
}

# ── Dense model ────────────────────────────────────────────────────────────
echo ""
echo "=== LingBot-Video dense ==="
MODEL_DENSE="LingBot/LingBot-Video"
vllm serve "$MODEL_DENSE" --omni --port $PORT &
SERVER_PID=$!
trap "pkill -f 'vllm serve' 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."
run_lingbot_bench "$MODEL_DENSE" "dense"
pkill -f "vllm serve" 2>/dev/null; sleep 6

# ── MoE model ─────────────────────────────────────────────────────────────
echo ""
echo "=== LingBot-Video MoE ==="
MODEL_MOE="LingBot/LingBot-Video-MoE"
vllm serve "$MODEL_MOE" --omni --port $PORT &
SERVER_PID=$!
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."
run_lingbot_bench "$MODEL_MOE" "moe"

echo ""
echo "Lab 3 complete. Results in $RESULTS_DIR"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. PSNR dense vs MoE?"
echo "     MoE should have higher PSNR (larger model = better quality)"
echo "     If similar PSNR → denoising steps are insufficient for both"
echo "  2. MAE degrades with concurrency?"
echo "     YES → decoder is being starved of compute steps; reduce concurrency or steps"
echo "  3. Dense vs MoE throughput (videos/sec)?"
echo "     MoE throughput < dense (expected — larger model)"
echo "     Tradeoff: use dense for high throughput, MoE for quality-critical applications"
