#!/usr/bin/env bash
# =============================================================================
# Phase 10 — Lab 2: SANA-Video 2B — CFG + Ulysses SP + Cache-DiT (new in 0.30.0)
#
# WHAT YOU LEARN:
#   SANA-Video 2B is new in 0.30.0. It supports THREE orthogonal speedups that
#   can be stacked:
#
#   1. CFG Parallel (--cfg-parallel-size 2):
#      Splits classifier-free guidance computation across GPUs.
#      Positive/negative guidance paths run on separate GPUs simultaneously.
#      Speedup: ~2× on the guidance step (requires 2+ GPUs).
#
#   2. Ulysses Sequence Parallelism (--usp):
#      Splits the sequence (spatial) dimension across GPUs.
#      Reduces per-GPU memory for long video sequences.
#      Speedup: ~1.5-2× for 720p+ (requires 2+ GPUs).
#
#   3. Cache-DiT (--cache-backend cache_dit):
#      Reuses KV tensors across denoising steps (steps share similar attention).
#      Speedup: ~1.3-1.5× with minimal quality loss (single GPU compatible).
#
# HARDWARE: Experiments A, B require 1 GPU; C requires 2 GPUs.
# Outputs: results/p10_lab2_sana_video/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="NVlabs/SANA-Video-2B"
PORT=8099
RESULTS_DIR="$(dirname "$0")/../results/p10_lab2_sana_video"
mkdir -p "$RESULTS_DIR"

GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)

echo "======================================================="
echo " Phase 10 | Lab 2: SANA-Video 2B (CFG + USP + Cache-DiT)"
echo "======================================================="
echo "  GPUs available: $GPU_COUNT"
echo ""

run_bench() {
  local label=$1
  local out_file="$RESULTS_DIR/${label}_result.json"
  echo "  Benchmarking $label ..."
  python3 "$VLLM_OMNI_DIR/benchmarks/diffusion/diffusion_benchmark_serving.py" \
    --base-url "http://localhost:$PORT" \
    --model "$MODEL" \
    --task t2v \
    --dataset random \
    --num-prompts 5 \
    --max-concurrency 1 \
    --width 832 --height 480 \
    --num-frames 49 --fps 24 \
    --num-inference-steps 20 \
    --output-file "$out_file"
}

stop_server() {
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 6
}

# ── A. Baseline (no parallelism, no cache) ─────────────────────────────────
echo "=== A. Baseline ==="
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "stop_server; exit" INT TERM EXIT
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."
run_bench "baseline"
stop_server

# ── B. Cache-DiT only (single GPU) ────────────────────────────────────────
echo ""
echo "=== B. Cache-DiT only ==="
vllm serve "$MODEL" --omni --port $PORT \
  --cache-backend cache_dit &
SERVER_PID=$!
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."
run_bench "cache_dit"
stop_server

# ── C. CFG Parallel + USP + Cache-DiT (2× GPU) ────────────────────────────
if [ "$GPU_COUNT" -ge 2 ]; then
  echo ""
  echo "=== C. CFG Parallel + USP + Cache-DiT (2× GPU) ==="
  vllm serve "$MODEL" --omni --port $PORT \
    --cfg-parallel-size 2 \
    --usp \
    --cache-backend cache_dit &
  SERVER_PID=$!
  until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
  echo "  Server ready."
  run_bench "cfg_usp_cache"
  stop_server
else
  echo ""
  echo "=== C. SKIPPED — requires 2× GPU (only $GPU_COUNT available) ==="
fi

echo ""
echo "Lab 2 complete. Results in $RESULTS_DIR"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. Cache-DiT speedup ratio (baseline / cache_dit E2E)?"
echo "     Expected: 1.3-1.5× — any more suggests quality may be impacted"
echo "  2. Does CFG+USP add >2× speedup over Cache-DiT alone?"
echo "     YES → parallelism fully additive; ideal for production on 2× GPUs"
echo "     NO  → inter-GPU communication overhead limits scaling"
echo "  3. Enable --enable-cache-dit-summary to see per-step cache hit rates"
