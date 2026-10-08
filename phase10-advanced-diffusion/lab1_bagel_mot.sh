#!/usr/bin/env bash
# =============================================================================
# Phase 10 — Lab 1: BAGEL 7B-MoT Continuous Batching (new in 0.30.0)
#
# WHAT YOU LEARN:
#   BAGEL uses a Mixture-of-Transformer (MoT) architecture where text and image
#   tokens share the same transformer layers. 0.30.0 added continuous batching
#   for BAGEL — multiple image generation requests now share denoising steps,
#   rather than each running its own isolated denoising loop.
#
# THE LLM ANALOGY:
#   Standard diffusion: each request runs full denoising loop independently
#                       (like LLM without batching)
#   BAGEL continuous:   multiple requests share each denoising step
#                       (like in-flight batching for LLMs)
#
# THREE EXPERIMENTS:
#   A. Serial (c=1) — baseline, no batching benefit
#   B. Batched (c=4) — continuous batching kicks in
#   C. Mixed prompt lengths via random dataset — tests MoT handling of variable
#      length text+image token sequences
#
# Outputs: results/p10_lab1_bagel_mot/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="ByteDance/BAGEL-7B-MoT"
PORT=8099
RESULTS_DIR="$(dirname "$0")/../results/p10_lab1_bagel_mot"
mkdir -p "$RESULTS_DIR"

echo "======================================================="
echo " Phase 10 | Lab 1: BAGEL 7B-MoT Continuous Batching"
echo "======================================================="

echo "[1/4] Starting BAGEL server with continuous step execution ..."
vllm serve "$MODEL" --omni --port $PORT \
  --step-execution continuous &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

# ── A. Serial c=1 baseline ─────────────────────────────────────────────────
echo ""
echo "[2/4] A. Serial requests (c=1) ..."
python3 "$VLLM_OMNI_DIR/benchmarks/diffusion/diffusion_benchmark_serving.py" \
  --base-url "http://localhost:$PORT" \
  --model "$MODEL" \
  --task t2i \
  --dataset random \
  --num-prompts 10 \
  --max-concurrency 1 \
  --width 512 --height 512 \
  --num-inference-steps 20 \
  --output-file "$RESULTS_DIR/c1_result.json"

# ── B. Batched c=4 ────────────────────────────────────────────────────────
echo ""
echo "[3/4] B. Batched requests (c=4, continuous batching) ..."
python3 "$VLLM_OMNI_DIR/benchmarks/diffusion/diffusion_benchmark_serving.py" \
  --base-url "http://localhost:$PORT" \
  --model "$MODEL" \
  --task t2i \
  --dataset random \
  --num-prompts 10 \
  --max-concurrency 4 \
  --width 512 --height 512 \
  --num-inference-steps 20 \
  --output-file "$RESULTS_DIR/c4_result.json"

# ── C. Mixed prompt lengths ────────────────────────────────────────────────
echo ""
echo "[4/4] C. Mixed prompt lengths (MoT variable-length tokens) ..."
python3 "$VLLM_OMNI_DIR/benchmarks/diffusion/diffusion_benchmark_serving.py" \
  --base-url "http://localhost:$PORT" \
  --model "$MODEL" \
  --task t2i \
  --dataset random \
  --num-prompts 10 \
  --max-concurrency 4 \
  --width 512 --height 512 \
  --num-inference-steps 20 \
  --output-file "$RESULTS_DIR/mixed_len_result.json"

echo ""
echo "Lab 1 complete. Results in $RESULTS_DIR"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. Throughput ratio c=4 / c=1?"
echo "     RATIO ≈ 4 → continuous batching is working perfectly"
echo "     RATIO < 2 → requests are not being batched together; check --step-execution"
echo "  2. Per-request latency at c=4 vs c=1?"
echo "     Latency should stay similar (or increase slightly) while throughput improves"
echo "     If per-request latency 4× at c=4 → serial scheduling, batching not applied"
echo "  3. Mixed length vs fixed length throughput?"
echo "     MoT handles variable token lengths natively; similar throughput expected"
echo "     Large degradation → MoT padding is wasteful; use --request-batch-max-wait-ms"
