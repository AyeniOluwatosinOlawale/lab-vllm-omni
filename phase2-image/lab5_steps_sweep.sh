#!/usr/bin/env bash
# =============================================================================
# Phase 2 — Lab 5: Inference Steps + Concurrency Sweep
#
# CONTEXT WINDOW EQUIVALENT FOR IMAGE DiT:
#   Images have no token-sequence context window. The equivalent dimension is
#   num_inference_steps — more steps = more denoising iterations = more
#   compute per request. Combined with concurrency, this reveals how the
#   DiT batch scheduler saturates under load.
#
# BOTTLENECK THIS EXPOSES:
#   DiT attention is O(tokens²) per step. At 1024×1024, each step processes
#   ~4096 spatial tokens. The bottleneck is GPU SM utilisation during
#   self-attention — not memory bandwidth. Adding concurrency stacks
#   requests into larger batches, which improves SM utilisation but
#   increases per-request latency.
#
#   Known pattern:
#     steps=5  (distilled)  → low latency, lower quality
#     steps=20 (balanced)   → moderate latency
#     steps=50 (full)       → high latency, best quality
#     At c>4: batching gains plateau, queue latency dominates
#
# OPTIMIZATION SIGNALS:
#   If E2E scales linearly with steps → compute-bound → use fewer steps
#     (distilled models: FLUX.2-klein needs only 8-20 steps)
#   If E2E does NOT scale linearly → memory-bound → add Ulysses SP
#   If throughput (img/s) plateaus at c=4 → SM saturated → add GPU via CFG parallel
#
# Outputs: results/p2_lab5_steps/steps<N>_c<M>/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="${T2I_MODEL:-black-forest-labs/FLUX.2-klein-4B}"
PORT=8099
RESULTS_DIR="$(dirname "$0")/../results"
STEPS=(5 20 50)
CONCURRENCY_LEVELS=(1 4 8)

echo "======================================================="
echo " Phase 2 | Lab 5: Steps (Context Equivalent) + Concurrency Sweep"
echo "======================================================="

echo "[1/2] Starting image server on port $PORT ..."
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

echo ""
echo "[2/2] Sweeping steps × concurrency ..."
for S in "${STEPS[@]}"; do
  for C in "${CONCURRENCY_LEVELS[@]}"; do
    OUT="$RESULTS_DIR/p2_lab5_steps/steps${S}_c${C}"
    mkdir -p "$OUT"
    echo ""
    echo "  -- steps=$S  concurrency=$C --"

    python3 "$VLLM_OMNI_DIR/benchmarks/diffusion/diffusion_benchmark_serving.py" \
      --base-url "http://localhost:$PORT" \
      --model "$MODEL" \
      --task t2i \
      --dataset random \
      --num-prompts 20 \
      --max-concurrency "$C" \
      --width 1024 --height 1024 \
      --num-inference-steps "$S" \
      --output-file "$OUT/result.json"
  done
done

echo ""
echo "Lab 5 complete. Results in results/p2_lab5_steps/"
echo ""
echo "BOTTLENECK ANALYSIS — check these patterns in the result JSONs:"
echo "  1. Does E2E latency scale linearly with steps?"
echo "     YES → compute-bound (DiT GEMM); use fewer steps or distilled model"
echo "     NO  → memory-bound (weight loading); use Ulysses SP to shard attention"
echo "  2. Does img/s throughput plateau after c=4?"
echo "     YES → SM saturation; add a second GPU via CFG Parallel (lab4)"
echo "  3. Is p99 latency >> p50 at high concurrency?"
echo "     YES → request queuing; increase max_num_seqs in deploy config"
echo ""
echo "Run phase6-benchmark/bottleneck_analysis.py to get automated findings."
