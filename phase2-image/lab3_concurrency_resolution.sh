#!/usr/bin/env bash
# =============================================================================
# Phase 2 — Lab 3: Concurrency and Resolution Sweep
#
# What you learn:
#   - Concurrency sweep: images/sec throughput vs concurrent requests
#   - Resolution sweep: latency scales ~quadratically with pixel area
#     because DiT attention is O(n²) in token count
#
# Outputs: results/p2_lab3_concurrency/  results/p2_lab3_resolution/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="${T2I_MODEL:-black-forest-labs/FLUX.2-klein-4B}"
PORT=8099
RESULTS_DIR="$(dirname "$0")/../results"
CONCURRENCY_LEVELS=(1 2 4 8)
RESOLUTIONS=("512 512" "768 768" "1024 1024")

echo "======================================================="
echo " Phase 2 | Lab 3: Concurrency + Resolution Sweep"
echo "======================================================="

echo "[1/3] Starting server on port $PORT ..."
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

echo ""
echo "[2/3] Concurrency sweep (fixed 1024×1024) ..."
for C in "${CONCURRENCY_LEVELS[@]}"; do
  OUT="$RESULTS_DIR/p2_lab3_concurrency/c$C"
  mkdir -p "$OUT"
  echo "  -- concurrency=$C --"
  python3 "$VLLM_OMNI_DIR/benchmarks/diffusion/diffusion_benchmark_serving.py" \
    --base-url "http://localhost:$PORT" \
    --model "$MODEL" \
    --task t2i \
    --dataset random \
    --num-prompts 20 \
    --max-concurrency "$C" \
    --width 1024 --height 1024 \
    --num-inference-steps 20 \
    --output-file "$OUT/result.json"
done

echo ""
echo "[3/3] Resolution sweep (fixed concurrency=4) ..."
for RES in "${RESOLUTIONS[@]}"; do
  W=$(echo "$RES" | cut -d' ' -f1)
  H=$(echo "$RES" | cut -d' ' -f2)
  OUT="$RESULTS_DIR/p2_lab3_resolution/${W}x${H}"
  mkdir -p "$OUT"
  echo "  -- ${W}×${H} --"
  python3 "$VLLM_OMNI_DIR/benchmarks/diffusion/diffusion_benchmark_serving.py" \
    --base-url "http://localhost:$PORT" \
    --model "$MODEL" \
    --task t2i \
    --dataset random \
    --num-prompts 10 \
    --max-concurrency 4 \
    --width "$W" --height "$H" \
    --num-inference-steps 20 \
    --output-file "$OUT/result.json"
done

echo ""
echo "Lab 3 complete."
echo "  Concurrency results: results/p2_lab3_concurrency/"
echo "  Resolution results:  results/p2_lab3_resolution/"
echo "Next: bash phase2-image/lab4_multi_gpu.sh"
