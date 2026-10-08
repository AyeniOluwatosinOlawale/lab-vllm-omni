#!/usr/bin/env bash
# =============================================================================
# Phase 3 — Lab 4: Video Concurrency Sweep
#
# What you learn:
#   Video jobs are long-running async tasks polled via /v1/videos.
#   --video-job-timeout guards against jobs that stall at high concurrency.
#   At c>2 for a 14B model, GPU memory starts to bind — observe throughput
#   plateau and latency blow-up.
#
#   Also tests the SLO attainment feature: the benchmark estimates expected
#   generation time and flags requests that miss by --slo-scale.
#
# Outputs: results/p3_lab4_concurrency/c<N>/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="${T2V_MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
PORT=8099
RESULTS_DIR="$(dirname "$0")/../results"
CONCURRENCY_LEVELS=(1 2 4)

echo "======================================================="
echo " Phase 3 | Lab 4: Video Concurrency Sweep"
echo "======================================================="

echo "[1/2] Starting video server on port $PORT ..."
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

echo ""
echo "[2/2] Concurrency sweep: ${CONCURRENCY_LEVELS[*]} ..."
for C in "${CONCURRENCY_LEVELS[@]}"; do
  OUT="$RESULTS_DIR/p3_lab4_concurrency/c$C"
  mkdir -p "$OUT"
  echo ""
  echo "  -- concurrency=$C --"
  python3 "$VLLM_OMNI_DIR/benchmarks/diffusion/diffusion_benchmark_serving.py" \
    --base-url "http://localhost:$PORT" \
    --model "$MODEL" \
    --task t2v \
    --dataset vbench \
    --num-prompts 10 \
    --max-concurrency "$C" \
    --warmup-requests 2 \
    --warmup-concurrency "$C" \
    --width 832 --height 480 \
    --num-frames 81 --fps 24 \
    --num-inference-steps 20 \
    --video-job-timeout 1800 \
    --slo \
    --slo-scale 3.0 \
    --output-file "$OUT/result.json"
done

echo ""
echo "Lab 4 complete. Results in results/p3_lab4_concurrency/"
echo "Phase 3 complete. Proceed to: bash phase4-tts/lab1_voice_clone.sh"
