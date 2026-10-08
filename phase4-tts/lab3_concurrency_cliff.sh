#!/usr/bin/env bash
# =============================================================================
# Phase 4 — Lab 3: Concurrency Cliff Experiment
#
# What you learn:
#   The TTS codec runs at batch_size=1 — requests cannot be batched through
#   the codec stage. At c>=8, TTFP jumps 4-6× while audio throughput
#   plateaus. This is a documented vLLM-Omni bottleneck (issue #272).
#
#   Historical data on H20-3e (Qwen3-TTS-1.7B):
#     c=1:  RTF 0.15, TTFP 165ms
#     c=4:  RTF 0.28, TTFP 412ms
#     c=8:  RTF 0.49, TTFP 1701ms   <-- cliff
#     c=16: RTF 0.72, TTFP 3355ms
#     c=32: RTF 0.77, TTFP 3772ms
#
#   Run this lab to reproduce on your hardware.
#
# Outputs: results/p4_lab3_cliff/  + cliff curve PNG
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
PORT=8000
RESULTS_DIR="$(dirname "$0")/../results/p4_lab3_cliff"
mkdir -p "$RESULTS_DIR"

echo "======================================================="
echo " Phase 4 | Lab 3: Concurrency Cliff"
echo "======================================================="

echo "[1/2] Starting TTS server on port $PORT ..."
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

echo ""
echo "[2/2] Running concurrency sweep 1 2 4 8 16 32 ..."
python3 "$VLLM_OMNI_DIR/benchmarks/tts/bench_tts.py" \
  --model "$MODEL" \
  --task voice_clone \
  --dataset-path "$VLLM_OMNI_DIR/benchmarks/build_dataset/seed_tts_smoke" \
  --host 127.0.0.1 \
  --port $PORT \
  --concurrency 1 2 4 8 16 32 \
  --num-prompts 20 \
  --num-warmups 2 \
  --output-dir "$RESULTS_DIR"

echo ""
echo "Plotting cliff curve ..."
python3 "$VLLM_OMNI_DIR/benchmarks/tts/plot_results.py" \
  --results "$RESULTS_DIR"/*.json \
  --output "$RESULTS_DIR/cliff_curve.png"

echo ""
echo "Lab 3 complete."
echo "  Results:    $RESULTS_DIR/"
echo "  Cliff plot: $RESULTS_DIR/cliff_curve.png"
echo "  Look for the TTFP and RTF jump between c=4 and c=8."
echo "Next: bash phase4-tts/lab4_quality_eval.sh"
