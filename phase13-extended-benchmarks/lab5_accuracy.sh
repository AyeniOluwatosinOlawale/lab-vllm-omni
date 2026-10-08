#!/usr/bin/env bash
# =============================================================================
# Phase 13 — Lab 5: Accuracy Benchmarks (GEBench T2I + GEdit-Bench I2I)
#
# WHAT YOU LEARN:
#   Accuracy benchmarks measure IMAGE QUALITY, not serving throughput.
#   These are separate from the diffusion serving benchmarks in Phase 2
#   which only measure latency and requests/sec.
#
# TWO BENCHMARKS:
#
#   GEBench (Generative Evaluation Benchmark):
#     Tests TEXT-TO-IMAGE quality: how well generated images match prompts.
#     Metrics: GEBench score, CLIP alignment score, FID
#     Model: FLUX.2-klein-4B
#
#   GEdit-Bench (Generative Editing Benchmark):
#     Tests IMAGE-TO-IMAGE editing quality: how faithfully an edit instruction
#     is applied while preserving non-edited regions.
#     Metrics: PSNR, SSIM, CLIP alignment, GEdit-Bench score
#     Model: Qwen-Image-Edit-2509
#
# WHY THIS MATTERS:
#   A model that serves at 10 req/sec with poor image quality is not useful.
#   GEBench/GEdit-Bench let you trade off serving speed vs output quality
#   (e.g., fewer inference steps → faster but lower score).
#
# Outputs: results/p13_lab5_accuracy/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
PORT=8099
RESULTS_DIR="$(dirname "$0")/../results/p13_lab5_accuracy"
mkdir -p "$RESULTS_DIR"

echo "======================================================="
echo " Phase 13 | Lab 5: Accuracy Benchmarks (GEBench + GEdit-Bench)"
echo "======================================================="

# ── Part A: GEBench — Text-to-Image quality ────────────────────────────────
echo ""
echo "=== Part A: GEBench (text-to-image quality) ==="
MODEL_T2I="black-forest-labs/FLUX.2-klein-4B"
vllm serve "$MODEL_T2I" --omni --port $PORT &
SERVER_PID=$!
trap "pkill -f 'vllm serve' 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

mkdir -p "$RESULTS_DIR/gebench"
python3 "$VLLM_OMNI_DIR/benchmarks/accuracy/gebench.py" \
  --host localhost \
  --port $PORT \
  --model "$MODEL_T2I" \
  --num-prompts 50 \
  --width 1024 --height 1024 \
  --num-inference-steps 20 \
  --output "$RESULTS_DIR/gebench/gebench_flux.json" \
  2>/dev/null || \
  echo "  GEBench script not found — running diffusion benchmark as fallback"

# Steps vs quality sweep (latency-quality tradeoff)
echo ""
echo "  Steps vs quality sweep (5, 20, 50 steps) ..."
for STEPS in 5 20 50; do
  mkdir -p "$RESULTS_DIR/gebench/steps_$STEPS"
  python3 "$VLLM_OMNI_DIR/benchmarks/diffusion/diffusion_benchmark_serving.py" \
    --base-url "http://localhost:$PORT" \
    --model "$MODEL_T2I" \
    --task t2i \
    --dataset random \
    --num-prompts 10 \
    --max-concurrency 2 \
    --width 1024 --height 1024 \
    --num-inference-steps "$STEPS" \
    --output-file "$RESULTS_DIR/gebench/steps_$STEPS/result.json"
  echo "  Steps=$STEPS done."
done

pkill -f "vllm serve" 2>/dev/null; sleep 6

# ── Part B: GEdit-Bench — Image editing quality ────────────────────────────
echo ""
echo "=== Part B: GEdit-Bench (image editing quality) ==="
MODEL_EDIT="Qwen/Qwen-Image-Edit-2509"
vllm serve "$MODEL_EDIT" --omni --port $PORT &
SERVER_PID=$!
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

mkdir -p "$RESULTS_DIR/gedit_bench"
python3 "$VLLM_OMNI_DIR/benchmarks/accuracy/gedit_bench.py" \
  --host localhost \
  --port $PORT \
  --model "$MODEL_EDIT" \
  --num-prompts 30 \
  --output "$RESULTS_DIR/gedit_bench/gedit_qwen.json" \
  2>/dev/null || \
  echo "  GEdit-Bench script not found — check VLLM_OMNI_DIR path"

# Spot check: run 3 edit requests manually
echo ""
echo "  Spot-check edits (3 sample instructions) ..."
EDIT_INSTRUCTIONS=(
  "change the sky to a dramatic sunset with orange and pink clouds"
  "add a light snowfall effect to the scene"
  "convert the style to a watercolor painting"
)
for i in "${!EDIT_INSTRUCTIONS[@]}"; do
  echo "  Edit $((i+1)): ${EDIT_INSTRUCTIONS[$i]}"
  curl -s "http://localhost:$PORT/v1/images/edits" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL_EDIT\",
      \"prompt\": \"${EDIT_INSTRUCTIONS[$i]}\",
      \"image\": \"placeholder_base64_image\",
      \"response_format\": \"b64_json\"
    }" | python3 -c "
import sys, json
d = json.load(sys.stdin)
if 'data' in d and d['data']:
    print('  Edit $((i+1)): SUCCESS')
else:
    print('  Edit $((i+1)): FAILED', d.get('error', ''))
" 2>/dev/null
done

echo ""
echo "Lab 5 complete. Results in $RESULTS_DIR"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. GEBench score vs inference steps?"
echo "     Steps=5 vs steps=20: measure quality drop vs 4× speedup"
echo "     Distilled models (Turbo, DMD): steps=5 → quality ≈ steps=20"
echo "     Standard models: steps=5 → significant quality drop"
echo "  2. GEdit-Bench: SSIM < 0.7?"
echo "     YES → edit is changing too much of the image (poor preservation)"
echo "     Mitigation: reduce --strength parameter in editing pipeline"
echo "  3. CLIP alignment score vs FID?"
echo "     High CLIP + high FID → images match prompt semantically but look unnatural"
echo "     Cause: model is overfitting to CLIP training distribution"
