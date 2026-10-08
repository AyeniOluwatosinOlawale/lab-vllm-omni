#!/usr/bin/env bash
# =============================================================================
# Phase 8 — Lab 2: Talking-Head Avatar (LongCat-Video-Avatar-1.5)
#
# WHAT YOU LEARN:
#   Avatar generation combines talking-head video synthesis with lip-sync audio.
#   Output is a composite MP4 with both video track and synchronised speech.
#   LongCat-Video-Avatar-1.5 supports single-speaker and multi-speaker modes.
#
# KEY CHARACTERISTICS:
#   - NVIDIA-only (no AMD/Intel support)
#   - Output: MP4 video with lip-synced audio
#   - Single-speaker: one face + one voice
#   - Multi-speaker: multiple faces in frame, each with distinct voice
#   - Generation serialises per speaker — concurrency > 1 queues requests
#
# HARDWARE NOTE:
#   Avatar at 480p fits in 1× 80GB GPU.
#   Multi-speaker mode with 2+ faces may require 2× GPUs.
#
# Outputs: results/p8_lab2_avatar/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="LongCat-AI/LongCat-Video-Avatar-1.5"
PORT=8099
RESULTS_DIR="$(dirname "$0")/../results/p8_lab2_avatar"
mkdir -p "$RESULTS_DIR"

AVATAR_PROMPTS=(
  "Welcome to our quarterly earnings call. We are pleased to report strong growth."
  "Today I will explain the three pillars of machine learning: data, model, and compute."
  "The weather forecast for tomorrow shows sunny skies with temperatures around 22 degrees."
)

echo "======================================================="
echo " Phase 8 | Lab 2: Talking-Head Avatar (LongCat-Video-Avatar-1.5)"
echo "======================================================="
echo "  NOTE: NVIDIA GPU required. AMD/Intel not supported."
echo ""

echo "[1/3] Starting avatar server on port $PORT ..."
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

# ── Single-speaker avatar ──────────────────────────────────────────────────
echo ""
echo "[2/3] Single-speaker avatar generation ..."
mkdir -p "$RESULTS_DIR/single_speaker"
python3 "$VLLM_OMNI_DIR/benchmarks/diffusion/diffusion_benchmark_serving.py" \
  --base-url "http://localhost:$PORT" \
  --model "$MODEL" \
  --task avatar \
  --dataset random \
  --num-prompts 5 \
  --max-concurrency 1 \
  --width 512 --height 512 \
  --num-inference-steps 20 \
  --output-file "$RESULTS_DIR/single_speaker/result.json"

# ── Concurrency comparison c=1 vs c=2 ─────────────────────────────────────
echo ""
echo "[3/3] Concurrency sweep (c=1 vs c=2) ..."
for C in 1 2; do
  mkdir -p "$RESULTS_DIR/concurrency_c$C"
  echo "  -- concurrency=$C --"
  python3 "$VLLM_OMNI_DIR/benchmarks/diffusion/diffusion_benchmark_serving.py" \
    --base-url "http://localhost:$PORT" \
    --model "$MODEL" \
    --task avatar \
    --dataset random \
    --num-prompts 6 \
    --max-concurrency "$C" \
    --width 512 --height 512 \
    --num-inference-steps 20 \
    --output-file "$RESULTS_DIR/concurrency_c$C/result.json"
done

echo ""
echo "Lab 2 complete. Results in $RESULTS_DIR"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. E2E latency at c=1 vs c=2?"
echo "     Avatar generation serialises per speaker (one face at a time)."
echo "     c=2 should show ~2× latency increase (full serialisation)."
echo "     If latency stays flat → pipeline has some parallelism (unexpected)."
echo "  2. Does audio lip-sync add overhead vs pure video generation?"
echo "     Compare to Phase 3 (pure video) at same resolution/steps."
echo "     Extra time = lip-sync audio alignment cost."
echo "  3. OOM at multi-speaker? → Use --vae-use-tiling or reduce resolution to 384p."
