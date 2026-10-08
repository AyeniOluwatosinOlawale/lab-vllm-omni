#!/usr/bin/env bash
# =============================================================================
# Phase 12 — Lab 2: π0.5 (Pi-Zero 0.5) — new in 0.30.0, NVIDIA + Intel
#
# WHAT YOU LEARN:
#   π0.5 is Physical Intelligence's updated robot policy, new in 0.30.0.
#   Key improvements over π0:
#     - Runs on NVIDIA AND Intel XPU (π0 was NVIDIA-only)
#     - Supports multi-frame observation (1, 2, 4 frames)
#     - Better action quality on manipulation tasks
#
# MULTI-FRAME OBSERVATION — THE CONTEXT WINDOW EQUIVALENT:
#   Just like Phase 1 context sweeps for LLMs, more observation frames =
#   richer history = better action quality, but higher memory and latency.
#   This is the Phase 5 equivalent for π0.5.
#
#   1 frame  — reactive policy (no history)
#   2 frames — minimal history
#   4 frames — full observation window
#
# TWO EXPERIMENTS:
#   A. π0 vs π0.5 latency comparison (same 3 manipulation tasks)
#   B. π0.5 observation frame sweep (1, 2, 4 frames)
#
# Outputs: results/p12_lab2_pi05/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
PORT=8091
RESULTS_DIR="$(dirname "$0")/../results/p12_lab2_pi05"
mkdir -p "$RESULTS_DIR"

TASKS=(
  "grasp the blue block and stack it on top of the red block"
  "wipe the table surface from left to right with the sponge"
  "open the drawer and retrieve the object inside"
)

bench_model() {
  local model=$1 label=$2 out_dir="$RESULTS_DIR/$label"
  mkdir -p "$out_dir"
  vllm bench serve --omni \
    --host localhost --port $PORT \
    --model "$model" \
    --backend openai-chat-omni \
    --dataset-name random \
    --num-prompts 20 --max-concurrency 1 \
    --random-input-len 128 --random-output-len 128 \
    --percentile-metrics ttft,tpot,e2el \
    --save-result \
    --result-dir "$out_dir"
}

echo "======================================================="
echo " Phase 12 | Lab 2: π0.5 vs π0 + Observation Frame Sweep"
echo "======================================================="

# ── A-1. π0 baseline ───────────────────────────────────────────────────────
echo ""
echo "=== A-1. π0 baseline ==="
MODEL_PI0="physicalintelligence/pi0"
vllm serve "$MODEL_PI0" --omni --port $PORT &
SERVER_PID=$!
trap "pkill -f 'vllm serve' 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready (π0)."
bench_model "$MODEL_PI0" "pi0_baseline"
pkill -f "vllm serve" 2>/dev/null; sleep 6

# ── A-2. π0.5 comparison ──────────────────────────────────────────────────
echo ""
echo "=== A-2. π0.5 (new in 0.30.0) ==="
MODEL_PI05="physicalintelligence/pi05"
vllm serve "$MODEL_PI05" --omni --port $PORT &
SERVER_PID=$!
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready (π0.5)."
bench_model "$MODEL_PI05" "pi05_baseline"

# ── B. Observation frame sweep ─────────────────────────────────────────────
echo ""
echo "=== B. π0.5 observation frame sweep (1, 2, 4 frames) ==="
echo "frame_count,latency_ms" > "$RESULTS_DIR/frame_sweep.csv"

for NUM_FRAMES in 1 2 4; do
  echo ""
  echo "  -- num_frames=$NUM_FRAMES --"
  mkdir -p "$RESULTS_DIR/frames_$NUM_FRAMES"
  for i in "${!TASKS[@]}"; do
    START_MS=$(date +%s%3N)
    curl -s "http://localhost:$PORT/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"$MODEL_PI05\",
        \"messages\": [{\"role\": \"user\", \"content\": \"${TASKS[$i]}\"}],
        \"modalities\": [\"text\"],
        \"max_tokens\": 128,
        \"extra_body\": {\"num_observation_frames\": $NUM_FRAMES}
      }" > /dev/null
    END_MS=$(date +%s%3N)
    LATENCY=$((END_MS - START_MS))
    echo "    Task $((i+1)), frames=$NUM_FRAMES: ${LATENCY}ms"
    echo "$NUM_FRAMES,$LATENCY" >> "$RESULTS_DIR/frame_sweep.csv"
  done
done

echo ""
echo "Lab 2 complete. Results in $RESULTS_DIR"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. π0 vs π0.5 TTFT?"
echo "     π0.5 should be faster or similar quality at same compute"
echo "     If π0.5 is slower → larger backbone model; tradeoff quality vs speed"
echo "  2. Frame sweep: does latency scale linearly with num_frames?"
echo "     LINEAR   → vision encoder is the bottleneck (each frame processed separately)"
echo "     SUB-LINEAR → frames share encoder computation (efficient multi-frame encoding)"
echo "     SUPER-LINEAR → attention over frame tokens is O(frames²) → compress history"
echo "  3. At 4 frames, is latency still < 20ms target?"
echo "     If not → use 2 frames for realtime deployment, 4 frames for batch eval"
