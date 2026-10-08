#!/usr/bin/env bash
# =============================================================================
# Phase 12 — Lab 1: GR00T N1.7 — NVIDIA Humanoid Robot Policy
#
# WHAT YOU LEARN:
#   GR00T N1.7 is NVIDIA's humanoid robot foundation model (NVIDIA-only GPU).
#   Unlike InternVLA-A1 (Phase 5) which uses Cosmos tokenizer for action tokens,
#   GR00T outputs joint velocities directly as floating-point tokens.
#
# OUTPUT COMPARISON:
#   InternVLA-A1: action tokens → Cosmos tokenizer decode → robot commands
#   GR00T N1.7:   joint velocity tokens (direct float output, no codec needed)
#
# KEY REQUIREMENT:
#   Action generation latency MUST be < robot control loop period.
#   Typical humanoid control: 50Hz = 20ms per step.
#   GR00T N1.7 target: < 20ms TTFT for real deployment.
#
# THREE TASKS:
#   1. Pick-and-place: reach + grasp + move + release
#   2. Push: contact and slide object
#   3. Reach: move end-effector to target position
#
# HARDWARE: NVIDIA GPU required. 1× 80GB is sufficient for N1.7 (small model).
# Outputs: results/p12_lab1_groot_n17/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="nvidia/GR00T-N1.7"
PORT=8091
RESULTS_DIR="$(dirname "$0")/../results/p12_lab1_groot_n17"
mkdir -p "$RESULTS_DIR"

TASKS=(
  "pick up the red block and place it in the blue container"
  "push the yellow cylinder to the right side of the table"
  "reach the green target marker above the workspace"
)

echo "======================================================="
echo " Phase 12 | Lab 1: GR00T N1.7 (NVIDIA Humanoid Policy)"
echo "======================================================="
echo "  NOTE: NVIDIA GPU required."
echo ""

echo "[1/3] Starting GR00T N1.7 server on port $PORT ..."
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

echo "task,latency_ms" > "$RESULTS_DIR/task_latency.csv"

# ── Task latency sweep ─────────────────────────────────────────────────────
echo ""
echo "[2/3] Task latency per action type ..."
for i in "${!TASKS[@]}"; do
  echo ""
  echo "  Task $((i+1)): ${TASKS[$i]}"
  START_MS=$(date +%s%3N)
  RESP=$(curl -s "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"messages\": [{
        \"role\": \"user\",
        \"content\": \"${TASKS[$i]}\"
      }],
      \"modalities\": [\"text\"],
      \"max_tokens\": 128
    }")
  END_MS=$(date +%s%3N)
  LATENCY=$((END_MS - START_MS))
  echo "  Latency: ${LATENCY} ms"
  echo "${TASKS[$i]//,/;},$LATENCY" >> "$RESULTS_DIR/task_latency.csv"
  echo "$RESP" | python3 -m json.tool | head -10
done

# ── Concurrency sweep c=1 vs c=2 ──────────────────────────────────────────
echo ""
echo "[3/3] Concurrency sweep (c=1 vs c=2) ..."
for C in 1 2; do
  mkdir -p "$RESULTS_DIR/concurrency_c$C"
  vllm bench serve --omni \
    --host localhost --port $PORT \
    --model "$MODEL" \
    --backend openai-chat-omni \
    --dataset-name random \
    --num-prompts 20 --max-concurrency "$C" \
    --random-input-len 128 --random-output-len 128 \
    --percentile-metrics ttft,tpot,e2el \
    --save-result \
    --result-dir "$RESULTS_DIR/concurrency_c$C"
done

echo ""
echo "Lab 1 complete. Results in $RESULTS_DIR"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. TTFT vs 20ms humanoid control loop target?"
echo "     TTFT < 20ms → GR00T can run at 50Hz realtime"
echo "     TTFT 20-50ms → run at 20-25Hz; acceptable for most manipulation"
echo "     TTFT > 100ms → too slow for realtime; use smaller model"
echo "  2. GR00T vs InternVLA-A1 (Phase 5) latency for same task?"
echo "     GR00T skips Cosmos tokenizer decode → should be faster"
echo "     If GR00T is slower → vision encoder is larger (N1.7 vs A1-3B)"
echo "  3. c=2 latency vs c=1?"
echo "     2× → fully serialised (GR00T may not support batched robot episodes)"
echo "     < 1.5× → some batching is happening"
