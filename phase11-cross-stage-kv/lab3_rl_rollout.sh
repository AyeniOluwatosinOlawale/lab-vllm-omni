#!/usr/bin/env bash
# =============================================================================
# Phase 11 — Lab 3: RL Rollout API (DreamZero-DROID) — new in 0.30.0
#
# WHAT YOU LEARN:
#   The RL rollout API (/v1/realtime/sessions/*) is entirely new in 0.30.0.
#   It treats the model as an ENVIRONMENT rather than an inference function.
#
# COMPARISON TO PHASE 5 (InternVLA-A1):
#   Phase 5: model IS the policy (observes world, outputs actions)
#   Phase 11 lab3: model IS the environment (takes actions, outputs next observation)
#   DreamZero-DROID generates video latents as the "world state" for RL policies.
#
# FIVE NEW HTTP ENDPOINTS:
#   POST /v1/realtime/sessions               → create session, get session_id
#   POST /v1/realtime/sessions/{id}/step     → advance one environment step
#   GET  /v1/realtime/sessions/{id}/status   → check session state
#   POST /v1/realtime/sessions/{id}/reset    → reset to initial state
#   POST /v1/realtime/sessions/{id}/close    → cleanup and free GPU memory
#
# SESSION LIFECYCLE:
#   create → [step × N] → reset → [step × N] → ... → close
#
# Outputs: results/p11_lab3_rl_rollout/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="DreamZero/DreamZero-DROID"
PORT=8091
RESULTS_DIR="$(dirname "$0")/../results/p11_lab3_rl_rollout"
mkdir -p "$RESULTS_DIR"

NUM_EPISODES=3
STEPS_PER_EPISODE=10

echo "======================================================="
echo " Phase 11 | Lab 3: RL Rollout API (DreamZero-DROID)"
echo "======================================================="

echo "[1/2] Starting DreamZero-DROID server on port $PORT ..."
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

echo ""
echo "[2/2] Running $NUM_EPISODES episodes × $STEPS_PER_EPISODE steps ..."

echo "episode,step,step_latency_ms,status" > "$RESULTS_DIR/rollout_log.csv"

for EP in $(seq 1 $NUM_EPISODES); do
  echo ""
  echo "  === Episode $EP / $NUM_EPISODES ==="

  # Create session
  SESSION_RESP=$(curl -s -X POST "http://localhost:$PORT/v1/realtime/sessions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"task\": \"manipulation\"}")
  SESSION_ID=$(echo "$SESSION_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('session_id',''))" 2>/dev/null)
  echo "  Session ID: $SESSION_ID"

  if [ -z "$SESSION_ID" ]; then
    echo "  ERROR: Failed to create session. Response: $SESSION_RESP"
    continue
  fi

  # Check session status
  curl -s "http://localhost:$PORT/v1/realtime/sessions/$SESSION_ID/status" \
    | python3 -m json.tool

  # Run steps
  for STEP in $(seq 1 $STEPS_PER_EPISODE); do
    START_MS=$(date +%s%3N)
    STEP_RESP=$(curl -s -X POST \
      "http://localhost:$PORT/v1/realtime/sessions/$SESSION_ID/step" \
      -H "Content-Type: application/json" \
      -d "{
        \"session_id\": \"$SESSION_ID\",
        \"action\": {\"joint_positions\": [0.0, 0.1, 0.2, 0.0, 0.0, 0.0], \"gripper\": 0.5}
      }")
    END_MS=$(date +%s%3N)
    LATENCY=$((END_MS - START_MS))
    STATUS=$(echo "$STEP_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null || echo "error")
    echo "    Step $STEP: ${LATENCY}ms  status=$STATUS"
    echo "$EP,$STEP,$LATENCY,$STATUS" >> "$RESULTS_DIR/rollout_log.csv"
  done

  # Reset session (start new episode without full session creation overhead)
  if [ "$EP" -lt "$NUM_EPISODES" ]; then
    curl -s -X POST "http://localhost:$PORT/v1/realtime/sessions/$SESSION_ID/reset" \
      -H "Content-Type: application/json" \
      -d "{\"session_id\": \"$SESSION_ID\"}" > /dev/null
    echo "  Session reset for next episode."
  fi

  # Close session on last episode
  if [ "$EP" -eq "$NUM_EPISODES" ]; then
    curl -s -X POST "http://localhost:$PORT/v1/realtime/sessions/$SESSION_ID/close" \
      -H "Content-Type: application/json" \
      -d "{\"session_id\": \"$SESSION_ID\"}" > /dev/null
    echo "  Session closed."
  fi
done

echo ""
echo "Lab 3 complete. Results in $RESULTS_DIR"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. Step latency: mean and P99 from rollout_log.csv?"
echo "     Target: < 50ms per step for 20Hz RL control loop"
echo "     > 100ms → model is too large for realtime RL; use smaller backbone"
echo "  2. Create vs reset latency?"
echo "     Create involves session state allocation; reset reuses existing state"
echo "     If reset latency ≈ create latency → state reuse not working; check logs"
echo "  3. Throughput: (NUM_EPISODES × STEPS_PER_EPISODE) / total_time (steps/sec)?"
echo "     Compare to VLA action generation (Phase 5) — different bottlenecks:"
echo "     RL rollout is compute-heavy (world-model); VLA is vision-encoder-heavy"
