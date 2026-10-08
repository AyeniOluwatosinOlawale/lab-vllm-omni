#!/usr/bin/env bash
# =============================================================================
# Phase 12 — Lab 3: OpenPI WebSocket Robot Policy — new in 0.30.0
#
# WHAT YOU LEARN:
#   0.30.0 added a dedicated WebSocket endpoint for robot policies:
#   ws /v1/realtime/robot/openpi
#
#   This is designed for REAL robot control loops that need:
#   - Sub-20ms action latency
#   - Persistent connection (no HTTP handshake overhead per request)
#   - Frame-by-frame observation streaming
#
# HTTP vs WebSocket for Robot Policies:
#   HTTP REST: +2-5ms per request for connection setup/teardown
#              Suitable for offline / batch evaluation
#   WebSocket: ~0ms overhead after connection; reuses session state
#              Required for realtime 50Hz control loops
#
# FLAGS:
#   --robot-openpi-idle-timeout N   seconds before idle session is closed
#
# TWO EXPERIMENTS:
#   A. HTTP REST latency (Phase 5 style) — baseline per-action overhead
#   B. WebSocket sustained 20Hz throughput — realtime performance
#
# Outputs: results/p12_lab3_openpi_websocket/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="physicalintelligence/pi0"
PORT=8091
RESULTS_DIR="$(dirname "$0")/../results/p12_lab3_openpi_websocket"
mkdir -p "$RESULTS_DIR"

NUM_STEPS=100  # number of steps to sustain for throughput measurement

echo "======================================================="
echo " Phase 12 | Lab 3: OpenPI WebSocket Robot Policy"
echo "======================================================="

echo "[1/3] Starting π0 server with OpenPI WebSocket support ..."
vllm serve "$MODEL" --omni --port $PORT \
  --robot-openpi-idle-timeout 60 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

# ── A. HTTP REST baseline ──────────────────────────────────────────────────
echo ""
echo "[2/3] A. HTTP REST baseline (20 actions via /v1/chat/completions) ..."
mkdir -p "$RESULTS_DIR/http_rest"
echo "request,latency_ms" > "$RESULTS_DIR/http_rest/latency.csv"
for i in $(seq 1 20); do
  START_MS=$(date +%s%3N)
  curl -s "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"messages\": [{\"role\": \"user\", \"content\": \"pick up the block\"}],
      \"modalities\": [\"text\"],
      \"max_tokens\": 64
    }" > /dev/null
  END_MS=$(date +%s%3N)
  LATENCY=$((END_MS - START_MS))
  echo "  Request $i: ${LATENCY}ms"
  echo "$i,$LATENCY" >> "$RESULTS_DIR/http_rest/latency.csv"
done

# ── B. WebSocket sustained throughput ─────────────────────────────────────
echo ""
echo "[3/3] B. WebSocket sustained $NUM_STEPS-step realtime test ..."
mkdir -p "$RESULTS_DIR/websocket"
python3 - << PYEOF
import asyncio, websockets, json, time, os, csv

RESULTS_DIR = "$RESULTS_DIR/websocket"
NUM_STEPS = $NUM_STEPS

async def openpi_websocket_test():
    uri = "ws://localhost:$PORT/v1/realtime/robot/openpi"
    latencies = []
    try:
        async with websockets.connect(uri, ping_interval=None, open_timeout=30) as ws:
            print(f"  WebSocket connected to {uri}")

            # Initial handshake
            await ws.send(json.dumps({
                "type": "session.init",
                "model": "$MODEL",
                "task": "pick up the red block"
            }))
            init_resp = await asyncio.wait_for(ws.recv(), timeout=15)
            print(f"  Init response: {init_resp[:100]}")

            # Sustained step loop
            overall_start = time.time()
            for step in range(NUM_STEPS):
                step_start = time.time()
                # Send observation
                await ws.send(json.dumps({
                    "type": "observation",
                    "step": step,
                    "image": "placeholder_base64_encoded_frame",
                    "joint_positions": [0.0] * 7
                }))
                # Receive action
                action_msg = await asyncio.wait_for(ws.recv(), timeout=5)
                step_end = time.time()
                latency_ms = (step_end - step_start) * 1000
                latencies.append(latency_ms)
                if step % 20 == 0:
                    print(f"    Step {step}/{NUM_STEPS}: {latency_ms:.1f}ms")

            overall_elapsed = time.time() - overall_start
            throughput = NUM_STEPS / overall_elapsed

            # Summary
            latencies_sorted = sorted(latencies)
            p50 = latencies_sorted[len(latencies_sorted)//2]
            p99 = latencies_sorted[int(len(latencies_sorted)*0.99)]
            mean = sum(latencies)/len(latencies)

            print(f"\n  === WebSocket Results ===")
            print(f"  Steps: {NUM_STEPS}  Total time: {overall_elapsed:.2f}s")
            print(f"  Throughput: {throughput:.1f} steps/sec  (target: 20 Hz = 20 steps/sec)")
            print(f"  Latency mean: {mean:.1f}ms  P50: {p50:.1f}ms  P99: {p99:.1f}ms")
            print(f"  (HTTP REST baseline: ~5ms overhead per request)")

            # Write CSV
            with open(f"{RESULTS_DIR}/ws_latency.csv", "w") as f:
                w = csv.writer(f)
                w.writerow(["step", "latency_ms"])
                for i, l in enumerate(latencies):
                    w.writerow([i, f"{l:.2f}"])

            with open(f"{RESULTS_DIR}/ws_summary.json", "w") as f:
                json.dump({
                    "num_steps": NUM_STEPS,
                    "throughput_hz": round(throughput, 2),
                    "latency_mean_ms": round(mean, 2),
                    "latency_p50_ms": round(p50, 2),
                    "latency_p99_ms": round(p99, 2)
                }, f, indent=2)

    except Exception as e:
        print(f"  WebSocket error: {e}")
        print("  (Server may not support OpenPI WebSocket yet; check 0.30.0 docs)")

asyncio.run(openpi_websocket_test())
PYEOF

echo ""
echo "Lab 3 complete. Results in $RESULTS_DIR"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. WebSocket throughput vs 20Hz target?"
echo "     > 20 steps/sec → can run at realtime 20Hz control loop"
echo "     < 20 steps/sec → action generation is too slow for realtime;"
echo "     use smaller policy model or reduce observation resolution"
echo "  2. HTTP REST P99 vs WebSocket P99?"
echo "     HTTP overhead = P99_http - P99_websocket (typically 2-5ms)"
echo "     Large gap → TCP handshake dominates; WebSocket essential for realtime"
echo "  3. WebSocket P99 > 50ms?"
echo "     YES → GPU inference is the bottleneck (not networking)"
echo "     Mitigation: reduce max_tokens, use smaller model, or add tensor parallelism"
