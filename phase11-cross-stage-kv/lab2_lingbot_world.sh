#!/usr/bin/env bash
# =============================================================================
# Phase 11 — Lab 2: LingBot-World 2.0 Interactive World-Model Streaming (0.30.0)
#
# WHAT YOU LEARN:
#   LingBot-World is a world model — it generates a consistent video of an
#   explorable environment. Unlike text-to-video (Phase 3), you can steer the
#   scene mid-generation by injecting camera control commands.
#
# NEW IN 0.30.0:
#   Stepwise generation with mid-stream camera control
#   Session-owned streaming VAE decode
#   Bounded async MP4 chunk transfer
#   WebSocket endpoint: ws /v1/realtime/video
#
# KEY METRIC — VIDEO_RTF:
#   VIDEO_RTF = video_generation_time / video_playback_duration
#   If VIDEO_RTF > 1.0: generation falls behind real-time → playback stutters
#   Target: VIDEO_RTF < 0.8 for smooth playback
#   Ulysses SP (--usp) is needed at 720p to keep VIDEO_RTF < 1.0
#
# THREE EXPERIMENTS:
#   A. Synchronous video generation (HTTP /v1/videos/sync)
#   B. WebSocket streaming (/v1/realtime/video) with mid-stream camera control
#   C. LingBot-World benchmark (TTFC, chunk inter-arrival, VIDEO_RTF, underruns)
#
# Outputs: results/p11_lab2_lingbot_world/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="LingBot/LingBot-World-2.0"
PORT=8099
RESULTS_DIR="$(dirname "$0")/../results/p11_lab2_lingbot_world"
mkdir -p "$RESULTS_DIR"

echo "======================================================="
echo " Phase 11 | Lab 2: LingBot-World 2.0 (World-Model Streaming)"
echo "======================================================="

echo "[1/4] Starting LingBot-World server with Ulysses SP ..."
vllm serve "$MODEL" --omni --port $PORT --usp &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

# ── A. Synchronous video generation ────────────────────────────────────────
echo ""
echo "[2/4] A. Synchronous world-model generation (/v1/videos/sync) ..."
mkdir -p "$RESULTS_DIR/sync"
PROMPTS=(
  "a forest clearing with sunlight filtering through tall oak trees"
  "a busy marketplace with colorful stalls and crowds of people"
  "an empty beach at sunset with gentle waves"
)
for i in "${!PROMPTS[@]}"; do
  echo "  Prompt $((i+1)): ${PROMPTS[$i]}"
  START_MS=$(date +%s%3N)
  curl -s -X POST "http://localhost:$PORT/v1/videos/sync" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"prompt\": \"${PROMPTS[$i]}\",
      \"width\": 832, \"height\": 480,
      \"num_frames\": 49,
      \"fps\": 24
    }" \
    --output "$RESULTS_DIR/sync/world_$i.mp4"
  END_MS=$(date +%s%3N)
  echo "  E2E: $((END_MS - START_MS)) ms  → world_$i.mp4"
done

# ── B. WebSocket streaming with camera control ─────────────────────────────
echo ""
echo "[3/4] B. WebSocket streaming with mid-stream camera control ..."
mkdir -p "$RESULTS_DIR/ws_stream"
python3 - << 'PYEOF'
import asyncio, websockets, json, os

RESULTS_DIR = os.environ.get('RESULTS_DIR', '.') + '/ws_stream'

async def world_model_stream():
    uri = "ws://localhost:8099/v1/realtime/video"
    try:
        async with websockets.connect(uri, ping_interval=None) as ws:
            # Send initial scene
            await ws.send(json.dumps({
                "type": "scene.create",
                "prompt": "a volcanic island with erupting volcano and lava flows",
                "width": 832, "height": 480,
                "fps": 24
            }))
            print("  Scene creation sent.")

            chunks = 0
            import time
            start = time.time()
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=30)
                data = json.loads(msg) if isinstance(msg, str) else {"type": "binary_chunk"}
                chunk_type = data.get("type", "unknown")

                if chunk_type == "video.chunk":
                    chunks += 1
                    if chunks == 3:
                        # Inject camera control mid-stream
                        await ws.send(json.dumps({
                            "type": "camera.control",
                            "action": "pan_right",
                            "speed": 0.5
                        }))
                        print(f"  Camera control injected at chunk {chunks}.")
                elif chunk_type == "video.done":
                    elapsed = time.time() - start
                    print(f"  Stream complete: {chunks} chunks in {elapsed:.1f}s")
                    break
    except Exception as e:
        print(f"  WebSocket error: {e}")

asyncio.run(world_model_stream())
PYEOF

# ── C. LingBot-World benchmark ─────────────────────────────────────────────
echo ""
echo "[4/4] C. LingBot-World realtime benchmark (TTFC, VIDEO_RTF, underruns) ..."
mkdir -p "$RESULTS_DIR/bench"
python3 "$VLLM_OMNI_DIR/benchmarks/lingbot_world/benchmark_lingbot_world_realtime.py" \
  --host localhost --port $PORT \
  --num-sessions 5 \
  --output "$RESULTS_DIR/bench/results.json" \
  2>/dev/null || echo "  (benchmark script not found; check VLLM_OMNI_DIR path)"

echo ""
echo "Lab 2 complete. Results in $RESULTS_DIR"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. VIDEO_RTF from benchmark?"
echo "     RTF > 1.0 at 480p → GPU is generation-bound; add more GPUs or reduce frames"
echo "     RTF < 0.8 → generation is faster than realtime; smooth playback guaranteed"
echo "  2. Chunk inter-arrival P99 > 100ms → jitter in chunk delivery"
echo "     Cause: VAE decode is blocking between chunks"
echo "     Fix: --enable-cuda-graph-decode for VAE (new in 0.30.0)"
echo "  3. Camera control response latency?"
echo "     < 50ms → world model correctly buffers and steers mid-stream (good)"
echo "     > 200ms → stepwise generation is not pipelined correctly"
