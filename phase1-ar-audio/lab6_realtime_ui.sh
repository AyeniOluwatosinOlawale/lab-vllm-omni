#!/usr/bin/env bash
# =============================================================================
# Phase 1 — Lab 6: Full-Duplex Realtime Web UI (VAD + Camera)
#
# What you learn:
#   OpenAI Realtime-compatible WebSocket protocol with:
#     - Server-side VAD: auto-commits turn after 500ms silence
#     - Barge-in: speak mid-response to interrupt
#     - Camera: JPEG frames sent alongside audio
#     - Playback acknowledgement: engine tracks what's been played
#
#   NOTE: async_chunk must be OFF for /v1/realtime endpoint.
#   NOTE: Update configs/qwen3_vad.yaml with the absolute path to
#         your silero_vad.onnx file before running.
#
#         Download: huggingface-cli download istupakov/silero-vad-onnx silero_vad.onnx
#
# Usage:
#   Terminal 1: bash phase1-ar-audio/lab6_realtime_ui.sh backend
#   Terminal 2: bash phase1-ar-audio/lab6_realtime_ui.sh ui
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"
BACKEND_PORT=8091
UI_PORT=7863
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VAD_CONFIG="$SCRIPT_DIR/configs/qwen3_vad.yaml"
MODE="${1:-backend}"

case "$MODE" in
  backend)
    echo "======================================================="
    echo " Phase 1 | Lab 6: Starting VAD Backend"
    echo "======================================================="
    echo ""
    echo "Config: $VAD_CONFIG"
    echo "  async_chunk is disabled automatically for /v1/realtime."
    echo ""
    vllm serve "$MODEL" \
      --omni \
      --no-async-chunk \
      --port $BACKEND_PORT \
      --deploy-config "$VAD_CONFIG"
    ;;

  ui)
    echo "======================================================="
    echo " Phase 1 | Lab 6: Starting Realtime Web UI"
    echo "======================================================="
    echo ""
    echo "Waiting for backend on port $BACKEND_PORT ..."
    until curl -sf "http://127.0.0.1:$BACKEND_PORT/health" > /dev/null 2>&1; do
      sleep 3
    done
    echo "Backend ready. Starting UI on http://localhost:$UI_PORT ..."
    echo ""
    python3 -m examples.online_serving.realtime_web \
      --profile qwen3-turn \
      --backend "ws://127.0.0.1:$BACKEND_PORT" \
      --vad \
      --port $UI_PORT
    ;;

  *)
    echo "Usage: $0 [backend|ui]"
    echo "  Run 'backend' in Terminal 1, then 'ui' in Terminal 2."
    exit 1
    ;;
esac

# -----------------------------------------------------------------------
# What to test in the browser (http://localhost:7863):
#
#   1. Basic VAD turn:
#      Speak a question → pause → server detects silence → response plays
#
#   2. Barge-in:
#      Start speaking while response is playing → interrupts generation
#
#   3. Camera mode:
#      Click [Camera] → allow browser access → speak about what you see
#      UI sends 1 JPEG/sec alongside audio
#
#   4. Multi-turn:
#      Complete 3+ turns on the same connection — verify conversation
#      history is maintained
# -----------------------------------------------------------------------
