#!/usr/bin/env bash
# =============================================================================
# Phase 7 — Lab 4: Ming-flash-omni-2.0 — Speech + Image from One Endpoint
#
# WHAT YOU LEARN:
#   Ming-flash-omni-2.0 is unique: a SINGLE server endpoint supports BOTH
#   audio output (speech synthesis) AND image generation. Switch output
#   modality by changing the `modalities` parameter — no server restart.
#
#   This is the only model in vllm-omni 0.30.0 that routes audio and image
#   generation through the same /v1/chat/completions endpoint.
#
# THREE EXPERIMENTS:
#   A. Text → text output (baseline)
#   B. Text → audio output (speech, /v1/chat/completions with modalities=audio)
#   C. Text → image output (/v1/images/generations from the same server)
#
# WHAT TO COMPARE:
#   TTFP for audio vs E2E latency for image — reveals which output pipeline
#   is faster and what the codec/decoder overhead is for each modality.
#
# HARDWARE: 2× 80GB GPU recommended (Ming-flash-omni-2.0 is a large MoE model).
# Outputs: results/p7_lab4_ming_omni/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="inclusiveai/Ming-flash-omni-2.0"
PORT=8091
RESULTS_DIR="$(dirname "$0")/../results/p7_lab4_ming_omni"
mkdir -p "$RESULTS_DIR"

PROMPTS=(
  "Describe the water cycle in two sentences."
  "What are the three laws of thermodynamics?"
  "Explain why the sky is blue."
  "How does a neural network learn?"
  "What is the speed of light?"
)

echo "======================================================="
echo " Phase 7 | Lab 4: Ming-flash-omni-2.0 (Audio + Image)"
echo "======================================================="

echo "[1/4] Starting Ming-flash-omni-2.0 server on port $PORT ..."
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

# ── A. Text output ─────────────────────────────────────────────────────────
echo ""
echo "[2/4] A. Text output (baseline) ..."
mkdir -p "$RESULTS_DIR/text"
vllm bench serve --omni \
  --host localhost --port $PORT \
  --model "$MODEL" \
  --backend openai-chat-omni \
  --dataset-name random \
  --num-prompts 20 --max-concurrency 4 \
  --extra-body '{"modalities":["text"]}' \
  --percentile-metrics ttft,tpot,e2el \
  --save-result \
  --result-dir "$RESULTS_DIR/text"

# ── B. Audio output ────────────────────────────────────────────────────────
echo ""
echo "[3/4] B. Audio output (speech from same endpoint) ..."
mkdir -p "$RESULTS_DIR/audio"
vllm bench serve --omni \
  --host localhost --port $PORT \
  --model "$MODEL" \
  --backend openai-chat-omni \
  --dataset-name random \
  --num-prompts 20 --max-concurrency 4 \
  --extra-body '{"modalities":["text","audio"]}' \
  --percentile-metrics ttft,tpot,e2el,audio_ttfp,audio_rtf \
  --save-result \
  --result-dir "$RESULTS_DIR/audio"

# ── C. Image output ────────────────────────────────────────────────────────
echo ""
echo "[4/4] C. Image output (/v1/images/generations from same server) ..."
mkdir -p "$RESULTS_DIR/image"
python3 "$VLLM_OMNI_DIR/benchmarks/diffusion/diffusion_benchmark_serving.py" \
  --base-url "http://localhost:$PORT" \
  --model "$MODEL" \
  --task t2i \
  --dataset random \
  --num-prompts 10 \
  --max-concurrency 2 \
  --width 512 --height 512 \
  --num-inference-steps 20 \
  --output-file "$RESULTS_DIR/image/result.json"

echo ""
echo "Lab 4 complete. Results in $RESULTS_DIR"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. Compare audio TTFP vs image E2E latency."
echo "     Audio TTFP << image E2E: codec streams first chunk before full decode"
echo "     Audio TTFP >> image E2E: codec has high startup cost; check stage 1 init"
echo "  2. Does switching modality mid-request work? (no server restart needed)"
echo "     If latency spikes on first request after modality switch → engine is"
echo "     reinitialising the output decoder; subsequent requests should be fast."
echo "  3. At c=4, does mixing audio and image requests increase latency?"
echo "     YES → output router has contention; serve modalities on separate ports"
