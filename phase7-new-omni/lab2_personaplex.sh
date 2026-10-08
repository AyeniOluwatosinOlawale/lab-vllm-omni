#!/usr/bin/env bash
# =============================================================================
# Phase 7 — Lab 2: PersonaPlex — Full-Duplex, 24 kHz (graduated in 0.30.0)
#
# WHAT YOU LEARN:
#   PersonaPlex graduated from experimental in 0.30.0. It runs full-duplex
#   at 24 kHz native output — compared to the standard 16 kHz of Qwen3-TTS.
#   Higher sample rate = richer audio quality but 50% more streaming bandwidth.
#
# KEY INSIGHT — RTF COMPARISON:
#   RTF is computed as: generation_time / audio_duration.
#   At 24 kHz, one second of audio is 24,000 samples (vs 16,000 at 16 kHz).
#   When comparing RTF between PersonaPlex (24 kHz) and Qwen3-TTS (16 kHz),
#   normalise by sample rate: RTF_norm = RTF × (native_rate / 16000).
#
# TWO EXPERIMENTS:
#   A. Full-duplex WebSocket benchmark (omniinteract dataset)
#   B. Realtime TTS via duplex endpoint — TTFP, RTF at 24 kHz
#
# HARDWARE: 1× 80GB GPU sufficient.
# Outputs: results/p7_lab2_personaplex/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="PersonaPlex/PersonaPlex-Duplex"
PORT=8091
RESULTS_DIR="$(dirname "$0")/../results/p7_lab2_personaplex"
mkdir -p "$RESULTS_DIR"

echo "======================================================="
echo " Phase 7 | Lab 2: PersonaPlex (24 kHz Full-Duplex)"
echo "======================================================="

echo "[1/3] Starting PersonaPlex server on port $PORT ..."
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

# ── A. Full-duplex WebSocket benchmark ─────────────────────────────────────
echo ""
echo "[2/3] A. Full-duplex benchmark (omniinteract dataset) ..."
mkdir -p "$RESULTS_DIR/duplex"
vllm bench serve --omni \
  --host localhost --port $PORT \
  --model "$MODEL" \
  --backend openai-realtime-duplex \
  --dataset-name omniinteract \
  --num-prompts 10 --max-concurrency 2 \
  --percentile-metrics audio_ttfp,audio_rtf,e2el \
  --save-result \
  --result-dir "$RESULTS_DIR/duplex"

# ── B. Realtime TTS via duplex endpoint ────────────────────────────────────
echo ""
echo "[3/3] B. Realtime TTS at 24 kHz ..."
mkdir -p "$RESULTS_DIR/tts_24k"
vllm bench serve --omni \
  --host localhost --port $PORT \
  --model "$MODEL" \
  --backend openai-realtime-tts \
  --dataset-name seed-tts-text \
  --dataset-path "$VLLM_OMNI_DIR/benchmarks/build_dataset/seed_tts_smoke" \
  --seed-tts-locale en \
  --num-prompts 20 --num-warmups 2 \
  --max-concurrency 1 \
  --request-rate inf \
  --percentile-metrics audio_ttfp,audio_rtf,audio_duration,audio_underrun \
  --save-result \
  --result-dir "$RESULTS_DIR/tts_24k"

echo ""
echo "Lab 2 complete. Results in $RESULTS_DIR"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. Normalise RTF for 24 kHz: RTF_norm = RTF × (24000/16000) = RTF × 1.5"
echo "     Compare RTF_norm to Qwen3-TTS from Phase 4 for an apples-to-apples comparison."
echo "  2. audio_underrun > 0 at c=2?"
echo "     YES → 24 kHz streaming requires 50% more bandwidth; reduce concurrency"
echo "  3. TTFP at duplex vs TTS endpoint?"
echo "     Duplex adds WebSocket session setup overhead (~5-15ms)."
echo "     If TTFP_duplex >> TTFP_tts, session handshake is the bottleneck."
