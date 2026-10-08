#!/usr/bin/env bash
# =============================================================================
# Phase 8 — Lab 4: /v1/audio/generate — OmniVoice + Stable-Audio-Open
#
# WHAT YOU LEARN:
#   /v1/audio/generate is a distinct endpoint from /v1/audio/speech:
#     /v1/audio/speech   → synthesise spoken text (TTS)
#     /v1/audio/generate → generate sound from a free-form description
#                          (ambient noise, sound effects, music ambience)
#
# NEW IN 0.30.0 — OmniVoice:
#   Variable-length attention enables batching requests of DIFFERENT durations
#   together. Previously, requests had to be padded to the same length, wasting
#   compute. At c=4 you should see significantly better GPU utilisation.
#
# TWO MODELS:
#   OmniVoice         — variable-length attention (0.30.0), general audio gen
#   Stable-Audio-Open — Stability AI sound design model
#
# List available voices first via GET /v1/audio/voices.
# Outputs: results/p8_lab4_audio_generate/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
PORT=8091
RESULTS_DIR="$(dirname "$0")/../results/p8_lab4_audio_generate"
mkdir -p "$RESULTS_DIR"

SOUND_PROMPTS=(
  "heavy rain falling on a metal roof with distant thunder"
  "busy city street with cars honking and people talking"
  "forest ambience with birds chirping and wind through trees"
  "ocean waves crashing on a rocky shore"
  "spaceship engine hum with electronic beeps"
)

echo "======================================================="
echo " Phase 8 | Lab 4: /v1/audio/generate (OmniVoice + Stable-Audio-Open)"
echo "======================================================="

# ── Part A: OmniVoice — variable-length attention ─────────────────────────
echo ""
echo "=== Part A: OmniVoice (variable-length attention, 0.30.0) ==="
MODEL_A="stabilityai/OmniVoice"
vllm serve "$MODEL_A" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

echo "  Available voices:"
curl -s "http://localhost:$PORT/v1/audio/voices" | python3 -m json.tool

mkdir -p "$RESULTS_DIR/omnivoice"
for C in 1 2 4; do
  echo ""
  echo "  -- OmniVoice concurrency=$C --"
  mkdir -p "$RESULTS_DIR/omnivoice/c$C"
  for i in "${!SOUND_PROMPTS[@]}"; do
    START_MS=$(date +%s%3N)
    curl -s "http://localhost:$PORT/v1/audio/generate" \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"$MODEL_A\",
        \"prompt\": \"${SOUND_PROMPTS[$i]}\",
        \"duration\": 10,
        \"response_format\": \"wav\"
      }" \
      --output "$RESULTS_DIR/omnivoice/c$C/sound_$i.wav"
    END_MS=$(date +%s%3N)
    echo "    Prompt $((i+1)): $((END_MS - START_MS)) ms"
  done
done

kill $SERVER_PID 2>/dev/null || true
sleep 5

# ── Part B: Stable-Audio-Open ─────────────────────────────────────────────
echo ""
echo "=== Part B: Stable-Audio-Open ==="
MODEL_B="stabilityai/stable-audio-open-1.0"
vllm serve "$MODEL_B" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

mkdir -p "$RESULTS_DIR/stable_audio"
for i in "${!SOUND_PROMPTS[@]}"; do
  START_MS=$(date +%s%3N)
  curl -s "http://localhost:$PORT/v1/audio/generate" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL_B\",
      \"prompt\": \"${SOUND_PROMPTS[$i]}\",
      \"duration\": 10,
      \"response_format\": \"wav\"
    }" \
    --output "$RESULTS_DIR/stable_audio/sound_$i.wav"
  END_MS=$(date +%s%3N)
  echo "  Prompt $((i+1)): $((END_MS - START_MS)) ms"
done

echo ""
echo "Lab 4 complete. Results in $RESULTS_DIR"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. OmniVoice c=1 vs c=4 throughput ratio?"
echo "     RATIO ≈ 4 → variable-length batching is working (ideal scaling)"
echo "     RATIO < 2 → requests aren't being batched; check --request-rate setting"
echo "     New in 0.30.0: variable-length attention removes the padding waste."
echo "  2. OmniVoice vs Stable-Audio-Open latency at c=1?"
echo "     OmniVoice uses diffusion; Stable-Audio-Open uses latent diffusion."
echo "     Latent diffusion is typically faster at the same quality level."
echo "  3. GET /v1/audio/voices returned results?"
echo "     Empty list → model does not support named voices (style-based only)"
