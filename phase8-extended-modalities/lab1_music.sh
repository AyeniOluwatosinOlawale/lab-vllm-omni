#!/usr/bin/env bash
# =============================================================================
# Phase 8 — Lab 1: Music Generation (MiniMax Music 3 + YuE2-3B)
#
# WHAT YOU LEARN:
#   Music generation uses the /v1/audio/generate endpoint — NOT /v1/audio/speech.
#   /v1/audio/speech synthesises text as speech; /v1/audio/generate creates
#   music, ambient sound, or scored audio from a free-form description prompt.
#   Output is a complete audio file (not a streaming chunk sequence).
#
# KEY DIFFERENCE FROM TTS:
#   TTS (Phase 4): input = text to speak, output = that text spoken aloud
#   Music gen:     input = description of desired sound, output = composed audio
#
# TWO MODELS:
#   MiniMax Music 3 — high-quality music composition with instruments + vocals
#   YuE2-3B         — text-to-music with ABC plan (structured musical notation)
#
# LATENCY PROFILE:
#   Music generation is dominated by the full generation length — a 30-second
#   track requires generating ~30s × sample_rate audio tokens before the first
#   byte can be sent (no streaming). E2E latency = generation time, not TTFP.
#
# Outputs: results/p8_lab1_music/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
PORT=8092
RESULTS_DIR="$(dirname "$0")/../results/p8_lab1_music"
mkdir -p "$RESULTS_DIR"

MUSIC_PROMPTS=(
  "upbeat electronic track with synthesizer lead and four-on-the-floor kick drum"
  "calm acoustic guitar fingerpicking with ambient forest background sounds"
  "orchestral cinematic piece with strings building to a brass crescendo"
  "lo-fi hip hop beat with vinyl crackle and mellow piano chords"
  "energetic jazz fusion with alto saxophone solo and walking bass line"
)

echo "======================================================="
echo " Phase 8 | Lab 1: Music Generation"
echo "======================================================="

# ── Part A: MiniMax Music 3 ────────────────────────────────────────────────
echo ""
echo "=== Part A: MiniMax Music 3 ==="
MODEL_A="MiniMaxAI/MiniMax-Music-3"
vllm serve "$MODEL_A" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

mkdir -p "$RESULTS_DIR/minimax_music3"
for i in "${!MUSIC_PROMPTS[@]}"; do
  echo "  Prompt $((i+1))/${#MUSIC_PROMPTS[@]}: ${MUSIC_PROMPTS[$i]}"
  START_MS=$(date +%s%3N)
  curl -s "http://localhost:$PORT/v1/audio/generate" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL_A\",
      \"prompt\": \"${MUSIC_PROMPTS[$i]}\",
      \"duration\": 20,
      \"response_format\": \"wav\"
    }" \
    --output "$RESULTS_DIR/minimax_music3/track_$i.wav"
  END_MS=$(date +%s%3N)
  echo "  E2E latency: $((END_MS - START_MS)) ms  → saved track_$i.wav"
done

kill $SERVER_PID 2>/dev/null || true
sleep 5

# ── Part B: YuE2-3B ───────────────────────────────────────────────────────
echo ""
echo "=== Part B: YuE2-3B (text-to-music with ABC plan) ==="
MODEL_B="YuE-AI/YuE2-3B"
vllm serve "$MODEL_B" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

mkdir -p "$RESULTS_DIR/yue2"
for i in "${!MUSIC_PROMPTS[@]}"; do
  echo "  Prompt $((i+1))/${#MUSIC_PROMPTS[@]}: ${MUSIC_PROMPTS[$i]}"
  START_MS=$(date +%s%3N)
  curl -s "http://localhost:$PORT/v1/audio/generate" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL_B\",
      \"prompt\": \"${MUSIC_PROMPTS[$i]}\",
      \"duration\": 20,
      \"response_format\": \"wav\"
    }" \
    --output "$RESULTS_DIR/yue2/track_$i.wav"
  END_MS=$(date +%s%3N)
  echo "  E2E latency: $((END_MS - START_MS)) ms  → saved track_$i.wav"
done

echo ""
echo "Lab 1 complete. Results in $RESULTS_DIR"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. Does E2E latency scale linearly with --duration?"
echo "     YES → autoregressive token generation (expected); use longer warmup"
echo "     NO  → VAE decode or post-processing is dominant for short durations"
echo "  2. MiniMax Music 3 vs YuE2-3B latency for same prompt?"
echo "     YuE2 uses ABC notation planning (extra pre-processing step)"
echo "     If YuE2 E2E >> MiniMax → ABC plan generation is the bottleneck"
echo "  3. GPU memory: music models need headroom for long sequence generation"
echo "     OOM at duration>30s → reduce --max-model-len or use quantization"
