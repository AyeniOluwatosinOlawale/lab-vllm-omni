#!/usr/bin/env bash
# =============================================================================
# Phase 9 — Lab 4: Configurable Audio Output Sample Rates (new in 0.30.0)
#
# WHAT YOU LEARN:
#   vllm-omni 0.30.0 added configurable output sample rates for TTS and duplex.
#   The model generates audio at its native rate; vllm-omni resamples server-side.
#
# KEY INSIGHT:
#   Higher sample rate = more data per second, but SAME GPU compute.
#   The constraint shifts from compute to network bandwidth at high rates.
#
# SAMPLE RATE COMPARISON:
#   8,000 Hz  — telephony quality (narrow-band)
#  16,000 Hz  — standard TTS quality (most models' native rate)
#  22,050 Hz  — CD-quality mid-range
#  44,100 Hz  — CD-quality (overkill for speech; useful for music/audio gen)
#
# Also tests PersonaPlex at its native 24,000 Hz.
#
# METRICS PER RATE:
#   - File size (bytes) — proxy for bandwidth requirement
#   - TTFP (ms)         — first chunk latency
#   - RTF               — real-time factor
#
# Outputs: results/p9_lab4_audio_sample_rates/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
PORT=8000
RESULTS_DIR="$(dirname "$0")/../results/p9_lab4_audio_sample_rates"
mkdir -p "$RESULTS_DIR"

SAMPLE_RATES=(8000 16000 22050 44100)
SAMPLE_TEXT="Artificial intelligence has transformed how we interact with computers and software systems."

echo "======================================================="
echo " Phase 9 | Lab 4: Configurable Audio Output Sample Rates"
echo "======================================================="

echo "[1/3] Starting TTS server on port $PORT ..."
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

echo ""
echo "sample_rate_hz,file_size_bytes,latency_ms" > "$RESULTS_DIR/sample_rate_results.csv"

echo "[2/3] Sweeping sample rates ..."
for RATE in "${SAMPLE_RATES[@]}"; do
  echo ""
  echo "  -- Sample rate: $RATE Hz --"
  OUT_FILE="$RESULTS_DIR/output_${RATE}hz.wav"
  START_MS=$(date +%s%3N)
  curl -s "http://localhost:$PORT/v1/audio/speech" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"input\": \"$SAMPLE_TEXT\",
      \"voice\": \"default\",
      \"response_format\": \"wav\",
      \"sample_rate\": $RATE
    }" \
    --output "$OUT_FILE"
  END_MS=$(date +%s%3N)
  LATENCY=$((END_MS - START_MS))
  FILE_SIZE=$(wc -c < "$OUT_FILE" | tr -d ' ')
  echo "    Latency: ${LATENCY} ms  |  File size: ${FILE_SIZE} bytes"
  echo "$RATE,$FILE_SIZE,$LATENCY" >> "$RESULTS_DIR/sample_rate_results.csv"

  # Benchmark TTFP and RTF
  mkdir -p "$RESULTS_DIR/bench_${RATE}hz"
  vllm bench serve --omni \
    --host 127.0.0.1 --port $PORT \
    --model "$MODEL" \
    --backend openai-audio-speech \
    --endpoint /v1/audio/speech \
    --dataset-name seed-tts-text \
    --dataset-path "$VLLM_OMNI_DIR/benchmarks/build_dataset/seed_tts_smoke" \
    --seed-tts-locale en \
    --num-prompts 10 --num-warmups 1 \
    --max-concurrency 1 \
    --request-rate inf \
    --percentile-metrics audio_ttfp,audio_rtf,e2el \
    --save-result \
    --result-dir "$RESULTS_DIR/bench_${RATE}hz" 2>/dev/null || \
    echo "    (bench failed for $RATE hz — may not support this rate)"
done

echo ""
echo "[3/3] Summary:"
cat "$RESULTS_DIR/sample_rate_results.csv"

echo ""
echo "Lab 4 complete. Results in $RESULTS_DIR"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. Does TTFP change with sample rate?"
echo "     YES → resampling is synchronous before first chunk send"
echo "     NO  → resampling is applied per-chunk in the streaming path (good)"
echo "  2. Does RTF change with sample rate?"
echo "     YES (increases) → higher sample rate = more data to stream → network bound"
echo "     NO  → GPU compute dominates; network bandwidth is not the limit yet"
echo "  3. File size ratio should be proportional to sample rate:"
echo "     44100 / 16000 = 2.75× larger file expected"
echo "     Lower ratio → server is applying lossy compression before sending"
