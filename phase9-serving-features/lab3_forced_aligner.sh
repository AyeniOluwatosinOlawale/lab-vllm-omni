#!/usr/bin/env bash
# =============================================================================
# Phase 9 — Lab 3: Forced Aligner — TTS Word Timestamps (new in 0.30.0)
#
# WHAT YOU LEARN:
#   The forced aligner runs as a lightweight CPU stage after the TTS codec
#   decode. It aligns the generated waveform against the input text to produce
#   per-word timestamps — enabling subtitles, karaoke display, and accessibility.
#
# HOW IT WORKS:
#   1. TTS model generates audio (GPU, as normal)
#   2. Forced aligner (CTC-based, CPU) aligns audio against input text
#   3. Response JSON includes word-level start/end times in seconds
#
# FLAGS:
#   --forced-aligner ctc-aligner    enables the aligner
#   --forced-aligner-device cpu     aligner runs on CPU (default)
#
# OVERHEAD:
#   Typically +5-15ms per request (aligner is small and runs alongside GPU decode)
#   Not a significant bottleneck for batch TTS.
#
# THREE EXPERIMENTS:
#   A. TTS without aligner — baseline TTFP and latency
#   B. TTS with aligner    — overhead measurement
#   C. Parse and display word timestamps from a sample response
#
# Outputs: results/p9_lab3_forced_aligner/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
PORT=8000
RESULTS_DIR="$(dirname "$0")/../results/p9_lab3_forced_aligner"
mkdir -p "$RESULTS_DIR"

SAMPLE_TEXT="The quick brown fox jumps over the lazy dog near the river bank on a sunny afternoon."

echo "======================================================="
echo " Phase 9 | Lab 3: Forced Aligner (Word Timestamps)"
echo "======================================================="

# ── A. Baseline: no aligner ────────────────────────────────────────────────
echo ""
echo "=== A. Baseline (no aligner) ==="
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

mkdir -p "$RESULTS_DIR/no_aligner"
vllm bench serve --omni \
  --host 127.0.0.1 --port $PORT \
  --model "$MODEL" \
  --backend openai-audio-speech \
  --endpoint /v1/audio/speech \
  --dataset-name seed-tts-text \
  --dataset-path "$VLLM_OMNI_DIR/benchmarks/build_dataset/seed_tts_smoke" \
  --seed-tts-locale en \
  --num-prompts 20 --num-warmups 2 \
  --max-concurrency 1 \
  --request-rate inf \
  --percentile-metrics audio_ttfp,audio_rtf,e2el \
  --save-result \
  --result-dir "$RESULTS_DIR/no_aligner"

kill $SERVER_PID 2>/dev/null || true
sleep 5

# ── B. With aligner ────────────────────────────────────────────────────────
echo ""
echo "=== B. With forced aligner (--forced-aligner ctc-aligner) ==="
vllm serve "$MODEL" --omni --port $PORT \
  --forced-aligner ctc-aligner \
  --forced-aligner-device cpu &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

mkdir -p "$RESULTS_DIR/with_aligner"
vllm bench serve --omni \
  --host 127.0.0.1 --port $PORT \
  --model "$MODEL" \
  --backend openai-audio-speech \
  --endpoint /v1/audio/speech \
  --dataset-name seed-tts-text \
  --dataset-path "$VLLM_OMNI_DIR/benchmarks/build_dataset/seed_tts_smoke" \
  --seed-tts-locale en \
  --num-prompts 20 --num-warmups 2 \
  --max-concurrency 1 \
  --request-rate inf \
  --percentile-metrics audio_ttfp,audio_rtf,e2el \
  --save-result \
  --result-dir "$RESULTS_DIR/with_aligner"

# ── C. Sample word timestamps ──────────────────────────────────────────────
echo ""
echo "=== C. Sample word-timestamp output ==="
curl -s "http://localhost:$PORT/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"input\": \"$SAMPLE_TEXT\",
    \"voice\": \"default\",
    \"response_format\": \"wav\"
  }" \
  --output "$RESULTS_DIR/sample_aligned.wav" \
  -D "$RESULTS_DIR/sample_headers.txt"

echo "  Response headers (look for X-Word-Timestamps):"
cat "$RESULTS_DIR/sample_headers.txt"

echo ""
echo "Lab 3 complete. Results in $RESULTS_DIR"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. Latency delta (with_aligner - no_aligner) per request?"
echo "     Delta < 10ms → aligner runs fully in parallel with GPU decode (good)"
echo "     Delta > 50ms → aligner is serialised after decode; check --forced-aligner-device"
echo "  2. Does aligner overhead scale with text length?"
echo "     YES (linear) → CTC alignment is O(n) in text length (expected)"
echo "     Use --forced-aligner-device cuda if CPU becomes the bottleneck at long texts"
echo "  3. Word timestamps present in response headers/body?"
echo "     Missing → model does not support aligner output format yet; check 0.30.0 docs"
