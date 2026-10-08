#!/usr/bin/env bash
# =============================================================================
# Phase 8 — Lab 3: ASR + Audio Understanding (MiMo-V2.5-ASR, MiMo-Audio-7B)
#
# WHAT YOU LEARN:
#   ASR (Automatic Speech Recognition) in vllm-omni is the REVERSE of TTS:
#     TTS (Phase 4): text → speech
#     ASR (this lab): speech → text
#
#   Both use /v1/chat/completions. For ASR, the audio is base64-encoded and
#   placed in the message content array alongside a transcription instruction.
#   Output is plain text tokens (same as LLM output).
#
# TWO MODELS:
#   MiMo-V2.5-ASR       — dedicated ASR model (speech → text transcript)
#   MiMo-Audio-7B-Instruct — audio QA model (listen to audio, answer questions)
#
# KEY METRICS:
#   WER (Word Error Rate) — if reference transcripts are available
#   TTFT                  — time to first transcription token
#   E2E latency           — total transcription time
#
# INPUT: audio files from seed_tts_smoke dataset (same as Phase 4)
# Outputs: results/p8_lab3_asr/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
PORT=8091
RESULTS_DIR="$(dirname "$0")/../results/p8_lab3_asr"
mkdir -p "$RESULTS_DIR"

SEED_TTS_DIR="$VLLM_OMNI_DIR/benchmarks/build_dataset/seed_tts_smoke"

echo "======================================================="
echo " Phase 8 | Lab 3: ASR + Audio Understanding"
echo "======================================================="

# ── Part A: MiMo-V2.5-ASR ─────────────────────────────────────────────────
echo ""
echo "=== Part A: MiMo-V2.5-ASR (speech transcription) ==="
MODEL_A="XiaomiMiMo/MiMo-V2.5-ASR"
vllm serve "$MODEL_A" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

mkdir -p "$RESULTS_DIR/mimo_asr"
vllm bench serve --omni \
  --host localhost --port $PORT \
  --model "$MODEL_A" \
  --backend openai-chat-omni \
  --dataset-name seed-tts-text \
  --dataset-path "$SEED_TTS_DIR" \
  --seed-tts-locale en \
  --num-prompts 20 --num-warmups 2 \
  --max-concurrency 4 \
  --extra-body '{"modalities":["text"]}' \
  --percentile-metrics ttft,tpot,e2el \
  --save-result \
  --result-dir "$RESULTS_DIR/mimo_asr"

kill $SERVER_PID 2>/dev/null || true
sleep 5

# ── Part B: MiMo-Audio-7B-Instruct ────────────────────────────────────────
echo ""
echo "=== Part B: MiMo-Audio-7B-Instruct (audio QA) ==="
MODEL_B="XiaomiMiMo/MiMo-Audio-7B-Instruct"
vllm serve "$MODEL_B" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

mkdir -p "$RESULTS_DIR/mimo_audio_qa"
# Audio QA: ask questions about audio content
vllm bench serve --omni \
  --host localhost --port $PORT \
  --model "$MODEL_B" \
  --backend openai-chat-omni \
  --dataset-name random-mm \
  --num-prompts 10 --max-concurrency 2 \
  --extra-body '{"modalities":["text"]}' \
  --percentile-metrics ttft,tpot,e2el \
  --save-result \
  --result-dir "$RESULTS_DIR/mimo_audio_qa"

echo ""
echo "Lab 3 complete. Results in $RESULTS_DIR"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. ASR TTFT vs TTS TTFP (Phase 4 comparison):"
echo "     ASR decodes audio tokens → text tokens (fast, ~LLM speed)"
echo "     TTS encodes text tokens → audio tokens (slower, codec bound)"
echo "     TTFT_ASR should be much lower than TTFP_TTS for same audio length."
echo "  2. Does ASR latency scale with audio duration?"
echo "     YES (linear) → audio encoder processes all frames before decoding"
echo "     YES (sub-linear) → chunked audio encoding (streaming-friendly)"
echo "  3. Audio QA TTFT vs ASR TTFT?"
echo "     Audio QA adds reasoning on top of transcription → expect higher TTFT"
echo "     Large gap → model is doing chain-of-thought before answering"
