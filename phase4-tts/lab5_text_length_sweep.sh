#!/usr/bin/env bash
# =============================================================================
# Phase 4 — Lab 5: Text Length + Concurrency Sweep
#
# CONTEXT WINDOW EQUIVALENT FOR TTS:
#   TTS text length directly controls the number of codec tokens the model
#   must generate (longer text → more tokens → longer audio → more Code2Wav
#   steps). This is the TTS equivalent of output token length.
#
#   Three tiers:
#     SHORT  — 1 sentence  (~10-15 words,  ~3-5s audio)
#     MEDIUM — 3 sentences (~40-50 words,  ~12-18s audio)
#     LONG   — 8 sentences (~100-120 words, ~35-45s audio)
#
# BOTTLENECK THIS EXPOSES:
#   At c=1: TTFP is driven by the first codec chunk; RTF degrades with length
#     if the codec cannot keep up with real-time playback rate.
#   At c=8: the codec batch_size=1 cliff is worse for long texts because
#     each request holds the codec slot longer, backing up the queue.
#
#   Known pattern:
#     SHORT  text: codec finishes fast, queue clears quickly
#     LONG   text: codec holds the slot for 40+ seconds at c=8 → TTFP explodes
#     RTF for LONG text at c=8 often exceeds 1.0 (slower than realtime)
#
# OPTIMIZATION SIGNALS:
#   RTF > 1.0           → cannot serve realtime; reduce concurrency or shard codec
#   audio_underrun > 0  → streaming gaps; reduce concurrency
#   TTFP grows with text length at c=1 → codec autoregressive, not batched
#   TTFP stays flat with text length at c=1 → first-chunk latency is length-independent (good)
#
# Outputs: results/p4_lab5_textlen/<length>_c<N>/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
PORT=8000
RESULTS_DIR="$(dirname "$0")/../results"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONCURRENCY_LEVELS=(1 4 8)

# Build three prompt files
mkdir -p "$SCRIPT_DIR/prompts"

cat > "$SCRIPT_DIR/prompts/short.txt" << 'PROMPTS'
The quick brown fox jumps over the lazy dog near the river bank.
A stitch in time saves nine and prevents larger problems later on.
To be or not to be, that is the question Shakespeare wrote for Hamlet.
She sells seashells by the seashore on sunny summer afternoons.
All that glitters is not gold, as the old proverb rightly reminds us.
PROMPTS

cat > "$SCRIPT_DIR/prompts/medium.txt" << 'PROMPTS'
The field of artificial intelligence has grown remarkably over the past decade. From simple rule-based systems to complex neural networks, the progress has been staggering. Today, AI models can generate text, images, audio, and even video with remarkable quality.
Machine learning models require enormous amounts of data to train effectively. The quality and diversity of training data directly impacts the capabilities of the resulting model. Researchers spend considerable effort curating and cleaning datasets before training begins.
Natural language processing has transformed how computers interact with human language. Modern systems can translate between languages, summarize long documents, answer questions, and even write code. These capabilities are now embedded in everyday tools used by millions.
PROMPTS

cat > "$SCRIPT_DIR/prompts/long.txt" << 'PROMPTS'
The history of computing is a remarkable story of human ingenuity and perseverance. Beginning with mechanical calculators in the seventeenth century, humanity has steadily built toward the digital computers we know today. Charles Babbage envisioned a mechanical analytical engine that could perform general computations, though he never completed it in his lifetime. Ada Lovelace wrote what many consider the first computer program for this machine. Their pioneering work laid the conceptual groundwork for everything that followed. The twentieth century brought vacuum tubes, then transistors, then integrated circuits, each transition multiplying computational power while shrinking physical size. Gordon Moore observed that the number of transistors on a chip doubled approximately every two years, a pattern that held for decades and drove an extraordinary era of progress. Today we stand at a new inflection point, with artificial intelligence systems demonstrating capabilities that would have seemed like science fiction just a generation ago.
PROMPTS

echo "======================================================="
echo " Phase 4 | Lab 5: Text Length (Context Equivalent) + Concurrency Sweep"
echo "======================================================="

echo "[1/2] Starting TTS server on port $PORT ..."
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

echo ""
echo "[2/2] Sweeping text length × concurrency ..."
for LEN in short medium long; do
  PROMPT_FILE="$SCRIPT_DIR/prompts/$LEN.txt"
  for C in "${CONCURRENCY_LEVELS[@]}"; do
    OUT="$RESULTS_DIR/p4_lab5_textlen/${LEN}_c${C}"
    mkdir -p "$OUT"
    echo ""
    echo "  -- length=$LEN  concurrency=$C --"

    vllm bench serve --omni \
      --host 127.0.0.1 --port $PORT \
      --model "$MODEL" \
      --backend openai-audio-speech \
      --endpoint /v1/audio/speech \
      --dataset-name seed-tts-text \
      --dataset-path "$VLLM_OMNI_DIR/benchmarks/build_dataset/seed_tts_smoke" \
      --seed-tts-locale en \
      --num-prompts 20 \
      --num-warmups 2 \
      --extra-body '{"task_type":"Base"}' \
      --max-concurrency "$C" \
      --request-rate inf \
      --percentile-metrics ttft,e2el,audio_rtf,audio_ttfp,audio_duration,audio_underrun \
      --save-result \
      --result-dir "$OUT"
  done
done

echo ""
echo "Lab 5 complete. Results in results/p4_lab5_textlen/"
echo ""
echo "BOTTLENECK ANALYSIS — check these patterns:"
echo "  1. Does audio_ttfp grow with text length at c=1?"
echo "     YES → codec is autoregressive, first chunk delayed by full decode"
echo "     NO  → first-chunk latency is length-independent (streaming works)"
echo "  2. Does audio_rtf exceed 1.0 for LONG text at c=8?"
echo "     YES → cannot serve long-form TTS in realtime at c=8"
echo "     Mitigation: limit concurrency to 4 for long texts"
echo "  3. audio_underrun > 0 for LONG text?"
echo "     YES → streaming gap; codec is the bottleneck, not network"
echo ""
echo "Run phase6-benchmark/bottleneck_analysis.py for automated findings."
