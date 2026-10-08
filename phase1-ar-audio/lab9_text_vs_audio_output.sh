#!/usr/bin/env bash
# =============================================================================
# Phase 1 — Lab 9: Text-Only vs Text+Audio Output Path
#
# What you learn:
#   Isolates the cost of audio synthesis by running identical workloads
#   with modalities=["text"] vs modalities=["text","audio"].
#
#   modalities=["text"]       → only Stage 0 (Thinker) runs
#   modalities=["text","audio"] → all 3 stages run (Thinker → Talker → Code2Wav)
#
#   Key deltas to observe:
#     TTFT       — should be identical (both wait for Thinker first token)
#     E2E        — text-only finishes when Thinker completes;
#                  text+audio finishes when Code2Wav completes
#     audio_ttfp — only present in text+audio run
#     audio_rtf  — only present in text+audio run
#
# Outputs: results/p1_lab9_text_only/  results/p1_lab9_text_audio/
# =============================================================================
set -euo pipefail
MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"
PORT=8091
RESULTS_DIR="$(dirname "$0")/../results"
mkdir -p "$RESULTS_DIR"

echo "======================================================="
echo " Phase 1 | Lab 9: Text-Only vs Text+Audio Output"
echo "======================================================="

echo "[1/3] Starting server (async_chunk ON) on port $PORT ..."
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

# Run A — text only (Thinker stage only)
echo ""
echo "[2/3] Benchmark A: modalities=[text] — Thinker only ..."
vllm bench serve --omni \
  --host localhost --port $PORT \
  --model "$MODEL" \
  --backend openai-chat-omni \
  --dataset-name random \
  --num-prompts 20 \
  --max-concurrency 4 \
  --random-input-len 2500 \
  --random-output-len 900 \
  --ignore-eos \
  --extra-body '{"modalities":["text"]}' \
  --percentile-metrics ttft,tpot,itl,e2el \
  --save-result \
  --result-dir "$RESULTS_DIR/p1_lab9_text_only"

# Run B — text + audio (all 3 stages)
echo ""
echo "[3/3] Benchmark B: modalities=[text,audio] — all 3 stages ..."
vllm bench serve --omni \
  --host localhost --port $PORT \
  --model "$MODEL" \
  --backend openai-chat-omni \
  --dataset-name random \
  --num-prompts 20 \
  --max-concurrency 4 \
  --random-input-len 2500 \
  --random-output-len 900 \
  --ignore-eos \
  --extra-body '{"modalities":["text","audio"]}' \
  --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf \
  --save-result \
  --result-dir "$RESULTS_DIR/p1_lab9_text_audio"

echo ""
echo "Lab 9 complete."
echo ""
echo "  Key comparison (open result JSONs):"
echo "    median_ttft_ms    — should be ~equal between A and B"
echo "    median_e2el_ms    — B is significantly higher (adds Talker + Code2Wav)"
echo "    median_audio_ttfp_ms — only in B (time to first audio packet)"
echo "    median_audio_rtf     — only in B (audio generation speed)"
echo ""
echo "Next: bash phase1-ar-audio/lab10_stage_profiler.sh"
