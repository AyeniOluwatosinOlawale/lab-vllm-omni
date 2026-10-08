#!/usr/bin/env bash
# =============================================================================
# Phase 1 — Lab 4: Concurrency Sweep
#
# What you learn:
#   How TTFT, TTFP, and RTF degrade under load as concurrent requests
#   compete for the 3-stage pipeline. The per-stage max_num_seqs: 64 in
#   the default deploy YAML is the hard ceiling.
#
#   Sweep: concurrency 1 → 2 → 4 → 8 → 16 → 32
#   Input: mixed multimodal (image + audio)
#   Output: text + audio
#
# Outputs: results/p1_lab4_concurrency/c<N>/
# =============================================================================
set -euo pipefail
MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"
PORT=8091
RESULTS_DIR="$(dirname "$0")/../results"
CONCURRENCY_LEVELS=(1 2 4 8 16 32)

echo "======================================================="
echo " Phase 1 | Lab 4: Concurrency Sweep"
echo "======================================================="

echo "[1/2] Starting server (async_chunk ON) on port $PORT ..."
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."

echo ""
echo "[2/2] Sweeping concurrency levels: ${CONCURRENCY_LEVELS[*]} ..."
for C in "${CONCURRENCY_LEVELS[@]}"; do
  OUT="$RESULTS_DIR/p1_lab4_concurrency/c$C"
  mkdir -p "$OUT"
  echo ""
  echo "  --- Concurrency: $C ---"

  vllm bench serve --omni \
    --host localhost --port $PORT \
    --model "$MODEL" \
    --backend openai-chat-omni \
    --dataset-name random-mm \
    --num-prompts 40 \
    --max-concurrency "$C" \
    --random-input-len 2500 \
    --random-output-len 900 \
    --random-mm-base-items-per-request 2 \
    --random-mm-limit-mm-per-prompt '{"image":1,"audio":1}' \
    --random-mm-bucket-config '{"(32, 32, 1)": 0.5, "(0, 1, 1)": 0.5}' \
    --ignore-eos \
    --extra-body '{"modalities":["text","audio"]}' \
    --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf \
    --save-result \
    --result-dir "$OUT"

  echo "  Done. Results: $OUT"
done

echo ""
echo "Lab 4 complete. Results in results/p1_lab4_concurrency/c*"
echo "Key observation: watch audio_ttfp_p50 rise steeply around c=8+."
echo "Next: bash phase1-ar-audio/lab5_context_sweep.sh"
