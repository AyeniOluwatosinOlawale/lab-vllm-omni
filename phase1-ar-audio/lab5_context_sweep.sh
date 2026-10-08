#!/usr/bin/env bash
# =============================================================================
# Phase 1 — Lab 5: Context Length Sweep
#
# What you learn:
#   TTFT cost grows with context budget because the KV cache occupies more
#   GPU memory, leaving less headroom for batching. Observe memory pressure
#   and whether audio output quality holds as output length increases.
#
#   Configs (in phase1-ar-audio/configs/):
#     ctx_8k.yaml   — max_num_batched_tokens=8192,  max_tokens=512
#     ctx_32k.yaml  — max_num_batched_tokens=32768, max_tokens=2048 (default)
#     ctx_64k.yaml  — max_num_batched_tokens=65536, max_tokens=4096
#
# Outputs: results/p1_lab5_context/<config>/
# =============================================================================
set -euo pipefail
MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"
PORT=8091
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/../results"
CONFIGS=(ctx_8k ctx_32k ctx_64k)

echo "======================================================="
echo " Phase 1 | Lab 5: Context Length Sweep"
echo "======================================================="

for CTX in "${CONFIGS[@]}"; do
  CONFIG_PATH="$SCRIPT_DIR/configs/$CTX.yaml"
  OUT="$RESULTS_DIR/p1_lab5_context/$CTX"
  mkdir -p "$OUT"

  echo ""
  echo "--- Config: $CTX ---"
  echo "  Starting server with $CONFIG_PATH ..."

  vllm serve "$MODEL" --omni --port $PORT \
    --deploy-config "$CONFIG_PATH" &
  SERVER_PID=$!

  until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
  echo "  Server ready."

  vllm bench serve --omni \
    --host localhost --port $PORT \
    --model "$MODEL" \
    --backend openai-chat-omni \
    --dataset-name random \
    --num-prompts 10 \
    --max-concurrency 2 \
    --random-input-len 2500 \
    --random-output-len 900 \
    --ignore-eos \
    --extra-body '{"modalities":["text","audio"]}' \
    --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf \
    --save-result \
    --result-dir "$OUT"

  echo "  Results saved to $OUT"
  kill $SERVER_PID 2>/dev/null || true
  sleep 10
done

echo ""
echo "Lab 5 complete. Results in results/p1_lab5_context/*"
echo "Key observation: TTFT_p50 should increase from ctx_8k → ctx_64k."
echo "Next: bash phase1-ar-audio/lab6_realtime_ui.sh"
