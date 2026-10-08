#!/usr/bin/env bash
# =============================================================================
# Phase 1 — Lab 7: Staged 3-Process Launch (Disaggregated Inference)
#
# What you learn:
#   The clearest hands-on demonstration of disaggregated inference.
#   Each of Qwen3-Omni's 3 stages runs as an independent process on its
#   own GPU, communicating via SharedMemoryConnector:
#
#     Stage 0 (Thinker)   — LLM prefill + decode; hosts the API server
#     Stage 1 (Talker)    — codec token predictor; receives from Stage 0
#     Stage 2 (Code2Wav)  — waveform synthesiser; receives from Stage 1
#
#   Compare staged vs unified (Lab 1) on TTFT and E2E to see whether
#   inter-process communication overhead is offset by independent
#   per-stage resource allocation.
#
# Requires: 2× GPUs minimum (Stage 0 on GPU 0, Stages 1+2 share GPU 1)
#           3× GPUs for fully isolated deployment
#
# Usage:
#   bash phase1-ar-audio/lab7_staged_launch.sh [bench]
#   Default: starts stages and prints curl smoke test.
#   Pass 'bench' to also run the vllm bench serve comparison.
# =============================================================================
set -euo pipefail
MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"
API_PORT=8091
MASTER_ADDR="127.0.0.1"
MASTER_PORT=26000
RESULTS_DIR="$(dirname "$0")/../results/p1_lab7_staged"
MODE="${1:-start}"
mkdir -p "$RESULTS_DIR"

cleanup() {
  echo "  Stopping all stage processes ..."
  kill "${STAGE_PIDS[@]}" 2>/dev/null || true
}

echo "======================================================="
echo " Phase 1 | Lab 7: Staged 3-Process Launch"
echo "======================================================="
echo "  Stage 0 (Thinker+API) → GPU 0"
echo "  Stage 1 (Talker)      → GPU 1"
echo "  Stage 2 (Code2Wav)    → GPU 1"
echo ""

declare -a STAGE_PIDS

# Stage 0 — Thinker + API server
echo "[1/4] Starting Stage 0 (Thinker + API) on GPU 0, port $API_PORT ..."
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" --omni \
  --port $API_PORT \
  --stage-id 0 \
  --omni-master-address $MASTER_ADDR \
  --omni-master-port $MASTER_PORT \
  --max-num-seqs 8 &
STAGE_PIDS+=($!)

sleep 5

# Stage 1 — Talker
echo "[2/4] Starting Stage 1 (Talker) on GPU 1 ..."
CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL" --omni \
  --stage-id 1 \
  --headless \
  --omni-master-address $MASTER_ADDR \
  --omni-master-port $MASTER_PORT \
  --max-num-seqs 4 &
STAGE_PIDS+=($!)

sleep 5

# Stage 2 — Code2Wav
echo "[3/4] Starting Stage 2 (Code2Wav) on GPU 1 ..."
CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL" --omni \
  --stage-id 2 \
  --headless \
  --omni-master-address $MASTER_ADDR \
  --omni-master-port $MASTER_PORT \
  --max-num-seqs 4 &
STAGE_PIDS+=($!)

trap cleanup INT TERM EXIT

echo ""
echo "[4/4] Waiting for API server health ..."
until curl -sf "http://localhost:$API_PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  All 3 stages connected. API ready on port $API_PORT."

# Smoke test
echo ""
echo "Smoke test (text-only) ..."
curl -s "http://localhost:$API_PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Describe vLLM in one sentence.\"}],
    \"modalities\": [\"text\"],
    \"max_tokens\": 50
  }" | python3 -m json.tool

# Optional benchmark
if [ "${MODE}" = "bench" ]; then
  echo ""
  echo "Benchmarking staged deployment (text+audio, c=4) ..."
  vllm bench serve --omni \
    --host localhost --port $API_PORT \
    --model "$MODEL" \
    --backend openai-chat-omni \
    --dataset-name random \
    --num-prompts 20 \
    --max-concurrency 4 \
    --random-input-len 2500 \
    --random-output-len 900 \
    --ignore-eos \
    --extra-body '{"modalities":["text","audio"]}' \
    --percentile-metrics ttft,tpot,e2el,audio_ttfp,audio_rtf \
    --save-result \
    --result-dir "$RESULTS_DIR"

  echo ""
  echo "Compare results/p1_lab1_sync_audio vs results/p1_lab7_staged"
  echo "  Staged may show lower per-stage memory pressure but adds IPC overhead."
fi

echo ""
echo "Lab 7 running. Ctrl+C to stop all stages."
wait
