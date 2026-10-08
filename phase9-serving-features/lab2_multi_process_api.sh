#!/usr/bin/env bash
# =============================================================================
# Phase 9 — Lab 2: Multi-Process API Serving (new in 0.30.0)
#
# WHAT YOU LEARN:
#   vllm-omni 0.30.0 added --api-server-count N, which spawns multiple FastAPI
#   processes that all share a single EngineCore. This parallelises the HTTP
#   request parsing and serialisation layer without duplicating model weights.
#
# HOW IT WORKS:
#   1 EngineCore ← shared by N FastAPI processes
#   Each FastAPI process handles its own connection pool and serialisation.
#   At high concurrency (c=32+), the single FastAPI process becomes the bottleneck
#   (JSON serialisation, async I/O), not the GPU. Multiple processes fix this.
#
# LIMITATION (0.30.0):
#   --api-server-count requires local EngineCore pipelines.
#   Not compatible with Ray-based distributed serving.
#
# TWO EXPERIMENTS:
#   A. api-server-count=1 (baseline) @ c=32
#   B. api-server-count=2           @ c=32
#
# Compare: request throughput, P99 latency, head-of-line blocking.
#
# Outputs: results/p9_lab2_multi_process_api/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"
PORT=8091
RESULTS_DIR="$(dirname "$0")/../results/p9_lab2_multi_process_api"
mkdir -p "$RESULTS_DIR"

run_bench() {
  local label=$1
  local out_dir="$RESULTS_DIR/$label"
  mkdir -p "$out_dir"
  echo "  Benchmarking $label ..."
  vllm bench serve --omni \
    --host localhost --port $PORT \
    --model "$MODEL" \
    --backend openai-chat-omni \
    --dataset-name random \
    --num-prompts 100 \
    --max-concurrency 32 \
    --random-input-len 512 --random-output-len 256 \
    --extra-body '{"modalities":["text"]}' \
    --percentile-metrics ttft,tpot,e2el \
    --save-result \
    --result-dir "$out_dir"
}

echo "======================================================="
echo " Phase 9 | Lab 2: Multi-Process API Serving"
echo "======================================================="

# ── Experiment A: 1 API server process (baseline) ─────────────────────────
echo ""
echo "=== Experiment A: api-server-count=1 (baseline) ==="
vllm serve "$MODEL" --omni \
  --port $PORT \
  --api-server-count 1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."
run_bench "api_count_1"
kill $SERVER_PID 2>/dev/null || true
sleep 8

# ── Experiment B: 2 API server processes ──────────────────────────────────
echo ""
echo "=== Experiment B: api-server-count=2 ==="
vllm serve "$MODEL" --omni \
  --port $PORT \
  --api-server-count 2 \
  --worker-backend multi_process \
  --omni-dp-size-local 1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."
run_bench "api_count_2"

echo ""
echo "Lab 2 complete. Results in $RESULTS_DIR"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. Throughput ratio (api_count_2 / api_count_1)?"
echo "     RATIO ≈ 2 → API layer was the bottleneck; multi-process helped"
echo "     RATIO ≈ 1 → GPU/EngineCore is the bottleneck; extra processes add no value"
echo "     (For most 30B models, ratio will be 1.0-1.3 since GPU dominates)"
echo "  2. P99 latency improvement at high concurrency?"
echo "     P99 improvement with api_count=2 → head-of-line blocking was real"
echo "  3. This feature is most useful for:"
echo "     - Small models (fast GPU) where JSON serialisation is the bottleneck"
echo "     - Very high concurrency (c > 64) workloads"
