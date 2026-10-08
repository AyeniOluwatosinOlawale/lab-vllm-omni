#!/usr/bin/env bash
# =============================================================================
# Phase 11 — Lab 1: Mooncake AR-to-DiT KV Handoff + NIXL Transport (0.30.0)
#
# WHAT YOU LEARN:
#   In disaggregated serving, the AR text encoder (Stage 0) must transfer its
#   hidden states to the DiT denoiser (Stage 1) before denoising can begin.
#   The speed of this transfer directly determines denoising start latency.
#
# THREE TRANSPORT MODES:
#   SharedMemory (SHM) — same node, single process, zero-copy mmap
#                        Bandwidth: memory bus speed (~300+ GB/s)
#   Mooncake            — RDMA-capable transport via NIXL layer
#                        Same node: similar to SHM; cross-node: NVLink/PCIe
#
# NIXL (Network Inference eXchange Layer):
#   The abstract transport layer above Mooncake. vllm-omni uses NIXL to
#   support multiple backends (SHM, Mooncake, UCX) without code changes.
#
# EXPERIMENT:
#   Compare stage handoff latency between SHM and Mooncake on the same node.
#   Use --enable-diffusion-pipeline-profiler to capture stage boundary times.
#
# HARDWARE: 2× GPU recommended (one per stage). Falls back to 1× GPU.
# Outputs: results/p11_lab1_mooncake_kv/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="Tencent-Hunyuan/HunyuanDiT-v1.2"
PORT=8099
RESULTS_DIR="$(dirname "$0")/../results/p11_lab1_mooncake_kv"
mkdir -p "$RESULTS_DIR"

# Write Mooncake deploy config
cat > "$RESULTS_DIR/mooncake_config.yaml" << 'EOF'
base_config: vllm_omni/deploy/hunyuan_image.yaml
kv_transfer:
  backend: mooncake
  transport: nixl
stages:
  - stage_id: 0
    gpu_memory_utilization: 0.45
  - stage_id: 1
    gpu_memory_utilization: 0.45
EOF

run_bench() {
  local label=$1
  echo "  Benchmarking: $label"
  python3 "$VLLM_OMNI_DIR/benchmarks/diffusion/diffusion_benchmark_serving.py" \
    --base-url "http://localhost:$PORT" \
    --model "$MODEL" \
    --task t2i \
    --dataset random \
    --num-prompts 10 \
    --max-concurrency 2 \
    --width 1024 --height 1024 \
    --num-inference-steps 20 \
    --output-file "$RESULTS_DIR/${label}_result.json"
}

echo "======================================================="
echo " Phase 11 | Lab 1: Mooncake KV Handoff (NIXL)"
echo "======================================================="

# ── A. SharedMemory baseline ───────────────────────────────────────────────
echo ""
echo "=== A. SharedMemory connector (baseline) ==="
vllm serve "$MODEL" --omni --port $PORT \
  --enable-diffusion-pipeline-profiler &
SERVER_PID=$!
trap "pkill -f 'vllm serve' 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."
run_bench "shm"
pkill -f "vllm serve" 2>/dev/null; sleep 6

# ── B. Mooncake + NIXL ────────────────────────────────────────────────────
echo ""
echo "=== B. Mooncake + NIXL connector ==="
vllm serve "$MODEL" --omni --port $PORT \
  --deploy-config "$RESULTS_DIR/mooncake_config.yaml" \
  --worker-backend multi_process \
  --enable-diffusion-pipeline-profiler &
SERVER_PID=$!
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."
run_bench "mooncake"

echo ""
echo "Lab 1 complete. Results in $RESULTS_DIR"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. Check stage handoff latency from profiler output:"
echo "     Look for 'stage0_end' to 'stage1_start' delta in profiler traces"
echo "     SHM handoff: typically < 1ms (same-node, zero-copy)"
echo "     Mooncake same-node: similar to SHM; benefit is cross-node capability"
echo "  2. Does E2E latency change between SHM and Mooncake?"
echo "     Similar → transport is not the bottleneck (GPU compute dominates)"
echo "     Mooncake slower → NIXL serialisation overhead on same-node; use SHM for local"
echo "  3. For cross-node (multi-node cluster):"
echo "     Mooncake with GPUDirect RDMA achieves NVLink bandwidth across nodes"
echo "     Benchmark with benchmarks/distributed/omni_connectors/ (Phase 13 lab4)"
