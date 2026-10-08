#!/usr/bin/env bash
# =============================================================================
# Phase 13 — Lab 4: Distributed RDMA Benchmarks (Mooncake Connectors)
#
# WHAT YOU LEARN:
#   Inter-stage KV transfer bandwidth directly limits throughput in
#   disaggregated serving. This lab measures raw connector bandwidth to
#   determine whether stage disaggregation is compute-bound or transfer-bound.
#
# THREE TRANSPORT MODES:
#   Copy      — GPU → CPU → network → CPU → GPU (PCIe limited, ~32 GB/s)
#   Zerocopy  — GPU → network → GPU (skips one CPU copy, ~50-60 GB/s on PCIe)
#   GPUDirect — GPU → network → GPU via RDMA (NVLink: ~600 GB/s, PCIe: ~32 GB/s)
#
# PAYLOAD SIZES:
#   1 KB   — latency-dominated (small KV tensors, per-request metadata)
#   1 MB   — typical text encoder hidden state (512 tokens × 2048 dim × BF16)
#   64 MB  — typical diffusion KV cache chunk
#   512 MB — full denoiser KV for large video DiT
#
# HARDWARE REQUIREMENT:
#   2 GPUs required (can be same node with NVLink or separate nodes with RDMA)
#   GPUDirect requires GPUDirect RDMA driver (NVIDIA ConnectX or similar NIC)
#
# Outputs: results/p13_lab4_rdma_distributed/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
RESULTS_DIR="$(dirname "$0")/../results/p13_lab4_rdma_distributed"
mkdir -p "$RESULTS_DIR"

PAYLOAD_SIZES=(1024 1048576 67108864 536870912)  # 1KB, 1MB, 64MB, 512MB
ITERATIONS=100

GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)

echo "======================================================="
echo " Phase 13 | Lab 4: RDMA Distributed Benchmarks"
echo "======================================================="
echo "  GPUs available: $GPU_COUNT"
echo ""

if [ "$GPU_COUNT" -lt 2 ]; then
  echo "  WARNING: 2 GPUs required for inter-GPU transfer benchmarks."
  echo "  Running in single-GPU simulation mode (measures CPU<->GPU bandwidth only)."
fi

BENCH_SCRIPT="$VLLM_OMNI_DIR/benchmarks/distributed/omni_connectors/bench_copy.py"

if [ ! -f "$BENCH_SCRIPT" ]; then
  echo "  ERROR: Benchmark script not found at $BENCH_SCRIPT"
  echo "  Check VLLM_OMNI_DIR=$VLLM_OMNI_DIR"
  exit 1
fi

echo "mode,payload_bytes,bandwidth_gbps,latency_us" > "$RESULTS_DIR/rdma_results.csv"

for MODE in copy zerocopy gpudirect; do
  echo ""
  echo "=== Mode: $MODE ==="
  for SIZE in "${PAYLOAD_SIZES[@]}"; do
    SIZE_MB=$(echo "scale=2; $SIZE/1048576" | bc 2>/dev/null || echo "$SIZE bytes")
    echo "  Payload: $SIZE bytes (${SIZE_MB} MB) ..."

    EXTRA_FLAG=""
    [ "$MODE" = "zerocopy" ] && EXTRA_FLAG="--zerocopy"
    [ "$MODE" = "gpudirect" ] && EXTRA_FLAG="--gpudirect"

    # shellcheck disable=SC2086
    python3 "$BENCH_SCRIPT" \
      --size-bytes "$SIZE" \
      --iterations "$ITERATIONS" \
      $EXTRA_FLAG \
      --output-csv "$RESULTS_DIR/bench_${MODE}_${SIZE}.csv" \
      2>/dev/null | tee /tmp/bench_out.txt || \
      echo "  (mode $MODE not supported on this hardware; skipping)"

    # Extract bandwidth and latency if available
    BANDWIDTH=$(python3 -c "
import sys
with open('/tmp/bench_out.txt') as f:
    for line in f:
        if 'bandwidth' in line.lower() or 'gb/s' in line.lower():
            print(line.strip())
            break
" 2>/dev/null || echo "N/A")
    echo "  $MODE,$SIZE,$BANDWIDTH" >> "$RESULTS_DIR/rdma_results.csv"
  done
done

echo ""
echo "Lab 4 complete. Results in $RESULTS_DIR"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. At 64MB payload (typical DiT KV chunk):"
echo "     Copy:      ~32 GB/s PCIe → 64MB transfer = ~2ms"
echo "     Zerocopy:  ~50 GB/s      → 64MB transfer = ~1.3ms"
echo "     GPUDirect: ~600 GB/s NVLink → 64MB transfer = ~0.1ms"
echo ""
echo "  2. Is stage disaggregation bottlenecked by transfer or compute?"
echo "     Transfer time < 1ms for 64MB → transfer is NOT the bottleneck"
echo "     Transfer time > 10ms         → use GPUDirect or collocate stages"
echo ""
echo "  3. At 1KB payload: latency-dominated regime"
echo "     High latency (>100µs) → connection setup overhead; use persistent connections"
