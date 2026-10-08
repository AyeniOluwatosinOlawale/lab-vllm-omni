#!/usr/bin/env bash
# =============================================================================
# Phase 5 — Lab 2: Concurrency + Observation Horizon Sweep
#
# CONCURRENCY FOR ROBOT POLICY:
#   InternVLA-A1 generates action tokens for one robot episode at a time.
#   Multiple concurrent episodes simulate a fleet of robots or a batch
#   evaluation scenario. Unlike LLMs, the output is fixed-length (one
#   action token sequence per step), so concurrency reveals scheduling
#   overhead rather than generation variability.
#
# CONTEXT WINDOW EQUIVALENT FOR ACTIONS:
#   The "context length" for a robot policy is the observation horizon:
#   how many past frames the model sees before generating the next action.
#   More history = richer context = better action quality, but higher
#   memory cost per request.
#
#   Three observation horizons:
#     1 frame  — reactive (no history)
#     4 frames — short history
#     8 frames — full observation window
#
# BOTTLENECK THIS EXPOSES:
#   Action generation is dominated by two costs:
#     1. Vision encoding  — ViT/VLA processes each observation frame
#     2. Cosmos decode    — tokenized action space reconstruction
#   At high concurrency, the Cosmos tokenizer decode becomes the
#   bottleneck (similar to the TTS codec cliff).
#
# OPTIMIZATION SIGNALS:
#   Latency grows linearly with num_frames → vision encoder bound
#     → use --tensor-parallel-size 2 to split vision encoder
#   Latency grows super-linearly → attention over observation tokens
#     → compress history with shorter sequences
#   Cosmos decode dominates → quantize tokenizer or reduce action space
#
# Requires: env vars from lab1_internvla.sh
# Outputs: results/p5_lab2_concurrency/  results/p5_lab2_horizon/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
RESULTS_DIR="$(dirname "$0")/../results"
CONCURRENCY_LEVELS=(1 2 4)
HORIZON_FRAMES=(1 4 8)

echo "======================================================="
echo " Phase 5 | Lab 2: Concurrency + Observation Horizon"
echo "======================================================="

# Validate env vars
MISSING=0
for VAR in INTERNVLA_A1_MODEL_DIR INTERNVLA_A1_DATASET_DIR INTERNVLA_A1_PROCESSOR_DIR INTERNVLA_A1_COSMOS_DIR; do
  if [ -z "${!VAR:-}" ]; then
    echo "  ERROR: $VAR is not set. Run lab1 first to confirm setup."
    MISSING=1
  fi
done
[ $MISSING -eq 1 ] && exit 1

export INTERNVLA_A1_MODEL_DIR INTERNVLA_A1_DATASET_DIR INTERNVLA_A1_PROCESSOR_DIR INTERNVLA_A1_COSMOS_DIR

cd "$VLLM_OMNI_DIR/examples/offline_inference/internvla_a1"

# Concurrency sweep — same horizon, vary concurrent episodes
echo ""
echo "[1/2] Concurrency sweep (fixed num_frames=4) ..."
for C in "${CONCURRENCY_LEVELS[@]}"; do
  OUT="$RESULTS_DIR/p5_lab2_concurrency/c$C"
  mkdir -p "$OUT"
  echo "  -- concurrency=$C episodes --"
  bash run.sh --num-samples "$C" --num-episodes 1 \
    2>&1 | tee "$OUT/run.log"
  bash collect_results.sh 2>&1 | tee "$OUT/collect.log"
  cp -r results/* "$OUT/" 2>/dev/null || true
done

# Observation horizon sweep — same concurrency, vary frames seen
echo ""
echo "[2/2] Observation horizon sweep (fixed concurrency=1) ..."
for F in "${HORIZON_FRAMES[@]}"; do
  OUT="$RESULTS_DIR/p5_lab2_horizon/frames$F"
  mkdir -p "$OUT"
  echo "  -- num_frames=$F observation frames --"
  bash run.sh --num-samples 1 --num-episodes 1 --num-frames "$F" \
    2>&1 | tee "$OUT/run.log"
  bash collect_results.sh 2>&1 | tee "$OUT/collect.log"
  cp -r results/* "$OUT/" 2>/dev/null || true
done

echo ""
echo "Lab 2 complete."
echo ""
echo "BOTTLENECK ANALYSIS — check these patterns in collect logs:"
echo "  1. Does latency scale linearly with num_frames?"
echo "     YES → vision encoder is the bottleneck → use TP=2"
echo "     SUPER-LINEAR → attention over observation tokens → shorten horizon"
echo "  2. Does throughput (episodes/s) drop sharply at c>2?"
echo "     YES → Cosmos decoder is serialised (like TTS codec cliff)"
echo "     Mitigation: quantize Cosmos tokenizer weights"
echo ""
echo "Run phase6-benchmark/bottleneck_analysis.py for automated findings."
