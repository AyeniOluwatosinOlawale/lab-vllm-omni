#!/usr/bin/env bash
# =============================================================================
# Phase 5 — Lab 1: InternVLA-A1 Robot Policy Inference
#
# What you learn:
#   The most unique output modality in vLLM-Omni: action token generation.
#   Instead of text/audio/image/video, the model outputs discrete action
#   tokens (joint positions / velocities) decoded via Cosmos tokenizer.
#   InternVLA-A1 takes visual observations + task language → robot actions.
#
# Required environment variables (set before running):
#   INTERNVLA_A1_MODEL_DIR     — path to InternVLA-A1-3B-ft-pen
#   INTERNVLA_A1_DATASET_DIR   — path to Genie1-Place_Markpen dataset
#   INTERNVLA_A1_PROCESSOR_DIR — path to Qwen3-VL-2B-Instruct
#   INTERNVLA_A1_COSMOS_DIR    — path to Cosmos-Tokenizer-CI8x8-SafeTensors
#
#   HF model: tenstep/Cosmos-Tokenizer-CI8x8-SafeTensors
#   Expected files: encoder.safetensors, decoder.safetensors
#
# Outputs: results/p5_lab1_actions/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
RESULTS_DIR="$(dirname "$0")/../results/p5_lab1_actions"
mkdir -p "$RESULTS_DIR"

echo "======================================================="
echo " Phase 5 | Lab 1: InternVLA-A1 Robot Policy"
echo "======================================================="

# Validate required env vars
MISSING=0
for VAR in INTERNVLA_A1_MODEL_DIR INTERNVLA_A1_DATASET_DIR INTERNVLA_A1_PROCESSOR_DIR INTERNVLA_A1_COSMOS_DIR; do
  if [ -z "${!VAR:-}" ]; then
    echo "  ERROR: $VAR is not set"
    MISSING=1
  else
    echo "  $VAR = ${!VAR}"
  fi
done
[ $MISSING -eq 1 ] && exit 1

export INTERNVLA_A1_MODEL_DIR
export INTERNVLA_A1_DATASET_DIR
export INTERNVLA_A1_PROCESSOR_DIR
export INTERNVLA_A1_COSMOS_DIR

echo ""
echo "[1/3] Single sample smoke test (num-samples=1, num-episodes=0) ..."
cd "$VLLM_OMNI_DIR/examples/offline_inference/internvla_a1"
bash run.sh --num-samples 1 --num-episodes 0 \
  2>&1 | tee "$RESULTS_DIR/smoke_test.log"

echo ""
echo "[2/3] Full episode (num-episodes=1) ..."
bash run.sh --num-episodes 1 \
  2>&1 | tee "$RESULTS_DIR/episode.log"

echo ""
echo "[3/3] Collecting results, metrics, and plots ..."
bash collect_results.sh 2>&1 | tee "$RESULTS_DIR/collect.log"
cp -r results/* "$RESULTS_DIR/" 2>/dev/null || true

echo ""
echo "Lab 1 complete."
echo "  Smoke log:   $RESULTS_DIR/smoke_test.log"
echo "  Episode log: $RESULTS_DIR/episode.log"
echo "  Metrics:     $RESULTS_DIR/"
echo ""
echo "Phase 5 complete. Proceed to: bash phase6-benchmark/run_all.sh"
