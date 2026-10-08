#!/usr/bin/env bash
# =============================================================================
# Phase 1 — Lab 8: Async Chunk Offline (Stage-Level Concurrency, No Server)
#
# What you learn:
#   end2end_async_chunk.py uses AsyncOmni instead of the synchronous Omni
#   class. Downstream stages (Talker, Code2Wav) start before the Thinker
#   finishes — chunk data flows directly between stage workers via
#   OmniChunkTransferAdapter, NOT through an orchestrator.
#
#   This is the offline equivalent of Lab 2 (async chunk online).
#   --max-in-flight controls how many requests overlap at the stage level.
#
#   Three sub-experiments:
#     A. Single prompt — measure wall-clock vs synchronous end2end.py
#     B. Multiple prompts with --max-in-flight 4
#     C. Text-only output (skip Talker + Code2Wav entirely)
#
# Requires: deploy YAML with async_chunk: true (default MoE config)
#           Minimum 2× GPUs matching the deploy config stage layout
#
# Outputs: results/p1_lab8_async_offline/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"
RESULTS_DIR="$(dirname "$0")/../results/p1_lab8_async_offline"
mkdir -p "$RESULTS_DIR"

echo "======================================================="
echo " Phase 1 | Lab 8: Async Chunk Offline"
echo "======================================================="

# A — Single prompt, text+audio output
echo ""
echo "[A] Single prompt — text+audio (async chunk, stage overlap) ..."
time python3 "$VLLM_OMNI_DIR/examples/offline_inference/qwen3_omni/end2end_async_chunk.py" \
  --model "$MODEL" \
  --query-type use_audio \
  --modalities text,audio \
  --output-dir "$RESULTS_DIR/single_audio" \
  2>&1 | tee "$RESULTS_DIR/single_audio.log"

# B — Multiple prompts with concurrency control
echo ""
echo "[B] Multiple prompts, --max-in-flight 4 ..."
time python3 "$VLLM_OMNI_DIR/examples/offline_inference/qwen3_omni/end2end_async_chunk.py" \
  --model "$MODEL" \
  --query-type text \
  --modalities text,audio \
  --txt-prompts "$VLLM_OMNI_DIR/examples/offline_inference/qwen3_omni/text_prompts_10.txt" \
  --max-in-flight 4 \
  --output-dir "$RESULTS_DIR/multi_inflight4" \
  2>&1 | tee "$RESULTS_DIR/multi_inflight4.log"

# C — Text-only output (skips Talker + Code2Wav stages)
echo ""
echo "[C] Text-only output (skip audio synthesis stages) ..."
time python3 "$VLLM_OMNI_DIR/examples/offline_inference/qwen3_omni/end2end_async_chunk.py" \
  --model "$MODEL" \
  --query-type text \
  --modalities text \
  --output-dir "$RESULTS_DIR/text_only" \
  2>&1 | tee "$RESULTS_DIR/text_only.log"

echo ""
echo "Lab 8 complete. Results in $RESULTS_DIR/"
echo ""
echo "  Compare wall-clock times:"
echo "    A (text+audio async chunk) vs Lab 2 online benchmark"
echo "    C (text-only) vs A — shows cost of audio synthesis stages"
echo "Next: bash phase1-ar-audio/lab9_text_vs_audio_output.sh"
