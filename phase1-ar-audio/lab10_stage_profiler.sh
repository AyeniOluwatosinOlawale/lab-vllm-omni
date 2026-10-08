#!/usr/bin/env bash
# =============================================================================
# Phase 1 — Lab 10: Stage Profiler
#
# What you learn:
#   --enable-profiler captures a PyTorch trace for each stage and writes
#   .json.gz trace files. Load them in chrome://tracing or Perfetto UI
#   (https://ui.perfetto.dev) to inspect per-stage kernel timings.
#
#   --profiler-stages lets you target individual stages:
#     0 = Thinker (LLM decode)
#     1 = Talker  (codec prediction)
#     2 = Code2Wav (waveform synthesis)
#
#   Three sub-experiments:
#     A. Profile all 3 stages on a single text+audio request
#     B. Profile Thinker only (Stage 0) — focus on LLM decode kernels
#     C. Profile Code2Wav only (Stage 2) — focus on waveform synthesis
#
#   After this lab you have trace files showing exactly where time is
#   spent inside each stage — the foundation for targeted optimisation.
#
# Outputs: results/p1_lab10_profiler/  (contains .json.gz trace files)
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"
RESULTS_DIR="$(dirname "$0")/../results/p1_lab10_profiler"
mkdir -p "$RESULTS_DIR"/{all_stages,thinker_only,code2wav_only}

echo "======================================================="
echo " Phase 1 | Lab 10: Stage Profiler"
echo "======================================================="

# A — Profile all 3 stages
echo ""
echo "[A] Profiling all 3 stages (1 prompt, text+audio) ..."
python3 "$VLLM_OMNI_DIR/examples/offline_inference/qwen3_omni/end2end.py" \
  --model "$MODEL" \
  --query-type use_audio \
  --modalities text,audio \
  --num-prompts 1 \
  --enable-profiler \
  --output-dir "$RESULTS_DIR/all_stages" \
  2>&1 | tee "$RESULTS_DIR/all_stages/run.log"

echo "  Traces saved to $RESULTS_DIR/all_stages/"

# B — Profile Thinker only (Stage 0)
echo ""
echo "[B] Profiling Stage 0 (Thinker) only ..."
python3 "$VLLM_OMNI_DIR/examples/offline_inference/qwen3_omni/end2end.py" \
  --model "$MODEL" \
  --query-type text \
  --modalities text \
  --num-prompts 1 \
  --enable-profiler \
  --profiler-stages 0 \
  --output-dir "$RESULTS_DIR/thinker_only" \
  2>&1 | tee "$RESULTS_DIR/thinker_only/run.log"

echo "  Stage 0 trace saved to $RESULTS_DIR/thinker_only/"

# C — Profile Code2Wav only (Stage 2)
echo ""
echo "[C] Profiling Stage 2 (Code2Wav) only ..."
python3 "$VLLM_OMNI_DIR/examples/offline_inference/qwen3_omni/end2end.py" \
  --model "$MODEL" \
  --query-type text \
  --modalities text,audio \
  --num-prompts 1 \
  --enable-profiler \
  --profiler-stages 2 \
  --output-dir "$RESULTS_DIR/code2wav_only" \
  2>&1 | tee "$RESULTS_DIR/code2wav_only/run.log"

echo "  Stage 2 trace saved to $RESULTS_DIR/code2wav_only/"

echo ""
echo "======================================================="
echo " Lab 10 complete."
echo ""
echo "  Trace files (.json.gz) are in:"
echo "    $RESULTS_DIR/all_stages/"
echo "    $RESULTS_DIR/thinker_only/"
echo "    $RESULTS_DIR/code2wav_only/"
echo ""
echo "  To view traces:"
echo "    1. Open https://ui.perfetto.dev"
echo "    2. Click 'Open trace file'"
echo "    3. Load any .json.gz file from the above directories"
echo ""
echo "  Look for:"
echo "    Stage 0: attention kernels, MoE routing, KV cache ops"
echo "    Stage 2: conv_transpose1d (waveform upsampling bottleneck)"
echo "======================================================="
