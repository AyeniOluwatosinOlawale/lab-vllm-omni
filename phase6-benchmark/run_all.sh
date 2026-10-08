#!/usr/bin/env bash
# =============================================================================
# Phase 6 — Cross-Pipeline Benchmark
#
# Runs one representative workload per pipeline type back-to-back and
# collects results into results/p6_summary.json for comparison.
#
# What you learn:
#   Unified view of all 5 output modalities on a single results table:
#     - AR+Audio (Qwen3-Omni): TTFT, E2E, TTFP, RTF
#     - Image (FLUX.2-klein-4B): request throughput, E2E latency
#     - Video (Wan2.2-T2V): request throughput, E2E latency
#     - TTS c=1 (Qwen3-TTS): TTFP, RTF, audio_underrun
#     - TTS c=8 (Qwen3-TTS): same — shows the codec cliff effect
#
# Outputs:
#   results/p6_omni/        results/p6_image/
#   results/p6_video/       results/p6_tts_c1/   results/p6_tts_c8/
#   results/p6_summary.json
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
RESULTS_DIR="$(dirname "$0")/../results"
mkdir -p "$RESULTS_DIR"

wait_for_health() {
  local port=$1
  echo "  Waiting for server on port $port ..."
  until curl -sf "http://localhost:$port/health" > /dev/null 2>&1; do sleep 5; done
  echo "  Server ready."
}

stop_server() {
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 8
}

echo "======================================================="
echo " Phase 6 | Cross-Pipeline Benchmark"
echo "======================================================="

# ------------------------------------------------------------------
# 1. AR + Audio — Qwen3-Omni
# ------------------------------------------------------------------
echo ""
echo "=== [1/5] AR+Audio: Qwen3-Omni ==="
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 &
wait_for_health 8091

vllm bench serve --omni \
  --host localhost --port 8091 \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --backend openai-chat-omni \
  --dataset-name random \
  --num-prompts 20 --max-concurrency 4 \
  --random-input-len 2500 --random-output-len 900 \
  --ignore-eos \
  --extra-body '{"modalities":["text","audio"]}' \
  --percentile-metrics ttft,tpot,e2el,audio_ttfp,audio_rtf \
  --save-result \
  --result-dir "$RESULTS_DIR/p6_omni"

stop_server

# ------------------------------------------------------------------
# 2. Image — FLUX.2-klein-4B
# ------------------------------------------------------------------
echo ""
echo "=== [2/5] Image: FLUX.2-klein-4B ==="
vllm serve black-forest-labs/FLUX.2-klein-4B --omni --port 8099 &
wait_for_health 8099

python3 "$VLLM_OMNI_DIR/benchmarks/diffusion/diffusion_benchmark_serving.py" \
  --base-url http://localhost:8099 \
  --model black-forest-labs/FLUX.2-klein-4B \
  --task t2i \
  --dataset random \
  --num-prompts 20 \
  --max-concurrency 4 \
  --width 1024 --height 1024 \
  --num-inference-steps 20 \
  --output-file "$RESULTS_DIR/p6_image.json"

stop_server

# ------------------------------------------------------------------
# 3. Video — Wan2.2-T2V
# ------------------------------------------------------------------
echo ""
echo "=== [3/5] Video: Wan2.2-T2V ==="
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni --port 8099 &
wait_for_health 8099

python3 "$VLLM_OMNI_DIR/benchmarks/diffusion/diffusion_benchmark_serving.py" \
  --base-url http://localhost:8099 \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --task t2v \
  --dataset random \
  --num-prompts 10 \
  --max-concurrency 2 \
  --width 832 --height 480 \
  --num-frames 81 --fps 24 \
  --num-inference-steps 20 \
  --video-job-timeout 1800 \
  --output-file "$RESULTS_DIR/p6_video.json"

stop_server

# ------------------------------------------------------------------
# 4. TTS — Qwen3-TTS concurrency=1 and concurrency=8 (cliff)
# ------------------------------------------------------------------
echo ""
echo "=== [4/5] TTS: Qwen3-TTS (c=1 and c=8) ==="
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base --omni --port 8000 &
wait_for_health 8000

for C in 1 8; do
  OUT="$RESULTS_DIR/p6_tts_c$C"
  mkdir -p "$OUT"
  echo "  -- TTS concurrency=$C --"
  vllm bench serve --omni \
    --host 127.0.0.1 --port 8000 \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --backend openai-audio-speech \
    --endpoint /v1/audio/speech \
    --dataset-name seed-tts-text \
    --dataset-path "$VLLM_OMNI_DIR/benchmarks/build_dataset/seed_tts_smoke" \
    --seed-tts-locale en \
    --num-prompts 20 --num-warmups 2 \
    --extra-body '{"task_type":"Base"}' \
    --max-concurrency "$C" \
    --request-rate inf \
    --percentile-metrics ttft,e2el,audio_rtf,audio_ttfp,audio_duration,audio_underrun \
    --save-result \
    --result-dir "$OUT"
done

stop_server

# ------------------------------------------------------------------
# 5. Summary
# ------------------------------------------------------------------
echo ""
echo "=== [5/6] Building cross-pipeline comparison ==="
python3 "$(dirname "$0")/plot_comparison.py" \
  --omni-dir       "$RESULTS_DIR/p6_omni" \
  --image-file     "$RESULTS_DIR/p6_image.json" \
  --video-file     "$RESULTS_DIR/p6_video.json" \
  --tts-c1-dir     "$RESULTS_DIR/p6_tts_c1" \
  --tts-c8-dir     "$RESULTS_DIR/p6_tts_c8" \
  --output         "$RESULTS_DIR/p6_summary.json" \
  --chart          "$RESULTS_DIR/p6_comparison.png"

echo ""
echo "=== [6/6] Bottleneck analysis — all pipelines ==="
python3 "$(dirname "$0")/bottleneck_analysis.py" \
  --results-dir "$RESULTS_DIR" \
  --output      "$RESULTS_DIR/bottleneck_report.json" \
  --chart       "$RESULTS_DIR/bottleneck_chart.png"

echo ""
echo "======================================================="
echo " Phase 6 complete."
echo "  Comparison JSON:    $RESULTS_DIR/p6_summary.json"
echo "  Comparison chart:   $RESULTS_DIR/p6_comparison.png"
echo "  Bottleneck report:  $RESULTS_DIR/bottleneck_report.json"
echo "  Bottleneck chart:   $RESULTS_DIR/bottleneck_chart.png"
echo "======================================================="
