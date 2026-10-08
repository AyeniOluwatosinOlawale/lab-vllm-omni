#!/usr/bin/env bash
# =============================================================================
# Phase 10 — Lab 3: HunyuanImage 3.0 Cross-Request Prefix Caching (new in 0.30.0)
#
# WHAT YOU LEARN:
#   0.30.0 introduced cross-REQUEST prefix caching for HunyuanImage 3.0.
#   When multiple users send image requests with the same text prefix (e.g.
#   "a professional portrait of..."), the text encoder KV computation is
#   cached and reused. The second+ request with the same prefix skips the
#   text encoder entirely.
#
# LLM ANALOGY:
#   This is vLLM's prefix caching applied to diffusion models. Same concept:
#   cache the KV of any shared prefix; reuse on matching requests.
#
# HOW TO MAXIMISE HIT RATE:
#   Use canonical prompt prefixes. The cache key is the exact token sequence.
#   Even one different token misses the cache. Wildcards do NOT work.
#
# TWO EXPERIMENTS:
#   A. No prefix cache — 20 requests with 5 shared prefixes (baseline)
#   B. Prefix cache ON — same 20 requests → expect latency reduction on hits
#
# Outputs: results/p10_lab3_hunyuan_prefix_cache/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="Tencent-Hunyuan/HunyuanDiT-v1.2"
PORT=8099
RESULTS_DIR="$(dirname "$0")/../results/p10_lab3_hunyuan_prefix_cache"
mkdir -p "$RESULTS_DIR"

# 20 prompts — 5 shared prefixes × 4 suffixes each
PROMPTS=(
  "a professional portrait of a woman in business attire, studio lighting"
  "a professional portrait of a man in casual clothes, outdoor background"
  "a professional portrait of a child with a bright smile, natural light"
  "a professional portrait of an elderly person, warm tones, soft focus"
  "a stunning landscape of mountain peaks at sunrise, golden hour"
  "a stunning landscape of ocean waves crashing on white sand beach"
  "a stunning landscape of dense forest with morning fog and light rays"
  "a stunning landscape of desert dunes under a starry night sky"
  "a detailed illustration of a futuristic city with flying vehicles"
  "a detailed illustration of an underwater kingdom with glowing fish"
  "a detailed illustration of a magical forest with fairy lights"
  "a detailed illustration of a steampunk airship in cloudy sky"
  "a close-up macro photograph of a dewdrop on a green leaf"
  "a close-up macro photograph of a butterfly wing pattern"
  "a close-up macro photograph of a honeybee collecting pollen"
  "a close-up macro photograph of a spider web with morning dew"
  "an oil painting of a serene lake reflecting autumn trees"
  "an oil painting of a thunderstorm over a medieval castle"
  "an oil painting of children playing in a sunny meadow"
  "an oil painting of an old lighthouse on a rocky cliff at dusk"
)

run_bench() {
  local label=$1
  local extra_args=$2
  local out_dir="$RESULTS_DIR/$label"
  mkdir -p "$out_dir"
  echo "  Benchmarking $label ..."
  python3 "$VLLM_OMNI_DIR/benchmarks/diffusion/diffusion_benchmark_serving.py" \
    --base-url "http://localhost:$PORT" \
    --model "$MODEL" \
    --task t2i \
    --dataset random \
    --num-prompts 20 \
    --max-concurrency 4 \
    --width 1024 --height 1024 \
    --num-inference-steps 20 \
    --output-file "$out_dir/result.json" \
    $extra_args
}

echo "======================================================="
echo " Phase 10 | Lab 3: HunyuanImage 3.0 Prefix Caching"
echo "======================================================="

# ── A. No prefix cache ─────────────────────────────────────────────────────
echo ""
echo "=== A. No prefix cache (baseline) ==="
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "pkill -f 'vllm serve' 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."
# shellcheck disable=SC2086
run_bench "no_cache" ""
pkill -f "vllm serve" 2>/dev/null; sleep 6

# ── B. Prefix cache enabled ────────────────────────────────────────────────
echo ""
echo "=== B. Cross-request prefix cache (--cache-backend prefix_paged) ==="
vllm serve "$MODEL" --omni --port $PORT \
  --cache-backend prefix_paged &
SERVER_PID=$!
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."
# shellcheck disable=SC2086
run_bench "prefix_cache" ""

echo ""
echo "Lab 3 complete. Results in $RESULTS_DIR"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. Latency for requests 5-20 in prefix_cache vs no_cache?"
echo "     Latency drop for same-prefix requests → cache hits working"
echo "     No improvement → prefixes are not matching (check tokenisation)"
echo "  2. Check cache hit rate in server logs:"
echo "     grep 'prefix_cache' server logs → shows hit/miss per request"
echo "  3. Does the cache benefit degrade with diverse prompts?"
echo "     YES (expected) → cache only helps when prefixes are shared"
echo "     Use this in production for: product image generation, avatar variants"
