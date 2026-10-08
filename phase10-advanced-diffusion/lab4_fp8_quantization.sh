#!/usr/bin/env bash
# =============================================================================
# Phase 10 — Lab 4: FP8 Quantization for Diffusion (new in 0.30.0)
#
# WHAT YOU LEARN:
#   FP8 diffusion quantization is new in 0.30.0 for Boogu-Image Turbo.
#   FP8 halves weight storage (vs BF16) and uses H100 native FP8 tensor cores
#   for matrix multiplications — yielding ~1.5-2× throughput improvement.
#
# WHY FP8 WORKS WELL FOR DIFFUSION (unlike LLMs):
#   Diffusion is iterative. Each denoising step introduces small rounding errors,
#   but these are averaged out across 20+ steps. The final image quality impact
#   is typically <0.5 PSNR, which is imperceptible.
#   LLMs accumulate errors autoregressively — each token depends on the last.
#   Diffusion models do not have this dependency chain.
#
# TWO PARTS:
#   A. Latency/throughput comparison (BF16 vs FP8)
#   B. Quality comparison (PSNR between same-seed BF16 and FP8 outputs)
#
# HARDWARE: H100 required for FP8 tensor core benefit (A100 has limited FP8).
# Outputs: results/p10_lab4_fp8_quantization/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="Boogu-AI/Boogu-Image-Turbo"
PORT=8099
RESULTS_DIR="$(dirname "$0")/../results/p10_lab4_fp8_quantization"
mkdir -p "$RESULTS_DIR"

# Write FP8 quantization config
cat > "$RESULTS_DIR/fp8_config.json" << 'EOF'
{
  "quant_type": "fp8",
  "skip_layers": ["norm", "text_encoder"],
  "activation_scheme": "dynamic",
  "weight_dtype": "fp8_e4m3"
}
EOF
echo "  FP8 config written to $RESULTS_DIR/fp8_config.json"

run_bench() {
  local label=$1
  python3 "$VLLM_OMNI_DIR/benchmarks/diffusion/diffusion_benchmark_serving.py" \
    --base-url "http://localhost:$PORT" \
    --model "$MODEL" \
    --task t2i \
    --dataset random \
    --num-prompts 20 \
    --max-concurrency 4 \
    --width 1024 --height 1024 \
    --num-inference-steps 8 \
    --output-file "$RESULTS_DIR/${label}_result.json"
}

echo "======================================================="
echo " Phase 10 | Lab 4: FP8 Quantization (Boogu-Image Turbo)"
echo "======================================================="

# ── A-1. BF16 baseline ─────────────────────────────────────────────────────
echo ""
echo "=== A-1. BF16 baseline ==="
vllm serve "$MODEL" --omni --port $PORT &
SERVER_PID=$!
trap "pkill -f 'vllm serve' 2>/dev/null; exit" INT TERM EXIT

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."
echo "  GPU memory (BF16):"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
  | awk '{print "  GPU: "$1" MiB"}' || true
run_bench "bf16"

# Save one reference image for PSNR comparison
curl -s "http://localhost:$PORT/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "prompt": "a red apple on a wooden table, studio lighting",
    "seed": 42,
    "width": 512, "height": 512,
    "num_inference_steps": 8,
    "response_format": "b64_json"
  }' | python3 -c "
import sys, json, base64
data = json.load(sys.stdin)
img_b64 = data['data'][0]['b64_json']
with open('$RESULTS_DIR/ref_bf16.png', 'wb') as f:
    f.write(base64.b64decode(img_b64))
print('  Reference BF16 image saved.')
" 2>/dev/null || echo "  (reference image save failed)"

pkill -f "vllm serve" 2>/dev/null; sleep 6

# ── A-2. FP8 ───────────────────────────────────────────────────────────────
echo ""
echo "=== A-2. FP8 quantized ==="
vllm serve "$MODEL" --omni --port $PORT \
  --force-cutlass-fp8 \
  --diffusion-quantization-config "$RESULTS_DIR/fp8_config.json" &
SERVER_PID=$!
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 5; done
echo "  Server ready."
echo "  GPU memory (FP8):"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
  | awk '{print "  GPU: "$1" MiB"}' || true
run_bench "fp8"

# ── B. Quality: PSNR comparison ───────────────────────────────────────────
echo ""
echo "=== B. Quality comparison (PSNR BF16 vs FP8, same seed) ==="
curl -s "http://localhost:$PORT/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "prompt": "a red apple on a wooden table, studio lighting",
    "seed": 42,
    "width": 512, "height": 512,
    "num_inference_steps": 8,
    "response_format": "b64_json"
  }' | python3 -c "
import sys, json, base64
data = json.load(sys.stdin)
img_b64 = data['data'][0]['b64_json']
with open('$RESULTS_DIR/ref_fp8.png', 'wb') as f:
    f.write(base64.b64decode(img_b64))
print('  FP8 reference image saved.')
" 2>/dev/null || echo "  (FP8 reference image save failed)"

# Compute PSNR if both images exist
python3 - <<'PYEOF'
import os
try:
    from PIL import Image
    import numpy as np
    bf16_path = os.path.join(os.environ.get('RESULTS_DIR', '.'), 'ref_bf16.png')
    fp8_path  = os.path.join(os.environ.get('RESULTS_DIR', '.'), 'ref_fp8.png')
    if os.path.exists(bf16_path) and os.path.exists(fp8_path):
        a = np.array(Image.open(bf16_path)).astype(float)
        b = np.array(Image.open(fp8_path)).astype(float)
        mse = np.mean((a - b) ** 2)
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
        print(f'  PSNR (BF16 vs FP8): {psnr:.2f} dB')
        print(f'  (>35 dB = imperceptible quality difference)')
    else:
        print('  (reference images not found; skip PSNR)')
except ImportError:
    print('  (Pillow not installed; skip PSNR: pip install Pillow)')
PYEOF

echo ""
echo "Lab 4 complete. Results in $RESULTS_DIR"
echo ""
echo "BOTTLENECK ANALYSIS:"
echo "  1. FP8 throughput / BF16 throughput?"
echo "     RATIO ≈ 1.5-2.0 on H100 (native FP8 tensor cores)"
echo "     RATIO ≈ 1.0 on A100 (limited FP8 support, no speedup expected)"
echo "  2. GPU memory: FP8 should use ~50% less memory than BF16"
echo "  3. PSNR > 35 dB → FP8 quality is production-acceptable"
echo "     PSNR < 30 dB → quantization is too aggressive; add more skip_layers"
