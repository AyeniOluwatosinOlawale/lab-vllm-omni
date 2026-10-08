#!/usr/bin/env bash
# =============================================================================
# Phase 3 — Lab 3: Speech-to-Video (Talking Head)
#
# What you learn:
#   S2V generates a talking-head video autoregressively: each clip's last
#   frames become the motion context for the next clip, producing a
#   seamless video that spans the full audio duration.
#   At 720p, self-attention runs over ~80K tokens — lower to 480p for
#   ~3.5× speedup.
#
#   Performance tips (all compatible):
#     --height 448 --width 832        480p  (~3.5× faster than 720p)
#     --num-inference-steps 5         few-step (~8× faster, lower quality)
#     --tensor-parallel-size 2        2-GPU TP (~1.4× faster)
#
# Outputs: results/p3_lab3_s2v/
# =============================================================================
set -euo pipefail
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../../vllm-omni}"
MODEL="${S2V_MODEL:-Wan-AI/Wan2.2-S2V-14B}"
RESULTS_DIR="$(dirname "$0")/../results/p3_lab3_s2v"
mkdir -p "$RESULTS_DIR"

echo "======================================================="
echo " Phase 3 | Lab 3: Speech-to-Video (Talking Head)"
echo "======================================================="

# Download reference assets from Wan2.2 repo
PORTRAIT="$RESULTS_DIR/portrait.png"
AUDIO="$RESULTS_DIR/audio.mp3"

if [ ! -f "$PORTRAIT" ]; then
  echo "[0/4] Downloading reference portrait ..."
  wget -q -O "$PORTRAIT" \
    "https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/examples/Five%20Hundred%20Miles.png"
fi
if [ ! -f "$AUDIO" ]; then
  echo "  Downloading reference audio ..."
  wget -q -O "$AUDIO" \
    "https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/examples/Five%20Hundred%20Miles.MP3"
fi

echo ""
echo "[1/4] 480p, 5 denoising steps (fast, lower quality) ..."
python3 "$VLLM_OMNI_DIR/examples/offline_inference/speech_to_video/speech_to_video.py" \
  --model "$MODEL" \
  --image "$PORTRAIT" \
  --audio "$AUDIO" \
  --prompt "A person singing expressively" \
  --height 448 --width 832 \
  --num-inference-steps 5 \
  --output "$RESULTS_DIR/s2v_480p_5steps.mp4"

echo ""
echo "[2/4] 480p, 40 denoising steps (full quality) ..."
python3 "$VLLM_OMNI_DIR/examples/offline_inference/speech_to_video/speech_to_video.py" \
  --model "$MODEL" \
  --image "$PORTRAIT" \
  --audio "$AUDIO" \
  --prompt "A person singing expressively" \
  --height 448 --width 832 \
  --num-inference-steps 40 \
  --output "$RESULTS_DIR/s2v_480p_40steps.mp4"

echo ""
echo "[3/4] With --enable-diffusion-pipeline-profiler to show stage timings ..."
python3 "$VLLM_OMNI_DIR/examples/offline_inference/speech_to_video/speech_to_video.py" \
  --model "$MODEL" \
  --image "$PORTRAIT" \
  --audio "$AUDIO" \
  --prompt "A person speaking naturally" \
  --height 448 --width 832 \
  --num-inference-steps 5 \
  --enable-diffusion-pipeline-profiler \
  --output "$RESULTS_DIR/s2v_profiled.mp4" \
  2>&1 | tee "$RESULTS_DIR/profiler.log"

echo ""
echo "[4/4] Results summary:"
ls -lh "$RESULTS_DIR"/*.mp4
echo ""
echo "  Compare s2v_480p_5steps.mp4 vs s2v_480p_40steps.mp4 for quality/speed tradeoff."
echo "  Check profiler.log for per-stage timing breakdown."
echo ""
echo "Lab 3 complete."
echo "Next: bash phase3-video/lab4_concurrency_sweep.sh"
