# vLLM-Omni: Complete Inference Engineering Lab

A phased, standalone lab covering **every feature of vllm-omni==0.30.0** — the unified serving runtime for omni-modality AI models.

**Pinned version:** `vllm-omni==0.30.0` (2026-09-25)

## What This Lab Covers

| Phase | Focus | Output Modality |
|---|---|---|
| 0 | Environment setup & smoke test | — |
| 1 | AR + Audio (Qwen3-Omni): baseline, async chunk, all input modalities, concurrency, context length, full-duplex VAD UI, staged 3-process launch, async chunk offline, text vs audio output, stage profiler | Text + Audio |
| 2 | DiT image generation: T2I, image editing, concurrency, multi-GPU CFG+Ulysses, steps sweep | Image |
| 3 | Diffusion video: T2V, I2V, S2V, concurrency, frame sweep | Video |
| 4 | TTS: voice clone, voice design, concurrency cliff, quality eval, text length sweep | Audio only |
| 5 | Robot VLA: InternVLA-A1 action generation, concurrency, observation horizon | Actions |
| 6 | Cross-pipeline benchmark — all 5 output types on one table | All |
| **7** | **New omni models (0.30.0 graduates): MiniCPM-o 4.5, PersonaPlex, AURA, Ming-flash-omni-2.0** | Text + Audio |
| **8** | **Extended modalities: music generation, talking-head avatar, ASR, /v1/audio/generate** | Audio / Video |
| **9** | **New serving features: sleep/wakeup API, multi-process API, forced aligner, audio sample rates** | — |
| **10** | **Advanced diffusion (0.30.0): BAGEL MoT continuous batching, SANA-Video, HunyuanImage prefix cache, FP8 quant** | Image / Video |
| **11** | **Cross-stage KV: Mooncake+NIXL handoff, LingBot-World world-model streaming, RL rollout API** | Video |
| **12** | **Extended VLA: GR00T N1.7, π0.5, OpenPI WebSocket robot policy** | Actions |
| **13** | **Extended benchmarks: Daily-Omni QA, OmniInteract duplex eval, LingBot-Video quality, RDMA, accuracy** | All |

## Repo Structure

```
lab-vllm-omni/
├── requirements.txt                   # pinned: vllm-omni==0.30.0
├── phase0-setup/
│   └── verify.sh                      # GPU check, version assert, smoke test
├── phase1-ar-audio/
│   ├── configs/                       # deploy YAML overlays (8K/32K/64K ctx, VAD)
│   ├── lab1_baseline.sh               # unified sync serving baseline
│   ├── lab2_async_chunk.sh            # async chunk stage pipelining
│   ├── lab3_all_modalities.sh         # all 7 input modality types
│   ├── lab4_concurrency_sweep.sh      # c=1→32
│   ├── lab5_context_sweep.sh          # context 8K/32K/64K
│   ├── lab6_realtime_ui.sh            # full-duplex VAD + camera UI
│   ├── lab7_staged_launch.sh          # 3-process disaggregated launch
│   ├── lab8_async_chunk_offline.sh    # AsyncOmni offline
│   ├── lab9_text_vs_audio_output.sh   # text-only vs audio output cost
│   └── lab10_stage_profiler.sh        # per-stage PyTorch trace
├── phase2-image/
│   ├── lab1_text_to_image.sh
│   ├── lab2_image_edit.sh
│   ├── lab3_concurrency_resolution.sh
│   ├── lab4_multi_gpu.sh              # CFG parallel + Ulysses SP
│   └── lab5_steps_sweep.sh
├── phase3-video/
│   ├── lab1_text_to_video.sh          # GPU_SAFE auto-detects 1 vs 2 GPUs
│   ├── lab2_image_to_video.sh
│   ├── lab3_speech_to_video.sh
│   ├── lab4_concurrency_sweep.sh
│   └── lab5_frames_sweep.sh
├── phase4-tts/
│   ├── lab1_voice_clone.sh
│   ├── lab2_voice_design.sh
│   ├── lab3_concurrency_cliff.sh
│   ├── lab4_quality_eval.sh
│   └── lab5_text_length_sweep.sh
├── phase5-actions/
│   ├── lab1_internvla.sh
│   └── lab2_concurrency_context.sh
├── phase6-benchmark/
│   ├── run_all.sh
│   ├── plot_comparison.py
│   └── bottleneck_analysis.py
├── phase7-new-omni/                   # NEW — 0.30.0 graduated models
│   ├── lab1_minicpmo.sh               # MiniCPM-o 4.5 (graduated from experimental)
│   ├── lab2_personaplex.sh            # PersonaPlex full-duplex 24 kHz
│   ├── lab3_aura.sh                   # AURA vision-only duplex + multi-turn
│   └── lab4_ming_omni.sh              # Ming-flash-omni-2.0: audio+image from one endpoint
├── phase8-extended-modalities/        # NEW — modalities not in phases 1-5
│   ├── lab1_music.sh                  # MiniMax Music 3 + YuE2-3B (/v1/audio/generate)
│   ├── lab2_avatar.sh                 # LongCat-Video-Avatar-1.5 talking-head
│   ├── lab3_asr.sh                    # MiMo-V2.5-ASR + MiMo-Audio-7B (speech→text)
│   └── lab4_audio_generate.sh        # OmniVoice variable-length attention + Stable-Audio-Open
├── phase9-serving-features/           # NEW — 0.30.0 serving API features
│   ├── lab1_sleep_wakeup.sh           # --enable-sleep-mode, /v1/omni/sleep|wakeup
│   ├── lab2_multi_process_api.sh      # --api-server-count N (new in 0.30.0)
│   ├── lab3_forced_aligner.sh         # --forced-aligner (TTS word timestamps)
│   └── lab4_audio_sample_rates.sh     # configurable output sample rates
├── phase10-advanced-diffusion/        # NEW — 0.30.0 diffusion features
│   ├── lab1_bagel_mot.sh              # BAGEL 7B-MoT continuous batching
│   ├── lab2_sana_video.sh             # SANA-Video: CFG+USP+Cache-DiT stacked
│   ├── lab3_hunyuan_prefix_cache.sh   # HunyuanImage3 cross-request prefix caching
│   └── lab4_fp8_quantization.sh       # FP8 quant for diffusion (Boogu-Image Turbo)
├── phase11-cross-stage-kv/            # NEW — disaggregated KV + world models
│   ├── lab1_mooncake_kv.sh            # Mooncake AR→DiT KV handoff (NIXL)
│   ├── lab2_lingbot_world.sh          # LingBot-World 2.0 interactive streaming
│   └── lab3_rl_rollout.sh             # RL rollout API (DreamZero-DROID)
├── phase12-extended-vla/              # NEW — more robot policies
│   ├── lab1_groot_n17.sh              # GR00T N1.7 NVIDIA humanoid policy
│   ├── lab2_pi05.sh                   # π0.5 (new in 0.30.0, NVIDIA+Intel)
│   └── lab3_openpi_websocket.sh       # OpenPI WebSocket for realtime control
├── phase13-extended-benchmarks/       # NEW — quality + distributed benchmarks
│   ├── lab1_daily_omni.sh             # Daily-Omni video QA (accuracy + latency)
│   ├── lab2_omniinteract.sh           # OmniInteract multi-turn duplex eval
│   ├── lab3_lingbot_video.sh          # LingBot-Video MAE/MSE/PSNR quality
│   ├── lab4_rdma_distributed.sh       # Mooncake RDMA copy/zerocopy/GPUDirect
│   └── lab5_accuracy.sh              # GEBench (T2I) + GEdit-Bench (I2I) quality
└── results/
```

## Prerequisites

```bash
pip install -r requirements.txt   # installs vllm-omni==0.30.0 + deps
```

- Python 3.10–3.13
- CUDA GPU: 1× A100/H100 80GB for single-GPU labs; 2× for multi-GPU labs
- vLLM-Omni source (for examples/ and benchmarks/):
  ```bash
  git clone https://github.com/vllm-project/vllm-omni
  export VLLM_OMNI_DIR=/path/to/vllm-omni
  ```

## Quick Start

```bash
git clone https://github.com/AyeniOluwatosinOlawale/lab-vllm-omni
cd lab-vllm-omni

# Verify environment (asserts vllm-omni==0.30.0)
bash phase0-setup/verify.sh

# Run in order, or jump to any phase
bash phase1-ar-audio/lab1_baseline.sh
bash phase7-new-omni/lab1_minicpmo.sh    # 0.30.0 new models
bash phase9-serving-features/lab1_sleep_wakeup.sh  # 0.30.0 new features
```

## Key Metrics Tracked

| Metric | Definition | Phases |
|---|---|---|
| TTFT | Time to first text token | 1, 7, 13 |
| TTFP | Time to first audio packet | 1, 4, 7, 9 |
| RTF | Real-time factor (wall-sec / audio-sec) | 1, 4, 7, 9 |
| E2E | End-to-end request latency | All |
| audio_underrun | Streaming gaps (RTF > 1.0) | 4, 7, 8 |
| img/s | Image generation throughput | 2, 10 |
| vid/s | Video generation throughput | 3, 10, 11 |
| VIDEO_RTF | World-model realtime factor | 11 |
| WER | Word error rate (TTS/ASR quality) | 4, 8 |
| PSNR | Image/video quality (dB) | 10, 13 |
| steps/sec | Robot action throughput | 5, 12 |
| turn_accuracy | Duplex conversation correctness | 13 |
| bandwidth GB/s | Inter-stage KV transfer rate | 11, 13 |

## 0.30.0 Feature Coverage

| Feature | Lab |
|---|---|
| Unified duplex engine (DuplexOmni) | phase7/lab2, phase7/lab3, phase13/lab2 |
| Sleep / Wakeup API | phase9/lab1 |
| Multi-process API serving | phase9/lab2 |
| Forced aligner (word timestamps) | phase9/lab3 |
| Configurable audio sample rates | phase9/lab4 |
| BAGEL MoT continuous batching | phase10/lab1 |
| SANA-Video (CFG+USP+Cache-DiT) | phase10/lab2 |
| HunyuanImage3 prefix caching | phase10/lab3 |
| FP8 diffusion quantization | phase10/lab4 |
| Mooncake + NIXL KV handoff | phase11/lab1 |
| LingBot-World world-model streaming | phase11/lab2 |
| RL rollout API | phase11/lab3 |
| GR00T N1.7 robot policy | phase12/lab1 |
| π0.5 (new VLA) | phase12/lab2 |
| OpenPI WebSocket endpoint | phase12/lab3 |
| MiniCPM-o 4.5 (graduated) | phase7/lab1 |
| PersonaPlex 24 kHz duplex | phase7/lab2 |
| Music generation (/v1/audio/generate) | phase8/lab1 |
| Avatar generation | phase8/lab2 |
| ASR (speech-to-text) | phase8/lab3 |
| OmniVoice variable-length attention | phase8/lab4 |
| Daily-Omni accuracy benchmark | phase13/lab1 |
| OmniInteract duplex eval | phase13/lab2 |
| RDMA copy/zerocopy/GPUDirect | phase13/lab4 |
| GEBench + GEdit-Bench accuracy | phase13/lab5 |

## Source

Built on [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) — version **0.30.0**.
