# vLLM-Omni: Complete Inference Engineering Lab

A phased, standalone lab for exploring every capability of [vLLM-Omni](https://github.com/vllm-project/vllm-omni) — the unified serving runtime for omni-modality AI models.

## What This Lab Covers

| Phase | Focus | Output Modality |
|---|---|---|
| 0 | Environment setup & smoke test | — |
| 1 | AR + Audio pipeline (Qwen3-Omni): unified baseline, async chunk, all input modalities, concurrency, context length, full-duplex VAD UI, **staged 3-process launch**, async chunk offline, text-only vs text+audio output path, stage profiler | Text + Audio |
| 2 | DiT image generation: text-to-image, image editing, CFG parallel | Image |
| 3 | Diffusion video: text-to-video, image-to-video, speech-to-video | Video |
| 4 | TTS pipeline: voice clone, voice design, concurrency cliff, quality eval | Audio only |
| 5 | Robot policy / action generation | Actions |
| 6 | Cross-pipeline benchmark — all 5 output types on one table | All |

## Repo Structure

```
lab-vllm-omni/
├── phase0-setup/
│   └── verify.sh
├── phase1-ar-audio/
│   ├── configs/                    # deploy YAML overlays
│   │   ├── ctx_8k.yaml
│   │   ├── ctx_32k.yaml
│   │   ├── ctx_64k.yaml
│   │   └── qwen3_vad.yaml
│   ├── lab1_baseline.sh            # unified sync serving baseline
│   ├── lab2_async_chunk.sh         # async chunk stage pipelining gain
│   ├── lab3_all_modalities.sh      # all 7 input modality types
│   ├── lab4_concurrency_sweep.sh   # concurrency c=1→32
│   ├── lab5_context_sweep.sh       # context length 8K/32K/64K
│   ├── lab6_realtime_ui.sh         # full-duplex VAD + camera web UI
│   ├── lab7_staged_launch.sh       # 3-process disaggregated launch
│   ├── lab8_async_chunk_offline.sh # async chunk offline (AsyncOmni)
│   ├── lab9_text_vs_audio_output.sh# text-only vs text+audio cost
│   └── lab10_stage_profiler.sh     # per-stage PyTorch trace
├── phase2-image/
│   ├── lab1_text_to_image.sh
│   ├── lab2_image_edit.sh
│   ├── lab3_concurrency_resolution.sh
│   └── lab4_multi_gpu.sh
├── phase3-video/
│   ├── lab1_text_to_video.sh
│   ├── lab2_image_to_video.sh
│   ├── lab3_speech_to_video.sh
│   └── lab4_concurrency_sweep.sh
├── phase4-tts/
│   ├── lab1_voice_clone.sh
│   ├── lab2_voice_design.sh
│   ├── lab3_concurrency_cliff.sh
│   └── lab4_quality_eval.sh
├── phase5-actions/
│   └── lab1_internvla.sh
├── phase6-benchmark/
│   ├── run_all.sh
│   └── plot_comparison.py
└── results/
```

## Prerequisites

- Python 3.10+
- CUDA GPU (minimum 1× A100/H100 80GB for Phase 1 unified launch; 2× for staged)
- vLLM-Omni installed (see Phase 0)

## Quick Start

```bash
# Clone the repo alongside vllm-omni
git clone https://github.com/AyeniOluwatosinOlawale/lab-vllm-omni
cd lab-vllm-omni

# Phase 0 first — verify environment
bash phase0-setup/verify.sh

# Then run phases in order
bash phase1-ar-audio/lab1_baseline.sh
```

## Key Metrics Tracked

| Metric | Definition | Phases |
|---|---|---|
| TTFT | Time to first text token | 1, 6 |
| TTFP | Time to first audio packet | 1, 4, 6 |
| RTF | Real-time factor (wall-sec / audio-sec) | 1, 4, 6 |
| E2E | End-to-end request latency | All |
| img/s | Image generation throughput | 2, 6 |
| vid/s | Video generation throughput | 3, 6 |
| WER | Word error rate (TTS quality) | 4 |
| SIM | Speaker similarity score | 4 |

## Source

Built on top of [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni).
All scripts reference examples and benchmarks from that repo — clone it alongside this one.

```bash
git clone https://github.com/vllm-project/vllm-omni
```
