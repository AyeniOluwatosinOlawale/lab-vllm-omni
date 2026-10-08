# vLLM-Omni: Complete Inference Engineering Lab

A phased, standalone lab for exploring every capability of [vLLM-Omni](https://github.com/vllm-project/vllm-omni) — the unified serving runtime for omni-modality AI models.

## What This Lab Covers

| Phase | Focus | Output Modality |
|---|---|---|
| 0 | Environment setup & smoke test | — |
| 1 | AR + Audio pipeline (Qwen3-Omni): full-duplex, stage pipelining, all input modalities, concurrency, context length | Text + Audio |
| 2 | DiT image generation: text-to-image, image editing, CFG parallel | Image |
| 3 | Diffusion video: text-to-video, image-to-video, speech-to-video | Video |
| 4 | TTS pipeline: voice clone, voice design, concurrency cliff, quality eval | Audio only |
| 5 | Robot policy / action generation | Actions |
| 6 | Cross-pipeline benchmark — all 5 output types on one table | All |

## Repo Structure

```
lab-vllm-omni/
├── phase0-setup/
├── phase1-ar-audio/
│   └── configs/          # deploy YAML overlays
├── phase2-image/
├── phase3-video/
├── phase4-tts/
├── phase5-actions/
├── phase6-benchmark/
└── results/              # all benchmark JSON outputs land here
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
