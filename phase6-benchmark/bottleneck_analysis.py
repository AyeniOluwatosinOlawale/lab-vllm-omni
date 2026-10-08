#!/usr/bin/env python3
"""
Phase 6 — Bottleneck Analysis
=============================================================================
Reads benchmark JSON results from all phases, identifies the bottleneck in
each pipeline, and emits a ranked optimization plan.

Usage:
    python3 phase6-benchmark/bottleneck_analysis.py --results-dir results/

Output:
    - Printed bottleneck report per pipeline
    - results/bottleneck_report.json
    - results/bottleneck_chart.png  (if matplotlib installed)

BOTTLENECK TAXONOMY
===================
Each pipeline has a known bottleneck class:

  AR+AUDIO (Phase 1)
  ├─ Thinker (Stage 0): memory-bandwidth-bound LLM decode
  │    Signal:  TTFT grows with context length
  │    Fix:     KV cache quantization (INT8/INT4), chunked prefill,
  │             Flash Attention, tensor parallel
  ├─ Talker (Stage 1): codec batch_size=1 serialisation
  │    Signal:  TTFP jumps 4-6× at c≥8 while throughput plateaus
  │    Fix:     codec batching (tracked: vllm-omni#272), reduce concurrency ≤4
  └─ Code2Wav (Stage 2): waveform upsampling (conv_transpose1d)
       Signal:  audio_rtf > 1.0 under load
       Fix:     CUDA graph capture, reduce Code2Wav max_num_seqs

  IMAGE DiT (Phase 2)
  ├─ Attention compute: O(tokens²) per step
  │    Signal:  E2E scales as steps² not linearly
  │    Fix:     fewer steps (distilled models), CFG Parallel, Ulysses SP
  └─ Memory bandwidth (VAE decode)
       Signal:  E2E scales linearly with steps
       Fix:     VAE slicing/tiling, increase batch size

  VIDEO DiT (Phase 3)
  ├─ Spatio-temporal attention: O((frames × spatial)²) per step
  │    Signal:  E2E grows quadratically with num_frames
  │    Fix:     lower resolution (-3.5× at 480p vs 720p), TP=2
  └─ Video job queuing at high concurrency
       Signal:  throughput (vid/s) flat at c≥2
       Fix:     use smaller model for fleet workloads

  TTS (Phase 4)
  ├─ Codec concurrency cliff at c≥8
  │    Signal:  TTFP jumps 4-6× between c=4 and c=8
  │    Fix:     keep concurrency ≤4, wait for codec batching (#272)
  └─ Codec throughput for long text
       Signal:  audio_rtf > 1.0 for LONG text at c=8
       Fix:     limit long-text requests to c≤2

  ACTIONS (Phase 5)
  ├─ Vision encoder (per-frame ViT)
  │    Signal:  latency scales linearly with num_frames
  │    Fix:     TP=2 to split vision encoder
  └─ Cosmos tokenizer decode
       Signal:  throughput drops sharply at c>2
       Fix:     quantize tokenizer, reduce action space
=============================================================================
"""

import argparse
import json
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_jsons(directory: str) -> list[dict]:
    results = []
    for p in Path(directory).rglob("*.json"):
        try:
            with open(p) as f:
                data = json.load(f)
                data["_source_file"] = str(p)
                results.append(data)
        except Exception:
            pass
    return results


def _get(d: dict, *keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


# ─────────────────────────────────────────────────────────────────────────────
# Bottleneck detection rules
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Finding:
    phase: str
    pipeline: str
    bottleneck: str
    severity: str          # CRITICAL | HIGH | MEDIUM | INFO
    evidence: str
    optimizations: list[str] = field(default_factory=list)


def analyse_phase1_ar_audio(results_dir: str) -> list[Finding]:
    findings = []

    # ── Lab 4: Concurrency sweep ──────────────────────────────────────────
    c_dirs = sorted(Path(results_dir).glob("p1_lab4_concurrency/c*"))
    if c_dirs:
        ttfp_by_c = {}
        for cdir in c_dirs:
            c = int(cdir.name.lstrip("c"))
            for r in find_jsons(str(cdir)):
                v = _get(r, "median_audio_ttfp_ms", "p50_audio_ttfp_ms")
                if v:
                    ttfp_by_c[c] = v

        if len(ttfp_by_c) >= 2:
            sorted_c = sorted(ttfp_by_c)
            # Detect cliff: TTFP ratio between c=8 and c=4
            c4 = ttfp_by_c.get(4)
            c8 = ttfp_by_c.get(8)
            if c4 and c8:
                ratio = c8 / c4
                if ratio >= 3.0:
                    findings.append(Finding(
                        phase="1 — AR+Audio",
                        pipeline="Qwen3-Omni Talker (Stage 1)",
                        bottleneck="Codec concurrency cliff (batch_size=1)",
                        severity="CRITICAL",
                        evidence=f"TTFP jumped {ratio:.1f}× from c=4 ({c4:.0f}ms) to c=8 ({c8:.0f}ms)",
                        optimizations=[
                            "Keep max concurrency ≤4 until codec batching ships (vllm-omni#272)",
                            "Use stage-overrides to reduce Stage 1 max_num_seqs to match codec capacity",
                            "For high-throughput workloads: route to text-only output (skip Talker + Code2Wav)",
                        ]
                    ))

    # ── Lab 5: Context length sweep ───────────────────────────────────────
    ctx_dirs = sorted(Path(results_dir).glob("p1_lab5_context/*"))
    ttft_by_ctx = {}
    for cdir in ctx_dirs:
        for r in find_jsons(str(cdir)):
            v = _get(r, "median_ttft_ms", "p50_ttft_ms")
            if v:
                ttft_by_ctx[cdir.name] = v

    if "ctx_8k" in ttft_by_ctx and "ctx_64k" in ttft_by_ctx:
        ratio = ttft_by_ctx["ctx_64k"] / ttft_by_ctx["ctx_8k"]
        if ratio > 2.0:
            findings.append(Finding(
                phase="1 — AR+Audio",
                pipeline="Qwen3-Omni Thinker (Stage 0)",
                bottleneck="KV cache memory pressure at long context",
                severity="HIGH",
                evidence=f"TTFT grew {ratio:.1f}× from ctx_8k ({ttft_by_ctx['ctx_8k']:.0f}ms) to ctx_64k ({ttft_by_ctx['ctx_64k']:.0f}ms)",
                optimizations=[
                    "Enable KV cache quantization: --kv-cache-dtype fp8",
                    "Use chunked prefill: --enable-chunked-prefill",
                    "Reduce max_num_batched_tokens if GPU memory pressure causes OOM",
                    "Use paged attention (default in vLLM) — already active",
                ]
            ))

    # ── Lab 9: Text-only vs text+audio ───────────────────────────────────
    text_only = find_jsons(os.path.join(results_dir, "p1_lab9_text_only"))
    text_audio = find_jsons(os.path.join(results_dir, "p1_lab9_text_audio"))
    if text_only and text_audio:
        e2e_text = _get(text_only[0], "median_e2el_ms")
        e2e_audio = _get(text_audio[0], "median_e2el_ms")
        if e2e_text and e2e_audio:
            audio_overhead_pct = (e2e_audio - e2e_text) / e2e_text * 100
            findings.append(Finding(
                phase="1 — AR+Audio",
                pipeline="Qwen3-Omni (Stage 1+2 overhead)",
                bottleneck="Audio synthesis adds E2E latency",
                severity="INFO",
                evidence=f"Text-only E2E: {e2e_text:.0f}ms | Text+Audio E2E: {e2e_audio:.0f}ms (+{audio_overhead_pct:.0f}%)",
                optimizations=[
                    "Use modalities=['text'] for API clients that discard audio",
                    "async_chunk reduces perceived latency by overlapping stages (Lab 2 vs Lab 1)",
                ]
            ))

    if not findings:
        findings.append(Finding(
            phase="1 — AR+Audio", pipeline="Qwen3-Omni",
            bottleneck="No results found",
            severity="INFO",
            evidence="Run phase1-ar-audio labs first",
            optimizations=[]
        ))
    return findings


def analyse_phase2_image(results_dir: str) -> list[Finding]:
    findings = []

    # ── Lab 5: Steps × concurrency ───────────────────────────────────────
    step_dirs = sorted(Path(results_dir).glob("p2_lab5_steps/steps*_c1"))
    e2e_by_steps = {}
    for d in step_dirs:
        steps = int(d.name.split("steps")[1].split("_")[0])
        for r in find_jsons(str(d)):
            v = _get(r, "median_latency_ms", "p50_latency_ms", "mean_latency_ms")
            if v:
                e2e_by_steps[steps] = v

    step_vals = sorted(e2e_by_steps)
    if len(step_vals) >= 2:
        s_lo, s_hi = step_vals[0], step_vals[-1]
        e_lo, e_hi = e2e_by_steps[s_lo], e2e_by_steps[s_hi]
        ideal_ratio = s_hi / s_lo
        actual_ratio = e_hi / e_lo
        if actual_ratio > ideal_ratio * 1.4:
            findings.append(Finding(
                phase="2 — Image",
                pipeline="Image DiT (attention)",
                bottleneck="Quadratic attention scaling with inference steps",
                severity="HIGH",
                evidence=f"Steps {s_lo}→{s_hi} ({ideal_ratio:.1f}× more compute): E2E grew {actual_ratio:.1f}× (super-linear)",
                optimizations=[
                    "Switch to a distilled model (FLUX.2-klein: 8-20 steps vs 50 for full)",
                    "Enable CFG Parallel on 2× GPUs (lab4) to halve per-step time",
                    "Use Ulysses Sequence Parallel for 1024×1024+ resolutions",
                ]
            ))
        else:
            findings.append(Finding(
                phase="2 — Image",
                pipeline="Image DiT (VAE / memory)",
                bottleneck="Memory-bandwidth bound (linear step scaling)",
                severity="MEDIUM",
                evidence=f"Steps {s_lo}→{s_hi}: E2E grew {actual_ratio:.1f}× (≈ linear — memory bound)",
                optimizations=[
                    "Increase batch size (--max-num-seqs) to improve SM utilisation",
                    "Enable --vae-use-slicing to reduce peak VAE memory",
                    "Use BF16 weights (default) rather than FP32",
                ]
            ))

    # ── Lab 3: Concurrency plateau ────────────────────────────────────────
    c_results = {}
    for cdir in Path(results_dir).glob("p2_lab3_concurrency/c*"):
        c = int(cdir.name.lstrip("c"))
        for r in find_jsons(str(cdir)):
            v = _get(r, "request_throughput")
            if v:
                c_results[c] = v

    sorted_c = sorted(c_results)
    if len(sorted_c) >= 3:
        # Check if throughput plateaued
        throughputs = [c_results[c] for c in sorted_c]
        last_gain = (throughputs[-1] - throughputs[-2]) / throughputs[-2]
        if last_gain < 0.05:
            findings.append(Finding(
                phase="2 — Image",
                pipeline="Image DiT (SM saturation)",
                bottleneck="Throughput plateaus at high concurrency — GPU SMs saturated",
                severity="MEDIUM",
                evidence=f"img/s gain from c={sorted_c[-2]} to c={sorted_c[-1]}: only {last_gain*100:.1f}%",
                optimizations=[
                    "Add a second GPU with CFG Parallel (lab4) to double SM capacity",
                    "Use Ulysses-Ring hybrid parallelism for very large resolutions",
                    "Reduce num_inference_steps to free SM budget for more concurrent requests",
                ]
            ))

    if not findings:
        findings.append(Finding(
            phase="2 — Image", pipeline="Image DiT",
            bottleneck="No results found",
            severity="INFO",
            evidence="Run phase2-image labs first",
            optimizations=[]
        ))
    return findings


def analyse_phase3_video(results_dir: str) -> list[Finding]:
    findings = []

    # ── Lab 5: Frames × concurrency ──────────────────────────────────────
    frame_dirs = sorted(Path(results_dir).glob("p3_lab5_frames/frames*_c1"))
    e2e_by_frames = {}
    for d in frame_dirs:
        frames = int(d.name.split("frames")[1].split("_")[0])
        for r in find_jsons(str(d)):
            v = _get(r, "median_latency_ms", "p50_latency_ms", "mean_latency_ms")
            if v:
                e2e_by_frames[frames] = v

    frame_vals = sorted(e2e_by_frames)
    if len(frame_vals) >= 2:
        f_lo, f_hi = frame_vals[0], frame_vals[-1]
        e_lo, e_hi = e2e_by_frames[f_lo], e2e_by_frames[f_hi]
        token_ratio = (f_hi / f_lo) ** 2  # attention is quadratic
        actual_ratio = e_hi / e_lo
        if actual_ratio > (f_hi / f_lo) * 1.5:
            findings.append(Finding(
                phase="3 — Video",
                pipeline="Video DiT (spatio-temporal attention)",
                bottleneck="Quadratic attention scaling with frame count",
                severity="CRITICAL",
                evidence=f"Frames {f_lo}→{f_hi}: E2E grew {actual_ratio:.1f}× (quadratic expected {token_ratio:.1f}×)",
                optimizations=[
                    "Use 480p instead of 720p (~3.5× fewer tokens, ~3.5× faster)",
                    "Use --tensor-parallel-size 2 to shard attention across GPUs",
                    "Use distilled model (fewer steps) — LingBot-Video at 2 steps",
                    "For fleet use: use LingBot-Video 1.3B (fits c>2 on single GPU)",
                ]
            ))

    # ── Lab 4: Concurrency ────────────────────────────────────────────────
    c_results = {}
    for cdir in Path(results_dir).glob("p3_lab4_concurrency/c*"):
        c = int(cdir.name.lstrip("c"))
        for r in find_jsons(str(cdir)):
            v = _get(r, "request_throughput")
            if v:
                c_results[c] = v

    if len(c_results) >= 2:
        throughputs = [c_results[c] for c in sorted(c_results)]
        if throughputs[-1] < throughputs[0] * 1.5:
            findings.append(Finding(
                phase="3 — Video",
                pipeline="Video DiT (job scheduling)",
                bottleneck="Throughput does not scale with concurrency",
                severity="HIGH",
                evidence="vid/s barely increases with concurrency — GPU fully occupied by single job",
                optimizations=[
                    "For single-GPU: keep concurrency=1 and use async polling (--video-job-timeout)",
                    "For throughput: use smaller model (Wan2.2-TI2V-5B) or lower resolution",
                    "Multi-GPU: Tensor Parallel across 2 GPUs allows c=2 without latency blowup",
                ]
            ))

    if not findings:
        findings.append(Finding(
            phase="3 — Video", pipeline="Video DiT",
            bottleneck="No results found",
            severity="INFO",
            evidence="Run phase3-video labs first",
            optimizations=[]
        ))
    return findings


def analyse_phase4_tts(results_dir: str) -> list[Finding]:
    findings = []

    # ── Lab 3: Concurrency cliff ───────────────────────────────────────────
    c_results = {}
    for cdir in Path(results_dir).glob("p4_lab3_cliff"):
        for r in find_jsons(str(cdir)):
            c = r.get("max_concurrency") or r.get("concurrency")
            v = _get(r, "median_audio_ttfp_ms", "p50_audio_ttfp_ms")
            if c and v:
                c_results[int(c)] = v

    if len(c_results) >= 2:
        c4 = c_results.get(4)
        c8 = c_results.get(8)
        if c4 and c8:
            ratio = c8 / c4
            findings.append(Finding(
                phase="4 — TTS",
                pipeline="Qwen3-TTS Codec (batch_size=1 cliff)",
                bottleneck="Codec serialisation cliff at c≥8",
                severity="CRITICAL",
                evidence=f"TTFP at c=4: {c4:.0f}ms → c=8: {c8:.0f}ms ({ratio:.1f}× increase)",
                optimizations=[
                    "Hard limit: serve at max_concurrency=4 for TTFP SLO compliance",
                    "Track vllm-omni#272 for codec batching — will remove the cliff",
                    "For high-throughput: deploy multiple single-GPU replicas behind a load balancer",
                    "Set VLLM_OMNI_BENCH_AUDIO_CONTINUITY_THRESHOLD_S=0.1 to alert on streaming gaps",
                ]
            ))

    # ── Lab 5: Text length × concurrency ─────────────────────────────────
    long_c8 = list(Path(results_dir).glob("p4_lab5_textlen/long_c8"))
    short_c8 = list(Path(results_dir).glob("p4_lab5_textlen/short_c8"))
    if long_c8 and short_c8:
        rtf_long = rtf_short = None
        for r in find_jsons(str(long_c8[0])):
            rtf_long = _get(r, "median_audio_rtf")
        for r in find_jsons(str(short_c8[0])):
            rtf_short = _get(r, "median_audio_rtf")
        if rtf_long and rtf_short:
            if rtf_long > 1.0:
                findings.append(Finding(
                    phase="4 — TTS",
                    pipeline="Qwen3-TTS (long-text at high concurrency)",
                    bottleneck="RTF > 1.0 for long text at c=8 — cannot serve realtime",
                    severity="CRITICAL",
                    evidence=f"RTF for long text at c=8: {rtf_long:.2f} (>1.0 = slower than realtime)",
                    optimizations=[
                        "Limit long-text requests to concurrency ≤2",
                        "Split long texts into ≤50-word chunks and stream audio chunks to client",
                        "Use sentence-boundary splitting before sending to TTS",
                    ]
                ))

    if not findings:
        findings.append(Finding(
            phase="4 — TTS", pipeline="Qwen3-TTS",
            bottleneck="No results found",
            severity="INFO",
            evidence="Run phase4-tts labs first",
            optimizations=[]
        ))
    return findings


def analyse_phase5_actions(results_dir: str) -> list[Finding]:
    findings = []

    horizon_dirs = sorted(Path(results_dir).glob("p5_lab2_horizon/frames*"))
    latency_by_frames = {}
    for d in horizon_dirs:
        frames = int(d.name.split("frames")[1])
        for log in d.glob("run.log"):
            text = log.read_text()
            for line in text.splitlines():
                if "latency" in line.lower() or "time" in line.lower():
                    latency_by_frames[frames] = line.strip()
                    break

    if len(latency_by_frames) >= 2:
        findings.append(Finding(
            phase="5 — Actions",
            pipeline="InternVLA-A1 (vision encoder)",
            bottleneck="Latency scales with observation horizon depth",
            severity="MEDIUM",
            evidence=str(latency_by_frames),
            optimizations=[
                "Use --tensor-parallel-size 2 to split vision encoder across GPUs",
                "Reduce observation horizon (fewer frames) if action quality permits",
                "Quantize Cosmos tokenizer weights (INT8) to reduce decode time",
            ]
        ))

    if not findings:
        findings.append(Finding(
            phase="5 — Actions", pipeline="InternVLA-A1",
            bottleneck="No results found — run phase5-actions/lab2_concurrency_context.sh first",
            severity="INFO",
            evidence="",
            optimizations=[]
        ))
    return findings


# ─────────────────────────────────────────────────────────────────────────────
# Cross-pipeline summary
# ─────────────────────────────────────────────────────────────────────────────

SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "INFO": 3}
SEVERITY_ICONS = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "INFO": "🔵"}


def print_report(all_findings: list[Finding]) -> None:
    sorted_findings = sorted(all_findings, key=lambda f: SEVERITY_ORDER.get(f.severity, 9))

    print("\n" + "=" * 72)
    print("  vLLM-Omni BOTTLENECK ANALYSIS REPORT")
    print("=" * 72)

    for f in sorted_findings:
        icon = SEVERITY_ICONS.get(f.severity, "  ")
        print(f"\n{icon} [{f.severity}] {f.phase} | {f.pipeline}")
        print(f"   Bottleneck:  {f.bottleneck}")
        print(f"   Evidence:    {f.evidence}")
        if f.optimizations:
            print("   Optimizations:")
            for i, opt in enumerate(f.optimizations, 1):
                print(f"     {i}. {opt}")

    print("\n" + "=" * 72)
    counts = {}
    for f in all_findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    print("  Summary: " + "  ".join(
        f"{SEVERITY_ICONS[s]} {counts.get(s, 0)} {s}"
        for s in ["CRITICAL", "HIGH", "MEDIUM", "INFO"]
    ))
    print("=" * 72 + "\n")


def make_chart(all_findings: list[Finding], output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("  matplotlib not installed — skipping chart")
        return

    severity_colors = {"CRITICAL": "#e74c3c", "HIGH": "#e67e22", "MEDIUM": "#f1c40f", "INFO": "#3498db"}
    phases = [f.phase.split(" — ")[1] for f in all_findings]
    severities = [f.severity for f in all_findings]
    colors = [severity_colors[s] for s in severities]
    scores = [3 - SEVERITY_ORDER[s] for s in severities]

    fig, ax = plt.subplots(figsize=(14, max(4, len(all_findings) * 0.6 + 2)))
    y = np.arange(len(all_findings))
    bars = ax.barh(y, scores, color=colors, edgecolor="white", height=0.7)

    labels = [f"{f.pipeline}: {f.bottleneck[:55]}..." if len(f.bottleneck) > 55 else f"{f.pipeline}: {f.bottleneck}"
              for f in all_findings]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Severity Score (3=Critical, 0=Info)")
    ax.set_title("vLLM-Omni Bottleneck Analysis — All Pipelines")
    ax.set_xlim(0, 3.5)

    patches = [mpatches.Patch(color=c, label=s) for s, c in severity_colors.items()]
    ax.legend(handles=patches, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Chart saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="vLLM-Omni cross-pipeline bottleneck analysis")
    parser.add_argument("--results-dir", default="results/", help="Root results directory")
    parser.add_argument("--output",      default=None,       help="JSON output path")
    parser.add_argument("--chart",       default=None,       help="PNG chart output path")
    args = parser.parse_args()

    rdir = args.results_dir
    output = args.output or os.path.join(rdir, "bottleneck_report.json")
    chart  = args.chart  or os.path.join(rdir, "bottleneck_chart.png")

    all_findings: list[Finding] = []
    all_findings += analyse_phase1_ar_audio(rdir)
    all_findings += analyse_phase2_image(rdir)
    all_findings += analyse_phase3_video(rdir)
    all_findings += analyse_phase4_tts(rdir)
    all_findings += analyse_phase5_actions(rdir)

    print_report(all_findings)

    with open(output, "w") as f:
        json.dump([asdict(x) for x in all_findings], f, indent=2)
    print(f"  Report JSON: {output}")

    make_chart(all_findings, chart)


if __name__ == "__main__":
    main()
