#!/usr/bin/env python3
"""
Phase 6 — Cross-Pipeline Comparison Plot

Reads benchmark JSON outputs from each pipeline phase and produces:
  1. A unified summary JSON (p6_summary.json)
  2. A side-by-side bar chart (p6_comparison.png)

Usage:
    python3 plot_comparison.py \
        --omni-dir      results/p6_omni \
        --image-file    results/p6_image.json \
        --video-file    results/p6_video.json \
        --tts-c1-dir    results/p6_tts_c1 \
        --tts-c8-dir    results/p6_tts_c8 \
        --output        results/p6_summary.json \
        --chart         results/p6_comparison.png
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def find_result_json(directory: str) -> dict:
    """Find the first .json result file in a directory."""
    for p in Path(directory).glob("*.json"):
        return load_json(str(p))
    raise FileNotFoundError(f"No JSON result found in {directory}")


def extract_omni(result_dir: str) -> dict:
    data = find_result_json(result_dir)
    return {
        "pipeline": "AR+Audio (Qwen3-Omni)",
        "output": "Text + Audio",
        "ttft_p50_ms":       data.get("median_ttft_ms"),
        "e2el_p50_ms":       data.get("median_e2el_ms"),
        "audio_ttfp_p50_ms": data.get("median_audio_ttfp_ms"),
        "audio_rtf_p50":     data.get("median_audio_rtf"),
        "throughput_req_s":  data.get("request_throughput"),
    }


def extract_diffusion(result_file: str, label: str, output_type: str) -> dict:
    data = load_json(result_file)
    return {
        "pipeline": label,
        "output": output_type,
        "e2el_p50_ms":      data.get("median_latency_ms") or data.get("p50_latency_ms"),
        "throughput_req_s": data.get("request_throughput"),
        "ttft_p50_ms":      None,
        "audio_ttfp_p50_ms": None,
        "audio_rtf_p50":    None,
    }


def extract_tts(result_dir: str, concurrency: int) -> dict:
    data = find_result_json(result_dir)
    return {
        "pipeline": f"TTS c={concurrency} (Qwen3-TTS)",
        "output": "Audio only",
        "ttfp_p50_ms":      data.get("median_audio_ttfp_ms"),
        "audio_rtf_p50":    data.get("median_audio_rtf"),
        "audio_underrun_p50": data.get("median_audio_underrun"),
        "throughput_req_s": data.get("request_throughput"),
        "ttft_p50_ms":      None,
        "e2el_p50_ms":      data.get("median_e2el_ms"),
        "audio_ttfp_p50_ms": data.get("median_audio_ttfp_ms"),
    }


def print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 80)
    print(f"{'Pipeline':<35} {'Output':<14} {'TTFT p50':>10} {'E2E p50':>10} {'TTFP p50':>10} {'RTF p50':>9} {'req/s':>7}")
    print("-" * 80)
    for r in rows:
        print(
            f"{r.get('pipeline',''):<35}"
            f" {r.get('output',''):<14}"
            f" {_fmt(r.get('ttft_p50_ms'))+'ms':>10}"
            f" {_fmt(r.get('e2el_p50_ms'))+'ms':>10}"
            f" {_fmt(r.get('audio_ttfp_p50_ms'))+'ms':>10}"
            f" {_fmt(r.get('audio_rtf_p50')):>9}"
            f" {_fmt(r.get('throughput_req_s')):>7}"
        )
    print("=" * 80 + "\n")


def _fmt(val) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def make_chart(rows: list[dict], output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not installed — skipping chart. pip install matplotlib")
        return

    labels = [r["pipeline"].split(" (")[0] for r in rows]
    e2el = [r.get("e2el_p50_ms") or 0 for r in rows]
    ttfp = [r.get("audio_ttfp_p50_ms") or 0 for r in rows]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, e2el, width, label="E2E p50 (ms)", color="steelblue")
    ax.bar(x + width / 2, ttfp, width, label="TTFP p50 (ms)", color="coral")

    ax.set_xlabel("Pipeline")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("vLLM-Omni Cross-Pipeline Latency Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Chart saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--omni-dir",    required=True)
    parser.add_argument("--image-file",  required=True)
    parser.add_argument("--video-file",  required=True)
    parser.add_argument("--tts-c1-dir",  required=True)
    parser.add_argument("--tts-c8-dir",  required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--chart",       required=True)
    args = parser.parse_args()

    rows = []

    def _try(fn, label):
        try:
            return fn()
        except Exception as e:
            print(f"  WARNING: could not load {label}: {e}")
            return None

    if r := _try(lambda: extract_omni(args.omni_dir), "AR+Audio"):
        rows.append(r)
    if r := _try(lambda: extract_diffusion(args.image_file, "Image (FLUX.2-klein-4B)", "Image"), "Image"):
        rows.append(r)
    if r := _try(lambda: extract_diffusion(args.video_file, "Video (Wan2.2-T2V)", "Video"), "Video"):
        rows.append(r)
    if r := _try(lambda: extract_tts(args.tts_c1_dir, 1), "TTS c=1"):
        rows.append(r)
    if r := _try(lambda: extract_tts(args.tts_c8_dir, 8), "TTS c=8"):
        rows.append(r)

    print_table(rows)

    with open(args.output, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  Summary JSON: {args.output}")

    make_chart(rows, args.chart)


if __name__ == "__main__":
    main()
