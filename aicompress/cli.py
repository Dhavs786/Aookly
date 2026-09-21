"""
Command-Line Interface (CLI) for AI-Based Image & Video Compression System.
Supports individual media compression, feature analysis, model training, and comparative benchmarking.
"""

import argparse
import sys
import json
from pathlib import Path

from aicompress.core.analyzer import MediaAnalyzer
from aicompress.core.optimizer import AdaptiveOptimizer
from aicompress.models.train_model import train_and_save_models
from benchmark import run_benchmark_suite


def main():
    parser = argparse.ArgumentParser(
        prog="ai-compress",
        description="AI-Based Content-Aware Image & Video Compression System"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: compress-image
    img_parser = subparsers.add_parser("compress-image", help="Compress an image using AI optimization")
    img_parser.add_argument("--input", "-i", required=True, help="Path to input image")
    img_parser.add_argument("--output", "-o", required=True, help="Path to output compressed image")
    img_parser.add_argument("--reduction", "-r", type=float, default=80.0, help="Target reduction percentage (default: 80)")
    img_parser.add_argument("--min-ssim", type=float, default=0.92, help="Minimum acceptable SSIM (default: 0.92)")

    # Command: compress-video
    vid_parser = subparsers.add_parser("compress-video", help="Compress a video using AI optimization")
    vid_parser.add_argument("--input", "-i", required=True, help="Path to input video")
    vid_parser.add_argument("--output", "-o", required=True, help="Path to output compressed video")
    vid_parser.add_argument("--reduction", "-r", type=float, default=70.0, help="Target reduction percentage (default: 70)")
    vid_parser.add_argument("--min-ssim", type=float, default=0.88, help="Minimum acceptable SSIM (default: 0.88)")

    # Command: analyze
    ana_parser = subparsers.add_parser("analyze", help="Extract and display media characteristics")
    ana_parser.add_argument("--input", "-i", required=True, help="Path to media file (image or video)")

    # Command: benchmark
    subparsers.add_parser("benchmark", help="Run full evaluation benchmark suite comparing AI vs baseline")

    # Command: train
    subparsers.add_parser("train", help="Train AI parameter prediction models")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    optimizer = AdaptiveOptimizer()

    if args.command == "compress-image":
        print(f"[*] Starting AI Image Compression for: {args.input}")
        rep = optimizer.optimize_image(
            input_path=args.input,
            output_path=args.output,
            target_reduction_pct=args.reduction,
            min_ssim=args.min_ssim
        )
        print("\n=== Compression Complete ===")
        print(f"Output File:       {rep.output_file}")
        print(f"Size Reduction:    {rep.reduction_percentage:.1f}% ({rep.original_size_bytes/1e6:.2f} MB -> {rep.compressed_size_bytes/1e6:.2f} MB)")
        print(f"SSIM Quality:      {rep.ssim:.4f}")
        print(f"PSNR Quality:      {rep.psnr_db:.1f} dB")
        print(f"Processing Time:   {rep.total_processing_time_sec:.2f}s")
        print(f"Predicted Codec:   {rep.final_params['codec'].upper()} (Quality: {rep.final_params['quality']})")

    elif args.command == "compress-video":
        print(f"[*] Starting AI Video Compression for: {args.input}")
        rep = optimizer.optimize_video(
            input_path=args.input,
            output_path=args.output,
            target_reduction_pct=args.reduction,
            min_ssim=args.min_ssim
        )
        print("\n=== Video Compression Complete ===")
        print(f"Output File:       {rep.output_file}")
        print(f"Size Reduction:    {rep.reduction_percentage:.1f}% ({rep.original_size_bytes/1e6:.2f} MB -> {rep.compressed_size_bytes/1e6:.2f} MB)")
        print(f"SSIM Quality:      {rep.ssim:.4f}")
        print(f"PSNR Quality:      {rep.psnr_db:.1f} dB")
        print(f"Processing Time:   {rep.total_processing_time_sec:.2f}s")
        print(f"Predicted Codec:   {rep.final_params['codec'].upper()} (CRF: {rep.final_params['crf']}, GOP: {rep.final_params['keyint']})")

    elif args.command == "analyze":
        p = Path(args.input)
        if not p.exists():
            print(f"Error: File not found {p}")
            sys.exit(1)
        suffix = p.suffix.lower()
        if suffix in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
            chars = MediaAnalyzer.analyze_image(p)
        else:
            chars = MediaAnalyzer.analyze_video(p)
        print(json.dumps(chars.to_dict(), indent=2))

    elif args.command == "benchmark":
        run_benchmark_suite()

    elif args.command == "train":
        train_and_save_models()


if __name__ == "__main__":
    main()
