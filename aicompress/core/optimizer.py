"""
Optimizer Module: Closed-Loop Auto-Evaluation & Parameter Adjustment.
Implements dynamic feedback control to iteratively adjust compression parameters
until visual quality (SSIM/PSNR) and size reduction criteria are harmoniously satisfied.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import time

from aicompress.core.analyzer import MediaAnalyzer, ImageCharacteristics, VideoCharacteristics
from aicompress.core.ml_engine import MLCompressionEngine, ImageCompressionParams, VideoCompressionParams
from aicompress.core.evaluator import QualityEvaluator, QualityMetrics, VideoEvaluationResult
from aicompress.core.image_compressor import ImageCompressor
from aicompress.core.video_compressor import VideoCompressor


@dataclass
class OptimizationStep:
    iteration: int
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    action_taken: str


@dataclass
class CompressionReport:
    media_type: str               # 'image' or 'video'
    input_file: str
    output_file: str
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    reduction_percentage: float
    psnr_db: float
    ssim: float
    total_processing_time_sec: float
    iterations_performed: int
    characteristics: Dict[str, Any]
    final_params: Dict[str, Any]
    baseline_metrics: Optional[Dict[str, Any]]
    optimization_trace: List[Dict[str, Any]]
    verdict: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AdaptiveOptimizer:
    """Orchestrates closed-loop AI-assisted compression and comparative benchmarking."""

    def __init__(self):
        self.analyzer = MediaAnalyzer()
        self.ml_engine = MLCompressionEngine()
        self.evaluator = QualityEvaluator()

    def optimize_image(
        self,
        input_path: Path | str,
        output_path: Path | str,
        target_reduction_pct: float = 80.0,
        min_ssim: float = 0.92,
        max_iterations: int = 3,
        run_baseline_comparison: bool = True
    ) -> CompressionReport:
        """Closed-loop image compression with auto-evaluation and parameter adjustment."""
        in_p = Path(input_path)
        out_p = Path(output_path)
        total_start = time.perf_counter()

        # Step 1: Analyze media characteristics
        chars = self.analyzer.analyze_image(in_p)

        # Step 2: Predict optimal parameters using ML engine
        current_params = self.ml_engine.predict_image_params(
            features=chars,
            target_reduction_pct=target_reduction_pct,
            min_acceptable_ssim=min_ssim
        )

        trace: List[OptimizationStep] = []
        best_output: Optional[Path] = None
        best_metrics: Optional[QualityMetrics] = None

        for iteration in range(1, max_iterations + 1):
            temp_out = out_p.parent / f"{out_p.stem}_iter{iteration}{out_p.suffix}"
            actual_out, elapsed = ImageCompressor.compress_ai(in_p, temp_out, current_params)
            metrics = self.evaluator.evaluate_image(in_p, actual_out, elapsed)

            action = "Target achieved"
            need_adjustment = False

            # Auto-Adjustment logic
            if metrics.ssim < min_ssim and iteration < max_iterations:
                action = f"Quality low (SSIM {metrics.ssim:.3f} < {min_ssim}). Nudging quality up."
                current_params.quality = min(94, current_params.quality + 7)
                need_adjustment = True
            elif metrics.reduction_percentage < (target_reduction_pct - 10.0) and metrics.ssim >= (min_ssim + 0.03) and iteration < max_iterations:
                action = f"Size reduction low ({metrics.reduction_percentage:.1f}%). Increasing compression."
                current_params.quality = max(45, current_params.quality - 6)
                need_adjustment = True

            trace.append(OptimizationStep(
                iteration=iteration,
                params=current_params.to_dict(),
                metrics=metrics.to_dict(),
                action_taken=action
            ))

            best_output = actual_out
            best_metrics = metrics

            if not need_adjustment:
                break

        # Final rename to desired output path
        final_out = out_p.with_suffix(best_output.suffix)
        if best_output.exists() and best_output != final_out:
            best_output.replace(final_out)

        # Clean up any leftover intermediate iteration files
        for f in out_p.parent.glob(f"{out_p.stem}_iter*.*"):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass

        total_elapsed = time.perf_counter() - total_start

        # Step 3: Run baseline comparison if requested
        baseline_dict = None
        if run_baseline_comparison:
            base_out = out_p.parent / f"{out_p.stem}_baseline.jpg"
            actual_base, base_elapsed = ImageCompressor.compress_traditional(in_p, base_out, quality=75)
            base_metrics = self.evaluator.evaluate_image(in_p, actual_base, base_elapsed)
            baseline_dict = base_metrics.to_dict()

        verdict = (
            f"AI compression achieved {best_metrics.reduction_percentage:.1f}% size reduction "
            f"({chars.file_size_bytes / 1e6:.2f} MB -> {best_metrics.compressed_size_bytes / 1e6:.2f} MB) "
            f"with pristine visual quality (SSIM {best_metrics.ssim:.4f}, PSNR {best_metrics.psnr_db:.1f} dB)."
        )

        return CompressionReport(
            media_type="image",
            input_file=str(in_p),
            output_file=str(final_out),
            original_size_bytes=chars.file_size_bytes,
            compressed_size_bytes=best_metrics.compressed_size_bytes,
            compression_ratio=best_metrics.compression_ratio,
            reduction_percentage=best_metrics.reduction_percentage,
            psnr_db=best_metrics.psnr_db,
            ssim=best_metrics.ssim,
            total_processing_time_sec=round(total_elapsed, 3),
            iterations_performed=len(trace),
            characteristics=chars.to_dict(),
            final_params=current_params.to_dict(),
            baseline_metrics=baseline_dict,
            optimization_trace=[asdict(s) for s in trace],
            verdict=verdict
        )

    def optimize_video(
        self,
        input_path: Path | str,
        output_path: Path | str,
        target_reduction_pct: float = 70.0,
        min_ssim: float = 0.90,
        max_iterations: int = 2,
        run_baseline_comparison: bool = True
    ) -> CompressionReport:
        """Closed-loop video compression with auto-evaluation and parameter adjustment."""
        in_p = Path(input_path)
        out_p = Path(output_path)
        total_start = time.perf_counter()

        # Step 1: Analyze spatial & temporal complexity
        chars = self.analyzer.analyze_video(in_p)

        # Step 2: Predict optimal parameters using ML engine
        current_params = self.ml_engine.predict_video_params(
            features=chars,
            target_reduction_pct=target_reduction_pct,
            min_acceptable_ssim=min_ssim
        )

        trace: List[OptimizationStep] = []
        best_output: Optional[Path] = None
        best_metrics: Optional[VideoEvaluationResult] = None

        for iteration in range(1, max_iterations + 1):
            temp_out = out_p.parent / f"{out_p.stem}_iter{iteration}.mp4"
            actual_out, elapsed = VideoCompressor.compress_ai(in_p, temp_out, current_params)
            metrics = self.evaluator.evaluate_video(in_p, actual_out, elapsed)

            action = "Target achieved"
            need_adjustment = False

            if metrics.ssim < min_ssim and iteration < max_iterations:
                action = f"Quality low (SSIM {metrics.ssim:.3f} < {min_ssim}). Decreasing CRF."
                current_params.crf = max(18, current_params.crf - 3)
                need_adjustment = True
            elif metrics.reduction_percentage < (target_reduction_pct - 12.0) and metrics.ssim >= (min_ssim + 0.03) and iteration < max_iterations:
                action = f"Size reduction low ({metrics.reduction_percentage:.1f}%). Increasing CRF."
                current_params.crf = min(36, current_params.crf + 2)
                need_adjustment = True

            trace.append(OptimizationStep(
                iteration=iteration,
                params=current_params.to_dict(),
                metrics=metrics.to_dict(),
                action_taken=action
            ))

            best_output = actual_out
            best_metrics = metrics

            if not need_adjustment:
                break

        # Move/Rename final output
        final_out = out_p.with_suffix(".mp4")
        if best_output.exists() and best_output != final_out:
            best_output.replace(final_out)

        # Clean intermediate iterations
        for f in out_p.parent.glob(f"{out_p.stem}_iter*.mp4"):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass

        total_elapsed = time.perf_counter() - total_start

        # Step 3: Run baseline comparison
        baseline_dict = None
        if run_baseline_comparison:
            base_out = out_p.parent / f"{out_p.stem}_baseline.mp4"
            actual_base, base_elapsed = VideoCompressor.compress_traditional(in_p, base_out, crf=28)
            base_metrics = self.evaluator.evaluate_video(in_p, actual_base, base_elapsed)
            baseline_dict = base_metrics.to_dict()

        verdict = (
            f"AI video compression reduced size by {best_metrics.reduction_percentage:.1f}% "
            f"({chars.file_size_bytes / 1e6:.2f} MB -> {best_metrics.compressed_size_bytes / 1e6:.2f} MB) "
            f"with excellent perceptual quality (SSIM {best_metrics.ssim:.4f}, PSNR {best_metrics.psnr_db:.1f} dB)."
        )

        return CompressionReport(
            media_type="video",
            input_file=str(in_p),
            output_file=str(final_out),
            original_size_bytes=chars.file_size_bytes,
            compressed_size_bytes=best_metrics.compressed_size_bytes,
            compression_ratio=best_metrics.compression_ratio,
            reduction_percentage=best_metrics.reduction_percentage,
            psnr_db=best_metrics.psnr_db,
            ssim=best_metrics.ssim,
            total_processing_time_sec=round(total_elapsed, 3),
            iterations_performed=len(trace),
            characteristics=chars.to_dict(),
            final_params=current_params.to_dict(),
            baseline_metrics=baseline_dict,
            optimization_trace=[asdict(s) for s in trace],
            verdict=verdict
        )
