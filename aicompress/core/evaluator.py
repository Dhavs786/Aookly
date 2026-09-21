"""
Evaluator Module: Objective & Perceptual Quality Evaluation.
Calculates SSIM, PSNR, Compression Ratio, Bitrate, Processing Speed,
and efficiency score between original and compressed media using high-performance
OpenCV and NumPy implementations (Wang et al., IEEE TIP 2004).
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from PIL import Image


@dataclass
class QualityMetrics:
    psnr_db: float
    ssim: float
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float      # e.g., 5.0x
    reduction_percentage: float   # e.g., 80.0%
    processing_time_sec: float
    throughput: str               # e.g., "12.4 MP/s" or "48.2 FPS"
    efficiency_score: float       # Composite Pareto score (SSIM * Reduction / Time penalty)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoEvaluationResult(QualityMetrics):
    psnr_min: float
    psnr_std: float
    ssim_min: float
    ssim_std: float
    original_bitrate_kbps: float
    compressed_bitrate_kbps: float
    frames_evaluated: int


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB)."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0  # Infinite PSNR
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Computes Structural Similarity Index (SSIM) based on Wang et al. (2004).
    Uses standard parameters: K1=0.01, K2=0.03, dynamic range L=255.
    """
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    # If RGB, compute per-channel SSIM and average
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        ssims = []
        for ch in range(3):
            s = _ssim_single_channel(img1[:, :, ch], img2[:, :, ch], c1, c2)
            ssims.append(s)
        return float(np.mean(ssims))
    return float(_ssim_single_channel(img1, img2, c1, c2))


def _ssim_single_channel(c1_img: np.ndarray, c2_img: np.ndarray, c1: float, c2: float) -> float:
    f1 = c1_img.astype(np.float64)
    f2 = c2_img.astype(np.float64)

    kernel_size = (11, 11)
    sigma = 1.5

    mu1 = cv2.GaussianBlur(f1, kernel_size, sigma)
    mu2 = cv2.GaussianBlur(f2, kernel_size, sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(f1 * f1, kernel_size, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(f2 * f2, kernel_size, sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(f1 * f2, kernel_size, sigma) - mu1_mu2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / (denominator + 1e-10)

    return float(np.mean(ssim_map))


class QualityEvaluator:
    """Evaluates perceptual similarity and compression metrics."""

    @staticmethod
    def _calculate_efficiency_score(ssim: float, reduction_pct: float, elapsed: float) -> float:
        """
        Computes balanced Pareto score:
        High reduction with high SSIM is rewarded; long elapsed time is penalized gracefully.
        """
        base = (ssim ** 2) * (reduction_pct / 100.0) * 100.0
        time_penalty = 1.0 + (0.05 * min(elapsed, 20.0))
        return round(base / time_penalty, 2)

    @classmethod
    def evaluate_image(
        cls,
        original_path: Path | str,
        compressed_path: Path | str,
        elapsed_time: float = 0.0
    ) -> QualityMetrics:
        """Compares original vs compressed image."""
        orig_p = Path(original_path)
        comp_p = Path(compressed_path)

        orig_size = orig_p.stat().st_size
        comp_size = comp_p.stat().st_size

        orig_img = cv2.imread(str(orig_p))
        if orig_img is None:
            orig_img = cv2.cvtColor(np.array(Image.open(orig_p).convert("RGB")), cv2.COLOR_RGB2BGR)

        comp_img = cv2.imread(str(comp_p))
        if comp_img is None:
            comp_img = cv2.cvtColor(np.array(Image.open(comp_p).convert("RGB")), cv2.COLOR_RGB2BGR)

        h1, w1 = orig_img.shape[:2]
        h2, w2 = comp_img.shape[:2]
        if (h1, w1) != (h2, w2):
            comp_img = cv2.resize(comp_img, (w1, h1), interpolation=cv2.INTER_LANCZOS4)

        orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        comp_rgb = cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB)

        psnr_val = calculate_psnr(orig_rgb, comp_rgb)
        ssim_val = calculate_ssim(orig_rgb, comp_rgb)

        comp_ratio = orig_size / max(comp_size, 1)
        reduction_pct = max(0.0, ((orig_size - comp_size) / max(orig_size, 1)) * 100.0)

        megapixels = (w1 * h1) / 1e6
        mp_s = f"{megapixels / max(elapsed_time, 0.001):.1f} MP/s"
        efficiency = cls._calculate_efficiency_score(ssim_val, reduction_pct, elapsed_time)

        return QualityMetrics(
            psnr_db=round(psnr_val, 2),
            ssim=round(ssim_val, 4),
            original_size_bytes=orig_size,
            compressed_size_bytes=comp_size,
            compression_ratio=round(comp_ratio, 2),
            reduction_percentage=round(reduction_pct, 2),
            processing_time_sec=round(elapsed_time, 3),
            throughput=mp_s,
            efficiency_score=efficiency
        )

    @classmethod
    def evaluate_video(
        cls,
        original_path: Path | str,
        compressed_path: Path | str,
        elapsed_time: float = 0.0,
        sample_frames: int = 15
    ) -> VideoEvaluationResult:
        """Compares original vs compressed video across sample frames."""
        orig_p = Path(original_path)
        comp_p = Path(compressed_path)

        orig_size = orig_p.stat().st_size
        comp_size = comp_p.stat().st_size

        cap_orig = cv2.VideoCapture(str(orig_p))
        cap_comp = cv2.VideoCapture(str(comp_p))

        orig_frames = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
        comp_frames = int(cap_comp.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap_orig.get(cv2.CAP_PROP_FPS)) or 30.0

        orig_duration = orig_frames / fps if fps > 0 else 1.0
        comp_duration = comp_frames / fps if fps > 0 else 1.0

        orig_bitrate = (orig_size * 8) / (orig_duration * 1000)
        comp_bitrate = (comp_size * 8) / (comp_duration * 1000)

        min_total_frames = min(orig_frames, comp_frames)
        if min_total_frames <= 0:
            cap_orig.release()
            cap_comp.release()
            raise ValueError("Cannot read video frames for evaluation.")

        step = max(1, min_total_frames // sample_frames)
        indices = list(range(0, min_total_frames, step))[:sample_frames]

        psnr_vals: List[float] = []
        ssim_vals: List[float] = []

        w_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for idx in indices:
            cap_orig.set(cv2.CAP_PROP_POS_FRAMES, idx)
            cap_comp.set(cv2.CAP_PROP_POS_FRAMES, idx)

            ret1, f1 = cap_orig.read()
            ret2, f2 = cap_comp.read()

            if not ret1 or not ret2 or f1 is None or f2 is None:
                continue

            if f1.shape != f2.shape:
                f2 = cv2.resize(f2, (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4)

            f1_rgb = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
            f2_rgb = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)

            p = calculate_psnr(f1_rgb, f2_rgb)
            s = calculate_ssim(f1_rgb, f2_rgb)

            psnr_vals.append(p)
            ssim_vals.append(s)

        cap_orig.release()
        cap_comp.release()

        avg_psnr = float(np.mean(psnr_vals)) if psnr_vals else 0.0
        min_psnr = float(np.min(psnr_vals)) if psnr_vals else 0.0
        std_psnr = float(np.std(psnr_vals)) if psnr_vals else 0.0

        avg_ssim = float(np.mean(ssim_vals)) if ssim_vals else 0.0
        min_ssim = float(np.min(ssim_vals)) if ssim_vals else 0.0
        std_ssim = float(np.std(ssim_vals)) if ssim_vals else 0.0

        comp_ratio = orig_size / max(comp_size, 1)
        reduction_pct = max(0.0, ((orig_size - comp_size) / max(orig_size, 1)) * 100.0)

        fps_speed = f"{len(psnr_vals) / max(elapsed_time, 0.001):.1f} FPS"
        efficiency = cls._calculate_efficiency_score(avg_ssim, reduction_pct, elapsed_time)

        return VideoEvaluationResult(
            psnr_db=round(avg_psnr, 2),
            ssim=round(avg_ssim, 4),
            original_size_bytes=orig_size,
            compressed_size_bytes=comp_size,
            compression_ratio=round(comp_ratio, 2),
            reduction_percentage=round(reduction_pct, 2),
            processing_time_sec=round(elapsed_time, 3),
            throughput=fps_speed,
            efficiency_score=efficiency,
            psnr_min=round(min_psnr, 2),
            psnr_std=round(std_psnr, 2),
            ssim_min=round(min_ssim, 4),
            ssim_std=round(std_ssim, 4),
            original_bitrate_kbps=round(orig_bitrate, 1),
            compressed_bitrate_kbps=round(comp_bitrate, 1),
            frames_evaluated=len(psnr_vals)
        )
