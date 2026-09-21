"""
Analyzer Module: Content-Aware Feature Extractor.
Extracts spatial and temporal complexity metrics from images and video streams
based on ITU-T P.910 video quality standards and computer vision metrics.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image


@dataclass
class ImageCharacteristics:
    width: int
    height: int
    megapixels: float
    aspect_ratio: float
    channels: int
    file_size_bytes: int
    spatial_information: float     # ITU-T P.910 SI (Sobel std dev)
    laplacian_variance: float      # Sharpness/edge energy
    shannon_entropy: float         # Information entropy (0-8 bits)
    color_richness: float          # Standard deviation of chroma/color channels
    high_freq_dct_ratio: float     # Ratio of high-frequency DCT energy
    noise_estimate: float          # Perceptual noise floor

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def feature_vector(self) -> np.ndarray:
        return np.array([
            self.megapixels,
            self.spatial_information,
            self.laplacian_variance,
            self.shannon_entropy,
            self.color_richness,
            self.high_freq_dct_ratio,
            self.noise_estimate
        ], dtype=np.float32)


@dataclass
class VideoCharacteristics:
    width: int
    height: int
    fps: float
    duration: float
    frame_count: int
    bitrate_kbps: float
    file_size_bytes: int
    avg_spatial_information: float    # Average SI across sampled frames
    avg_laplacian_variance: float     # Average sharpness
    avg_shannon_entropy: float        # Average luminance entropy
    temporal_information_mean: float  # ITU-T P.910 TI mean (motion energy)
    temporal_information_max: float   # ITU-T P.910 TI peak (fast action/scene cuts)
    motion_vector_energy: float       # Dense optical flow magnitude mean
    scene_cut_count: int              # Detected scene changes
    color_richness: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def feature_vector(self) -> np.ndarray:
        return np.array([
            self.width * self.height / 1e6,
            self.fps,
            self.avg_spatial_information,
            self.avg_laplacian_variance,
            self.avg_shannon_entropy,
            self.temporal_information_mean,
            self.temporal_information_max,
            self.motion_vector_energy,
            float(self.scene_cut_count),
            self.color_richness
        ], dtype=np.float32)


class MediaAnalyzer:
    """Intelligent content-aware media analyzer."""

    @staticmethod
    def _compute_entropy(gray: np.ndarray) -> float:
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        hist = hist / (hist.sum() + 1e-10)
        non_zero = hist[hist > 0]
        return float(-np.sum(non_zero * np.log2(non_zero)))

    @staticmethod
    def _compute_high_freq_dct_ratio(gray: np.ndarray) -> float:
        # Resize to fixed dimension for consistent DCT analysis
        target_size = (256, 256)
        resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA).astype(np.float32)
        dct = cv2.dct(resized)
        abs_dct = np.abs(dct)
        total_energy = np.sum(abs_dct) + 1e-10

        # Mask lower frequency triangular region
        h, w = target_size
        y, x = np.ogrid[:h, :w]
        low_freq_mask = (x + y) < (h // 2)
        high_freq_energy = np.sum(abs_dct[~low_freq_mask])
        return float(high_freq_energy / total_energy)

    @staticmethod
    def _estimate_noise(gray: np.ndarray) -> float:
        # Donoho and Johnstone median absolute deviation (MAD) noise estimation
        h, w = gray.shape
        if h < 8 or w < 8:
            return 0.0
        # High pass Laplacian kernel
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sigma = np.median(np.abs(lap)) / 0.6745
        return float(sigma)

    @classmethod
    def analyze_image(cls, image_path: Path | str) -> ImageCharacteristics:
        """Extract spatial features from an image file."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        file_size = path.stat().st_size
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            # Fallback to Pillow
            pil_img = Image.open(path).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        h, w, c = img_bgr.shape
        megapixels = (h * w) / 1e6
        aspect_ratio = w / max(h, 1)

        # Convert to YUV / Gray
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

        # Spatial Information (ITU-T P.910): std dev of Sobel filtered luminance
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.hypot(sobel_x, sobel_y)
        spatial_info = float(np.std(sobel_mag))

        # Laplacian variance (texture & sharpness)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Shannon entropy
        entropy = cls._compute_entropy(gray)

        # Color richness: standard deviation of Cr and Cb chroma
        cr_std = np.std(ycrcb[:, :, 1])
        cb_std = np.std(ycrcb[:, :, 2])
        color_richness = float((cr_std + cb_std) / 2.0)

        # High-frequency DCT ratio
        dct_ratio = cls._compute_high_freq_dct_ratio(gray)

        # Noise estimate
        noise = cls._estimate_noise(gray)

        return ImageCharacteristics(
            width=w,
            height=h,
            megapixels=round(megapixels, 4),
            aspect_ratio=round(aspect_ratio, 3),
            channels=c,
            file_size_bytes=file_size,
            spatial_information=round(spatial_info, 2),
            laplacian_variance=round(lap_var, 2),
            shannon_entropy=round(entropy, 3),
            color_richness=round(color_richness, 2),
            high_freq_dct_ratio=round(dct_ratio, 4),
            noise_estimate=round(noise, 3)
        )

    @classmethod
    def analyze_video(cls, video_path: Path | str, sample_frames: int = 15) -> VideoCharacteristics:
        """Extract spatial and temporal complexity metrics from a video file."""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")

        file_size = path.stat().st_size
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0.0
        bitrate_kbps = (file_size * 8) / (duration * 1000) if duration > 0 else 0.0

        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Unable to read frame count for: {path}")

        # Choose uniform frame sampling indices
        step = max(1, total_frames // sample_frames)
        sample_indices = list(range(0, total_frames, step))[:sample_frames]

        si_list = []
        lap_list = []
        entropy_list = []
        ti_list = []
        motion_energies = []
        chroma_stds = []
        scene_cuts = 0

        prev_gray: Optional[np.ndarray] = None
        prev_small_gray: Optional[np.ndarray] = None

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

            # Spatial Information (SI)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.hypot(sobel_x, sobel_y)
            si_list.append(float(np.std(mag)))

            # Laplacian variance
            lap_list.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

            # Shannon entropy
            entropy_list.append(cls._compute_entropy(gray))

            # Chroma std
            chroma_stds.append(float((np.std(ycrcb[:, :, 1]) + np.std(ycrcb[:, :, 2])) / 2.0))

            # Temporal Information (TI) and Optical Flow
            if prev_gray is not None and prev_gray.shape == gray.shape:
                # Frame difference standard deviation (ITU-T P.910)
                diff = gray.astype(np.float64) - prev_gray.astype(np.float64)
                ti_val = float(np.std(diff))
                ti_list.append(ti_val)

                # Scene cut detection (large jump in mean absolute difference)
                mad = np.mean(np.abs(diff))
                if mad > 45.0:
                    scene_cuts += 1

                # Fast Optical Flow on downscaled frame
                small_w = min(320, width)
                small_h = int(height * (small_w / max(width, 1)))
                curr_small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

                if prev_small_gray is not None and prev_small_gray.shape == curr_small.shape:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_small_gray, curr_small, None,
                        pyr_scale=0.5, levels=2, winsize=11,
                        iterations=2, poly_n=5, poly_sigma=1.1, flags=0
                    )
                    mag_flow, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    motion_energies.append(float(np.mean(mag_flow)))

                prev_small_gray = curr_small
            else:
                small_w = min(320, width)
                small_h = int(height * (small_w / max(width, 1)))
                prev_small_gray = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

            prev_gray = gray

        cap.release()

        avg_si = float(np.mean(si_list)) if si_list else 0.0
        avg_lap = float(np.mean(lap_list)) if lap_list else 0.0
        avg_entropy = float(np.mean(entropy_list)) if entropy_list else 0.0
        ti_mean = float(np.mean(ti_list)) if ti_list else 0.0
        ti_max = float(np.max(ti_list)) if ti_list else 0.0
        motion_energy = float(np.mean(motion_energies)) if motion_energies else 0.0
        color_richness = float(np.mean(chroma_stds)) if chroma_stds else 0.0

        return VideoCharacteristics(
            width=width,
            height=height,
            fps=round(fps, 2),
            duration=round(duration, 2),
            frame_count=total_frames,
            bitrate_kbps=round(bitrate_kbps, 2),
            file_size_bytes=file_size,
            avg_spatial_information=round(avg_si, 2),
            avg_laplacian_variance=round(avg_lap, 2),
            avg_shannon_entropy=round(avg_entropy, 3),
            temporal_information_mean=round(ti_mean, 2),
            temporal_information_max=round(ti_max, 2),
            motion_vector_energy=round(motion_energy, 3),
            scene_cut_count=scene_cuts,
            color_richness=round(color_richness, 2)
        )
