"""
Video Compressor Module.
Interfaces with FFmpeg to apply high-efficiency video coding (HEVC/H.265, H.264)
with dynamic CRF, adaptive quantization, custom GOP structure, and psycho-visual tuning.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import subprocess
import time
import shutil

from aicompress.core.ml_engine import VideoCompressionParams


class VideoCompressor:
    """Encodes video with traditional baseline or AI-predicted parameters."""

    FFMPEG_CMD = shutil.which("ffmpeg") or "ffmpeg"

    @classmethod
    def compress_traditional(
        cls,
        input_path: Path | str,
        output_path: Path | str,
        crf: int = 28
    ) -> Tuple[Path, float]:
        """Traditional fixed baseline using H.264 (CRF=28, medium preset, default GOP)."""
        in_p = Path(input_path)
        out_p = Path(output_path).with_suffix(".mp4")
        out_p.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            cls.FFMPEG_CMD,
            "-y",
            "-i", str(in_p),
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(out_p)
        ]

        start_time = time.perf_counter()
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        elapsed = time.perf_counter() - start_time

        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg baseline compression failed: {proc.stderr[-500:]}")

        return out_p, elapsed

    @classmethod
    def compress_ai(
        cls,
        input_path: Path | str,
        output_path: Path | str,
        params: VideoCompressionParams
    ) -> Tuple[Path, float]:
        """AI-assisted compression applying predicted codec, CRF, GOP, AQ, and tuning."""
        in_p = Path(input_path)
        out_p = Path(output_path).with_suffix(".mp4")
        out_p.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            cls.FFMPEG_CMD,
            "-y",
            "-i", str(in_p),
            "-c:v", params.codec,
            "-crf", str(params.crf),
            "-preset", params.preset,
            "-g", str(params.keyint),
            "-bf", str(params.b_frames),
            "-pix_fmt", "yuv420p"
        ]

        # Codec-specific psycho-visual & adaptive quantization parameters
        if params.codec == "libx265":
            x265_opts = [
                f"aq-mode={params.aq_mode}",
                f"aq-strength={params.aq_strength}",
                "repeat-headers=1"
            ]
            cmd.extend(["-x265-params", ":".join(x265_opts)])
            # Optional tune for libx265
            if params.tune in ["grain", "animation"]:
                cmd.extend(["-tune", params.tune])
        elif params.codec == "libx264":
            if params.tune:
                cmd.extend(["-tune", params.tune])

        # Intelligent scale filter
        if params.scale_factor < 0.999:
            # Maintain 2-divisible dimensions for YUV420p
            vf = f"scale=trunc(iw*{params.scale_factor}/2)*2:trunc(ih*{params.scale_factor}/2)*2"
            cmd.extend(["-vf", vf])

        # Audio and container web optimization
        cmd.extend([
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(out_p)
        ])

        start_time = time.perf_counter()
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        elapsed = time.perf_counter() - start_time

        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg AI compression failed: {proc.stderr[-500:]}")

        return out_p, elapsed
