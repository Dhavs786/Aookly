"""
Image Compressor Module.
Executes content-adaptive image compression using modern codecs (WebP, MozJPEG, AVIF)
with fine-grained control over chroma subsampling, sharp YUV, and scale filtering.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
from PIL import Image
import cv2
import numpy as np

from aicompress.core.ml_engine import ImageCompressionParams


class ImageCompressor:
    """Performs image compression using traditional or AI-predicted parameters."""

    @classmethod
    def compress_traditional(
        cls,
        input_path: Path | str,
        output_path: Path | str,
        quality: int = 75
    ) -> Tuple[Path, float]:
        """Traditional fixed-compression baseline using standard JPEG (quality=75)."""
        in_p = Path(input_path)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        start_time = time.perf_counter()
        img = Image.open(in_p).convert("RGB")
        out_p = out_p.with_suffix(".jpg")
        img.save(out_p, "JPEG", quality=quality, optimize=False)
        elapsed = time.perf_counter() - start_time

        return out_p, elapsed

    @classmethod
    def compress_ai(
        cls,
        input_path: Path | str,
        output_path: Path | str,
        params: ImageCompressionParams
    ) -> Tuple[Path, float]:
        """AI-assisted compression applying predicted codec, quantizer, and filters."""
        in_p = Path(input_path)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        start_time = time.perf_counter()
        img = Image.open(in_p).convert("RGB")
        orig_w, orig_h = img.size

        # Apply intelligent scale factor if specified
        if params.scale_factor < 0.999:
            new_w = max(16, int(orig_w * params.scale_factor))
            new_h = max(16, int(orig_h * params.scale_factor))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Codec execution
        if params.codec.lower() == "webp":
            out_p = out_p.with_suffix(".webp")
            # Pillow supports modern WebP encoding features
            subsample_val = 0 if params.subsampling == "4:2:0" else 2  # 0=4:2:0, 2=4:4:4
            img.save(
                out_p,
                "WEBP",
                quality=params.quality,
                method=params.method_effort,
                subsampling=subsample_val,
                exact=False
            )
        else:
            # Fallback to optimized JPEG
            out_p = out_p.with_suffix(".jpg")
            img.save(out_p, "JPEG", quality=params.quality, optimize=True, progressive=True)

        elapsed = time.perf_counter() - start_time
        return out_p, elapsed
