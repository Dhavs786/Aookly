"""
Asset Generation Script: Creates realistic high-resolution sample inputs.
Generates:
- samples/input/sample_image_10mb.png (~10 MB) with complex procedural texture & detail.
- samples/input/sample_video_50mb.mp4 (~50 MB) with multi-frequency motion and scene transitions.
"""

from pathlib import Path
import time
import cv2
import numpy as np
import subprocess
import shutil

INPUT_DIR = Path(__file__).resolve().parent.parent / "samples" / "input"
INPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_high_res_image(output_path: Path):
    """Generates ~10MB high-detail PNG image with rich multi-frequency textures and structure."""
    print(f"[*] Generating ~10 MB high-resolution test image at: {output_path.name}...")
    width, height = 4000, 2600  # 10.4 Megapixels

    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    # Multi-frequency organic textures (coherent wavelets)
    wave1 = np.sin(xx * 24.0 + np.cos(yy * 18.0))
    wave2 = np.cos(yy * 32.0 + np.sin(xx * 20.0))
    wave3 = np.sin(np.sqrt((xx - 0.5)**2 + (yy - 0.5)**2) * 60.0)
    wave4 = np.sin(xx * 120.0) * np.cos(yy * 120.0) * 0.15

    r = np.clip((xx * 160 + (wave1 * 35) + (wave3 * 40) + (wave4 * 20) + 40), 0, 255).astype(np.uint8)
    g = np.clip((yy * 150 + (wave2 * 45) + (wave1 * 25) + 30), 0, 255).astype(np.uint8)
    b = np.clip(((1.0 - xx) * 180 + (wave3 * 35) + (wave2 * 30) + 50), 0, 255).astype(np.uint8)

    img = np.dstack([b, g, r])

    # Structured geometrical overlays
    step = 50
    for gx in range(0, width, step * 2):
        cv2.line(img, (gx, 0), (gx, height), (90, 90, 100), 1)
    for gy in range(0, height, step * 2):
        cv2.line(img, (0, gy), (width, gy), (90, 90, 100), 1)

    center = (width // 2, height // 2)
    for rad in range(100, 950, 40):
        cv2.circle(img, center, rad, (255, 220, 50), 2, lineType=cv2.LINE_AA)

    # High-contrast sharp text overlays
    cv2.putText(img, "AI-BASED IMAGE COMPRESSION TEST BENCHMARK", (120, 180),
                cv2.FONT_HERSHEY_DUPLEX, 2.2, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img, "Resolution: 3200x2200 (7.0 MP) | Perceptual Quality Benchmark", (120, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (180, 255, 200), 2, cv2.LINE_AA)
    cv2.putText(img, "Target: 10 MB -> ~2 MB (80% Size Reduction with High Perceptual Fidelity)", (120, 330),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (220, 230, 255), 2, cv2.LINE_AA)

    # Save as uncompressed/lightly-compressed PNG to yield ~10MB
    cv2.imwrite(str(output_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    size_mb = output_path.stat().st_size / 1e6
    print(f"    Generated image: {size_mb:.2f} MB ({output_path.stat().st_size:,} bytes)")


def generate_high_res_video(output_path: Path):
    """
    Generates a ~50MB video with rich continuous temporal motion,
    dynamic rotating particles, moving text, and natural gradient sweeps.
    """
    print(f"[*] Generating ~50 MB test video at: {output_path.name}...")
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"

    # Use ffmpeg lavfi to generate 15 seconds of dynamic animated cellnoise and plasma
    # with high entropy and motion vectors at 1920x1080 30fps with 28 Mbps bitrate
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "lavfi",
        "-i", "cellauto=s=1920x1080:r=30:rule=30:p=100",
        "-f", "lavfi",
        "-i", "mptestsrc=rate=30:size=1920x1080",
        "-f", "lavfi",
        "-i", "sine=f=440:b=4:d=14",
        "-filter_complex",
        "[0:v]trim=duration=14,format=yuv420p[c];"
        "[1:v]trim=duration=14,format=yuv420p[m];"
        "[c][m]blend=all_mode='screen':all_opacity=0.6[vblend];"
        "[vblend]drawtext=text='AI-BASED VIDEO COMPRESSION BENCHMARK':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=120,"
        "drawtext=text='Evaluation Target\\: 50 MB -> ~15 MB (~70% Reduction)':fontcolor=yellow:fontsize=36:x=(w-text_w)/2:y=200,"
        "drawtext=text='Dynamic Temporal Motion & Texture Stress Test':fontcolor=cyan:fontsize=30:x=(w-text_w)/2:y=280[outv]",
        "-map", "[outv]",
        "-map", "2:a",
        "-c:v", "libx264",
        "-b:v", "28M",
        "-minrate", "25M",
        "-maxrate", "32M",
        "-bufsize", "40M",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        str(output_path)
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if proc.returncode != 0 or output_path.stat().st_size < 10_000_000:
        # High bitrate fallback generating complex particle frames
        print("    Synthesizing animated frames via OpenCV writer...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw, vh = 1920, 1080
        fps = 30
        num_frames = 300  # 10 seconds
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (vw, vh))

        rng = np.random.default_rng(42)
        particles = rng.uniform(0, [vw, vh], size=(300, 2))
        velocities = rng.uniform(-10, 10, size=(300, 2))

        for fi in range(num_frames):
            frame = np.zeros((vh, vw, 3), dtype=np.uint8)
            # Animated gradient background
            shift = int(fi * 4) % vw
            gx = ((np.arange(vw) + shift) % 256).astype(np.uint8)
            frame[:, :, 0] = gx[None, :]
            frame[:, :, 1] = 120
            frame[:, :, 2] = (255 - gx)[None, :]

            # Update and draw particles
            particles += velocities
            particles[:, 0] = particles[:, 0] % vw
            particles[:, 1] = particles[:, 1] % vh

            for px, py in particles:
                cv2.circle(frame, (int(px), int(py)), 8, (255, 255, 255), -1)

            # High-frequency texture noise
            noise = rng.integers(-30, 31, (vh, vw, 3), dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            cv2.putText(frame, f"AI VIDEO COMPRESSION TEST - Frame {fi:04d}", (100, 120),
                        cv2.FONT_HERSHEY_DUPLEX, 1.8, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, "Target: 50 MB -> ~15 MB", (100, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 2, cv2.LINE_AA)

            writer.write(frame)
        writer.release()

    size_mb = output_path.stat().st_size / 1e6
    print(f"    Generated video: {size_mb:.2f} MB ({output_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    img_target = INPUT_DIR / "sample_image_10mb.png"
    vid_target = INPUT_DIR / "sample_video_50mb.mp4"

    generate_high_res_image(img_target)
    generate_high_res_video(vid_target)
    print("\n>>> Sample assets created successfully in samples/input/")
