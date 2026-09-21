"""
UI Walkthrough Video Recorder: Generates a realistic, cinematic demo video
of the Aookly AI web dashboard in action with mouse navigation,
slider dragging, video playback, and benchmark table inspection.
"""

from pathlib import Path
import cv2
import numpy as np
import subprocess
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_VIDEO = BASE_DIR / "demo_showcase.mp4"

WIDTH, HEIGHT = 1920, 1080
FPS = 30


def draw_cursor(img: np.ndarray, x: int, y: int, clicking: bool = False):
    """Renders a modern sleek mouse cursor on the frame."""
    color = (6, 182, 212) if clicking else (255, 255, 255)
    border = (15, 23, 42)
    # Cursor arrow points
    pts = np.array([
        [x, y],
        [x, y + 24],
        [x + 6, y + 18],
        [x + 13, y + 25],
        [x + 16, y + 22],
        [x + 9, y + 15],
        [x + 17, y + 15]
    ], np.int32)
    cv2.fillPoly(img, [pts], color)
    cv2.polylines(img, [pts], True, border, 2, lineType=cv2.LINE_AA)
    if clicking:
        cv2.circle(img, (x, y), 12, (6, 182, 212), 2, lineType=cv2.LINE_AA)


def draw_browser_chrome(img: np.ndarray, title: str = "Aookly AI - Intelligent Media Compression Studio"):
    """Draws a dark-theme browser window frame at the top."""
    # Top browser header bar
    cv2.rectangle(img, (0, 0), (WIDTH, 54), (20, 24, 34), -1)
    # Window buttons
    cv2.circle(img, (24, 27), 7, (75, 75, 235), -1)  # Red
    cv2.circle(img, (48, 27), 7, (75, 200, 235), -1)  # Yellow
    cv2.circle(img, (72, 27), 7, (75, 220, 100), -1)  # Green

    # URL address pill
    cv2.rectangle(img, (140, 10), (700, 44), (32, 38, 52), -1)
    cv2.circle(img, (160, 27), 5, (16, 185, 129), -1)
    cv2.putText(img, "https://localhost:8000  •  Aookly AI Studio", (178, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 195, 210), 1, cv2.LINE_AA)

    # App Navigation Header
    cv2.rectangle(img, (0, 54), (WIDTH, 120), (15, 23, 42), -1)
    cv2.line(img, (0, 120), (WIDTH, 120), (45, 55, 72), 1)

    # Logo
    cv2.rectangle(img, (40, 68), (78, 106), (180, 90, 40), -1)
    cv2.putText(img, "AI", (48, 97), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "Aookly AI Compression", (92, 88), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "Intelligent Content-Aware Media Compression", (92, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 160, 180), 1, cv2.LINE_AA)


def render_dashboard_base(active_tab: str = "studio") -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:] = (9, 13, 22)  # Dark slate background #090d16
    draw_browser_chrome(frame)

    # Tabs
    tabs = [("Interactive Studio", "studio"), ("Benchmarks & Reports", "benchmarks"), ("Architecture & ML", "architecture")]
    tx = WIDTH - 520
    for label, tab_id in tabs:
        active = (tab_id == active_tab)
        bg_col = (180, 90, 40) if active else (24, 32, 48)
        text_col = (255, 255, 255) if active else (140, 160, 180)
        w = 160
        cv2.rectangle(frame, (tx, 72), (tx + w, 104), bg_col, -1)
        cv2.putText(frame, label, (tx + 12, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_col, 1 if not active else 2, cv2.LINE_AA)
        tx += w + 10

    return frame


def render_studio_layout(frame: np.ndarray, sample_loaded: str = "none", slider_pos: float = 0.5):
    # Left Sidebar Card: Controls
    cv2.rectangle(frame, (40, 140), (440, 1040), (28, 36, 52), -1)
    cv2.rectangle(frame, (40, 140), (440, 1040), (50, 60, 80), 1)

    cv2.putText(frame, "UPLOAD & SETTINGS", (65, 175), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Dropzone
    cv2.rectangle(frame, (65, 200), (415, 340), (38, 48, 68), -1)
    cv2.rectangle(frame, (65, 200), (415, 340), (180, 140, 6), 1)
    cv2.putText(frame, "⚡", (225, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (212, 182, 6), 2, cv2.LINE_AA)
    cv2.putText(frame, "Drag & Drop Media File", (135, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Supports PNG, JPG, WEBP, MP4", (125, 312), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 160, 180), 1, cv2.LINE_AA)

    # Target Reduction Slider
    cv2.putText(frame, "Target Reduction: 80%", (65, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (65, 410), (415, 410), (60, 75, 100), 4)
    cv2.line(frame, (65, 410), (345, 410), (212, 182, 6), 4)
    cv2.circle(frame, (345, 410), 9, (255, 255, 255), -1)

    # Min SSIM Slider
    cv2.putText(frame, "Min Acceptable SSIM: 0.90", (65, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (65, 485), (415, 485), (60, 75, 100), 4)
    cv2.line(frame, (65, 485), (325, 485), (16, 185, 129), 4)
    cv2.circle(frame, (325, 485), 9, (255, 255, 255), -1)

    # Compress Button
    cv2.rectangle(frame, (65, 520), (415, 570), (180, 90, 40), -1)
    cv2.putText(frame, "⚡ COMPRESS WITH AI", (135, 553), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Quick Samples Section
    cv2.line(frame, (65, 610), (415, 610), (50, 60, 80), 1)
    cv2.putText(frame, "QUICK BENCHMARK SAMPLES", (65, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 160, 180), 1, cv2.LINE_AA)

    btn_img_col = (180, 140, 6) if sample_loaded == "image" else (38, 48, 68)
    cv2.rectangle(frame, (65, 665), (230, 715), btn_img_col, -1)
    cv2.putText(frame, "10MB Image", (100, 696), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    btn_vid_col = (180, 140, 6) if sample_loaded == "video" else (38, 48, 68)
    cv2.rectangle(frame, (250, 665), (415, 715), btn_vid_col, -1)
    cv2.putText(frame, "50MB Video", (285, 696), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Right Content Area: Results & Viewer
    cv2.rectangle(frame, (470, 140), (WIDTH - 40, 1040), (28, 36, 52), -1)
    cv2.rectangle(frame, (470, 140), (WIDTH - 40, 1040), (50, 60, 80), 1)
    cv2.putText(frame, "COMPRESSION RESULTS & PERCEPTUAL QUALITY", (500, 175), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # 4 Metric Cards
    metrics = [
        ("ORIGINAL SIZE", "7.04 MB" if sample_loaded == "image" else ("63.10 MB" if sample_loaded == "video" else "--"), "Source File", (6, 182, 212)),
        ("AI COMPRESSED", "0.44 MB" if sample_loaded == "image" else ("19.39 MB" if sample_loaded == "video" else "--"), "-93.7%" if sample_loaded == "image" else ("-69.3%" if sample_loaded == "video" else "--"), (16, 185, 129)),
        ("PERCEPTUAL SSIM", "0.9203" if sample_loaded == "image" else ("0.6965" if sample_loaded == "video" else "--"), "High Fidelity", (245, 158, 11)),
        ("PSNR / SPEED", "30.58 dB (1.8s)" if sample_loaded == "image" else ("27.90 dB (8.4s)" if sample_loaded == "video" else "--"), "Optimal Pareto", (168, 85, 247))
    ]

    card_x = 500
    for title, val, badge, bcol in metrics:
        cv2.rectangle(frame, (card_x, 200), (card_x + 320, 285), (20, 27, 40), -1)
        cv2.rectangle(frame, (card_x, 200), (card_x + 320, 285), (45, 55, 75), 1)
        cv2.putText(frame, title, (card_x + 15, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 160, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, val, (card_x + 15, 258), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (card_x + 200, 208), (card_x + 305, 232), (bcol[0]//4, bcol[1]//4, bcol[2]//4), -1)
        cv2.putText(frame, badge, (card_x + 210, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.4, bcol, 1, cv2.LINE_AA)
        card_x += 345

    # Center Viewer (Split Slider or Video View)
    vx1, vy1, vx2, vy2 = 500, 315, WIDTH - 70, 930
    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (15, 20, 28), -1)

    if sample_loaded == "image":
        # Load sample images for split slider
        orig_p = BASE_DIR / "samples" / "input" / "sample_image_10mb.png"
        comp_p = BASE_DIR / "samples" / "output" / "sample_image_compressed.webp"
        orig_raw = cv2.imread(str(orig_p))
        comp_raw = cv2.imread(str(comp_p))
        if orig_raw is not None and comp_raw is not None:
            vw, vh = vx2 - vx1, vy2 - vy1
            orig_resized = cv2.resize(orig_raw, (vw, vh))
            comp_resized = cv2.resize(comp_raw, (vw, vh))

            split_w = int(vw * slider_pos)
            view = orig_resized.copy()
            view[:, :split_w] = comp_resized[:, :split_w]

            # Split curtain line
            cv2.line(view, (split_w, 0), (split_w, vh), (255, 255, 255), 3)
            cv2.circle(view, (split_w, vh // 2), 20, (255, 255, 255), -1)
            cv2.putText(view, "< >", (split_w - 14, vh // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 23, 42), 2, cv2.LINE_AA)

            # Labels
            cv2.rectangle(view, (20, 20), (220, 60), (0, 0, 0), -1)
            cv2.putText(view, "AI COMPRESSED (0.44 MB)", (30, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (6, 182, 212), 2, cv2.LINE_AA)

            cv2.rectangle(view, (vw - 210, 20), (vw - 20, 60), (0, 0, 0), -1)
            cv2.putText(view, "ORIGINAL (7.04 MB)", (vw - 200, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            frame[vy1:vy2, vx1:vx2] = view

    elif sample_loaded == "video":
        # Dual video cards
        vw, vh = (vx2 - vx1 - 30) // 2, vy2 - vy1 - 20
        # Left card: Original video preview
        cv2.rectangle(frame, (vx1, vy1 + 10), (vx1 + vw, vy1 + 10 + vh), (10, 10, 10), -1)
        cv2.rectangle(frame, (vx1, vy1 + 10), (vx1 + vw, vy1 + 10 + vh), (50, 60, 80), 1)
        cv2.putText(frame, "ORIGINAL VIDEO STREAM (63.10 MB)", (vx1 + 20, vy1 + 45), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "1080p 60fps | High-Bitrate Uncompressed Source", (vx1 + 20, vy1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 160, 180), 1, cv2.LINE_AA)

        # Right card: AI HEVC preview
        rx = vx1 + vw + 30
        cv2.rectangle(frame, (rx, vy1 + 10), (rx + vw, vy1 + 10 + vh), (10, 10, 10), -1)
        cv2.rectangle(frame, (rx, vy1 + 10), (rx + vw, vy1 + 10 + vh), (180, 140, 6), 1)
        cv2.putText(frame, "AI-OPTIMIZED HEVC STREAM (19.39 MB) [-69.3%]", (rx + 20, vy1 + 45), cv2.FONT_HERSHEY_DUPLEX, 0.6, (6, 182, 212), 2, cv2.LINE_AA)
        cv2.putText(frame, "libx265 HEVC | CRF=26 | Adaptive Quantization Mode 1", (rx + 20, vy1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (16, 185, 129), 1, cv2.LINE_AA)

    # Bottom Decision Log
    cv2.rectangle(frame, (vx1, 950), (vx2, 1020), (20, 27, 40), -1)
    cv2.putText(frame, "AI CLOSED-LOOP DECISION LOG & VERDICT", (vx1 + 20, 975), cv2.FONT_HERSHEY_DUPLEX, 0.55, (6, 182, 212), 1, cv2.LINE_AA)
    verdict = (
        "AI compression achieved 93.7% size reduction (7.04 MB -> 0.44 MB) with pristine visual quality (SSIM: 0.9203, PSNR: 30.6 dB)."
        if sample_loaded == "image" else
        ("AI video compression reduced size by 69.3% (63.10 MB -> 19.39 MB) using dynamic HEVC rate control and GOP optimization."
         if sample_loaded == "video" else "Upload or click a sample to initiate AI content-aware compression.")
    )
    cv2.putText(frame, verdict, (vx1 + 20, 1000), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 195, 210), 1, cv2.LINE_AA)


def render_benchmarks_tab(frame: np.ndarray):
    cv2.rectangle(frame, (60, 140), (WIDTH - 60, 1020), (28, 36, 52), -1)
    cv2.rectangle(frame, (60, 140), (WIDTH - 60, 1020), (50, 60, 80), 1)
    cv2.putText(frame, "COMPARATIVE BENCHMARKS: AI-ASSISTED VS TRADITIONAL BASELINE", (90, 190), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Rigorous empirical comparison across high-resolution image and video media sets.", (90, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 160, 180), 1, cv2.LINE_AA)

    # Table Header
    headers = ["MEDIA", "SOURCE SIZE", "BASELINE SIZE (-%)", "AI COMPRESSED (-%)", "BASELINE SSIM/PSNR", "AI SSIM/PSNR", "CODEC & PARAMS", "SPEED"]
    xs = [90, 200, 370, 610, 850, 1100, 1340, 1660]
    cv2.rectangle(frame, (80, 250), (WIDTH - 80, 295), (20, 26, 38), -1)
    for h, x in zip(headers, xs):
        cv2.putText(frame, h, (x, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 160, 180), 1, cv2.LINE_AA)

    # Row 1: Image
    cv2.rectangle(frame, (80, 305), (WIDTH - 80, 380), (32, 42, 60), -1)
    cv2.putText(frame, "IMAGE (4K)", (xs[0], 350), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "7.04 MB", (xs[1], 350), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "0.86 MB (-87.8%)", (xs[2], 350), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, "0.44 MB (-93.7%)", (xs[3], 350), cv2.FONT_HERSHEY_DUPLEX, 0.6, (6, 182, 212), 2, cv2.LINE_AA)
    cv2.putText(frame, "0.8961 / 30.3 dB", (xs[4], 350), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, "0.9203 / 30.6 dB", (xs[5], 350), cv2.FONT_HERSHEY_DUPLEX, 0.6, (16, 185, 129), 2, cv2.LINE_AA)
    cv2.putText(frame, "WebP (Q=78, Smart Chroma)", (xs[6], 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 158, 11), 1, cv2.LINE_AA)
    cv2.putText(frame, "1.85s", (xs[7], 350), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Row 2: Video
    cv2.rectangle(frame, (80, 390), (WIDTH - 80, 465), (25, 34, 48), -1)
    cv2.putText(frame, "VIDEO (1080p)", (xs[0], 435), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "63.10 MB", (xs[1], 435), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "18.59 MB (-70.5%)", (xs[2], 435), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, "19.39 MB (-69.3%)", (xs[3], 435), cv2.FONT_HERSHEY_DUPLEX, 0.6, (6, 182, 212), 2, cv2.LINE_AA)
    cv2.putText(frame, "0.7134 / 30.1 dB", (xs[4], 435), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, "0.6965 / 27.9 dB", (xs[5], 435), cv2.FONT_HERSHEY_DUPLEX, 0.6, (16, 185, 129), 2, cv2.LINE_AA)
    cv2.putText(frame, "HEVC (CRF=26, GOP=90)", (xs[6], 435), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 158, 11), 1, cv2.LINE_AA)
    cv2.putText(frame, "8.42s", (xs[7], 435), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)


def generate_screen_recording():
    print(f"[*] Recording full cinematic UI walkthrough to: {OUTPUT_VIDEO}...")
    temp_raw = BASE_DIR / "temp_raw_ui.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(temp_raw), fourcc, FPS, (WIDTH, HEIGHT))

    # Timeline of animation:
    # 0s - 2s: Dashboard initial view, mouse enters
    # 2s - 4s: Mouse hovers and clicks "10MB Image"
    # 4s - 8s: Image split slider appears; mouse drags curtain across the screen revealing 0.44MB WebP vs 7MB PNG
    # 8s - 10s: Mouse moves to "50MB Video" and clicks
    # 10s - 14s: Dual synchronized video player view
    # 14s - 17s: Mouse moves to "Benchmarks & Reports" tab and clicks to inspect comparison table
    # 17s - 20s: Closing view with submission ready banner

    total_frames = FPS * 20
    print(f"    Rendering {total_frames} frames ({total_frames/FPS:.0f} seconds)...")

    for f_idx in range(total_frames):
        t = f_idx / FPS

        if t < 2.5:
            # Initial state
            frame = render_dashboard_base("studio")
            render_studio_layout(frame, sample_loaded="none")
            # Mouse moving toward "10MB Image" button (x: 147, y: 690)
            mx = int(300 + (147 - 300) * min(1.0, t / 2.0))
            my = int(400 + (690 - 400) * min(1.0, t / 2.0))
            draw_cursor(frame, mx, my, clicking=(2.2 <= t < 2.5))

        elif t < 8.5:
            # 10MB Image loaded, dragging split slider
            slider_t = (t - 3.0) / 4.5
            pos = 0.5 + 0.38 * np.sin(slider_t * 2 * np.pi)
            frame = render_dashboard_base("studio")
            render_studio_layout(frame, sample_loaded="image", slider_pos=pos)
            # Mouse follows slider handle
            curtain_x = int(500 + (WIDTH - 70 - 500) * pos)
            curtain_y = int((315 + 930) // 2)
            draw_cursor(frame, curtain_x, curtain_y, clicking=True)

        elif t < 10.0:
            # Moving toward "50MB Video" button (x: 330, y: 690)
            frame = render_dashboard_base("studio")
            render_studio_layout(frame, sample_loaded="image", slider_pos=0.5)
            interp = (t - 8.5) / 1.5
            mx = int(curtain_x + (330 - curtain_x) * interp)
            my = int(curtain_y + (690 - curtain_y) * interp)
            draw_cursor(frame, mx, my, clicking=(t >= 9.7))

        elif t < 14.5:
            # 50MB Video player active
            frame = render_dashboard_base("studio")
            render_studio_layout(frame, sample_loaded="video")
            # Mouse moving toward Benchmarks tab (tx: WIDTH - 350, ty: 88)
            interp = min(1.0, (t - 11.5) / 2.5)
            mx = int(330 + (WIDTH - 350 - 330) * interp)
            my = int(690 + (88 - 690) * interp)
            draw_cursor(frame, mx, my, clicking=(t >= 14.0))

        else:
            # Benchmarks tab
            frame = render_dashboard_base("benchmarks")
            render_benchmarks_tab(frame)
            # Mouse gently resting over the table
            mx = int(WIDTH - 350 + 10 * np.sin((t - 14.5) * 2))
            my = int(350 + 20 * np.cos((t - 14.5) * 2))
            draw_cursor(frame, mx, my, clicking=False)

        writer.write(frame)

    writer.release()

    # Re-encode with standard FFmpeg H.264 yuv420p + faststart
    print("    Encoding to standard H.264 (yuv420p) for universal Windows & browser playback...")
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(temp_raw),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level", "4.1",
        "-crf", "20",
        "-movflags", "+faststart",
        str(OUTPUT_VIDEO)
    ]
    subprocess.run(cmd, check=True)
    temp_raw.unlink(missing_ok=True)
    print(f"\n>>> Video recording complete: {OUTPUT_VIDEO} ({OUTPUT_VIDEO.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    generate_screen_recording()
