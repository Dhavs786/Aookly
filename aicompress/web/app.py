"""
Web Dashboard Module: FastAPI Interactive Application.
Serves modern web UI for real-time AI compression, side-by-side comparison,
metric visualization, and benchmark inspection.
"""

from pathlib import Path
import shutil
import time
import json
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from aicompress.core.optimizer import AdaptiveOptimizer
from aicompress.core.analyzer import MediaAnalyzer

app = FastAPI(title="AI-Based Image & Video Compression Studio", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = BASE_DIR / "samples" / "web_uploads"
OUTPUT_DIR = BASE_DIR / "samples" / "web_outputs"
STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/samples_out", StaticFiles(directory=str(BASE_DIR / "samples" / "output")), name="samples_out")
app.mount("/samples_in", StaticFiles(directory=str(BASE_DIR / "samples" / "input")), name="samples_in")

optimizer = AdaptiveOptimizer()


@app.get("/", response_class=HTMLResponse)
async def index():
    html_file = TEMPLATE_DIR / "index.html"
    if not html_file.exists():
        return HTMLResponse("<h1>Web Dashboard Template Not Found</h1>")
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))


@app.get("/api/benchmarks")
async def get_benchmarks():
    bench_file = BASE_DIR / "benchmark_results.json"
    if bench_file.exists():
        try:
            with open(bench_file, "r", encoding="utf-8") as f:
                return JSONResponse(content=json.load(f))
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)
    return JSONResponse(content=[])


@app.post("/api/compress")
async def compress_media(
    file: UploadFile = File(...),
    target_reduction: float = Form(80.0),
    min_ssim: float = Form(0.90)
):
    timestamp = int(time.time() * 1000)
    safe_name = f"{timestamp}_{file.filename}"
    input_path = UPLOAD_DIR / safe_name

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    suffix = input_path.suffix.lower()
    is_image = suffix in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
    is_video = suffix in [".mp4", ".mov", ".mkv", ".avi", ".webm"]

    if not is_image and not is_video:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PNG, JPG, WEBP, or MP4/MOV.")

    try:
        if is_image:
            out_path = OUTPUT_DIR / f"{input_path.stem}_ai.webp"
            report = optimizer.optimize_image(
                input_path=input_path,
                output_path=out_path,
                target_reduction_pct=target_reduction,
                min_ssim=min_ssim,
                run_baseline_comparison=True
            )
            download_url = f"/outputs/{Path(report.output_file).name}"
            baseline_url = f"/outputs/{Path(report.output_file).stem}_baseline.jpg"
        else:
            out_path = OUTPUT_DIR / f"{input_path.stem}_ai.mp4"
            report = optimizer.optimize_video(
                input_path=input_path,
                output_path=out_path,
                target_reduction_pct=target_reduction,
                min_ssim=min_ssim,
                max_iterations=1,
                run_baseline_comparison=True
            )
            download_url = f"/outputs/{Path(report.output_file).name}"
            baseline_url = f"/outputs/{Path(report.output_file).stem}_baseline.mp4"

        res = report.to_dict()
        res["download_url"] = download_url
        res["baseline_url"] = baseline_url
        res["original_url"] = f"/outputs/{safe_name}"

        # Copy original to output dir for web comparison viewing
        web_orig = OUTPUT_DIR / safe_name
        shutil.copyfile(input_path, web_orig)

        return JSONResponse(content=res)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
