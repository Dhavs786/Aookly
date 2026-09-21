"""
Packaging Script: Creates complete project ZIP archive for assignment submission.
Generates ai_compression_system.zip containing all source code, models,
sample inputs/outputs, benchmarks, demo video, and documentation.
"""

from pathlib import Path
import zipfile
import os

BASE_DIR = Path(__file__).resolve().parent.parent
ZIP_PATH = BASE_DIR / "ai_compression_system.zip"

EXCLUDE_DIRS = {"__pycache__", ".git", ".pytest_cache", "web_uploads", "web_outputs"}
EXCLUDE_EXTS = {".pyc", ".pyo", ".tmp"}


def create_submission_zip():
    print(f"[*] Packaging submission ZIP at: {ZIP_PATH.name}...")
    file_count = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(BASE_DIR):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in files:
                p = Path(root) / file
                if p.suffix in EXCLUDE_EXTS or p.name == ZIP_PATH.name:
                    continue

                rel_path = p.relative_to(BASE_DIR)
                zf.write(p, arcname=rel_path)
                file_count += 1
                total_uncompressed_bytes += p.stat().st_size

    zip_size_mb = ZIP_PATH.stat().st_size / 1e6
    print(f"    Packaged {file_count} files.")
    print(f"    Uncompressed: {total_uncompressed_bytes / 1e6:.2f} MB")
    print(f"    ZIP Archive:  {zip_size_mb:.2f} MB")
    print(f"\n>>> Project ZIP ready for submission: {ZIP_PATH}")


if __name__ == "__main__":
    create_submission_zip()
