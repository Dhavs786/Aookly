from setuptools import setup, find_packages

setup(
    name="aicompress",
    version="1.0.0",
    description="AI-Based Content-Aware Image & Video Compression System",
    author="AI/ML Technical Evaluation",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.20.0",
        "python-multipart>=0.0.6",
    ],
    entry_points={
        "console_scripts": [
            "ai-compress=aicompress.cli:main",
        ],
    },
)
