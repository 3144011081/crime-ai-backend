# Production Dockerfile for CrimeAI Backend (Hugging Face Spaces & Cloud Compatible)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system runtime dependencies for OpenCV, PyTorch, and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up user 1000 required by Hugging Face Spaces
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Install Python requirements (CPU-optimized for cloud instances)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code with proper ownership
COPY --chown=user:user . /app

# Ensure writable directories exist and permissions are set
RUN mkdir -p /app/outputs/uploads /app/outputs/evidence /app/outputs/reports /app/videos && \
    chown -R user:user /app

USER user

# Expose standard HF Spaces port
EXPOSE 7860

# Run monoserver
CMD ["sh", "-c", "python run.py --host 0.0.0.0 --port ${PORT:-7860}"]
