# Use NVIDIA CUDA base image for GPU acceleration (optional but recommended for performance)
# Ubuntu 22.04 LTS provides long-term stability
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    portaudio19-dev \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    git \
    wget \
    curl \
    sox \
    libsox-dev \
    libsox-fmt-all \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip to latest stable version
RUN pip install --upgrade pip==23.3.1

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for input/output
RUN mkdir -p /app/input /app/output /app/models

# Set permissions
RUN chmod +x /app/*.py 2>/dev/null || true

# Expose port for Gradio interface (if using)
EXPOSE 7860

# Set environment variables optimized for RTX 4050 (6GB VRAM)
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV TOKENIZERS_PARALLELISM=false
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_LAUNCH_BLOCKING=0

# Default command
CMD ["python3", "main.py"]