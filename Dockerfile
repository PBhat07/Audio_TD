# This allows the version to be passed from the host environment
ARG CUDA_VER=12.4.0
FROM nvidia/cuda:${CUDA_VER}-devel-ubuntu22.04

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
    git-lfs \
    libavcodec-dev \
    libavformat-dev \
    && rm -rf /var/lib/apt/lists/*

# Install cuDNN for CUDA 12
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12 && \
    rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip to fixed stable version for reproducibility
RUN pip install --upgrade pip==23.3.1

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

RUN pip cache purge
# Install Python dependencies (with PyTorch from NVIDIA index)
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121


# Copy application code
COPY . .

# Create directories for input/output and models
RUN mkdir -p /app/input /app/output /app/models

# Set permissions
RUN chmod +x /app/*.py 2>/dev/null || true

# Expose port for Gradio interface (if using)
EXPOSE 7860

# Set environment variables optimized for RTX 4050 (6GB VRAM)
ENV CUDA_VISIBLE_DEVICES=0
ENV TOKENIZERS_PARALLELISM=false
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_LAUNCH_BLOCKING=0

# Default command
CMD ["python3", "main.py"]