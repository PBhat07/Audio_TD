#!/bin/bash

# Audio_TD Setup Script for Ubuntu 22.04 + WSL2
# Run with: chmod +x setup.sh && ./setup.sh

set -e

echo "Setting up Audio_TD for Ubuntu 22.04 + WSL2..."

# Check if NVIDIA GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Install NVIDIA drivers on Windows first."
    exit 1
fi

echo "NVIDIA GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Fixed CUDA version for Docker
CUDA_VER="12.4.0"
echo "Using CUDA version $CUDA_VER for Docker builds"

# Update system packages
sudo apt update && sudo apt upgrade -y

# Check if Docker CLI is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker not found. Enable WSL integration in Docker Desktop."
    exit 1
fi
echo "Docker detected: $(docker --version)"

# Add user to docker group if needed
if ! groups "$USER" | grep -q '\bdocker\b'; then
    sudo usermod -aG docker "$USER"
    echo "User '$USER' added to 'docker' group. Close terminal and reopen to apply changes."
    exit 0
fi

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker for NVIDIA
sudo nvidia-ctk runtime configure --runtime=docker

# Test NVIDIA Docker integration
TEST_IMAGE="nvidia/cuda:${CUDA_VER}-base-ubuntu22.04"
if docker run --rm --gpus all $TEST_IMAGE nvidia-smi > /dev/null 2>&1; then
    echo "NVIDIA Docker integration working with CUDA $CUDA_VER."
else
    echo "Warning: NVIDIA Docker integration may not be active yet."
    echo "Run 'wsl --shutdown', reopen Ubuntu, and test with:"
    echo "docker run --rm --gpus all $TEST_IMAGE nvidia-smi"
fi

# Create project directories
mkdir -p input output models logs

# Create .env file with only Hugging Face token
if [ ! -f .env ]; then
    echo "HUGGING_FACE_TOKEN=" > .env
    echo "Created .env file. Add your Hugging Face token to HUGGING_FACE_TOKEN."
fi

# Build Docker image
docker compose build --build-arg CUDA_VER=$CUDA_VER

echo "Setup complete. To run:"
echo "1. Place your audio files in the 'input/' directory"
echo "2. Run the application: docker compose up"
