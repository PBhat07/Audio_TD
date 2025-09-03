#!/bin/bash

# Audio_TD Setup Script for Ubuntu 22.04 with RTX 4050
# Run with: chmod +x setup.sh && ./setup.sh

set -e

echo "🚀 Setting up Audio_TD for RTX 4050 on Ubuntu 22.04..."

# Check if running on Ubuntu 22.04
if ! grep -q "22.04" /etc/os-release; then
    echo "⚠️  Warning: This script is optimized for Ubuntu 22.04"
fi

# Check if NVIDIA GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi

echo "✅ NVIDIA GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Update system packages
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo "ℹ️  Please log out and back in to use Docker without sudo"
else
    echo "✅ Docker already installed"
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    echo "🔧 Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
else
    echo "✅ Docker Compose already installed"
fi

# Install NVIDIA Container Toolkit
echo "🎮 Installing NVIDIA Container Toolkit..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker for NVIDIA
echo " Configuring Docker for NVIDIA..."
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Create project directories
echo " Creating project directories..."
mkdir -p input output models logs

# Create environment file optimized for RTX 4050
echo "📝 Creating optimized .env file..."
cat > .env << EOF
# RTX 4050 Optimized Settings
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
TOKENIZERS_PARALLELISM=false

# Model Settings (Memory Optimized)
WHISPER_MODEL=large-v2
DIARIZATION_MODEL=pyannote/speaker-diarization-3.1

# Processing Settings
MAX_SPEAKERS=6
MIN_SEGMENT_LENGTH=0.5
CONFIDENCE_THRESHOLD=0.7
MAX_CHUNK_LENGTH=30
BATCH_SIZE=1

# Output Settings
OUTPUT_FORMAT=json
INCLUDE_TIMESTAMPS=true
EOF

# Test NVIDIA Docker integration
echo "🧪 Testing NVIDIA Docker integration..."
if docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo " NVIDIA Docker integration working!"
else
    echo " NVIDIA Docker integration failed. Please check your setup."
    exit 1
fi

echo "
Setup complete! Your system is ready for Audio_TD.

Next steps:
1. Place your audio files in the 'input/' directory
2. Build the Docker image: docker-compose build
3. Run the application: docker-compose up

RTX 4050 is optimized for:
- Processing speed: ~1.5-2x realtime
- Memory usage: 4-5GB VRAM
- Recommended audio length: Up to 60 minutes per file

Need help? Check the README.md file for detailed instructions.
"