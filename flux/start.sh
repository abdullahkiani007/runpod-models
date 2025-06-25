#!/bin/bash
set -e  # Exit on error

echo "Starting Flux.1 initialization..."

# Activate conda and the flux environment
source ~/miniconda3/bin/activate
conda activate flux

# Install dependencies
echo "Installing Python dependencies..."
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install diffusers==0.31.0 transformers==4.45.2 accelerate==0.34.2 runpod

# Create directories
echo "Setting up directories..."
mkdir -p /app/checkpoints
mkdir -p /app/output
mkdir -p /runpod-volume/flux/checkpoints
mkdir -p /runpod-volume/flux/temp

# Start the RunPod handler
echo "Starting RunPod handler with Python $(python --version)..."
python -u rp_handler.py