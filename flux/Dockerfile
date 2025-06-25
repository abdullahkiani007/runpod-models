# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH=/root/miniconda3/bin:$PATH

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN mkdir -p ~/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh \
    && bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 \
    && rm ~/miniconda3/miniconda.sh \
    && ~/miniconda3/bin/conda init bash

# Create app directory
WORKDIR /app

# Create checkpoints and output directories
RUN mkdir -p /app/checkpoints /app/output /runpod-volume/flux/checkpoints /runpod-volume/flux/temp

# Create Python 3.10 conda environment
SHELL ["/bin/bash", "-c"]
RUN source ~/miniconda3/bin/activate && \
    conda create -y -n flux python=3.10

# Copy scripts
COPY rp_handler.py /app/rp_handler.py
COPY startup.sh /app/startup.sh

# Make startup script executable
RUN chmod +x /app/startup.sh

# Set working directory
WORKDIR /app

# Run startup script
CMD ["/app/startup.sh"]