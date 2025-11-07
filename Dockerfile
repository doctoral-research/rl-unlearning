FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    swig \
    libglu1-mesa-dev \
    libgl1-mesa-dev \
    libosmesa6-dev \
    xvfb \
    patchelf \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install project in editable mode
RUN pip install -e .

# Create directories
RUN mkdir -p /workspace/experiments /workspace/outputs

# Set environment variables
ENV PYTHONPATH=/workspace:$PYTHONPATH
ENV HYDRA_FULL_ERROR=1

# Default command
CMD ["python", "src/train.py"]
