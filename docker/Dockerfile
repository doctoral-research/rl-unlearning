# RL-Unlearning CPU Dockerfile (~2GB)
#
# Build:  docker build -f docker/Dockerfile -t rl-unlearning .
# Run:    docker run rl-unlearning

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    swig \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Install CPU-only PyTorch + dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Runtime (minimal)
# =============================================================================
FROM python:3.12-slim

WORKDIR /app

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    swig \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code and configs
COPY src/ ./src/
COPY configs/ ./configs/
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1
ENV HYDRA_FULL_ERROR=1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-c", "import torch; print(f'RL-Unlearning ready. PyTorch {torch.__version__}')"]
