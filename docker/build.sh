#!/bin/bash
set -e

# RL-Unlearning Docker Build Script
# Usage:
#   ./docker/build.sh              # Build CPU image (~2GB)
#   ./docker/build.sh --gpu        # Build GPU image (~8GB)
#   ./docker/build.sh --gpu -t my-tag

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE_NAME="rl-unlearning"
TAG="latest"
GPU=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU=true; shift ;;
        -t|--tag) TAG="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ "$GPU" = true ]; then
    echo "Building GPU image: ${IMAGE_NAME}:${TAG}-gpu"
    docker build -f "$SCRIPT_DIR/Dockerfile.GPU" -t "${IMAGE_NAME}:${TAG}-gpu" "$PROJECT_DIR"
else
    echo "Building CPU image: ${IMAGE_NAME}:${TAG}"
    docker build -f "$SCRIPT_DIR/Dockerfile" -t "${IMAGE_NAME}:${TAG}" "$PROJECT_DIR"
fi

echo "Build complete!"
