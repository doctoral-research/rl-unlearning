#!/bin/bash
set -e

# RL-Unlearning Docker Run Script
# Usage:
#   ./docker/run.sh                    # Run CPU image
#   ./docker/run.sh --gpu              # Run GPU image
#   ./docker/run.sh -- python src/train.py   # Custom command

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE_NAME="rl-unlearning"
TAG="latest"
GPU=false
CMD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU=true; shift ;;
        -t|--tag) TAG="$2"; shift 2 ;;
        --) shift; CMD="$*"; break ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

DOCKER_ARGS=(
    --rm -it
    -v "$PROJECT_DIR/src:/app/src"
    -v "$PROJECT_DIR/configs:/app/configs"
    -v "$PROJECT_DIR/experiments:/app/experiments"
)

if [ -f "$PROJECT_DIR/.env" ]; then
    DOCKER_ARGS+=(--env-file "$PROJECT_DIR/.env")
fi

if [ "$GPU" = true ]; then
    FULL_TAG="${TAG}-gpu"
    DOCKER_ARGS+=(--gpus all)
else
    FULL_TAG="${TAG}"
fi

IMAGE="${IMAGE_NAME}:${FULL_TAG}"

if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "Image $IMAGE not found. Build it first with: ./docker/build.sh $([ "$GPU" = true ] && echo '--gpu')"
    exit 1
fi

if [ -n "$CMD" ]; then
    docker run "${DOCKER_ARGS[@]}" "$IMAGE" $CMD
else
    docker run "${DOCKER_ARGS[@]}" "$IMAGE" /bin/bash
fi
