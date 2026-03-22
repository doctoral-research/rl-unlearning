#!/usr/bin/env bash
# =============================================================================
# Full experiment pipeline: train + unlearn + evaluate for all envs & methods.
#
# Usage (inside Docker container):
#   cd /workspace && bash scripts/run_all.sh
#
# Usage (from host with docker-compose):
#   docker compose run --rm rl-unlearning bash /workspace/scripts/run_all.sh
#
# This script:
#   1. Cleans previous experiment data
#   2. Trains a PPO agent on each environment
#   3. Runs each unlearning method on each trained agent
#   4. Runs evaluation (comparison + plots + videos) for each environment
# =============================================================================
set -euo pipefail

# Resolve paths relative to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Load environment variables (WANDB_API_KEY, etc.)
if [ -f "$REPO_ROOT/.env" ]; then
    set -a; source "$REPO_ROOT/.env"; set +a
    echo "Loaded .env"
fi

SRC_DIR="$REPO_ROOT/src"
ENVS=("cartpole" "lunarlander" "acrobot")
METHODS=("trajectory_selective" "strategy_inversion" "retain_protection")

echo "============================================================"
echo "  RL-Unlearning Full Experiment Pipeline"
echo "============================================================"
echo "Environments: ${ENVS[*]}"
echo "Methods:      ${METHODS[*]}"
echo "Repo root:    $REPO_ROOT"
echo ""

# ------------------------------------------------------------------
# Step 0: Clean previous data
# ------------------------------------------------------------------
echo "[0/3] Cleaning previous experiment data..."
rm -rf experiments/
rm -rf outputs/   # Old Hydra output dir (if any)
rm -rf wandb/     # wandb local run storage
echo "  Cleaned."
echo ""

# ------------------------------------------------------------------
# Step 1: Train on each environment
# ------------------------------------------------------------------
echo "[1/3] Training PPO agents..."
for env in "${ENVS[@]}"; do
    echo ""
    echo "  >>> Training on $env ..."
    python "$SRC_DIR/train.py" \
        env="$env" \
        logging=wandb \
        wandb.group="train" \
        wandb.tags="[train,$env]" \
        || { echo "  FAILED: train $env"; exit 1; }
    echo "  <<< Done training $env"
done
echo ""
echo "  All training complete."
echo ""

# ------------------------------------------------------------------
# Step 2: Unlearn with each method on each environment
# ------------------------------------------------------------------
echo "[2/3] Running unlearning methods..."
for env in "${ENVS[@]}"; do
    for method in "${METHODS[@]}"; do
        echo ""
        echo "  >>> Unlearning $method on $env ..."
        python "$SRC_DIR/unlearn.py" \
            env="$env" \
            unlearn="$method" \
            logging=wandb \
            wandb.group="unlearn" \
            wandb.tags="[unlearn,$env,$method]" \
            || { echo "  FAILED: unlearn $method on $env"; exit 1; }
        echo "  <<< Done $method on $env"
    done
done
echo ""
echo "  All unlearning complete."
echo ""

# ------------------------------------------------------------------
# Step 3: Evaluate (comparison plots + videos)
# ------------------------------------------------------------------
echo "[3/3] Running evaluation..."
for env in "${ENVS[@]}"; do
    echo ""
    echo "  >>> Evaluating $env ..."
    python "$SRC_DIR/evaluate.py" \
        env="$env" \
        logging=wandb \
        wandb.group="evaluate" \
        wandb.tags="[evaluate,$env]" \
        || { echo "  FAILED: evaluate $env"; exit 1; }
    echo "  <<< Done evaluating $env"
done
echo ""

echo "============================================================"
echo "  All experiments complete!"
echo ""
echo "  Results:"
echo "    Checkpoints: experiments/checkpoints/"
echo "    Videos:      experiments/outputs/*/videos/"
echo "    Plots:       experiments/outputs/*/evaluation_comparison.png"
echo "    Metrics:     experiments/outputs/*/evaluation_metrics.json"
echo "============================================================"
