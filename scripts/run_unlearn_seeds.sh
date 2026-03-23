#!/usr/bin/env bash
# =============================================================================
# Run unlearning + evaluation with multiple seeds on existing trained models.
# The trained models are from seed 42; we vary only the unlearning seed.
#
# For each new seed, we copy the trained final_model.pt AND trajectories.pkl
# into seed-specific dirs so unlearn.py can find them.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ -f "$REPO_ROOT/.env" ]; then
    set -a; source "$REPO_ROOT/.env"; set +a
    echo "Loaded .env"
fi

SRC_DIR="$REPO_ROOT/src"
ENVS=("cartpole" "lunarlander" "acrobot")
METHODS=("trajectory_selective" "strategy_inversion" "retain_protection")
SEEDS=(43 44)
TRAIN_SEED=42

echo "============================================================"
echo "  Multi-seed Unlearning (seeds: ${SEEDS[*]})"
echo "  Using trained models from seed $TRAIN_SEED"
echo "============================================================"
echo ""

for seed in "${SEEDS[@]}"; do
    echo "==================== SEED $seed ===================="

    # Copy trained models and trajectories into seed-specific dirs
    for env in "${ENVS[@]}"; do
        # Checkpoint dir (final_model.pt)
        src_ckpt="experiments/checkpoints/${env}_ppo_seed${TRAIN_SEED}"
        dst_ckpt="experiments/checkpoints/${env}_ppo_seed${seed}"
        mkdir -p "$dst_ckpt"
        cp "$src_ckpt/final_model.pt" "$dst_ckpt/final_model.pt"
        echo "  Copied trained model: $src_ckpt -> $dst_ckpt"

        # Output dir (trajectories.pkl)
        src_out="experiments/outputs/${env}_ppo_seed${TRAIN_SEED}"
        dst_out="experiments/outputs/${env}_ppo_seed${seed}"
        mkdir -p "$dst_out"
        if [ -f "$src_out/trajectories.pkl" ]; then
            cp "$src_out/trajectories.pkl" "$dst_out/trajectories.pkl"
            echo "  Copied trajectories: $src_out -> $dst_out"
        fi
    done

    # ------------------------------------------------------------------
    # Unlearn with each method on each environment
    # ------------------------------------------------------------------
    for env in "${ENVS[@]}"; do
        for method in "${METHODS[@]}"; do
            echo ""
            echo "  >>> Unlearning $method on $env (seed $seed) ..."
            python "$SRC_DIR/unlearn.py" \
                env="$env" \
                unlearn="$method" \
                seed="$seed" \
                logging=wandb \
                wandb.group="unlearn_seed${seed}" \
                wandb.tags="[unlearn,$env,$method,seed${seed}]" \
                || { echo "  FAILED: unlearn $method on $env seed $seed"; exit 1; }
            echo "  <<< Done $method on $env (seed $seed)"
        done
    done

    # ------------------------------------------------------------------
    # Evaluate each environment for this seed
    # ------------------------------------------------------------------
    for env in "${ENVS[@]}"; do
        echo ""
        echo "  >>> Evaluating $env (seed $seed) ..."
        python "$SRC_DIR/evaluate.py" \
            env="$env" \
            seed="$seed" \
            logging=wandb \
            wandb.group="evaluate_seed${seed}" \
            wandb.tags="[evaluate,$env,seed${seed}]" \
            || { echo "  FAILED: evaluate $env seed $seed"; exit 1; }
        echo "  <<< Done evaluating $env (seed $seed)"
    done
done

echo ""
echo "============================================================"
echo "  All multi-seed experiments complete!"
echo "  Seeds: ${SEEDS[*]}"
echo "============================================================"

# Fix file ownership if running in Docker as root
if [ "$(id -u)" = "0" ] && [ -n "${HOST_UID:-}" ]; then
    echo "Fixing file ownership to ${HOST_UID}:${HOST_GID:-$HOST_UID}..."
    chown -R "${HOST_UID}:${HOST_GID:-$HOST_UID}" experiments/ wandb/ 2>/dev/null || true
fi
