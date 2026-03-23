#!/bin/bash
set -euo pipefail

# Train baseline models for all 3 environments (seed 42).
#
# Usage (docker):
#   docker compose run --rm rl-unlearning bash scripts/train_baselines.sh
# Usage (local):
#   PYTHON=/path/to/python bash scripts/train_baselines.sh

PYTHON="${PYTHON:-python}"
ENVS=("cartpole" "acrobot" "lunarlander")
SEED=42

echo "============================================"
echo "  Training baseline models (seed=${SEED})"
echo "============================================"

for ENV in "${ENVS[@]}"; do
    EXP_NAME="${ENV}_ppo_seed${SEED}"
    echo ""
    echo ">>> Training ${EXP_NAME} ..."

    $PYTHON src/train.py \
        env="${ENV}" \
        seed="${SEED}" \
        logging=wandb \
        experiment_name="${EXP_NAME}" \
        wandb.group="${ENV}/train" \
        wandb.job_type=train \
        wandb.tags="[${ENV},train,seed${SEED}]"

    echo ">>> Done: ${EXP_NAME}"
    echo "    Model: experiments/checkpoints/${EXP_NAME}/final_model.pt"
    echo "    Trajectories: experiments/outputs/${EXP_NAME}/trajectories.pkl"
done

echo ""
echo "============================================"
echo "  All baselines trained."
echo "============================================"
