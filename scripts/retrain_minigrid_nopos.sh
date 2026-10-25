#!/bin/bash
# Stage 0 retrain: MiniGrid baselines WITHOUT the privileged (x, y, dir)
# prefix in the policy observation. Scenarios now read agent pose through
# the MiniGrid FeatureView instead (configs set scenario_view: minigrid).
#
# Usage: bash scripts/retrain_minigrid_nopos.sh [env] [seeds...]
set -uo pipefail

ENV_NAME="${1:-fourrooms}"; shift || true
SEEDS=("${@:-42 43 44 45 46}")
[ ${#SEEDS[@]} -eq 1 ] && read -ra SEEDS <<< "${SEEDS[0]}"

# Cap torch threads so parallel seeds don't oversubscribe the machine.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# CPU by default, for two reasons:
#  1. Speed. The policy is a 2835->64->64 MLP updated at batch 64, far too
#     small to amortise CUDA kernel-launch and transfer overhead. Measured
#     6700 steps/s on CPU vs 2555 on a 5090 for the PPO update.
#  2. Stability. Five concurrent CUDA contexts produced a simultaneous
#     segfault across all five workers (2026-09-15), losing ~1.7M steps
#     each. CPU sidesteps the shared failure mode entirely.
# Override with: DEVICE=cuda bash scripts/retrain_minigrid_nopos.sh ...
DEVICE="${DEVICE:-cpu}"

# Resume from the newest periodic checkpoint if one exists. Checkpoints
# carry optimizer moments, normaliser state and RNG streams, so a resumed
# run continues the learner exactly; only the in-flight episodes are lost.
# Set RESUME=false to force a clean start.
RESUME="${RESUME:-auto}"

mkdir -p experiments/logs/retrain
pids=()
for s in "${SEEDS[@]}"; do
  log="experiments/logs/retrain/${ENV_NAME}_seed${s}.log"
  echo "launching ${ENV_NAME} seed=${s} -> ${log}"
  python src/train_vec.py env="${ENV_NAME}" seed="${s}" +num_envs=8 \
      device="${DEVICE}" resume="${RESUME}" \
      experiment_name="${ENV_NAME}_ppo_seed${s}" \
      > "${log}" 2>&1 &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

echo "---- results ----"
for s in "${SEEDS[@]}"; do
  ckpt="experiments/checkpoints/${ENV_NAME}_ppo_seed${s}/final_model.pt"
  if [ -f "$ckpt" ]; then
    echo "OK   seed ${s}: $(grep -c 'Eval at step' "experiments/logs/retrain/${ENV_NAME}_seed${s}.log") evals, $ckpt"
    tail -3 "experiments/logs/retrain/${ENV_NAME}_seed${s}.log" | grep -i eval || true
  else
    echo "FAIL seed ${s}: no checkpoint"; fail=1
  fi
done
exit $fail
