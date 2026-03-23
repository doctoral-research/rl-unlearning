#!/bin/bash
set -euo pipefail

# Run full unlearning sweep for a given seed across all 9 scenarios × 3 methods.
# Each run: unlearn → evaluate (with scenario-aware metrics + wandb logging).
#
# Usage (docker):
#   docker compose run --rm rl-unlearning bash scripts/run_sweep.sh 42
# Usage (local):
#   PYTHON=/path/to/python bash scripts/run_sweep.sh 42

PYTHON="${PYTHON:-python}"
UNLEARN_SEED="${1:?Usage: $0 <unlearn_seed>}"
TRAIN_SEED=42

# Tuned hyperparameters
UNLEARN_LR=1e-5
NUM_STEPS=500
EVAL_FREQ=20

# env -> scenarios mapping
declare -A ENV_SCENARIOS
ENV_SCENARIOS[cartpole]="cartpole/left_only cartpole/centered cartpole/high_reward"
ENV_SCENARIOS[acrobot]="acrobot/no_clockwise acrobot/slow_swing acrobot/high_reward"
ENV_SCENARIOS[lunarlander]="lunarlander/center_landing lunarlander/no_tilt lunarlander/high_reward"

METHODS=("trajectory_selective" "strategy_inversion" "retain_protection")

# Summary CSV
SUMMARY_DIR="experiments/outputs"
SUMMARY_CSV="${SUMMARY_DIR}/sweep_seed${UNLEARN_SEED}_summary.csv"
mkdir -p "${SUMMARY_DIR}"
echo "env,scenario,method,seed,mean_return,std_return,retain_stability,scenario_forget_eff,scenario_selectivity,forget_region_frac,forget_traj_frac,conditioned_action_rate" > "${SUMMARY_CSV}"

TOTAL=0
PASSED=0
FAILED=0
FAILED_RUNS=""

echo "============================================"
echo "  Unlearning sweep — seed=${UNLEARN_SEED}"
echo "  9 scenarios × 3 methods = 27 runs"
echo "============================================"

for ENV in cartpole acrobot lunarlander; do
    BASELINE_CKPT="experiments/checkpoints/${ENV}_ppo_seed${TRAIN_SEED}"
    BASELINE_OUTPUT="experiments/outputs/${ENV}_ppo_seed${TRAIN_SEED}"

    # Verify baseline exists
    if [ ! -f "${BASELINE_CKPT}/final_model.pt" ]; then
        echo "ERROR: Baseline model not found: ${BASELINE_CKPT}/final_model.pt"
        echo "       Run scripts/train_baselines.sh first."
        exit 1
    fi

    for SCENARIO in ${ENV_SCENARIOS[$ENV]}; do
        SCENARIO_SHORT="${SCENARIO#*/}"  # e.g. "left_only" from "cartpole/left_only"
        BASELINE_ROW_WRITTEN=false

        for METHOD in "${METHODS[@]}"; do
            TOTAL=$((TOTAL + 1))
            RUN_NAME="${ENV}_${SCENARIO_SHORT}_${METHOD}_seed${UNLEARN_SEED}"
            EXP_DIR="experiments/outputs/${RUN_NAME}"

            echo ""
            echo ">>> [${TOTAL}/27] ${RUN_NAME}"

            # ---- Unlearn ----
            if $PYTHON src/unlearn.py \
                env="${ENV}" \
                unlearn="${METHOD}" \
                seed="${UNLEARN_SEED}" \
                scenario="${SCENARIO}" \
                experiment_name="${RUN_NAME}" \
                +baseline_dir="${BASELINE_CKPT}" \
                +baseline_output_dir="${BASELINE_OUTPUT}" \
                unlearn_lr="${UNLEARN_LR}" \
                num_unlearn_steps="${NUM_STEPS}" \
                eval_frequency="${EVAL_FREQ}" \
                logging=wandb \
                wandb.group="${ENV}/${SCENARIO_SHORT}" \
                wandb.tags="[${ENV},${SCENARIO_SHORT},${METHOD},seed${UNLEARN_SEED}]" \
                2>&1 | tee "${EXP_DIR}/unlearn.log"; then
                echo "    Unlearn OK"
            else
                echo "    FAILED: unlearn ${RUN_NAME}"
                FAILED=$((FAILED + 1))
                FAILED_RUNS="${FAILED_RUNS}\n  ${RUN_NAME} (unlearn)"
                continue
            fi

            # ---- Evaluate ----
            if $PYTHON src/evaluate.py \
                env="${ENV}" \
                seed="${UNLEARN_SEED}" \
                scenario="${SCENARIO}" \
                experiment_name="${RUN_NAME}" \
                +baseline_dir="${BASELINE_CKPT}" \
                +baseline_output_dir="${BASELINE_OUTPUT}" \
                logging=wandb \
                wandb.group="${ENV}/${SCENARIO_SHORT}" \
                wandb.tags="[${ENV},${SCENARIO_SHORT},eval,seed${UNLEARN_SEED}]" \
                2>&1 | tee "${EXP_DIR}/evaluate.log"; then
                echo "    Evaluate OK"
            else
                echo "    FAILED: evaluate ${RUN_NAME}"
                FAILED=$((FAILED + 1))
                FAILED_RUNS="${FAILED_RUNS}\n  ${RUN_NAME} (evaluate)"
                continue
            fi

            PASSED=$((PASSED + 1))

            # ---- Collect results into summary CSV ----
            # Write baseline row once per scenario (from first successful eval)
            WRITE_BASELINE="${BASELINE_ROW_WRITTEN}"
            $PYTHON -c "
import json, csv

exp_dir = '${EXP_DIR}'
env = '${ENV}'
scenario = '${SCENARIO}'
method = '${METHOD}'
seed = '${UNLEARN_SEED}'
write_baseline = '${WRITE_BASELINE}' != 'true'

# Load scenario metrics CSV (has both baseline and method rows)
baseline_scn = {}
method_scn = {}
try:
    with open(f'{exp_dir}/scenario_metrics.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['model'] == 'baseline':
                baseline_scn = row
            else:
                method_scn = row
except Exception:
    pass

# Load evaluation metrics JSON
metrics = {}
baseline_metrics = {}
try:
    with open(f'{exp_dir}/evaluation_metrics.json') as f:
        data = json.load(f)
    baseline_metrics = data.get('baseline_scenario', {})
    for k, v in data.items():
        if k not in ('baseline_scenario',):
            metrics = v
            break
except Exception:
    pass

def row(env, scenario, method, seed, met, scn):
    return (f'{env},{scenario},{method},{seed},'
            f'{met.get(\"mean_retain_return\", met.get(\"return_mean\", \"\"))},'
            f'{met.get(\"std_retain_return\", met.get(\"return_std\", \"\"))},'
            f'{met.get(\"retain_stability_index\", met.get(\"retain_stability\", \"\"))},'
            f'{scn.get(\"scenario_forget_effectiveness\", \"\")},'
            f'{scn.get(\"scenario_selectivity\", \"\")},'
            f'{scn.get(\"forget_region_fraction\", \"\")},'
            f'{scn.get(\"forget_traj_fraction\", \"\")},'
            f'{scn.get(\"conditioned_action_rate\", \"\")}')

# Baseline row
if write_baseline and baseline_scn:
    bl_ret = {'mean_retain_return': baseline_scn.get('return_mean', ''),
              'std_retain_return': baseline_scn.get('return_std', '')}
    # Get baseline return from eval results CSV
    try:
        with open(f'{exp_dir}/evaluation_results.csv') as f:
            import statistics
            rdr = csv.DictReader(f)
            bl_returns = [float(r['return']) for r in rdr if r['model'] == 'baseline']
            if bl_returns:
                bl_ret['mean_retain_return'] = f'{statistics.mean(bl_returns):.4f}'
                bl_ret['std_retain_return'] = f'{statistics.stdev(bl_returns):.4f}' if len(bl_returns) > 1 else '0'
    except Exception:
        pass
    print(row(env, scenario, 'baseline', seed, bl_ret, baseline_scn))

# Method row
print(row(env, scenario, method, seed, metrics, method_scn))
" >> "${SUMMARY_CSV}"
            BASELINE_ROW_WRITTEN=true

        done  # methods
    done  # scenarios
done  # envs

echo ""
echo "============================================"
echo "  Sweep complete — seed=${UNLEARN_SEED}"
echo "  Passed: ${PASSED}/27  Failed: ${FAILED}/27"
if [ -n "${FAILED_RUNS}" ]; then
    echo -e "  Failed runs:${FAILED_RUNS}"
fi
echo "  Summary: ${SUMMARY_CSV}"
echo "============================================"
