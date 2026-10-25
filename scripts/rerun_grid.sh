#!/bin/bash
# Re-run the curated online-unlearning grid on the Stage-0 baselines.
#
# Differs from the original grid in two ways that change what the numbers
# mean, so results are NOT comparable to the pre-2026-09 figures:
#   1. Baselines trained with include_pos=false -- the policy no longer
#      receives privileged absolute position, and scenarios resolve agent
#      pose through the MiniGrid FeatureView instead.
#   2. The disjoint objective now genuinely zeroes the policy gradient on
#      forget transitions. Previously PPO's internal advantage
#      normalisation reintroduced a systematic advantage there.
set -uo pipefail

ENV_NAME="${ENV_NAME:-fourrooms}"
SCENARIO="${SCENARIO:-fourrooms/avoid_bottom_right}"
SEEDS=(${SEEDS:-42 43 44 45 46})
ITERS="${ITERS:-200}"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# label:hydra-overrides
METHODS=(
  "noop_disjoint:+unlearn_method=no_op +disjoint_objective=true"
  "is_replay_disjoint:+unlearn_method=is_replay +disjoint_objective=true"
  "neg_reward_0.1:+unlearn_method=no_op +disjoint_objective=false +negative_reward_penalty=0.1"
  "neg_reward_1.0:+unlearn_method=no_op +disjoint_objective=false +negative_reward_penalty=1.0"
  "neggrad:+unlearn_method=gradient_reversal +disjoint_objective=false"
  "kl_neggrad:+unlearn_method=gradient_reversal +disjoint_objective=false +kl_anchor=true"
  "npo:+unlearn_method=npo +disjoint_objective=false"
)

mkdir -p experiments/logs/rerun
run() {  # label overrides seed
  local label="$1" ov="$2" s="$3"
  local ck="experiments/checkpoints/${ENV_NAME}_ppo_seed${s}/final_model.pt"
  [ -f "$ck" ] || { echo "SKIP seed $s: no baseline"; return 1; }
  # Skip work already on disk, so an interrupted grid resumes.
  # Hydra names the dir <env>_seed<s>_<method>_<label>, so glob the method
  # part rather than assuming the label alone identifies the run.
  local out
  out=$(ls -d experiments/outputs/online_unlearn_${ENV_NAME}_seed${s}_*_${label} 2>/dev/null | head -1)
  if [ -n "$out" ] && [ -f "${out}/history.json" ]; then
    echo "  skip seed=${s} ${label} (done)"; return 0
  fi
  local log="experiments/logs/rerun/${ENV_NAME}_seed${s}_${label}.log"
  python src/online_unlearn.py env="${ENV_NAME}" seed="${s}" device=cpu \
      scenario="${SCENARIO}" +load_baseline="${ck}" \
      +online_iters="${ITERS}" +run_suffix="${label}" \
      $ov > "$log" 2>&1
}

fail=0
for s in "${SEEDS[@]}"; do
  pids=()
  for m in "${METHODS[@]}"; do
    label="${m%%:*}"; ov="${m#*:}"
    echo "seed=${s} ${label}"
    run "$label" "$ov" "$s" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
done

echo "---- unlearned checkpoints ----"
ls -d experiments/outputs/online_unlearn_${ENV_NAME}_seed*_* 2>/dev/null | wc -l
exit $fail
