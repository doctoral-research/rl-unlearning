#!/bin/bash
set -euo pipefail

# Collect trajectories from existing trained models.
# Use this after train_baselines.sh or when trajectories were lost.
#
# Usage (docker):
#   docker compose run --rm rl-unlearning bash scripts/collect_trajectories.sh
# Usage (local):
#   PYTHON=/path/to/python bash scripts/collect_trajectories.sh

PYTHON="${PYTHON:-python}"
SEED=42
NUM_EPISODES=1000

declare -A ENV_TIMESTEPS
ENV_TIMESTEPS[cartpole]=500000
ENV_TIMESTEPS[acrobot]=1000000
ENV_TIMESTEPS[lunarlander]=2000000

echo "============================================"
echo "  Collecting trajectories from trained models"
echo "============================================"

for ENV in cartpole acrobot lunarlander; do
    EXP_NAME="${ENV}_ppo_seed${SEED}"
    CKPT="experiments/checkpoints/${EXP_NAME}/final_model.pt"
    OUT_DIR="experiments/outputs/${EXP_NAME}"

    if [ ! -f "${CKPT}" ]; then
        echo "SKIP: ${CKPT} not found"
        continue
    fi

    if [ -f "${OUT_DIR}/trajectories.pkl" ]; then
        echo "SKIP: ${OUT_DIR}/trajectories.pkl already exists"
        continue
    fi

    echo ">>> Collecting trajectories for ${ENV} ..."
    mkdir -p "${OUT_DIR}"

    $PYTHON -c "
import sys
sys.path.insert(0, 'src')
import gymnasium as gym
import numpy as np
import pickle
from agents import PPOAgent
from utils import set_seed

set_seed(${SEED})
env_ids = {'cartpole': 'CartPole-v1', 'acrobot': 'Acrobot-v1', 'lunarlander': 'LunarLander-v3'}
env = gym.make(env_ids['${ENV}'])

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

agent = PPOAgent(
    observation_dim=obs_dim,
    action_dim=act_dim,
    action_type='discrete',
    hidden_dims=[64, 64],
    device='cpu',
)
agent.load('${CKPT}')
print(f'Loaded model from ${CKPT}')

trajectories = []
for ep in range(${NUM_EPISODES}):
    obs, _ = env.reset(seed=${SEED} * 100 + ep)
    traj = {'observations': [], 'actions': [], 'rewards': [], 'dones': []}
    done = False
    while not done:
        action, _ = agent.select_action(obs, deterministic=False)
        traj['observations'].append(obs.copy())
        traj['actions'].append(action.copy() if isinstance(action, np.ndarray) else action)
        obs, reward, terminated, truncated, _ = env.step(action)
        traj['rewards'].append(reward)
        traj['dones'].append(terminated or truncated)
        done = terminated or truncated
    trajectories.append(traj)

env.close()
with open('${OUT_DIR}/trajectories.pkl', 'wb') as f:
    pickle.dump(trajectories, f)
print(f'Saved {len(trajectories)} trajectories to ${OUT_DIR}/trajectories.pkl')
returns = [sum(t['rewards']) for t in trajectories]
print(f'Mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}')
"

    echo "    Done: ${OUT_DIR}/trajectories.pkl"
done

echo ""
echo "============================================"
echo "  Trajectory collection complete."
echo "============================================"
