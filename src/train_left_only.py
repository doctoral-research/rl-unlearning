"""
Full-retrain baseline: train a CartPole agent from scratch that only balances left.

Uses a reward wrapper that penalizes the agent whenever cart_position > 0,
creating an agent that learns to balance exclusively on the left side.

This serves as the "gold standard" baseline — the best possible left-only
agent achievable through standard training. The comparison with unlearning
methods measures efficiency: how many steps does each approach need?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gymnasium as gym
import numpy as np
import torch
import pickle
import json
import argparse
from tqdm import tqdm

from agents import PPOAgent
from evaluate import evaluate_agent_with_trajectories, compute_scenario_metrics
from scenarios import ForgetScenario
from utils import set_seed, get_device, setup_logger, record_videos


class LeftOnlyCartPole(gym.Wrapper):
    """CartPole wrapper that penalizes right-side behavior.

    When the cart is on the right side (position > 0), the reward is
    replaced with a penalty. This trains the agent to balance only on
    the left side from scratch.
    """

    def __init__(self, env, penalty=-1.0):
        super().__init__(env)
        self.penalty = penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cart_position = obs[0]
        if cart_position > 0.0:
            reward = self.penalty
        return obs, reward, terminated, truncated, info


def train_left_only(seed=42, experiment_name=None, total_timesteps=500_000):
    """Train a CartPole agent with left-only reward shaping."""

    set_seed(seed)
    device = get_device("auto")

    if experiment_name is None:
        experiment_name = f"cartpole_left_only_full_retrain_seed{seed}"

    checkpoint_dir = f"experiments/checkpoints/{experiment_name}"
    output_dir = f"experiments/outputs/{experiment_name}"
    log_dir = f"experiments/logs/{experiment_name}"

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = setup_logger("train_left_only", log_file=f"{log_dir}/train.log")
    logger.info(f"Training left-only CartPole agent, seed={seed}")

    # Create wrapped environment
    base_env = gym.make("CartPole-v1")
    env = LeftOnlyCartPole(base_env, penalty=-1.0)

    # Unwrapped env for evaluation (standard CartPole rewards)
    eval_env = gym.make("CartPole-v1")

    # Same architecture and hyperparams as the baseline training
    agent = PPOAgent(
        observation_dim=4,
        action_dim=2,
        action_type="discrete",
        hidden_dims=[64, 64],
        activation="tanh",
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        clip_value=True,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        n_epochs=10,
        batch_size=64,
        device=str(device),
    )

    # Load scenario for metric tracking
    scenario = ForgetScenario.load("configs/scenarios/cartpole/left_only.yaml")

    # Training loop (same structure as train.py)
    n_steps = 2048
    eval_frequency = 10_000
    num_eval_episodes = 10

    global_step = 0
    episode_count = 0
    episode_rewards = []
    next_eval_step = eval_frequency
    metrics_history = []

    rollout_buffer = {
        "observations": [], "actions": [], "rewards": [],
        "dones": [], "log_probs": [], "values": [],
    }

    obs, _ = env.reset(seed=seed)
    episode_reward = 0

    pbar = tqdm(total=total_timesteps, desc="Training left-only")

    while global_step < total_timesteps:
        for _ in range(n_steps):
            action, info = agent.select_action(obs, deterministic=False)

            rollout_buffer["observations"].append(obs)
            rollout_buffer["actions"].append(action)
            rollout_buffer["log_probs"].append(info["log_prob"])
            rollout_buffer["values"].append(info["value"])

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            rollout_buffer["rewards"].append(reward)
            rollout_buffer["dones"].append(done)

            episode_reward += reward
            episode_length = 0
            global_step += 1
            obs = next_obs

            if done:
                episode_rewards.append(episode_reward)
                episode_count += 1
                obs, _ = env.reset()
                episode_reward = 0

        # Compute advantages and update
        next_value = agent.network.get_value(
            torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0).to(device)
        ).item()

        advantages, returns = agent.compute_gae(
            torch.as_tensor(np.asarray(rollout_buffer["rewards"]), dtype=torch.float32),
            torch.as_tensor(np.asarray(rollout_buffer["values"]), dtype=torch.float32),
            torch.as_tensor(np.asarray(rollout_buffer["dones"]), dtype=torch.float32),
            next_value,
        )

        rollout_buffer["advantages"] = advantages.numpy()
        rollout_buffer["returns"] = returns.numpy()
        agent.update(rollout_buffer)

        for key in rollout_buffer:
            rollout_buffer[key] = []

        # Evaluate on STANDARD CartPole (not wrapped) with scenario metrics
        if global_step >= next_eval_step:
            eval_returns, eval_trajs = evaluate_agent_with_trajectories(
                agent, eval_env, num_eval_episodes, seed=seed * 1000
            )
            scn_metrics = compute_scenario_metrics(scenario, eval_trajs)

            mean_return = float(np.mean(eval_returns))
            forget_region_frac = scn_metrics.get("forget_region_fraction", 0.0)
            forget_traj_frac = scn_metrics.get("forget_traj_fraction", 0.0)

            metrics_history.append({
                "step": global_step,
                "mean_return": mean_return,
                "forget_region_fraction": forget_region_frac,
                "forget_traj_fraction": forget_traj_frac,
            })

            logger.info(
                f"Step {global_step:>7d} | "
                f"Return: {mean_return:.1f} | "
                f"Region Frac: {forget_region_frac:.4f} | "
                f"Traj Frac: {forget_traj_frac:.2f}"
            )

            next_eval_step += eval_frequency

        pbar.update(n_steps)

    pbar.close()

    # Save model (as unlearned_ so evaluate.py can discover it)
    unlearned_path = f"{checkpoint_dir}/unlearned_full_retrain.pt"
    agent.save(unlearned_path)
    logger.info(f"Saved model to {unlearned_path}")

    # Save metrics history
    with open(f"{output_dir}/retrain_metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)

    # Final evaluation
    final_returns, final_trajs = evaluate_agent_with_trajectories(
        agent, eval_env, num_eval_episodes * 5, seed=seed * 1000
    )
    final_scn = compute_scenario_metrics(scenario, final_trajs)
    logger.info(f"Final: return={np.mean(final_returns):.1f}, "
                f"forget_region_frac={final_scn.get('forget_region_fraction', 0):.4f}, "
                f"forget_traj_frac={final_scn.get('forget_traj_fraction', 0):.2f}")

    env.close()
    eval_env.close()

    # Record videos
    record_videos(
        agent, "CartPole-v1", f"{output_dir}/videos",
        label="full_retrain", num_videos=3, seed=seed * 2000,
    )

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    # Also accept Hydra-style positional args from the shell script
    args, unknown = parser.parse_known_args()

    # Parse Hydra-style key=value args
    for arg in unknown:
        if "=" in arg:
            key, val = arg.split("=", 1)
            if key == "seed":
                args.seed = int(val)
            elif key == "experiment_name":
                args.experiment_name = val
            elif key == "total_timesteps":
                args.total_timesteps = int(val)

    train_left_only(
        seed=args.seed,
        experiment_name=args.experiment_name,
        total_timesteps=args.total_timesteps,
    )
