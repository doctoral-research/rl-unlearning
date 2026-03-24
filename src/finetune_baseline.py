"""
Fine-tuning baseline: load a pre-trained agent and fine-tune with reward shaping.

This is the fair warm-start comparison to our unlearning method — both start
from the same checkpoint, but fine-tuning uses environment interaction with
shaped rewards rather than gradient manipulation on stored trajectories.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gymnasium as gym
import numpy as np
import torch
import json
import argparse
from tqdm import tqdm

from agents import PPOAgent
from evaluate import evaluate_agent_with_trajectories, compute_scenario_metrics
from scenarios import ForgetScenario
from utils import set_seed, get_device, setup_logger, record_videos


class RewardShapingWrapper(gym.Wrapper):
    """Generic reward shaping wrapper that penalizes forget-region states."""

    def __init__(self, env, scenario, penalty=-1.0):
        super().__init__(env)
        self.scenario = scenario
        self.penalty = penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.scenario.matches_state(obs, action):
            reward = self.penalty
        return obs, reward, terminated, truncated, info


def finetune_baseline(
    env_id,
    scenario_path,
    checkpoint_path,
    seed=42,
    experiment_name=None,
    total_timesteps=50_000,
    learning_rate=3e-4,
    observation_dim=4,
    action_dim=2,
    action_type="discrete",
    hidden_dims=(64, 64),
    activation="tanh",
):
    """Fine-tune a pre-trained agent with reward shaping."""

    set_seed(seed)
    device = get_device("auto")

    scenario = ForgetScenario.load(scenario_path)

    if experiment_name is None:
        experiment_name = f"finetune_seed{seed}"

    checkpoint_dir = f"experiments/checkpoints/{experiment_name}"
    output_dir = f"experiments/outputs/{experiment_name}"
    log_dir = f"experiments/logs/{experiment_name}"

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = setup_logger("finetune", log_file=f"{log_dir}/finetune.log")
    logger.info(f"Fine-tuning {env_id}, scenario={scenario.name}, seed={seed}")

    # Create wrapped environment for training
    base_env = gym.make(env_id)
    env = RewardShapingWrapper(base_env, scenario, penalty=-1.0)

    # Unwrapped env for evaluation
    eval_env = gym.make(env_id)

    # Create agent with same architecture
    agent = PPOAgent(
        observation_dim=observation_dim,
        action_dim=action_dim,
        action_type=action_type,
        hidden_dims=list(hidden_dims),
        activation=activation,
        learning_rate=learning_rate,
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

    # Load pre-trained checkpoint (warm start)
    agent.load(checkpoint_path)
    logger.info(f"Loaded pre-trained model from {checkpoint_path}")

    # Training loop
    n_steps = 2048
    eval_frequency = 2_000
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

    pbar = tqdm(total=total_timesteps, desc="Fine-tuning")

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

        # Evaluate on STANDARD env (not wrapped)
        if global_step >= next_eval_step:
            eval_returns, eval_trajs = evaluate_agent_with_trajectories(
                agent, eval_env, num_eval_episodes, seed=seed * 1000
            )
            scn_metrics = compute_scenario_metrics(scenario, eval_trajs)

            mean_return = float(np.mean(eval_returns))
            forget_region_frac = scn_metrics.get("forget_region_fraction", 0.0)

            metrics_history.append({
                "step": global_step,
                "mean_return": mean_return,
                "forget_region_fraction": forget_region_frac,
            })

            logger.info(
                f"Step {global_step:>7d} | "
                f"Return: {mean_return:.1f} | "
                f"Region Frac: {forget_region_frac:.4f}"
            )

            next_eval_step += eval_frequency

        pbar.update(n_steps)

    pbar.close()

    # Save model
    unlearned_path = f"{checkpoint_dir}/unlearned_finetune.pt"
    agent.save(unlearned_path)
    logger.info(f"Saved model to {unlearned_path}")

    # Save metrics history
    with open(f"{output_dir}/finetune_metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)

    # Final evaluation
    final_returns, final_trajs = evaluate_agent_with_trajectories(
        agent, eval_env, num_eval_episodes * 5, seed=seed * 1000
    )
    final_scn = compute_scenario_metrics(scenario, final_trajs)
    logger.info(f"Final: return={np.mean(final_returns):.1f}, "
                f"forget_region_frac={final_scn.get('forget_region_fraction', 0):.4f}")

    env.close()
    eval_env.close()

    record_videos(
        agent, env_id, f"{output_dir}/videos",
        label="finetune", num_videos=3, seed=seed * 2000,
    )

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, required=True)
    parser.add_argument("--scenario_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--total_timesteps", type=int, default=50_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--observation_dim", type=int, default=4)
    parser.add_argument("--action_dim", type=int, default=2)
    parser.add_argument("--action_type", type=str, default="discrete")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--activation", type=str, default="tanh")
    args = parser.parse_args()

    finetune_baseline(
        env_id=args.env_id,
        scenario_path=args.scenario_path,
        checkpoint_path=args.checkpoint_path,
        seed=args.seed,
        experiment_name=args.experiment_name,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        observation_dim=args.observation_dim,
        action_dim=args.action_dim,
        action_type=args.action_type,
        hidden_dims=tuple(args.hidden_dims),
        activation=args.activation,
    )
