"""
Training script for RL agents
"""
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

import hydra
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from agents import PPOAgent
from utils import set_seed, get_device, create_directories, setup_logger, save_checkpoint, record_videos
from utils.buffers import TrajectoryBuffer


def init_wandb(cfg: DictConfig):
    """Initialize WandB if configured."""
    if cfg.get("backend") == "wandb" and cfg.get("wandb", {}).get("enabled", False):
        import wandb
        run_name = f"train_{cfg.env_name}_{cfg.name}_seed{cfg.seed}"
        run = wandb.init(
            project=cfg.wandb.get("project", "rl-unlearning"),
            entity=cfg.wandb.get("entity", None),
            name=run_name,
            group=cfg.wandb.get("group", None),
            job_type=cfg.wandb.get("job_type", "train"),
            tags=list(cfg.wandb.get("tags", [])),
            notes=cfg.wandb.get("notes", None),
            mode=cfg.wandb.get("mode", "online"),
            save_code=cfg.wandb.get("save_code", True),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        return run
    return None


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    """Main training function."""
    
    # Setup
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    create_directories(OmegaConf.to_container(cfg, resolve=True))
    
    logger = setup_logger(
        "train",
        log_file=f"{cfg.log_dir}/train.log"
    )
    
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Device: {device}")
    
    # Initialize WandB
    wandb_run = init_wandb(cfg)
    if wandb_run:
        logger.info(f"WandB run: {wandb_run.url}")
    
    # Create environment (routes MiniGrid envs through the flat-pos wrapper)
    from utils.env_wrappers import make_env
    env = make_env(
        cfg.env_id, seed=cfg.seed,
        exploration_bonus=cfg.get("exploration_bonus", None),
        exploration_beta=cfg.get("exploration_beta", 0.05),
        exploration_anneal_steps=cfg.get("exploration_anneal_steps", 0),
        exploration_hash_scale=cfg.get("exploration_hash_scale", 1.0),
        obs_encoding=cfg.get("obs_encoding", "image"),
        fixed_goal_pos=cfg.get("fixed_goal_pos", None),
    )
    
    # Create agent
    agent = PPOAgent(
        observation_dim=cfg.observation_dim,
        action_dim=cfg.action_dim,
        action_type=cfg.action_type,
        hidden_dims=cfg.network.hidden_dims,
        activation=cfg.network.activation,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_epsilon=cfg.clip_epsilon,
        clip_value=cfg.clip_value,
        value_coef=cfg.value_coef,
        entropy_coef=cfg.entropy_coef,
        max_grad_norm=cfg.max_grad_norm,
        n_epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        device=str(device),
    )
    
    # Trajectory buffer for saving experiences
    trajectory_buffer = TrajectoryBuffer(capacity=1000)

    # Optional Intrinsic Curiosity Module (Pathak et al. 2017). Needed
    # for sparse-reward envs where count-based bonus isn't enough — e.g.
    # Taxi-v3, where vanilla PPO and PPO+count-based both train at -200.
    icm = None
    if cfg.get("use_icm", False):
        from utils.icm import ICM
        icm = ICM(
            obs_dim=cfg.observation_dim,
            action_dim=cfg.action_dim,
            feature_dim=int(cfg.get("icm_feature_dim", 32)),
            hidden=int(cfg.get("icm_hidden", 64)),
            eta=float(cfg.get("icm_eta", 0.1)),
            beta=float(cfg.get("icm_beta", 0.2)),
            lr=float(cfg.get("icm_lr", 1e-3)),
            device=str(device),
        )
        logger.info(
            f"ICM enabled: feature_dim={icm.feature_dim}, eta={icm.eta}, "
            f"beta={icm.beta}"
        )

    # Training loop
    logger.info("Starting training...")
    
    global_step = 0
    episode_count = 0
    episode_rewards = []
    episode_lengths = []
    next_eval_step = cfg.training.eval_frequency
    next_save_step = cfg.training.save_frequency
    
    # Rollout buffer
    rollout_buffer = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "log_probs": [],
        "values": [],
    }
    
    obs, _ = env.reset(seed=cfg.seed)
    episode_reward = 0
    episode_length = 0
    current_trajectory = {"observations": [], "actions": [], "rewards": [], "dones": []}
    
    pbar = tqdm(total=cfg.training.total_timesteps, desc="Training")
    
    while global_step < cfg.training.total_timesteps:
        # Collect n_steps of experience
        for _ in range(cfg.n_steps):
            action, info = agent.select_action(obs, deterministic=False)
            
            rollout_buffer["observations"].append(obs)
            rollout_buffer["actions"].append(action)
            rollout_buffer["log_probs"].append(info["log_prob"])
            rollout_buffer["values"].append(info["value"])
            
            current_trajectory["observations"].append(obs.copy())
            current_trajectory["actions"].append(action.copy() if isinstance(action, np.ndarray) else action)
            
            # Step environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if icm is not None:
                reward = reward + icm.intrinsic_reward(obs, action, next_obs)

            rollout_buffer["rewards"].append(reward)
            rollout_buffer["dones"].append(done)
            
            current_trajectory["rewards"].append(reward)
            current_trajectory["dones"].append(done)
            
            episode_reward += reward
            episode_length += 1
            global_step += 1
            
            obs = next_obs
            
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_count += 1
                
                # Save trajectory
                trajectory_buffer.add_trajectory(current_trajectory.copy())
                
                # Reset
                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0
                current_trajectory = {"observations": [], "actions": [], "rewards": [], "dones": []}
                
                # Log episode stats
                log_freq = cfg.get("log_frequency", 10)
                if wandb_run:
                    import wandb
                    wandb.log({
                        "episode/reward": episode_rewards[-1],
                        "episode/length": episode_lengths[-1],
                        "episode/count": episode_count,
                    }, step=global_step)
                if episode_count % log_freq == 0:
                    mean_reward = np.mean(episode_rewards[-log_freq:])
                    mean_length = np.mean(episode_lengths[-log_freq:])
                    logger.info(
                        f"Episode {episode_count} | "
                        f"Step {global_step} | "
                        f"Mean Reward: {mean_reward:.2f} | "
                        f"Mean Length: {mean_length:.2f}"
                    )
        
        # Compute advantages
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
        
        # Update policy
        update_info = agent.update(rollout_buffer)

        # Update ICM on the same rollout (next_obs from buffer shift; the
        # last step's next_obs is the current `obs`).
        if icm is not None:
            obs_arr = np.asarray(rollout_buffer["observations"], dtype=np.float32)
            act_arr = np.asarray(rollout_buffer["actions"])
            next_obs_arr = np.concatenate(
                [obs_arr[1:], np.asarray(obs, dtype=np.float32)[None]], axis=0,
            )
            icm_info = icm.update(obs_arr, act_arr, next_obs_arr)
            update_info.update(icm_info)
        
        if wandb_run:
            import wandb
            wandb.log({
                "train/policy_loss": update_info["policy_loss"],
                "train/value_loss": update_info["value_loss"],
                "train/entropy": update_info["entropy"],
            }, step=global_step)
        
        # Clear rollout buffer
        for key in rollout_buffer:
            rollout_buffer[key] = []
        
        # Evaluation
        if global_step >= next_eval_step:
            eval_rewards = []
            eval_seed = cfg.seed * 1000
            for ep_i in range(cfg.training.num_eval_episodes):
                eval_obs, _ = env.reset(seed=eval_seed + ep_i)
                eval_reward = 0
                eval_done = False
                
                while not eval_done:
                    eval_action, _ = agent.select_action(eval_obs, deterministic=True)
                    eval_obs, eval_r, eval_terminated, eval_truncated, _ = env.step(eval_action)
                    eval_reward += eval_r
                    eval_done = eval_terminated or eval_truncated
                
                eval_rewards.append(eval_reward)
            
            mean_eval_reward = np.mean(eval_rewards)
            logger.info(f"Eval at step {global_step}: Mean Reward = {mean_eval_reward:.2f}")
            
            if wandb_run:
                import wandb
                wandb.log({
                    "eval/mean_reward": mean_eval_reward,
                    "eval/std_reward": np.std(eval_rewards),
                }, step=global_step)
            next_eval_step += cfg.training.eval_frequency

        # Save checkpoint
        if global_step >= next_save_step:
            checkpoint_path = f"{cfg.training.checkpoint_dir}/checkpoint_{global_step}.pt"
            Path(cfg.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                agent,
                agent.optimizer,
                global_step,
                {"mean_reward": np.mean(episode_rewards[-100:]) if episode_rewards else 0.0},
                checkpoint_path,
            )
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            next_save_step += cfg.training.save_frequency

        pbar.update(cfg.n_steps)
    
    pbar.close()
    
    # Save final model
    final_path = f"{cfg.training.checkpoint_dir}/final_model.pt"
    agent.save(final_path)
    logger.info(f"Training completed. Final model saved to {final_path}")
    
    # Save trajectories
    import pickle
    traj_path = f"{cfg.output_dir}/trajectories.pkl"
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    with open(traj_path, "wb") as f:
        pickle.dump(trajectory_buffer.get_all(), f)
    logger.info(f"Saved {len(trajectory_buffer)} trajectories to {traj_path}")
    
    env.close()

    # Record behavior videos after training
    video_dir = f"{cfg.output_dir}/videos"
    video_seed = cfg.seed * 2000
    logger.info("Recording post-training behavior videos...")
    video_paths = record_videos(
        agent, cfg.env_id, video_dir, label="trained",
        num_videos=3, seed=video_seed,
    )
    if wandb_run and video_paths:
        import wandb
        for vp in video_paths:
            wandb.log({"videos/trained": wandb.Video(vp, fps=30)})

    if wandb_run:
        import wandb
        wandb.finish()
        logger.info("WandB run finished.")


if __name__ == "__main__":
    train()
