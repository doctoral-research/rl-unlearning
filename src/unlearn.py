"""
Unlearning script
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
import pickle
from tqdm import tqdm

from agents import PPOAgent
from unlearning import TrajectorySelectiveForgetting, StrategyInversion, RetainProtection
from metrics import UnlearningMetrics, evaluate_trajectory_similarity
from utils import set_seed, get_device, setup_logger


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def unlearn(cfg: DictConfig):
    """Main unlearning function."""
    
    # Setup
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    
    logger = setup_logger(
        "unlearn",
        log_file=f"{cfg.log_dir}/unlearn.log"
    )
    
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Unlearning method: {cfg.method}")
    
    # Load trained agent
    checkpoint_path = f"{cfg.training.checkpoint_dir}/final_model.pt"
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Create environment
    env = gym.make(cfg.env_id)
    
    # Create agent
    agent = PPOAgent(
        observation_dim=cfg.observation_dim,
        action_dim=cfg.action_dim,
        action_type=cfg.action_type,
        hidden_dims=cfg.network.hidden_dims,
        activation=cfg.network.activation,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        device=str(device),
    )
    
    # Load checkpoint
    agent.load(checkpoint_path)
    logger.info(f"Loaded agent from {checkpoint_path}")
    
    # Load trajectories
    traj_path = f"{cfg.output_dir}/trajectories.pkl"
    if Path(traj_path).exists():
        with open(traj_path, "rb") as f:
            trajectories = pickle.load(f)
        logger.info(f"Loaded {len(trajectories)} trajectories")
    else:
        logger.warning("No trajectories found, generating random forget targets")
        trajectories = []
    
    # Initialize unlearning method
    if cfg.method == "trajectory_selective":
        unlearning_method = TrajectorySelectiveForgetting(
            agent=agent,
            forget_strength=cfg.forget_strength,
            loss_decomposition=cfg.loss_decomposition,
            target_selection_method=cfg.target_selection.method,
            target_selection_threshold=cfg.target_selection.threshold,
            gradient_reversal=cfg.negative_learning.gradient_reversal,
            loss_weights=OmegaConf.to_container(cfg.loss_weights),
            device=str(device),
        )
        
        # Identify forget trajectories
        toxic_states = cfg.get("toxic_states", None)
        if toxic_states:
            toxic_states = [np.array(s) for s in toxic_states]
        
        forget_indices = unlearning_method.identify_forget_trajectories(
            trajectories,
            toxic_states=toxic_states,
        )
        logger.info(f"Identified {len(forget_indices)} trajectories to forget")
        
    elif cfg.method == "strategy_inversion":
        unlearning_method = StrategyInversion(
            agent=agent,
            env=env,
            num_seeds=cfg.num_seeds,
            num_iterations=cfg.num_iterations,
            learning_rate=cfg.learning_rate,
            search_method=cfg.search.method,
            device=str(device),
        )
        
        # Generate synthetic forget states
        logger.info("Generating synthetic forget states...")
        forget_states = unlearning_method.generate_forget_states()
        logger.info(f"Generated {len(forget_states)} forget states")
        
    elif cfg.method == "retain_protection":
        retain_protection = RetainProtection(
            agent=agent,
            metaplasticity_enabled=cfg.metaplasticity.enabled,
            mask_type=cfg.metaplasticity.mask_type,
            mask_threshold=cfg.metaplasticity.threshold,
            distillation_enabled=cfg.distillation.enabled,
            distillation_temperature=cfg.distillation.temperature,
            distillation_alpha=cfg.distillation.alpha,
            device=str(device),
        )
        logger.info("Retain protection initialized")
        
        # This method is used in combination with others
        # For now, we'll just demonstrate the protection mechanism
    
    else:
        logger.error(f"Unknown unlearning method: {cfg.method}")
        return
    
    # Initialize metrics
    metrics_tracker = UnlearningMetrics()
    
    # Evaluate baseline performance
    logger.info("Evaluating baseline performance...")
    baseline_returns = []
    for _ in range(cfg.training.num_eval_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        baseline_returns.append(episode_reward)
    
    metrics_tracker.set_baseline(baseline_returns)
    logger.info(f"Baseline performance: {np.mean(baseline_returns):.2f} ± {np.std(baseline_returns):.2f}")
    
    # Unlearning loop
    logger.info("Starting unlearning process...")
    
    num_unlearn_steps = 1000
    batch_size = cfg.batch_size
    
    for step in tqdm(range(num_unlearn_steps), desc="Unlearning"):
        # 1. Sample forget and retain trajectories
        if trajectories and forget_indices and cfg.method == "trajectory_selective":
            # Sample a batch of forget trajectories
            num_forget = min(batch_size // 2, len(forget_indices))
            sample_forget_indices = np.random.choice(
                forget_indices, 
                size=num_forget,
                replace=False
            )
            
            # Collect forget batch data
            forget_obs, forget_acts, forget_rews = [], [], []
            for idx in sample_forget_indices:
                traj = trajectories[idx]
                forget_obs.extend(traj["observations"])
                forget_acts.extend(traj["actions"])
                forget_rews.extend(traj["rewards"])
            
            # Convert to tensors
            forget_batch = {
                "observations": torch.FloatTensor(np.array(forget_obs)).to(device),
                "actions": torch.LongTensor(forget_acts).to(device) if cfg.action_type == "discrete" 
                          else torch.FloatTensor(forget_acts).to(device),
                "rewards": torch.FloatTensor(forget_rews).to(device),
            }
            
            # 2. Apply unlearning method
            loss_info = unlearning_method.unlearn_step(forget_batch, retain_batch=None)
        
        if step % 100 == 0:
            # Evaluate
            eval_returns = []
            for _ in range(cfg.training.num_eval_episodes):
                obs, _ = env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    action, _ = agent.select_action(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    done = terminated or truncated
                
                eval_returns.append(episode_reward)
            
            # Compute forget scores
            forget_scores = []
            if trajectories and cfg.method == "trajectory_selective":
                for idx in forget_indices[:10]:  # Sample subset
                    score = evaluate_trajectory_similarity(
                        agent, trajectories[idx], str(device)
                    )
                    forget_scores.append(score)
            
            # Compute metrics
            if forget_scores:
                metrics = metrics_tracker.compute_all_metrics(
                    forget_scores=forget_scores,
                    retain_returns=eval_returns,
                    baseline_returns=baseline_returns,
                )
                
                logger.info(
                    f"Step {step} | "
                    f"Forget Eff: {metrics['forget_effectiveness']:.3f} | "
                    f"Retain Stab: {metrics['retain_stability_index']:.3f} | "
                    f"Selectivity: {metrics['selectivity']:.3f}"
                )
    
    # Final evaluation
    logger.info("Final evaluation...")
    final_returns = []
    for _ in range(cfg.training.num_eval_episodes * 2):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        final_returns.append(episode_reward)
    
    logger.info(f"Final performance: {np.mean(final_returns):.2f} ± {np.std(final_returns):.2f}")
    
    # Save unlearned model
    unlearned_path = f"{cfg.training.checkpoint_dir}/unlearned_model.pt"
    agent.save(unlearned_path)
    logger.info(f"Saved unlearned model to {unlearned_path}")
    
    env.close()


if __name__ == "__main__":
    unlearn()
