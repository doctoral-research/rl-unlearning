"""
Evaluation and comparison script
"""
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

import hydra
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from agents import PPOAgent
from metrics import UnlearningMetrics, evaluate_trajectory_similarity
from utils import set_seed, get_device, setup_logger


def init_wandb(cfg: DictConfig):
    """Initialize WandB if configured."""
    if cfg.get("backend") == "wandb" and cfg.get("wandb", {}).get("enabled", False):
        import wandb
        run = wandb.init(
            project=cfg.wandb.get("project", "rl-unlearning"),
            entity=cfg.wandb.get("entity", None),
            group=cfg.wandb.get("group", None),
            job_type="evaluate",
            tags=list(cfg.wandb.get("tags", [])),
            mode=cfg.wandb.get("mode", "online"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        return run
    return None


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def evaluate(cfg: DictConfig):
    """Evaluate and compare baseline vs unlearned models."""
    
    # Setup
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    
    logger = setup_logger("evaluate", log_file=f"{cfg.log_dir}/evaluate.log")
    
    # Initialize WandB
    wandb_run = init_wandb(cfg)
    if wandb_run:
        logger.info(f"WandB run: {wandb_run.url}")
    
    # Create environment
    env = gym.make(cfg.env_id)
    
    # Load baseline model
    baseline_path = f"{cfg.training.checkpoint_dir}/final_model.pt"
    baseline_agent = PPOAgent(
        observation_dim=cfg.observation_dim,
        action_dim=cfg.action_dim,
        action_type=cfg.action_type,
        hidden_dims=cfg.network.hidden_dims,
        device=str(device),
    )
    
    if Path(baseline_path).exists():
        baseline_agent.load(baseline_path)
        logger.info(f"Loaded baseline model from {baseline_path}")
    else:
        logger.error(f"Baseline model not found: {baseline_path}")
        return
    
    # Load unlearned model
    unlearned_path = f"{cfg.training.checkpoint_dir}/unlearned_model.pt"
    unlearned_agent = PPOAgent(
        observation_dim=cfg.observation_dim,
        action_dim=cfg.action_dim,
        action_type=cfg.action_type,
        hidden_dims=cfg.network.hidden_dims,
        device=str(device),
    )
    
    if Path(unlearned_path).exists():
        unlearned_agent.load(unlearned_path)
        logger.info(f"Loaded unlearned model from {unlearned_path}")
    else:
        logger.warning(f"Unlearned model not found: {unlearned_path}")
        unlearned_agent = None
    
    # Load trajectories
    traj_path = f"{cfg.output_dir}/trajectories.pkl"
    if Path(traj_path).exists():
        with open(traj_path, "rb") as f:
            trajectories = pickle.load(f)
        logger.info(f"Loaded {len(trajectories)} trajectories")
    else:
        trajectories = []
    
    # Evaluation
    results = {
        "baseline": {"returns": [], "forget_scores": []},
        "unlearned": {"returns": [], "forget_scores": []},
    }
    
    num_eval_episodes = 50
    
    # Evaluate baseline
    logger.info("Evaluating baseline agent...")
    for _ in range(num_eval_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = baseline_agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        results["baseline"]["returns"].append(episode_reward)
    
    # Compute forget scores for baseline
    if trajectories:
        for traj in trajectories[:20]:
            score = evaluate_trajectory_similarity(baseline_agent, traj, str(device))
            results["baseline"]["forget_scores"].append(score)
    
    # Evaluate unlearned
    if unlearned_agent:
        logger.info("Evaluating unlearned agent...")
        for _ in range(num_eval_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = unlearned_agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            results["unlearned"]["returns"].append(episode_reward)
        
        # Compute forget scores for unlearned
        if trajectories:
            for traj in trajectories[:20]:
                score = evaluate_trajectory_similarity(unlearned_agent, traj, str(device))
                results["unlearned"]["forget_scores"].append(score)
    
    # Compute metrics
    metrics_tracker = UnlearningMetrics()
    
    if unlearned_agent and results["unlearned"]["forget_scores"]:
        metrics = metrics_tracker.compute_all_metrics(
            forget_scores=results["unlearned"]["forget_scores"],
            retain_returns=results["unlearned"]["returns"],
            baseline_returns=results["baseline"]["returns"],
        )
        
        logger.info("\n=== Unlearning Metrics ===")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        if wandb_run:
            import wandb
            wandb.log({f"eval/{k}": v for k, v in metrics.items()})
    
    # Print comparison
    logger.info("\n=== Performance Comparison ===")
    logger.info(f"Baseline Return: {np.mean(results['baseline']['returns']):.2f} ± {np.std(results['baseline']['returns']):.2f}")
    if unlearned_agent:
        logger.info(f"Unlearned Return: {np.mean(results['unlearned']['returns']):.2f} ± {np.std(results['unlearned']['returns']):.2f}")
    
    if results["baseline"]["forget_scores"] and results["unlearned"]["forget_scores"]:
        logger.info(f"Baseline Forget Score: {np.mean(results['baseline']['forget_scores']):.4f}")
        logger.info(f"Unlearned Forget Score: {np.mean(results['unlearned']['forget_scores']):.4f}")
    
    # Log summary to WandB
    if wandb_run:
        import wandb
        summary = {
            "eval/baseline_return_mean": np.mean(results["baseline"]["returns"]),
            "eval/baseline_return_std": np.std(results["baseline"]["returns"]),
        }
        if unlearned_agent:
            summary["eval/unlearned_return_mean"] = np.mean(results["unlearned"]["returns"])
            summary["eval/unlearned_return_std"] = np.std(results["unlearned"]["returns"])
        wandb.log(summary)
    
    # Save results
    results_df = pd.DataFrame({
        "model": ["baseline"] * len(results["baseline"]["returns"]) + 
                 (["unlearned"] * len(results["unlearned"]["returns"]) if unlearned_agent else []),
        "return": results["baseline"]["returns"] + 
                  (results["unlearned"]["returns"] if unlearned_agent else []),
    })
    
    results_path = f"{cfg.output_dir}/evaluation_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved results to {results_path}")
    
    # Plot comparison
    if unlearned_agent:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Returns comparison
        sns.boxplot(data=results_df, x="model", y="return", ax=axes[0])
        axes[0].set_title("Return Comparison")
        axes[0].set_ylabel("Episode Return")
        
        # Forget scores comparison
        if results["baseline"]["forget_scores"] and results["unlearned"]["forget_scores"]:
            forget_df = pd.DataFrame({
                "model": ["baseline"] * len(results["baseline"]["forget_scores"]) + 
                         ["unlearned"] * len(results["unlearned"]["forget_scores"]),
                "forget_score": results["baseline"]["forget_scores"] + results["unlearned"]["forget_scores"],
            })
            sns.boxplot(data=forget_df, x="model", y="forget_score", ax=axes[1])
            axes[1].set_title("Forget Score Comparison (Lower = More Forgotten)")
            axes[1].set_ylabel("Trajectory Similarity Score")
        
        plt.tight_layout()
        plot_path = f"{cfg.output_dir}/evaluation_comparison.png"
        plt.savefig(plot_path, dpi=150)
        logger.info(f"Saved plot to {plot_path}")
    
    env.close()
    
    if wandb_run:
        import wandb
        wandb.finish()
        logger.info("WandB run finished.")


if __name__ == "__main__":
    evaluate()
