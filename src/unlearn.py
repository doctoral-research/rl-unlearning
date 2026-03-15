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


def init_wandb(cfg: DictConfig):
    """Initialize WandB if configured."""
    if cfg.get("backend") == "wandb" and cfg.get("wandb", {}).get("enabled", False):
        import wandb
        run = wandb.init(
            project=cfg.wandb.get("project", "rl-unlearning"),
            entity=cfg.wandb.get("entity", None),
            group=cfg.wandb.get("group", None),
            job_type="unlearn",
            tags=list(cfg.wandb.get("tags", [])),
            mode=cfg.wandb.get("mode", "online"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        return run
    return None


def evaluate_agent(agent, env, num_episodes, deterministic=True):
    """Run evaluation episodes and return list of episode returns."""
    returns = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = agent.select_action(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        returns.append(episode_reward)
    return returns


def sample_batch_from_trajectories(
    trajectories, indices, batch_size, action_type, device
):
    """Sample a batch of transitions from the given trajectory indices."""
    num_sample = min(batch_size, len(indices))
    sampled_indices = np.random.choice(indices, size=num_sample, replace=False)

    obs_list, act_list, rew_list = [], [], []
    for idx in sampled_indices:
        traj = trajectories[idx]
        obs_list.extend(traj["observations"])
        act_list.extend(traj["actions"])
        rew_list.extend(traj["rewards"])

    # Sub-sample transitions if too many were collected
    n_transitions = len(obs_list)
    if n_transitions > batch_size:
        chosen = np.random.choice(n_transitions, size=batch_size, replace=False)
        obs_list = [obs_list[i] for i in chosen]
        act_list = [act_list[i] for i in chosen]
        rew_list = [rew_list[i] for i in chosen]

    batch = {
        "observations": torch.as_tensor(np.array(obs_list), dtype=torch.float32).to(device),
        "actions": (
            torch.as_tensor(np.array(act_list), dtype=torch.long).to(device)
            if action_type == "discrete"
            else torch.as_tensor(np.array(act_list), dtype=torch.float32).to(device)
        ),
        "rewards": torch.as_tensor(np.array(rew_list), dtype=torch.float32).to(device),
    }
    return batch


def compute_forget_scores(agent, trajectories, forget_indices, device, max_eval=10):
    """Compute trajectory similarity scores for forget trajectories."""
    scores = []
    eval_indices = forget_indices[:max_eval] if len(forget_indices) > max_eval else forget_indices
    for idx in eval_indices:
        score = evaluate_trajectory_similarity(agent, trajectories[idx], str(device))
        scores.append(score)
    return scores


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

    # Initialize WandB
    wandb_run = init_wandb(cfg)
    if wandb_run:
        logger.info(f"WandB run: {wandb_run.url}")

    # Load trained agent
    checkpoint_path = f"{cfg.training.checkpoint_dir}/final_model.pt"
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    # Create environment
    env = gym.make(cfg.env_id)

    # Create agent with the TRAINING learning rate (from agent config, not unlearn config)
    # Note: unlearn configs may override cfg.learning_rate (e.g. strategy_inversion
    # sets search_lr for state optimization). We always use the agent's original LR here.
    agent_lr = cfg.get("agent_lr", 3e-4)  # fallback to PPO default
    agent = PPOAgent(
        observation_dim=cfg.observation_dim,
        action_dim=cfg.action_dim,
        action_type=cfg.action_type,
        hidden_dims=cfg.network.hidden_dims,
        activation=cfg.network.activation,
        learning_rate=agent_lr,
        gamma=cfg.gamma,
        device=str(device),
    )

    # Load checkpoint (restores network weights AND optimizer state from training)
    agent.load(checkpoint_path)
    logger.info(f"Loaded agent from {checkpoint_path}")

    # Create a FRESH optimizer for unlearning with a much smaller LR.
    # Using the training optimizer is problematic: it carries accumulated momentum
    # from training and its LR is tuned for learning, not unlearning.
    unlearn_lr = cfg.get("unlearn_lr", 1e-5)
    agent.optimizer = torch.optim.Adam(
        agent.network.parameters(), lr=unlearn_lr, eps=1e-5
    )
    logger.info(f"Created fresh unlearning optimizer with lr={unlearn_lr}")

    # Load trajectories
    traj_path = f"{cfg.output_dir}/trajectories.pkl"
    if Path(traj_path).exists():
        with open(traj_path, "rb") as f:
            trajectories = pickle.load(f)
        logger.info(f"Loaded {len(trajectories)} trajectories")
    else:
        logger.warning("No trajectories found, generating random forget targets")
        trajectories = []

    # ---- Initialize unlearning method and identify forget/retain sets ----
    forget_indices = []
    retain_indices = []
    forget_states = []
    unlearning_method = None
    retain_protection = None

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

        # If no toxic states matched, use reward-based selection as fallback:
        # forget the top-performing trajectories (high-reward region unlearning)
        if not forget_indices and trajectories:
            traj_returns = [sum(t["rewards"]) for t in trajectories]
            num_forget = max(1, int(len(trajectories) * 0.1))
            forget_indices = list(np.argsort(traj_returns)[-num_forget:])
            logger.info(f"No toxic states provided; using top-{num_forget} reward trajectories as forget targets")

        logger.info(f"Identified {len(forget_indices)} trajectories to forget")

    elif cfg.method == "strategy_inversion":
        unlearning_method = TrajectorySelectiveForgetting(
            agent=agent,
            forget_strength=cfg.get("negative_signal", {}).get("strength", 0.1),
            loss_decomposition=True,
            target_selection_method="similarity",
            target_selection_threshold=0.8,
            gradient_reversal=True,
            device=str(device),
        )

        strategy_inv = StrategyInversion(
            agent=agent,
            env=env,
            num_seeds=cfg.num_seeds,
            num_iterations=cfg.num_iterations,
            learning_rate=cfg.get("search_lr", 0.01),
            search_method=cfg.search.method,
            temperature=cfg.search.get("temperature", 1.0),
            noise_scale=cfg.search.get("noise_scale", 0.1),
            device=str(device),
        )

        # Generate synthetic forget states
        logger.info("Generating synthetic forget states...")
        forget_states = strategy_inv.generate_forget_states()
        logger.info(f"Generated {len(forget_states)} forget states")

        # For evaluation: identify closest real trajectories to the forget states
        if trajectories and forget_states:
            forget_indices = _match_trajectories_to_states(
                trajectories, forget_states, max_matches=max(1, int(len(trajectories) * 0.1))
            )
            logger.info(f"Matched {len(forget_indices)} trajectories near forget states")

    elif cfg.method == "retain_protection":
        # Retain protection wraps another unlearning method (trajectory_selective by default)
        # Use higher forget_strength here since masks will protect retained knowledge
        unlearning_method = TrajectorySelectiveForgetting(
            agent=agent,
            forget_strength=cfg.get("forget_strength", 0.3),
            loss_decomposition=True,
            target_selection_method="similarity",
            target_selection_threshold=0.8,
            gradient_reversal=True,
            device=str(device),
        )

        retain_protection = RetainProtection(
            agent=agent,
            metaplasticity_enabled=cfg.metaplasticity.enabled,
            mask_type=cfg.metaplasticity.mask_type,
            mask_threshold=cfg.metaplasticity.threshold,
            consolidation_strength=cfg.metaplasticity.get("consolidation_strength", 0.9),
            distillation_enabled=cfg.distillation.enabled,
            distillation_temperature=cfg.distillation.temperature,
            distillation_alpha=cfg.distillation.alpha,
            device=str(device),
        )
        logger.info("Retain protection initialized (wrapping trajectory_selective)")

        # Identify forget targets (same as trajectory_selective)
        if trajectories:
            traj_returns = [sum(t["rewards"]) for t in trajectories]
            num_forget = max(1, int(len(trajectories) * 0.1))
            forget_indices = list(np.argsort(traj_returns)[-num_forget:])
            logger.info(f"Using top-{num_forget} reward trajectories as forget targets")

    else:
        logger.error(f"Unknown unlearning method: {cfg.method}")
        return

    # Build retain indices (all trajectories not in the forget set)
    all_indices = set(range(len(trajectories)))
    retain_indices = sorted(all_indices - set(forget_indices))
    logger.info(f"Retain set: {len(retain_indices)} trajectories")

    # Compute importance masks from retain data before unlearning starts
    if retain_protection is not None and retain_protection.metaplasticity_enabled and retain_indices:
        logger.info("Computing importance masks from retain data...")
        retain_batches = []
        for i in range(0, len(retain_indices), cfg.batch_size):
            batch_idx = retain_indices[i:i + cfg.batch_size]
            batch = sample_batch_from_trajectories(
                trajectories, batch_idx, cfg.batch_size, cfg.action_type, device
            )
            retain_batches.append(batch)

        importances = retain_protection.compute_importance_scores(retain_batches)
        retain_protection.update_masks(importances)
        logger.info("Importance masks computed and applied")

    # Initialize metrics
    metrics_tracker = UnlearningMetrics()

    # Evaluate baseline performance
    logger.info("Evaluating baseline performance...")
    num_eval_episodes = cfg.training.num_eval_episodes
    baseline_returns = evaluate_agent(agent, env, num_eval_episodes)
    metrics_tracker.set_baseline(baseline_returns)
    logger.info(f"Baseline performance: {np.mean(baseline_returns):.2f} ± {np.std(baseline_returns):.2f}")

    # Compute baseline forget scores for tracking progress
    baseline_forget_scores = []
    if trajectories and forget_indices:
        baseline_forget_scores = compute_forget_scores(agent, trajectories, forget_indices, device)
        logger.info(f"Baseline forget score: {np.mean(baseline_forget_scores):.4f}")

    # ---- Unlearning loop ----
    logger.info("Starting unlearning process...")

    num_unlearn_steps = cfg.get("num_unlearn_steps", 500)
    batch_size = cfg.batch_size
    eval_frequency = cfg.get("eval_frequency", 50)
    retain_stability_threshold = cfg.get("retain_stability_threshold", 0.5)
    best_selectivity = 0.0
    best_state_dict = None
    stopped_early = False
    all_metrics_history = []

    for step in tqdm(range(num_unlearn_steps), desc="Unlearning"):
        step_metrics = {}

        if cfg.method == "strategy_inversion" and forget_states:
            # Build forget batch from synthetic states with current policy actions
            n_states = min(batch_size, len(forget_states))
            state_indices = np.random.choice(len(forget_states), size=n_states, replace=True)
            batch_states = np.array([forget_states[i] for i in state_indices])
            obs_tensor = torch.as_tensor(batch_states, dtype=torch.float32).to(device)

            # Get current policy's actions for these states (to unlearn them)
            with torch.no_grad():
                actions, _, _, _ = agent.network.get_action_and_value(obs_tensor)

            forget_batch = {
                "observations": obs_tensor,
                "actions": actions.to(device),
                "rewards": torch.zeros(len(batch_states)).to(device),
            }

            # Sample retain batch
            retain_batch = None
            if retain_indices:
                retain_batch = sample_batch_from_trajectories(
                    trajectories, retain_indices, batch_size, cfg.action_type, device
                )

            step_metrics = unlearning_method.unlearn_step(forget_batch, retain_batch)

        elif cfg.method == "retain_protection" and forget_indices and trajectories:
            # Sample forget batch
            forget_batch = sample_batch_from_trajectories(
                trajectories, forget_indices, batch_size, cfg.action_type, device
            )

            # Sample retain batch
            retain_batch = None
            if retain_indices:
                retain_batch = sample_batch_from_trajectories(
                    trajectories, retain_indices, batch_size, cfg.action_type, device
                )

            # Use protected_update which handles masking + distillation + unlearning
            step_metrics = retain_protection.protected_update(
                forget_batch, retain_batch, unlearning_method=unlearning_method
            )

        elif cfg.method == "trajectory_selective" and forget_indices and trajectories:
            # Sample forget batch
            forget_batch = sample_batch_from_trajectories(
                trajectories, forget_indices, batch_size, cfg.action_type, device
            )

            # Sample retain batch
            retain_batch = None
            if retain_indices:
                retain_batch = sample_batch_from_trajectories(
                    trajectories, retain_indices, batch_size, cfg.action_type, device
                )

            step_metrics = unlearning_method.unlearn_step(forget_batch, retain_batch)

        # Periodic evaluation
        if step % eval_frequency == 0:
            eval_returns = evaluate_agent(agent, env, num_eval_episodes)

            # Compute forget scores
            forget_scores = []
            if trajectories and forget_indices:
                forget_scores = compute_forget_scores(
                    agent, trajectories, forget_indices, device
                )

            # Compute metrics
            if forget_scores:
                metrics = metrics_tracker.compute_all_metrics(
                    forget_scores=forget_scores,
                    retain_returns=eval_returns,
                    baseline_returns=baseline_returns,
                )
                metrics.update(step_metrics)
                all_metrics_history.append({"step": step, **metrics})

                logger.info(
                    f"Step {step} | "
                    f"Forget Eff: {metrics['forget_effectiveness']:.3f} | "
                    f"Retain Stab: {metrics['retain_stability_index']:.3f} | "
                    f"Selectivity: {metrics['selectivity']:.3f} | "
                    f"Mean Forget Score: {metrics['mean_forget_score']:.4f}"
                )

                if wandb_run:
                    import wandb
                    wandb.log(
                        {f"unlearn/{k}": v for k, v in metrics.items()},
                        step=step,
                    )

                # Track best model by selectivity (forget effectiveness * retain stability)
                if metrics["selectivity"] > best_selectivity:
                    best_selectivity = metrics["selectivity"]
                    best_state_dict = {
                        k: v.clone() for k, v in agent.network.state_dict().items()
                    }

                # Early stopping: if retain stability drops below threshold,
                # stop and restore the best model found so far
                if (
                    step > 0
                    and metrics["retain_stability_index"] < retain_stability_threshold
                ):
                    logger.warning(
                        f"Early stopping at step {step}: retain stability "
                        f"{metrics['retain_stability_index']:.3f} < "
                        f"{retain_stability_threshold:.3f}"
                    )
                    stopped_early = True
                    break

        # Update retain protection masks periodically
        if (
            retain_protection is not None
            and retain_protection.metaplasticity_enabled
            and step > 0
            and step % cfg.metaplasticity.get("update_frequency", 1000) == 0
            and retain_indices
        ):
            logger.info(f"Step {step}: updating importance masks...")
            retain_batches = []
            for i in range(0, min(len(retain_indices), batch_size * 4), batch_size):
                batch_idx = retain_indices[i:i + batch_size]
                batch = sample_batch_from_trajectories(
                    trajectories, batch_idx, batch_size, cfg.action_type, device
                )
                retain_batches.append(batch)
            importances = retain_protection.compute_importance_scores(retain_batches)
            retain_protection.update_masks(importances)

    # Restore best model if we stopped early or if current model is worse
    if best_state_dict is not None and stopped_early:
        agent.network.load_state_dict(best_state_dict)
        logger.info(f"Restored best model (selectivity={best_selectivity:.4f})")

    # ---- Final evaluation ----
    logger.info("Final evaluation...")
    final_returns = evaluate_agent(agent, env, num_eval_episodes * 2)
    logger.info(f"Final performance: {np.mean(final_returns):.2f} ± {np.std(final_returns):.2f}")

    # Final forget scores
    if trajectories and forget_indices:
        final_forget_scores = compute_forget_scores(agent, trajectories, forget_indices, device)
        final_metrics = metrics_tracker.compute_all_metrics(
            forget_scores=final_forget_scores,
            retain_returns=final_returns,
            baseline_returns=baseline_returns,
        )
        logger.info("--- Final Metrics ---")
        for k, v in final_metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        if wandb_run:
            import wandb
            wandb.log({f"unlearn/final/{k}": v for k, v in final_metrics.items()})

    # Save unlearned model (named by method for multi-method comparison)
    unlearned_path = f"{cfg.training.checkpoint_dir}/unlearned_{cfg.method}.pt"
    Path(unlearned_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(unlearned_path)
    logger.info(f"Saved unlearned model to {unlearned_path}")

    # Save metrics history
    if all_metrics_history:
        import json
        metrics_path = f"{cfg.output_dir}/unlearning_metrics_{cfg.method}.json"
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(all_metrics_history, f, indent=2)
        logger.info(f"Saved metrics history to {metrics_path}")

    env.close()

    if wandb_run:
        import wandb
        wandb.finish()
        logger.info("WandB run finished.")


def _match_trajectories_to_states(trajectories, forget_states, max_matches=10):
    """Find trajectory indices whose observations are closest to the forget states."""
    forget_states_arr = np.array(forget_states)
    mean_forget = forget_states_arr.mean(axis=0)

    # Score each trajectory by how close its mean observation is to the forget region
    scores = []
    for i, traj in enumerate(trajectories):
        mean_obs = np.mean(traj["observations"], axis=0)
        dist = np.linalg.norm(mean_obs - mean_forget)
        scores.append((dist, i))

    scores.sort()
    return [idx for _, idx in scores[:max_matches]]


if __name__ == "__main__":
    unlearn()
