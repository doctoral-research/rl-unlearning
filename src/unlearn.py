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
from utils import set_seed, get_device, setup_logger, record_videos


def init_wandb(cfg: DictConfig):
    """Initialize WandB if configured."""
    if cfg.get("backend") == "wandb" and cfg.get("wandb", {}).get("enabled", False):
        import wandb
        run_name = f"unlearn_{cfg.env_name}_{cfg.method}_seed{cfg.seed}"
        run = wandb.init(
            project=cfg.wandb.get("project", "rl-unlearning"),
            entity=cfg.wandb.get("entity", None),
            name=run_name,
            group=cfg.wandb.get("group", None),
            job_type="unlearn",
            tags=list(cfg.wandb.get("tags", [])),
            mode=cfg.wandb.get("mode", "online"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        return run
    return None


def evaluate_agent(agent, env, num_episodes, deterministic=True, seed=None):
    """Run evaluation episodes and return list of episode returns."""
    returns = []
    for i in range(num_episodes):
        reset_kwargs = {"seed": seed + i} if seed is not None else {}
        obs, _ = env.reset(**reset_kwargs)
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
    """Sample a batch of transitions from the given trajectory indices.

    Pools all transitions from the specified trajectories and randomly
    samples ``batch_size`` transitions.  This gives different mini-batches
    each call even when the trajectory set is small.
    """
    # Pool all transitions from the designated trajectories
    obs_list, act_list, rew_list = [], [], []
    for idx in indices:
        traj = trajectories[idx]
        obs_list.extend(traj["observations"])
        act_list.extend(traj["actions"])
        rew_list.extend(traj["rewards"])

    n_transitions = len(obs_list)
    if n_transitions == 0:
        # Edge case: empty trajectories
        raise ValueError("No transitions found in the specified trajectory indices")

    # Randomly sample batch_size transitions (with replacement if needed)
    replace = n_transitions < batch_size
    chosen = np.random.choice(n_transitions, size=min(batch_size, n_transitions) if not replace else batch_size, replace=replace)
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
    if len(forget_indices) > max_eval:
        # Random subsample to avoid bias from index ordering
        eval_indices = list(np.random.choice(forget_indices, size=max_eval, replace=False))
    else:
        eval_indices = forget_indices
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

        # Use real trajectory observations as seeds instead of random OOD states
        if trajectories:
            strategy_inv.set_reference_states(trajectories)
            logger.info("Set reference states from training trajectories")

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

        # Identify forget targets using the same logic as trajectory_selective:
        # try toxic_states first, fall back to reward-based selection.
        toxic_states = cfg.get("toxic_states", None)
        if toxic_states:
            toxic_states = [np.array(s) for s in toxic_states]

        forget_indices = unlearning_method.identify_forget_trajectories(
            trajectories,
            toxic_states=toxic_states,
        )

        if not forget_indices and trajectories:
            traj_returns = [sum(t["rewards"]) for t in trajectories]
            num_forget = max(1, int(len(trajectories) * 0.1))
            forget_indices = list(np.argsort(traj_returns)[-num_forget:])
            logger.info(f"No toxic states provided; using top-{num_forget} reward trajectories as forget targets")
        else:
            logger.info(f"Identified {len(forget_indices)} trajectories to forget via toxic state matching")

    else:
        logger.error(f"Unknown unlearning method: {cfg.method}")
        return

    # Build retain indices (all trajectories not in the forget set)
    all_indices = set(range(len(trajectories)))
    retain_indices = sorted(all_indices - set(forget_indices))
    logger.info(f"Retain set: {len(retain_indices)} trajectories")

    # Pre-compute frozen original-policy actions for strategy_inversion
    # so the unlearning target doesn't shift as the policy changes.
    frozen_forget_actions = None
    if cfg.method == "strategy_inversion" and forget_states:
        logger.info("Pre-computing original policy actions for forget states...")
        all_states = np.array(forget_states)
        obs_tensor = torch.as_tensor(all_states, dtype=torch.float32).to(device)
        with torch.no_grad():
            # Use argmax (most likely action) rather than a stochastic sample.
            # Unlearning the policy's *preferred* action at each state gives a
            # stronger, more targeted signal than unlearning a random sample.
            action_output, _ = agent.network(obs_tensor)
            if agent.action_type == "discrete":
                frozen_forget_actions = torch.argmax(action_output, dim=-1).to(device)
            else:
                frozen_forget_actions = action_output.to(device)
        logger.info(f"Cached {len(frozen_forget_actions)} frozen actions")

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

    # Evaluate baseline performance (seeded for reproducibility)
    logger.info("Evaluating baseline performance...")
    num_eval_episodes = cfg.training.num_eval_episodes
    eval_seed = cfg.seed * 1000
    baseline_returns = evaluate_agent(agent, env, num_eval_episodes, seed=eval_seed)
    metrics_tracker.set_baseline(baseline_returns)
    logger.info(f"Baseline performance: {np.mean(baseline_returns):.2f} ± {np.std(baseline_returns):.2f}")

    # Record pre-unlearning behavior videos
    video_dir = f"{cfg.output_dir}/videos"
    video_seed = cfg.seed * 2000
    logger.info("Recording pre-unlearning behavior videos...")
    record_videos(
        agent, cfg.env_id, video_dir,
        label=f"before_{cfg.method}",
        num_videos=3, seed=video_seed,
    )

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
            # Build forget batch from synthetic states with FROZEN original actions.
            # Using the original policy's actions prevents the moving-target problem
            # where the policy oscillates as we unlearn its current action each step.
            n_states = min(batch_size, len(forget_states))
            state_indices = np.random.choice(len(forget_states), size=n_states, replace=True)
            batch_states = np.array([forget_states[i] for i in state_indices])
            obs_tensor = torch.as_tensor(batch_states, dtype=torch.float32).to(device)

            forget_batch = {
                "observations": obs_tensor,
                "actions": frozen_forget_actions[state_indices],
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
            eval_returns = evaluate_agent(agent, env, num_eval_episodes, seed=eval_seed)

            # Compute forget scores
            forget_scores = []
            if trajectories and forget_indices:
                forget_scores = compute_forget_scores(
                    agent, trajectories, forget_indices, device
                )

            # Compute metrics (with baseline forget scores for relative effectiveness)
            if forget_scores:
                metrics = metrics_tracker.compute_all_metrics(
                    forget_scores=forget_scores,
                    retain_returns=eval_returns,
                    baseline_returns=baseline_returns,
                    baseline_forget_scores=baseline_forget_scores,
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
    final_returns = evaluate_agent(agent, env, num_eval_episodes * 2, seed=eval_seed)
    logger.info(f"Final performance: {np.mean(final_returns):.2f} ± {np.std(final_returns):.2f}")

    # Final forget scores
    if trajectories and forget_indices:
        final_forget_scores = compute_forget_scores(agent, trajectories, forget_indices, device)
        final_metrics = metrics_tracker.compute_all_metrics(
            forget_scores=final_forget_scores,
            retain_returns=final_returns,
            baseline_returns=baseline_returns,
            baseline_forget_scores=baseline_forget_scores,
        )

        # Compute AUFC from the time series of mean forget scores
        if all_metrics_history:
            forget_score_series = [m["mean_forget_score"] for m in all_metrics_history]
            step_series = [m["step"] for m in all_metrics_history]
            final_metrics["aufc"] = metrics_tracker.compute_aufc(
                forget_score_series, step_series
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

    # Record post-unlearning behavior videos
    logger.info("Recording post-unlearning behavior videos...")
    post_paths = record_videos(
        agent, cfg.env_id, video_dir,
        label=f"after_{cfg.method}",
        num_videos=3, seed=video_seed,
    )
    if wandb_run and post_paths:
        import wandb
        for vp in post_paths:
            wandb.log({f"videos/after_{cfg.method}": wandb.Video(vp, fps=30)})

    if wandb_run:
        import wandb
        wandb.finish()
        logger.info("WandB run finished.")


def _match_trajectories_to_states(trajectories, forget_states, max_matches=10):
    """Find trajectory indices whose observations are closest to the forget states.

    Uses minimum per-state distance rather than mean-observation distance,
    so a trajectory that *passes through* the forget region is matched even
    if it spends most of its time elsewhere.
    """
    forget_states_arr = np.array(forget_states)  # (N_forget, obs_dim)

    scores = []
    for i, traj in enumerate(trajectories):
        obs = np.array(traj["observations"])  # (T, obs_dim)
        # Minimum distance from any observation in the trajectory to any forget state
        # Efficient: compute pairwise distances between obs and forget_states
        # Use broadcasting: (T,1,D) - (1,N,D) -> (T,N,D) -> (T,N) -> scalar
        diffs = obs[:, None, :] - forget_states_arr[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)  # (T, N_forget)
        min_dist = dists.min()
        scores.append((min_dist, i))

    scores.sort()
    return [idx for _, idx in scores[:max_matches]]


if __name__ == "__main__":
    unlearn()
