"""
Evaluation and comparison script.

Compares a baseline (trained) model against all unlearned variants found
in the same checkpoint directory.  Unlearned checkpoints are expected to
follow the naming convention ``unlearned_{method}.pt``.
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
import json
import matplotlib.pyplot as plt
import seaborn as sns

from agents import PPOAgent
from metrics import UnlearningMetrics, evaluate_trajectory_similarity
from utils import set_seed, get_device, setup_logger, record_videos


def init_wandb(cfg: DictConfig):
    """Initialize WandB if configured."""
    if cfg.get("backend") == "wandb" and cfg.get("wandb", {}).get("enabled", False):
        import wandb
        run_name = f"eval_{cfg.env_name}_seed{cfg.seed}"
        run = wandb.init(
            project=cfg.wandb.get("project", "rl-unlearning"),
            entity=cfg.wandb.get("entity", None),
            name=run_name,
            group=cfg.wandb.get("group", None),
            job_type="evaluate",
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


def compute_forget_scores(agent, trajectories, device, max_eval=20):
    """Compute trajectory similarity scores."""
    scores = []
    for traj in trajectories[:max_eval]:
        score = evaluate_trajectory_similarity(agent, traj, str(device))
        scores.append(score)
    return scores


def load_agent(cfg, device, checkpoint_path):
    """Create and load a PPOAgent from checkpoint."""
    agent = PPOAgent(
        observation_dim=cfg.observation_dim,
        action_dim=cfg.action_dim,
        action_type=cfg.action_type,
        hidden_dims=cfg.network.hidden_dims,
        device=str(device),
    )
    agent.load(checkpoint_path)
    return agent


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def evaluate(cfg: DictConfig):
    """Evaluate and compare baseline vs all unlearned models."""

    # Setup
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    logger = setup_logger("evaluate", log_file=f"{cfg.log_dir}/evaluate.log")
    logger.info(f"Experiment: {cfg.experiment_name}")

    # Initialize WandB
    wandb_run = init_wandb(cfg)
    if wandb_run:
        logger.info(f"WandB run: {wandb_run.url}")

    # Create environment
    env = gym.make(cfg.env_id)

    # ---- Load baseline model ----
    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    baseline_path = checkpoint_dir / "final_model.pt"

    if not baseline_path.exists():
        logger.error(f"Baseline model not found: {baseline_path}")
        return

    baseline_agent = load_agent(cfg, device, str(baseline_path))
    logger.info(f"Loaded baseline model from {baseline_path}")

    # ---- Discover all unlearned models ----
    unlearned_paths = sorted(checkpoint_dir.glob("unlearned_*.pt"))
    if not unlearned_paths:
        logger.warning(f"No unlearned models found in {checkpoint_dir}")
        logger.info("Expected files like: unlearned_trajectory_selective.pt, "
                     "unlearned_strategy_inversion.pt, unlearned_retain_protection.pt")
        logger.info("Run unlearn.py first, then re-run evaluate.py")
        return

    # Extract method names from filenames: unlearned_{method}.pt -> method
    methods = {}
    for path in unlearned_paths:
        method_name = path.stem.replace("unlearned_", "")
        methods[method_name] = load_agent(cfg, device, str(path))
        logger.info(f"Loaded unlearned model: {method_name} from {path}")

    logger.info(f"Found {len(methods)} unlearned model(s): {list(methods.keys())}")

    # ---- Load trajectories ----
    traj_path = f"{cfg.output_dir}/trajectories.pkl"
    if Path(traj_path).exists():
        with open(traj_path, "rb") as f:
            trajectories = pickle.load(f)
        logger.info(f"Loaded {len(trajectories)} trajectories")
    else:
        trajectories = []
        logger.warning("No trajectories found for forget score evaluation")

    # ---- Evaluate all models ----
    num_eval_episodes = 50

    results = {}

    # Baseline (seeded for reproducibility)
    eval_seed = cfg.seed * 1000
    logger.info("Evaluating baseline agent...")
    baseline_returns = evaluate_agent(baseline_agent, env, num_eval_episodes, seed=eval_seed)
    baseline_forget_scores = compute_forget_scores(baseline_agent, trajectories, device) if trajectories else []
    results["baseline"] = {
        "returns": baseline_returns,
        "forget_scores": baseline_forget_scores,
    }
    logger.info(f"  Baseline return: {np.mean(baseline_returns):.2f} ± {np.std(baseline_returns):.2f}")

    # Each unlearned method
    metrics_tracker = UnlearningMetrics()
    all_metrics = {}

    for method_name, agent in methods.items():
        logger.info(f"Evaluating {method_name}...")
        method_returns = evaluate_agent(agent, env, num_eval_episodes, seed=eval_seed)
        method_forget_scores = compute_forget_scores(agent, trajectories, device) if trajectories else []

        results[method_name] = {
            "returns": method_returns,
            "forget_scores": method_forget_scores,
        }

        logger.info(f"  {method_name} return: {np.mean(method_returns):.2f} ± {np.std(method_returns):.2f}")

        # Compute unlearning metrics (with baseline forget scores for relative effectiveness)
        if method_forget_scores:
            metrics = metrics_tracker.compute_all_metrics(
                forget_scores=method_forget_scores,
                retain_returns=method_returns,
                baseline_returns=baseline_returns,
                baseline_forget_scores=baseline_forget_scores,
            )
            all_metrics[method_name] = metrics

    # ---- Print comparison table ----
    logger.info("\n" + "=" * 70)
    logger.info(f"  EVALUATION RESULTS — {cfg.env_id} ({cfg.experiment_name})")
    logger.info("=" * 70)

    # Performance table
    header = f"{'Model':<25} {'Return (mean±std)':<22} {'Forget Score':<15}"
    logger.info(header)
    logger.info("-" * 62)

    for model_name, data in results.items():
        ret_str = f"{np.mean(data['returns']):.2f} ± {np.std(data['returns']):.2f}"
        fs_str = f"{np.mean(data['forget_scores']):.4f}" if data['forget_scores'] else "N/A"
        logger.info(f"{model_name:<25} {ret_str:<22} {fs_str:<15}")

    # Metrics table
    if all_metrics:
        logger.info("\n" + "-" * 62)
        logger.info(f"{'Method':<25} {'Forget Eff':<12} {'Retain Stab':<12} {'Selectivity':<12}")
        logger.info("-" * 62)
        for method_name, metrics in all_metrics.items():
            logger.info(
                f"{method_name:<25} "
                f"{metrics['forget_effectiveness']:<12.4f} "
                f"{metrics['retain_stability_index']:<12.4f} "
                f"{metrics['selectivity']:<12.4f}"
            )

    logger.info("=" * 70)

    # ---- Log to WandB ----
    if wandb_run:
        import wandb
        summary = {
            "eval/baseline_return_mean": np.mean(baseline_returns),
            "eval/baseline_return_std": np.std(baseline_returns),
        }
        for method_name, data in results.items():
            if method_name == "baseline":
                continue
            summary[f"eval/{method_name}/return_mean"] = np.mean(data["returns"])
            summary[f"eval/{method_name}/return_std"] = np.std(data["returns"])
            if method_name in all_metrics:
                for k, v in all_metrics[method_name].items():
                    summary[f"eval/{method_name}/{k}"] = v
        wandb.log(summary)

    # ---- Save results ----
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV with all returns
    rows = []
    for model_name, data in results.items():
        for ret in data["returns"]:
            rows.append({"model": model_name, "return": ret})
    results_df = pd.DataFrame(rows)
    results_csv = output_dir / "evaluation_results.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Saved results to {results_csv}")

    # JSON with all metrics
    if all_metrics:
        metrics_json = output_dir / "evaluation_metrics.json"
        with open(metrics_json, "w") as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_json}")

    # ---- Plots ----
    _plot_comparison(results, all_metrics, cfg, output_dir, logger)

    env.close()

    # ---- Record behavior videos for each model ----
    video_dir = output_dir / "videos"
    video_seed = cfg.seed * 2000
    all_video_paths = {}

    logger.info("Recording behavior videos...")
    # Baseline
    baseline_vids = record_videos(
        baseline_agent, cfg.env_id, str(video_dir),
        label="baseline", num_videos=3, seed=video_seed,
    )
    all_video_paths["baseline"] = baseline_vids

    # Each unlearned method
    for method_name, agent in methods.items():
        vids = record_videos(
            agent, cfg.env_id, str(video_dir),
            label=method_name, num_videos=3, seed=video_seed,
        )
        all_video_paths[method_name] = vids

    if wandb_run and any(all_video_paths.values()):
        import wandb
        for model_name, paths in all_video_paths.items():
            for vp in paths:
                wandb.log({f"videos/{model_name}": wandb.Video(vp, fps=30)})

    if wandb_run:
        import wandb
        wandb.finish()
        logger.info("WandB run finished.")


def _plot_comparison(results, all_metrics, cfg, output_dir, logger):
    """Generate comparison plots."""
    model_names = list(results.keys())
    n_models = len(model_names)

    if n_models < 2:
        return

    has_forget = any(results[m]["forget_scores"] for m in model_names)
    n_cols = 3 if (has_forget and all_metrics) else (2 if has_forget else 1)

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    fig.suptitle(f"Unlearning Comparison — {cfg.env_id}", fontsize=14, fontweight="bold")

    # 1. Returns box plot
    rows = []
    for model_name, data in results.items():
        for ret in data["returns"]:
            rows.append({"Model": model_name, "Episode Return": ret})
    returns_df = pd.DataFrame(rows)
    sns.boxplot(data=returns_df, x="Model", y="Episode Return", ax=axes[0])
    axes[0].set_title("Return Comparison")
    axes[0].tick_params(axis="x", rotation=30)

    # 2. Forget scores box plot
    if has_forget:
        rows = []
        for model_name, data in results.items():
            for score in data["forget_scores"]:
                rows.append({"Model": model_name, "Forget Score": score})
        forget_df = pd.DataFrame(rows)
        sns.boxplot(data=forget_df, x="Model", y="Forget Score", ax=axes[1])
        axes[1].set_title("Forget Score (Lower = More Forgotten)")
        axes[1].tick_params(axis="x", rotation=30)

    # 3. Metrics bar chart
    if all_metrics and n_cols >= 3:
        metric_keys = ["forget_effectiveness", "retain_stability_index", "selectivity"]
        metric_labels = ["Forget Eff.", "Retain Stab.", "Selectivity"]
        x = np.arange(len(metric_keys))
        width = 0.8 / len(all_metrics)

        for i, (method_name, metrics) in enumerate(all_metrics.items()):
            values = [metrics.get(k, 0) for k in metric_keys]
            axes[2].bar(x + i * width, values, width, label=method_name)

        axes[2].set_xticks(x + width * (len(all_metrics) - 1) / 2)
        axes[2].set_xticklabels(metric_labels)
        axes[2].set_ylim(0, 1.05)
        axes[2].set_title("Unlearning Metrics")
        axes[2].legend(fontsize=8)

    plt.tight_layout()
    plot_path = output_dir / "evaluation_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    evaluate()
