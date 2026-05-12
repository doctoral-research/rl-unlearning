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
from scenarios import ForgetScenario
from utils import set_seed, get_device, setup_logger, record_videos


def init_wandb(cfg: DictConfig):
    """Initialize WandB if configured."""
    if cfg.get("backend") == "wandb" and cfg.get("wandb", {}).get("enabled", False):
        import wandb
        scenario_name = cfg.get("scenario", "none")
        if scenario_name and "/" in str(scenario_name):
            scenario_name = str(scenario_name).split("/")[-1]
        run_name = f"eval/{cfg.env_name}/{scenario_name}/seed{cfg.seed}"
        group = cfg.wandb.get("group", None) or f"{cfg.env_name}/{scenario_name}"
        tags = list(cfg.wandb.get("tags", []))
        tags.extend([cfg.env_name, "eval"])
        if scenario_name and scenario_name != "none":
            tags.append(scenario_name)
        run = wandb.init(
            project=cfg.wandb.get("project", "rl-unlearning"),
            entity=cfg.wandb.get("entity", None),
            name=run_name,
            group=group,
            job_type="evaluate",
            tags=tags,
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


def evaluate_agent_with_trajectories(agent, env, num_episodes, deterministic=True, seed=None):
    """Run evaluation episodes, recording full trajectories (obs, actions, rewards).

    Returns (returns, trajectories) where trajectories is a list of dicts
    with keys: observations, actions, rewards.
    """
    returns = []
    trajectories = []
    for i in range(num_episodes):
        reset_kwargs = {"seed": seed + i} if seed is not None else {}
        obs, _ = env.reset(**reset_kwargs)
        episode_reward = 0
        done = False
        ep_obs, ep_actions, ep_rewards = [obs.copy()], [], []
        while not done:
            action, _ = agent.select_action(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_obs.append(obs.copy())
            ep_actions.append(action)
            ep_rewards.append(reward)
            episode_reward += reward
            done = terminated or truncated
        returns.append(episode_reward)
        trajectories.append({
            "observations": np.array(ep_obs),
            "actions": np.array(ep_actions),
            "rewards": np.array(ep_rewards),
        })
    return returns, trajectories


def compute_scenario_metrics(scenario, eval_trajectories):
    """Compute scenario-aware behavioral metrics from evaluation trajectories.

    For step-level scenarios (any_step, all_steps, proportion):
      - forget_region_fraction: fraction of all steps that match the forget condition
      - per_episode_match_rate: list of per-episode match fractions

    For trajectory-level scenarios (top_percentile, trajectory):
      - forget_traj_fraction: fraction of trajectories that match forget criteria

    For action-conditioned scenarios (conditions with target=action):
      - conditioned_action_rate: P(forbidden action | state condition holds)

    Returns a dict of metric_name -> value.
    """
    metrics = {}
    n_episodes = len(eval_trajectories)
    if n_episodes == 0:
        return metrics

    has_step_conditions = scenario.match_mode in ("any_step", "all_steps", "proportion")
    has_action_conditions = _scenario_has_action_conditions(scenario)

    if has_step_conditions:
        total_steps = 0
        matching_steps = 0
        per_episode_rates = []

        # For action-conditioned: count state-condition-only matches and full matches
        state_only_steps = 0
        full_match_steps = 0

        for traj in eval_trajectories:
            obs = traj["observations"]
            actions = traj["actions"]
            ep_matches = 0
            ep_state_matches = 0
            ep_full_matches = 0
            n_steps = len(actions)  # obs has n_steps+1 entries

            for t in range(n_steps):
                total_steps += 1
                act = actions[t]
                if scenario._step_matches(obs[t], act):
                    matching_steps += 1
                    ep_matches += 1

                if has_action_conditions:
                    # Check state conditions only (ignore action conditions)
                    if _state_only_matches(scenario, obs[t]):
                        state_only_steps += 1
                        ep_state_matches += 1
                        # Check full match (state + action)
                        if scenario._step_matches(obs[t], act):
                            full_match_steps += 1
                            ep_full_matches += 1

            per_episode_rates.append(ep_matches / max(n_steps, 1))

        metrics["forget_region_fraction"] = matching_steps / max(total_steps, 1)
        metrics["per_episode_match_rate_mean"] = np.mean(per_episode_rates)
        metrics["per_episode_match_rate_std"] = np.std(per_episode_rates)

        if has_action_conditions:
            if state_only_steps > 0:
                metrics["conditioned_action_rate"] = full_match_steps / state_only_steps
            else:
                metrics["conditioned_action_rate"] = 0.0
            metrics["state_trigger_fraction"] = state_only_steps / max(total_steps, 1)

        # Per-dimension stats for the first group's first state condition
        _add_dimension_stats(scenario, eval_trajectories, metrics)

    # Trajectory-level match fraction (works for all modes)
    forget_indices = scenario.identify_forget_trajectories(eval_trajectories)
    metrics["forget_traj_fraction"] = len(forget_indices) / max(n_episodes, 1)

    # Top-percentile specific: return distribution stats
    if scenario.match_mode == "top_percentile":
        returns = [sum(t["rewards"]) for t in eval_trajectories]
        metrics["return_mean"] = float(np.mean(returns))
        metrics["return_std"] = float(np.std(returns))
        metrics["return_max"] = float(np.max(returns))
        metrics["return_p90"] = float(np.percentile(returns, 90))

    return metrics


def _scenario_has_action_conditions(scenario):
    """Check if any group has action-targeted conditions."""
    for group in scenario.groups:
        for cond in group.get("conditions", []):
            if cond.get("target") == "action":
                return True
    return False


def _state_only_matches(scenario, obs):
    """Check if observation matches state conditions of any group (ignoring action conditions)."""
    from scenarios import _OPS
    for group in scenario.groups:
        if "conditions" not in group:
            continue
        group_ok = True
        for cond in group["conditions"]:
            if cond.get("target", "state") != "state":
                continue  # skip action conditions
            val = float(obs[cond["dim"]])
            op_fn = _OPS.get(cond["op"])
            if op_fn is None or not op_fn(val, cond["value"]):
                group_ok = False
                break
        if group_ok:
            return True
    return False


def _add_dimension_stats(scenario, eval_trajectories, metrics):
    """Add per-dimension statistics for the primary scenario dimensions."""
    for group in scenario.groups:
        for cond in group.get("conditions", []):
            if cond.get("target", "state") != "state":
                continue
            dim = cond["dim"]
            dim_name = cond.get("dim_name", f"dim{dim}")
            all_vals = []
            for traj in eval_trajectories:
                obs = traj["observations"]
                # obs has one more entry than actions; use all
                all_vals.extend(obs[:, dim].tolist())
            if all_vals:
                arr = np.array(all_vals)
                metrics[f"{dim_name}_mean"] = float(np.mean(arr))
                metrics[f"{dim_name}_std"] = float(np.std(arr))
                # Fraction of steps satisfying this condition
                from scenarios import _OPS
                op_fn = _OPS.get(cond["op"])
                if op_fn:
                    frac = np.mean([op_fn(v, cond["value"]) for v in all_vals])
                    metrics[f"{dim_name}_condition_frac"] = float(frac)
            return  # only first state dimension


def compute_scenario_forget_effectiveness(scenario, baseline_metrics, unlearned_metrics):
    """Compute scenario-aware forget effectiveness, dispatched by scenario type.

    Each scenario type uses the most meaningful primary metric:

    - Step-level with action conditions (no_right_push):
        Uses conditioned_action_rate — P(forbidden action | triggering state).
        Reduction in this rate is the truest measure of action unlearning.

    - Step-level state-only (left_only, centered, slow_swing, no_tilt, ...):
        Uses forget_region_fraction — % of steps in the forbidden region.

    - top_percentile (high_reward):
        Uses return_p90 drop relative to baseline. The goal is to degrade
        peak performance, so we measure how much the 90th-percentile return fell.

    - trajectory-level:
        Uses forget_traj_fraction — % of trajectories matching forget criteria.

    All return a value in [0, 1] where 1 = complete forgetting.
    """
    has_action_cond = _scenario_has_action_conditions(scenario)

    # 1. Action-conditioned scenarios: use conditioned_action_rate
    if has_action_cond:
        key = "conditioned_action_rate"
        if key in baseline_metrics and key in unlearned_metrics:
            base_rate = baseline_metrics[key]
            unlearned_rate = unlearned_metrics[key]
            if base_rate > 1e-6:
                return max(0.0, (base_rate - unlearned_rate) / base_rate)

    # 2. Step-level state-only scenarios: use forget_region_fraction
    if scenario.match_mode in ("any_step", "all_steps", "proportion"):
        key = "forget_region_fraction"
        if key in baseline_metrics and key in unlearned_metrics:
            base_frac = baseline_metrics[key]
            unlearned_frac = unlearned_metrics[key]
            if base_frac > 1e-6:
                return max(0.0, (base_frac - unlearned_frac) / base_frac)

    # 3. Top-percentile scenarios: use return_p90 degradation
    if scenario.match_mode == "top_percentile":
        if "return_p90" in baseline_metrics and "return_p90" in unlearned_metrics:
            base_p90 = baseline_metrics["return_p90"]
            unlearned_p90 = unlearned_metrics["return_p90"]
            scale = max(abs(base_p90), 1.0)
            return max(0.0, min(1.0, (base_p90 - unlearned_p90) / scale))

    # 4. Trajectory-level fallback: use forget_traj_fraction
    key = "forget_traj_fraction"
    if key in baseline_metrics and key in unlearned_metrics:
        base_frac = baseline_metrics[key]
        unlearned_frac = unlearned_metrics[key]
        if base_frac > 1e-6:
            return max(0.0, (base_frac - unlearned_frac) / base_frac)

    return 0.0


def _log_scenario_summary(logger, model_name, sm):
    """Log a one-line scenario summary for a model, handling missing keys."""
    parts = [f"  {model_name}:"]
    for key, label in [
        ("forget_region_fraction", "region_frac"),
        ("conditioned_action_rate", "action_rate"),
        ("forget_traj_fraction", "traj_frac"),
        ("scenario_forget_effectiveness", "scn_forget_eff"),
        ("return_p90", "return_p90"),
    ]:
        if key in sm:
            parts.append(f"{label}={sm[key]:.4f}")
    logger.info(" ".join(parts))


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

    # Create environment (routes MiniGrid envs through the flat-pos wrapper)
    from utils.env_wrappers import make_env
    env = make_env(
        cfg.env_id, seed=cfg.seed,
        obs_encoding=cfg.get("obs_encoding", "image"),
        fixed_goal_pos=cfg.get("fixed_goal_pos", None),
    )

    # ---- Load baseline model ----
    baseline_dir = Path(cfg.get("baseline_dir", cfg.training.checkpoint_dir))
    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    baseline_path = baseline_dir / "final_model.pt"

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
    baseline_output = cfg.get("baseline_output_dir", cfg.output_dir)
    traj_path = f"{baseline_output}/trajectories.pkl"
    if Path(traj_path).exists():
        with open(traj_path, "rb") as f:
            trajectories = pickle.load(f)
        logger.info(f"Loaded {len(trajectories)} trajectories")
    else:
        trajectories = []
        logger.warning("No trajectories found for forget score evaluation")

    # ---- Load scenario (if configured) ----
    scenario = ForgetScenario.from_config(cfg)
    if scenario:
        logger.info(f"Scenario loaded: {scenario.name} ({scenario.match_mode})")

    # ---- Evaluate all models ----
    num_eval_episodes = 50

    results = {}
    scenario_results = {}  # model_name -> scenario metrics dict

    # Baseline (seeded for reproducibility)
    eval_seed = cfg.seed * 1000
    logger.info("Evaluating baseline agent...")
    baseline_returns, baseline_eval_trajs = evaluate_agent_with_trajectories(
        baseline_agent, env, num_eval_episodes, seed=eval_seed
    )
    baseline_forget_scores = compute_forget_scores(baseline_agent, trajectories, device) if trajectories else []
    results["baseline"] = {
        "returns": baseline_returns,
        "forget_scores": baseline_forget_scores,
    }
    logger.info(f"  Baseline return: {np.mean(baseline_returns):.2f} ± {np.std(baseline_returns):.2f}")

    if scenario:
        baseline_scenario_metrics = compute_scenario_metrics(scenario, baseline_eval_trajs)
        scenario_results["baseline"] = baseline_scenario_metrics
        _log_scenario_summary(logger, "Baseline", baseline_scenario_metrics)

    # Each unlearned method
    metrics_tracker = UnlearningMetrics()
    all_metrics = {}

    for method_name, agent in methods.items():
        logger.info(f"Evaluating {method_name}...")
        method_returns, method_eval_trajs = evaluate_agent_with_trajectories(
            agent, env, num_eval_episodes, seed=eval_seed
        )
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

        # Scenario-aware metrics
        if scenario:
            method_scenario_metrics = compute_scenario_metrics(scenario, method_eval_trajs)
            scenario_forget_eff = compute_scenario_forget_effectiveness(
                scenario, baseline_scenario_metrics, method_scenario_metrics
            )
            method_scenario_metrics["scenario_forget_effectiveness"] = scenario_forget_eff
            # Scenario selectivity: scenario_forget_eff * retain_stability
            retain_stab = all_metrics.get(method_name, {}).get("retain_stability_index", 1.0)
            method_scenario_metrics["scenario_selectivity"] = scenario_forget_eff * retain_stab
            scenario_results[method_name] = method_scenario_metrics
            _log_scenario_summary(logger, method_name, method_scenario_metrics)

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

    # Original return-based metrics table
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

    # Scenario-aware metrics table
    if scenario and scenario_results:
        logger.info("\n" + "=" * 70)
        logger.info(f"  SCENARIO-AWARE METRICS — {scenario.name}")
        logger.info("=" * 70)

        # Determine which columns to show
        has_action_cond = any("conditioned_action_rate" in m for m in scenario_results.values())
        first_dim_key = None
        for m in scenario_results.values():
            for k in m:
                if k.endswith("_condition_frac"):
                    first_dim_key = k
                    break
            if first_dim_key:
                break

        header_parts = [f"{'Model':<25}", f"{'Forget Frac':<13}", f"{'Traj Frac':<11}"]
        if has_action_cond:
            header_parts.append(f"{'Act Rate':<10}")
        if first_dim_key:
            header_parts.append(f"{first_dim_key:<20}")
        # Only show eff/sel for non-baseline
        header_parts.extend([f"{'Scn Forget':<12}", f"{'Scn Select':<12}"])
        logger.info(" ".join(header_parts))
        logger.info("-" * 100)

        for model_name, sm in scenario_results.items():
            parts = [f"{model_name:<25}"]
            parts.append(f"{sm.get('forget_region_fraction', 0):<13.4f}")
            parts.append(f"{sm.get('forget_traj_fraction', 0):<11.4f}")
            if has_action_cond:
                parts.append(f"{sm.get('conditioned_action_rate', 0):<10.4f}")
            if first_dim_key:
                parts.append(f"{sm.get(first_dim_key, 0):<20.4f}")
            if model_name == "baseline":
                parts.extend(["  —         ", "  —         "])
            else:
                parts.append(f"{sm.get('scenario_forget_effectiveness', 0):<12.4f}")
                parts.append(f"{sm.get('scenario_selectivity', 0):<12.4f}")
            logger.info(" ".join(parts))

        # Also print dimension-level stats
        for model_name, sm in scenario_results.items():
            dim_stats = [(k, v) for k, v in sm.items() if k.endswith("_mean") or k.endswith("_std")]
            if dim_stats:
                logger.info(f"  {model_name}: " + ", ".join(f"{k}={v:.4f}" for k, v in dim_stats))

    logger.info("=" * 70)

    # ---- Log to WandB ----
    if wandb_run:
        import wandb
        summary = {
            "eval/baseline_return_mean": np.mean(baseline_returns),
            "eval/baseline_return_std": np.std(baseline_returns),
        }
        if "baseline" in scenario_results:
            for k, v in scenario_results["baseline"].items():
                if isinstance(v, (int, float)):
                    summary[f"eval/baseline/scenario/{k}"] = v
        for method_name, data in results.items():
            if method_name == "baseline":
                continue
            summary[f"eval/{method_name}/return_mean"] = np.mean(data["returns"])
            summary[f"eval/{method_name}/return_std"] = np.std(data["returns"])
            if method_name in all_metrics:
                for k, v in all_metrics[method_name].items():
                    summary[f"eval/{method_name}/{k}"] = v
            if method_name in scenario_results:
                for k, v in scenario_results[method_name].items():
                    if isinstance(v, (int, float)):
                        summary[f"eval/{method_name}/scenario/{k}"] = v
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

    # JSON with all metrics (include scenario metrics)
    combined_metrics = {}
    for method_name in all_metrics:
        combined_metrics[method_name] = dict(all_metrics[method_name])
        if method_name in scenario_results:
            for k, v in scenario_results[method_name].items():
                if isinstance(v, (int, float)):
                    combined_metrics[method_name][f"scenario_{k}"] = v
    # Also save baseline scenario metrics
    if "baseline" in scenario_results:
        combined_metrics["baseline_scenario"] = {
            k: v for k, v in scenario_results["baseline"].items()
            if isinstance(v, (int, float))
        }
    if combined_metrics:
        metrics_json = output_dir / "evaluation_metrics.json"
        with open(metrics_json, "w") as f:
            json.dump(combined_metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_json}")

    # Scenario metrics CSV (one row per model, easy to parse)
    if scenario_results:
        scenario_rows = []
        for model_name, sm in scenario_results.items():
            row = {"model": model_name}
            row.update({k: v for k, v in sm.items() if isinstance(v, (int, float))})
            scenario_rows.append(row)
        scenario_df = pd.DataFrame(scenario_rows)
        scenario_csv = output_dir / "scenario_metrics.csv"
        scenario_df.to_csv(scenario_csv, index=False)
        logger.info(f"Saved scenario metrics to {scenario_csv}")

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
