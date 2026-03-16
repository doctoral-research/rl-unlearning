"""
Comprehensive plotting for RL unlearning experiments.

Usage:
    # Plot a single experiment folder
    python src/plot_results.py experiments/outputs/cartpole_ppo_seed42

    # Plot by environment name (finds matching folders)
    python src/plot_results.py --env cartpole

    # Plot all environments
    python src/plot_results.py --all

    # Cross-environment comparison
    python src/plot_results.py --compare
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

# ── Style ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
METHOD_COLORS = {
    "trajectory_selective": "#2196F3",
    "strategy_inversion": "#FF9800",
    "retain_protection": "#4CAF50",
    "baseline": "#9E9E9E",
}
METHOD_LABELS = {
    "trajectory_selective": "Trajectory Selective",
    "strategy_inversion": "Strategy Inversion",
    "retain_protection": "Retain Protection",
    "baseline": "Baseline",
}
METHODS = ["trajectory_selective", "strategy_inversion", "retain_protection"]


def _label(method):
    return METHOD_LABELS.get(method, method)


def _color(method):
    return METHOD_COLORS.get(method, "#666666")


# ── Data loading ───────────────────────────────────────────────────────────

def load_experiment(folder: Path):
    """Load all data from an experiment output folder."""
    data = {"folder": folder, "name": folder.name}

    # Evaluation metrics (final)
    metrics_path = folder / "evaluation_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            data["eval_metrics"] = json.load(f)

    # Evaluation returns (per-episode)
    csv_path = folder / "evaluation_results.csv"
    if csv_path.exists():
        data["eval_returns"] = pd.read_csv(csv_path)

    # Unlearning progress (per-step, per-method)
    data["unlearn_progress"] = {}
    for method in METHODS:
        progress_path = folder / f"unlearning_metrics_{method}.json"
        if progress_path.exists():
            with open(progress_path) as f:
                data["unlearn_progress"][method] = json.load(f)

    # Training curve from log
    log_dir = folder.parent.parent / "logs" / folder.name
    train_log = log_dir / "train.log"
    if train_log.exists():
        data["train_curve"] = _parse_train_log(train_log)

    return data


def _parse_train_log(log_path: Path):
    """Parse episode-level stats from train.log."""
    pattern = re.compile(
        r"Episode (\d+) \| Step (\d+) \| Mean Reward: ([\d\.\-]+)"
    )
    episodes, steps, rewards = [], [], []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                episodes.append(int(m.group(1)))
                steps.append(int(m.group(2)))
                rewards.append(float(m.group(3)))
    if episodes:
        return {"episode": episodes, "step": steps, "mean_reward": rewards}
    return None


def find_experiment_folders(base_dir: Path, env_name: str = None):
    """Find experiment output folders, optionally filtering by env."""
    outputs_dir = base_dir / "experiments" / "outputs"
    if not outputs_dir.exists():
        return []
    folders = sorted(d for d in outputs_dir.iterdir() if d.is_dir())
    if env_name:
        folders = [d for d in folders if d.name.startswith(env_name)]
    return folders


# ── Individual experiment plots ────────────────────────────────────────────

def plot_experiment(data: dict, save_dir: Path = None):
    """Generate all plots for a single experiment."""
    if save_dir is None:
        save_dir = data["folder"]
    save_dir.mkdir(parents=True, exist_ok=True)

    name = data["name"]
    env_label = name.split("_ppo")[0].replace("_", " ").title()

    plots_made = []

    # 1. Training curve
    if data.get("train_curve"):
        _plot_training_curve(data["train_curve"], env_label, save_dir)
        plots_made.append("training_curve.png")

    # 2. Return distributions
    if data.get("eval_returns") is not None:
        _plot_return_distributions(data["eval_returns"], env_label, save_dir)
        plots_made.append("return_distributions.png")

    # 3. Metrics radar chart
    if data.get("eval_metrics"):
        _plot_metrics_radar(data["eval_metrics"], env_label, save_dir)
        plots_made.append("metrics_radar.png")

    # 4. Metrics bar chart
    if data.get("eval_metrics"):
        _plot_metrics_bars(data["eval_metrics"], env_label, save_dir)
        plots_made.append("metrics_bars.png")

    # 5. Unlearning dynamics (per-method over steps)
    if data.get("unlearn_progress"):
        _plot_unlearning_dynamics(data["unlearn_progress"], env_label, save_dir)
        plots_made.append("unlearning_dynamics.png")

    # 6. Loss curves during unlearning
    if data.get("unlearn_progress"):
        _plot_loss_curves(data["unlearn_progress"], env_label, save_dir)
        plots_made.append("loss_curves.png")

    # 7. Forget-retain tradeoff
    if data.get("unlearn_progress"):
        _plot_forget_retain_tradeoff(data["unlearn_progress"], env_label, save_dir)
        plots_made.append("forget_retain_tradeoff.png")

    # 8. Combined summary dashboard
    _plot_dashboard(data, env_label, save_dir)
    plots_made.append("dashboard.png")

    return plots_made


def _plot_training_curve(train_data, env_label, save_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_data["step"], train_data["mean_reward"],
            color="#1976D2", linewidth=2)
    ax.fill_between(train_data["step"], train_data["mean_reward"],
                    alpha=0.15, color="#1976D2")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Episode Return")
    ax.set_title(f"Training Curve — {env_label}")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.tight_layout()
    plt.savefig(save_dir / "training_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_return_distributions(df, env_label, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Box plot
    order = ["baseline"] + [m for m in METHODS if m in df["model"].values]
    palette = {m: _color(m) for m in order}
    sns.boxplot(data=df, x="model", y="return", hue="model", order=order,
                palette=palette, ax=axes[0], fliersize=3, legend=False)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Episode Return")
    axes[0].set_title("Return Distribution")
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels([_label(m) for m in order], rotation=25, ha="right")

    # Violin plot
    sns.violinplot(data=df, x="model", y="return", hue="model", order=order,
                   palette=palette, ax=axes[1], inner="quartile", cut=0, legend=False)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Episode Return")
    axes[1].set_title("Return Density")
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels([_label(m) for m in order], rotation=25, ha="right")

    fig.suptitle(f"Evaluation Returns — {env_label}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "return_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_metrics_radar(metrics, env_label, save_dir):
    categories = ["Forget\nEffectiveness", "Retain\nStability", "Selectivity", "AUFC\n(inverted)"]
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for method in METHODS:
        if method not in metrics:
            continue
        m = metrics[method]
        # Invert AUFC so lower (better forgetting) = higher on radar
        values = [
            m.get("forget_effectiveness", 0),
            m.get("retain_stability_index", 0),
            m.get("selectivity", 0),
            1.0 - m.get("aufc", 0),
        ]
        values += values[:1]
        ax.plot(angles, values, "-o", label=_label(method),
                color=_color(method), linewidth=2, markersize=5)
        ax.fill(angles, values, alpha=0.1, color=_color(method))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Unlearning Metrics — {env_label}", pad=20, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(save_dir / "metrics_radar.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_metrics_bars(metrics, env_label, save_dir):
    metric_keys = [
        ("forget_effectiveness", "Forget Effectiveness"),
        ("retain_stability_index", "Retain Stability Index"),
        ("selectivity", "Selectivity"),
        ("aufc", "AUFC (lower=better)"),
    ]
    methods_present = [m for m in METHODS if m in metrics]
    n_metrics = len(metric_keys)

    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    for ax, (key, label) in zip(axes, metric_keys):
        values = [metrics[m].get(key, 0) for m in methods_present]
        colors = [_color(m) for m in methods_present]
        bars = ax.bar(range(len(methods_present)), values, color=colors)
        ax.set_xticks(range(len(methods_present)))
        ax.set_xticklabels([_label(m) for m in methods_present],
                           rotation=30, ha="right", fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, max(1.05, max(values) * 1.1) if values else 1.05)
        # Value labels on bars
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Final Metrics — {env_label}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "metrics_bars.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_unlearning_dynamics(progress, env_label, save_dir):
    metrics_to_plot = [
        ("forget_effectiveness", "Forget Effectiveness", True),
        ("retain_stability_index", "Retain Stability Index", True),
        ("selectivity", "Selectivity", True),
        ("mean_forget_score", "Mean Forget Score", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, (key, label, ylim01) in zip(axes, metrics_to_plot):
        for method, steps_data in progress.items():
            x = [s["step"] for s in steps_data]
            y = [s.get(key, 0) for s in steps_data]
            ax.plot(x, y, "-o", label=_label(method), color=_color(method),
                    linewidth=2, markersize=4)
        ax.set_xlabel("Unlearning Step")
        ax.set_ylabel(label)
        ax.set_title(label)
        if ylim01:
            ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=8)

    fig.suptitle(f"Unlearning Dynamics — {env_label}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "unlearning_dynamics.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_loss_curves(progress, env_label, save_dir):
    # Gather all loss keys across all methods
    all_loss_keys = set()
    for method, steps_data in progress.items():
        for step in steps_data:
            for k in step:
                if "loss" in k:
                    all_loss_keys.add(k)

    loss_keys = sorted(all_loss_keys)
    if not loss_keys:
        return

    n = len(loss_keys)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, key in zip(axes_flat, loss_keys):
        for method, steps_data in progress.items():
            x = [s["step"] for s in steps_data if key in s]
            y = [s[key] for s in steps_data if key in s]
            if x:
                ax.plot(x, y, "-o", label=_label(method), color=_color(method),
                        linewidth=2, markersize=4)
        ax.set_xlabel("Unlearning Step")
        ax.set_title(key.replace("_", " ").title(), fontsize=10)
        ax.legend(fontsize=8)

    # Hide unused axes
    for ax in axes_flat[len(loss_keys):]:
        ax.set_visible(False)

    fig.suptitle(f"Loss Curves — {env_label}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_forget_retain_tradeoff(progress, env_label, save_dir):
    fig, ax = plt.subplots(figsize=(7, 6))

    for method, steps_data in progress.items():
        fe = [s.get("forget_effectiveness", 0) for s in steps_data]
        rsi = [s.get("retain_stability_index", 0) for s in steps_data]
        # Plot trajectory with arrow direction
        ax.plot(fe, rsi, "-o", label=_label(method), color=_color(method),
                linewidth=2, markersize=5, alpha=0.8)
        # Mark start and end
        if len(fe) >= 2:
            ax.annotate("", xy=(fe[-1], rsi[-1]),
                        xytext=(fe[-2], rsi[-2]),
                        arrowprops=dict(arrowstyle="->", color=_color(method),
                                        lw=2))
        ax.scatter(fe[0], rsi[0], color=_color(method), s=100,
                   zorder=5, edgecolors="black", linewidths=1)

    ax.set_xlabel("Forget Effectiveness")
    ax.set_ylabel("Retain Stability Index")
    ax.set_title(f"Forget–Retain Tradeoff — {env_label}", fontweight="bold")
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9)

    # Ideal corner annotation
    ax.annotate("Ideal →", xy=(1.0, 1.0), fontsize=9, color="green",
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="green", alpha=0.7))

    plt.tight_layout()
    plt.savefig(save_dir / "forget_retain_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_dashboard(data, env_label, save_dir):
    """Single-page summary dashboard."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    fig.suptitle(f"Unlearning Dashboard — {env_label}", fontsize=16, fontweight="bold")

    # 1. Training curve (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    if data.get("train_curve"):
        tc = data["train_curve"]
        ax1.plot(tc["step"], tc["mean_reward"], color="#1976D2", linewidth=2)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Mean Return")
        ax1.set_title("Training Curve", fontsize=11)
        ax1.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    else:
        ax1.text(0.5, 0.5, "No training data", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Training Curve", fontsize=11)

    # 2. Return distributions (top-center)
    ax2 = fig.add_subplot(gs[0, 1])
    if data.get("eval_returns") is not None:
        df = data["eval_returns"]
        order = ["baseline"] + [m for m in METHODS if m in df["model"].values]
        palette = {m: _color(m) for m in order}
        sns.boxplot(data=df, x="model", y="return", hue="model", order=order,
                    palette=palette, ax=ax2, fliersize=2, legend=False)
        ax2.set_xticks(range(len(order)))
        ax2.set_xticklabels([_label(m).split()[0] for m in order],
                            rotation=30, ha="right", fontsize=8)
        ax2.set_xlabel("")
        ax2.set_ylabel("Return")
        ax2.set_title("Eval Returns", fontsize=11)

    # 3. Metrics bars (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    if data.get("eval_metrics"):
        metrics = data["eval_metrics"]
        methods_present = [m for m in METHODS if m in metrics]
        keys = ["forget_effectiveness", "retain_stability_index", "selectivity"]
        labels = ["Forget\nEff.", "Retain\nStab.", "Select."]
        x = np.arange(len(keys))
        width = 0.8 / max(len(methods_present), 1)
        for i, method in enumerate(methods_present):
            vals = [metrics[method].get(k, 0) for k in keys]
            ax3.bar(x + i * width, vals, width, label=_label(method),
                    color=_color(method))
        ax3.set_xticks(x + width * (len(methods_present) - 1) / 2)
        ax3.set_xticklabels(labels, fontsize=9)
        ax3.set_ylim(0, 1.05)
        ax3.set_title("Final Metrics", fontsize=11)
        ax3.legend(fontsize=7, loc="upper left")

    # 4. Unlearning dynamics — forget effectiveness (bottom-left)
    ax4 = fig.add_subplot(gs[1, 0])
    if data.get("unlearn_progress"):
        for method, steps_data in data["unlearn_progress"].items():
            x = [s["step"] for s in steps_data]
            y = [s.get("forget_effectiveness", 0) for s in steps_data]
            ax4.plot(x, y, "-o", label=_label(method), color=_color(method),
                     linewidth=2, markersize=3)
        ax4.set_xlabel("Unlearning Step")
        ax4.set_ylabel("Forget Effectiveness")
        ax4.set_title("Forgetting Over Time", fontsize=11)
        ax4.set_ylim(-0.02, 1.05)
        ax4.legend(fontsize=7)

    # 5. Unlearning dynamics — retain stability (bottom-center)
    ax5 = fig.add_subplot(gs[1, 1])
    if data.get("unlearn_progress"):
        for method, steps_data in data["unlearn_progress"].items():
            x = [s["step"] for s in steps_data]
            y = [s.get("retain_stability_index", 0) for s in steps_data]
            ax5.plot(x, y, "-o", label=_label(method), color=_color(method),
                     linewidth=2, markersize=3)
        ax5.set_xlabel("Unlearning Step")
        ax5.set_ylabel("RSI")
        ax5.set_title("Retain Stability Over Time", fontsize=11)
        ax5.set_ylim(-0.02, 1.05)
        ax5.legend(fontsize=7)

    # 6. Forget-retain tradeoff (bottom-right)
    ax6 = fig.add_subplot(gs[1, 2])
    if data.get("unlearn_progress"):
        for method, steps_data in data["unlearn_progress"].items():
            fe = [s.get("forget_effectiveness", 0) for s in steps_data]
            rsi = [s.get("retain_stability_index", 0) for s in steps_data]
            ax6.plot(fe, rsi, "-o", label=_label(method), color=_color(method),
                     linewidth=2, markersize=4)
            ax6.scatter(fe[0], rsi[0], color=_color(method), s=60,
                        zorder=5, edgecolors="black", linewidths=0.8)
        ax6.set_xlabel("Forget Effectiveness")
        ax6.set_ylabel("Retain Stability")
        ax6.set_title("Forget–Retain Tradeoff", fontsize=11)
        ax6.set_xlim(-0.02, 1.05)
        ax6.set_ylim(-0.02, 1.05)
        ax6.legend(fontsize=7)

    plt.savefig(save_dir / "dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── Cross-environment comparison ───────────────────────────────────────────

def plot_cross_environment(all_data: list, save_dir: Path):
    """Compare metrics across environments."""
    save_dir.mkdir(parents=True, exist_ok=True)

    envs = []
    for d in all_data:
        env_label = d["name"].split("_ppo")[0].replace("_", " ").title()
        envs.append(env_label)

    # 1. Final metrics comparison across envs
    metric_keys = [
        ("forget_effectiveness", "Forget Effectiveness"),
        ("retain_stability_index", "Retain Stability Index"),
        ("selectivity", "Selectivity"),
    ]

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(5 * len(metric_keys), 5))
    if len(metric_keys) == 1:
        axes = [axes]

    for ax, (key, label) in zip(axes, metric_keys):
        x = np.arange(len(envs))
        width = 0.8 / len(METHODS)
        for i, method in enumerate(METHODS):
            values = []
            for d in all_data:
                m = d.get("eval_metrics", {}).get(method, {})
                values.append(m.get(key, 0))
            bars = ax.bar(x + i * width, values, width,
                          label=_label(method), color=_color(method))
            for bar, v in zip(bars, values):
                if v > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x + width)
        ax.set_xticklabels(envs, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_title(label, fontsize=12)
        ax.legend(fontsize=8)

    fig.suptitle("Cross-Environment Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "cross_env_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Return comparison across envs (normalized)
    fig, axes = plt.subplots(1, len(all_data), figsize=(5 * len(all_data), 5),
                             sharey=False)
    if len(all_data) == 1:
        axes = [axes]

    for ax, d, env_label in zip(axes, all_data, envs):
        if d.get("eval_returns") is not None:
            df = d["eval_returns"]
            order = ["baseline"] + [m for m in METHODS if m in df["model"].values]
            palette = {m: _color(m) for m in order}
            sns.boxplot(data=df, x="model", y="return", hue="model", order=order,
                        palette=palette, ax=ax, fliersize=2, legend=False)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([_label(m).split()[0] for m in order],
                               rotation=30, ha="right", fontsize=8)
            ax.set_xlabel("")
        ax.set_title(env_label, fontsize=12)

    fig.suptitle("Evaluation Returns Across Environments",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "cross_env_returns.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Training curves side by side
    fig, axes = plt.subplots(1, len(all_data), figsize=(5 * len(all_data), 4),
                             sharey=False)
    if len(all_data) == 1:
        axes = [axes]

    for ax, d, env_label in zip(axes, all_data, envs):
        if d.get("train_curve"):
            tc = d["train_curve"]
            ax.plot(tc["step"], tc["mean_reward"], color="#1976D2", linewidth=2)
            ax.fill_between(tc["step"], tc["mean_reward"],
                            alpha=0.15, color="#1976D2")
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.set_title(env_label, fontsize=12)
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean Return")

    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "cross_env_training.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Heatmap: methods × envs for each metric
    for key, label in metric_keys:
        matrix = []
        for method in METHODS:
            row = []
            for d in all_data:
                m = d.get("eval_metrics", {}).get(method, {})
                row.append(m.get(key, 0))
            matrix.append(row)

        fig, ax = plt.subplots(figsize=(max(5, len(envs) * 1.5), 4))
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(envs)))
        ax.set_xticklabels(envs, fontsize=10)
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels([_label(m) for m in METHODS], fontsize=10)

        # Annotate cells
        for i in range(len(METHODS)):
            for j in range(len(envs)):
                ax.text(j, i, f"{matrix[i][j]:.3f}",
                        ha="center", va="center", fontsize=11, fontweight="bold",
                        color="white" if matrix[i][j] < 0.4 else "black")

        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"{label} — Methods × Environments", fontsize=13, fontweight="bold")
        plt.tight_layout()
        safe_key = key.replace("_", "-")
        plt.savefig(save_dir / f"heatmap_{safe_key}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Cross-environment plots saved to {save_dir}/")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive plots for RL unlearning experiments."
    )
    parser.add_argument("folder", nargs="?", default=None,
                        help="Path to experiment output folder (e.g. experiments/outputs/cartpole_ppo_seed42)")
    parser.add_argument("--env", type=str, default=None,
                        help="Environment name to find (e.g. cartpole, lunarlander, acrobot)")
    parser.add_argument("--all", action="store_true",
                        help="Plot all experiments found")
    parser.add_argument("--compare", action="store_true",
                        help="Generate cross-environment comparison plots")
    parser.add_argument("--base-dir", type=str, default=".",
                        help="Project base directory (default: current dir)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory for plots (default: same as data folder)")

    args = parser.parse_args()
    base_dir = Path(args.base_dir)

    # Determine plot output base
    plots_base = Path(args.output_dir) if args.output_dir else base_dir / "plots"

    if args.folder:
        # Single experiment folder
        folder = Path(args.folder)
        if not folder.exists():
            print(f"Error: {folder} does not exist")
            sys.exit(1)
        data = load_experiment(folder)
        out = plots_base / folder.name
        plots = plot_experiment(data, save_dir=out)
        print(f"Generated {len(plots)} plots in {out}/:")
        for p in plots:
            print(f"  {p}")

    elif args.env:
        # Find by environment name
        folders = find_experiment_folders(base_dir, args.env)
        if not folders:
            print(f"No experiment folders found for env '{args.env}'")
            sys.exit(1)
        for folder in folders:
            data = load_experiment(folder)
            out = plots_base / folder.name
            plots = plot_experiment(data, save_dir=out)
            print(f"Generated {len(plots)} plots in {out}/:")
            for p in plots:
                print(f"  {p}")

    elif args.all or args.compare:
        folders = find_experiment_folders(base_dir)
        if not folders:
            print("No experiment folders found")
            sys.exit(1)

        all_data = []
        for folder in folders:
            data = load_experiment(folder)
            all_data.append(data)
            if args.all:
                out = plots_base / folder.name
                plots = plot_experiment(data, save_dir=out)
                print(f"Generated {len(plots)} plots in {out}/:")
                for p in plots:
                    print(f"  {p}")

        if args.compare or args.all:
            compare_dir = plots_base / "comparison"
            plot_cross_environment(all_data, compare_dir)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
