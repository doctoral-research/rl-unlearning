"""Headline plot for the on-policy online unlearning paper / poster.

Compares 4 conditions on Empty-8x8:
- shared objective + no_op control
- shared objective + is_replay (the chicken-and-egg / tug-of-war setup)
- disjoint objective + no_op   (zero PPO grad on forget transitions)
- disjoint objective + is_replay (the proposed method)

Three panels:
  (1) n_forget transitions per rollout
  (2) eval/forget_effectiveness over iters
  (3) eval/retain_stability over iters

Usage:
    python scripts/plot_disjoint_result.py \\
        --output docs/images/results/online_unlearn/disjoint_result_empty8x8.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Map (label, dir_suffix, color, linestyle)
CONDITIONS = [
    ("no_op (shared objective)",       "noop_shared",           "#999999", "--"),
    ("is_replay (shared objective)",   "is_replay_shared",       "#cc6600", "--"),
    ("no_op (disjoint objective)",     "noop_disjoint",       "#3366cc", "-"),
    ("is_replay (disjoint objective)", "is_replay_disjoint", "#cc0000", "-"),
]


def _stack_metric(histories: List[list], key: str):
    """Return (iters, mean, std) across multiple history runs for a metric.

    Some metrics are logged every iteration (`n_forget_transitions`); others
    only on eval iterations (`eval/forget_effectiveness`). We use the shared
    set of iterations on which all seeds have the metric.
    """
    per_seed_x_y = []
    for hist in histories:
        xs = [h["iter"] for h in hist if key in h]
        ys = [h[key] for h in hist if key in h]
        if xs:
            per_seed_x_y.append((xs, ys))
    if not per_seed_x_y:
        return np.array([]), np.array([]), np.array([])
    # Different seeds may have slightly different lengths if eval timing
    # differs at the boundary. Truncate to the shortest.
    n = min(len(ys) for _, ys in per_seed_x_y)
    iters = per_seed_x_y[0][0][:n]
    Y = np.array([ys[:n] for _, ys in per_seed_x_y], dtype=np.float32)
    return np.asarray(iters), Y.mean(axis=0), Y.std(axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="empty8x8")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--output", required=True)
    args = p.parse_args()

    out_root = ROOT / "experiments" / "outputs"
    # Group histories by condition, gathering all seeds we find.
    grouped = {}
    for label, suffix, color, ls in CONDITIONS:
        hists = []
        for seed in args.seeds:
            d = out_root / f"online_unlearn_{args.env}_seed{seed}_{suffix}"
            hf = d / "history.json"
            if not hf.exists():
                print(f"missing: {hf}")
                continue
            hists.append(json.loads(hf.read_text()))
        if hists:
            grouped[label] = (hists, color, ls)

    if not grouped:
        raise SystemExit("no histories found")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    def plot_panel(ax, key, ylabel, title, ylim=None):
        for label, (hists, color, ls) in grouped.items():
            iters, mean, std = _stack_metric(hists, key)
            if len(iters) == 0:
                continue
            ax.plot(iters, mean, color=color, linestyle=ls, linewidth=1.8,
                    label=label, marker="o", markersize=3)
            if len(hists) > 1:
                ax.fill_between(iters, mean - std, mean + std,
                                color=color, alpha=0.15, linewidth=0)
        ax.set_xlabel("rollout iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if ylim is not None:
            ax.set_ylim(*ylim)
            ax.axhline(ylim[1], color="black", linestyle=":", alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    plot_panel(axes[0], "n_forget_transitions",
               "# forget-region transitions / rollout",
               "On-policy forget-data over time")
    plot_panel(axes[1], "eval/forget_effectiveness",
               "forget effectiveness", "Forget effectiveness", ylim=(0, 1.05))
    plot_panel(axes[2], "eval/retain_stability",
               "retain stability", "Retain stability", ylim=(0, 1.05))

    n_seeds = max(len(v[0]) for v in grouped.values())
    fig.suptitle(
        f"On-policy online RL unlearning — {args.env} "
        f"(mean ± std across {n_seeds} seeds)",
        fontsize=12,
    )
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
