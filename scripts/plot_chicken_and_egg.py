"""Plot the chicken-and-egg diagnostic for on-policy online unlearning.

For each `online_unlearn_*` output directory, reads history.json and
plots forget-region transitions per rollout iteration. Across the
methods, the same env, same baseline checkpoint, same scenario.

Headline figure: on-policy methods drive forget transitions toward
zero, halting the unlearning prematurely.

Usage:
    python scripts/plot_chicken_and_egg.py --env empty8x8 \\
        --output docs/images/results/chicken_and_egg_empty8x8.png

It picks up everything under experiments/outputs/online_unlearn_{env}_*
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, help="env name (empty8x8, fourrooms)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    pattern = f"online_unlearn_{args.env}_seed{args.seed}_*"
    dirs = sorted((ROOT / "experiments" / "outputs").glob(pattern))
    if not dirs:
        raise SystemExit(f"No output dirs matching {pattern}")

    methods = {}
    for d in dirs:
        hf = d / "history.json"
        if not hf.exists():
            continue
        name = d.name.split(f"seed{args.seed}_")[-1]
        methods[name] = json.loads(hf.read_text())
    if not methods:
        raise SystemExit("No history.json files found.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # Panel 1: forget transitions per rollout — the chicken-and-egg signal
    ax = axes[0]
    for name, hist in methods.items():
        xs = [h["iter"] for h in hist]
        ys = [h["n_forget_transitions"] for h in hist]
        ax.plot(xs, ys, label=name, marker="o", markersize=2, linewidth=1.5)
    ax.set_xlabel("rollout iteration")
    ax.set_ylabel("# forget-region transitions per rollout")
    ax.set_title("Chicken-and-egg: forget data dries up")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # Panel 2: forget effectiveness from periodic eval
    ax = axes[1]
    for name, hist in methods.items():
        xs = [h["iter"] for h in hist if "eval/forget_effectiveness" in h]
        ys = [h["eval/forget_effectiveness"] for h in hist if "eval/forget_effectiveness" in h]
        ax.plot(xs, ys, label=name, marker="o", markersize=3, linewidth=1.5)
    ax.set_xlabel("rollout iteration")
    ax.set_ylabel("forget effectiveness")
    ax.set_title("Forget effectiveness over time")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # Panel 3: retain stability
    ax = axes[2]
    for name, hist in methods.items():
        xs = [h["iter"] for h in hist if "eval/retain_stability" in h]
        ys = [h["eval/retain_stability"] for h in hist if "eval/retain_stability" in h]
        ax.plot(xs, ys, label=name, marker="o", markersize=3, linewidth=1.5)
    ax.set_xlabel("rollout iteration")
    ax.set_ylabel("retain stability")
    ax.set_title("Retain stability over time")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    fig.suptitle(f"On-policy online unlearning diagnostic ({args.env})", fontsize=12)
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
