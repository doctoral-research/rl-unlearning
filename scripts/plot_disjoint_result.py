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

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

# Map (label, dir_suffix, color, linestyle)
CONDITIONS = [
    ("no_op (shared objective)",       "no_op_noop",           "#999999", "--"),
    ("is_replay (shared objective)",   "is_replay_w2.0",       "#cc6600", "--"),
    ("no_op (disjoint objective)",     "no_op_disjoint",       "#3366cc", "-"),
    ("is_replay (disjoint objective)", "is_replay_disjoint_w2", "#cc0000", "-"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="empty8x8")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    out_root = ROOT / "experiments" / "outputs"
    histories = {}
    for label, suffix, color, ls in CONDITIONS:
        d = out_root / f"online_unlearn_{args.env}_seed{args.seed}_{suffix}"
        hf = d / "history.json"
        if not hf.exists():
            print(f"missing: {hf}")
            continue
        histories[label] = (json.loads(hf.read_text()), color, ls)

    if not histories:
        raise SystemExit("no histories found")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # Panel 1: forget transitions per rollout
    ax = axes[0]
    for label, (hist, color, ls) in histories.items():
        xs = [h["iter"] for h in hist]
        ys = [h["n_forget_transitions"] for h in hist]
        ax.plot(xs, ys, label=label, color=color, linestyle=ls, linewidth=1.6)
    ax.set_xlabel("rollout iteration")
    ax.set_ylabel("# forget-region transitions / rollout")
    ax.set_title("On-policy forget-data over time")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    # Panel 2: forget_effectiveness over iters
    ax = axes[1]
    for label, (hist, color, ls) in histories.items():
        xs = [h["iter"] for h in hist if "eval/forget_effectiveness" in h]
        ys = [h["eval/forget_effectiveness"] for h in hist
              if "eval/forget_effectiveness" in h]
        ax.plot(xs, ys, label=label, color=color, linestyle=ls, marker="o",
                markersize=3, linewidth=1.6)
    ax.set_xlabel("rollout iteration")
    ax.set_ylabel("forget effectiveness")
    ax.set_title("Forget effectiveness")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.3)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # Panel 3: retain stability over iters
    ax = axes[2]
    for label, (hist, color, ls) in histories.items():
        xs = [h["iter"] for h in hist if "eval/retain_stability" in h]
        ys = [h["eval/retain_stability"] for h in hist
              if "eval/retain_stability" in h]
        ax.plot(xs, ys, label=label, color=color, linestyle=ls, marker="o",
                markersize=3, linewidth=1.6)
    ax.set_xlabel("rollout iteration")
    ax.set_ylabel("retain stability")
    ax.set_title("Retain stability")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.3)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        f"On-policy online RL unlearning — {args.env} / avoid_bottom_left",
        fontsize=12,
    )
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
