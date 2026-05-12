"""Headline plot for the Brittle Unlearning result.

X axis: relearning attack iterations (continued PPO with normal env
reward starting from an unlearned checkpoint).
Y axis: forget_effectiveness measured each `relearn_eval_every` iters.

Lines: one per method (noop_disjoint, is_replay_disjoint, neg_reward_X).
Shaded bands: mean ± 1 std across seeds.

A flat line at 1.0 = unlearning persists. A line that crashes back
toward baseline = unlearning was cosmetic.

Usage:
    python scripts/plot_relearn_attack.py --env fourrooms \\
        --seeds 42 43 44 45 46 \\
        --output docs/images/results/online_unlearn/relearn_attack_fourrooms.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

METHODS = [
    ("noop disjoint",        "noop_disjoint",       "#3366cc", "-"),
    ("is_replay disjoint",   "is_replay_disjoint",  "#cc0000", "-"),
    ("neg reward pen=0.1",   "neg_reward_0.1",      "#888888", "--"),
    ("neg reward pen=1.0",   "neg_reward_1.0",      "#444444", ":"),
]


def _stack(histories, key):
    rows = []
    for h in histories:
        xs = [r["iter"] for r in h if key in r]
        ys = [r[key] for r in h if key in r]
        if xs:
            rows.append((xs, ys))
    if not rows:
        return np.array([]), np.array([]), np.array([])
    n = min(len(ys) for _, ys in rows)
    iters = rows[0][0][:n]
    Y = np.array([ys[:n] for _, ys in rows], dtype=np.float32)
    return np.asarray(iters), Y.mean(axis=0), Y.std(axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="fourrooms")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out_root = ROOT / "experiments" / "outputs"

    grouped = {}
    for label, suffix, color, ls in METHODS:
        hs = []
        for s in args.seeds:
            f = out_root / f"relearn_attack_{args.env}_seed{s}_{suffix}" / "history.json"
            if not f.exists():
                print(f"missing: {f}")
                continue
            hs.append(json.loads(f.read_text()))
        if hs:
            grouped[label] = (hs, color, ls)

    if not grouped:
        raise SystemExit("no histories found")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    # Panel 1: forget_eff over relearning iters
    ax = axes[0]
    for label, (hs, color, ls) in grouped.items():
        iters, mean, std = _stack(hs, "eval/forget_effectiveness")
        if len(iters) == 0: continue
        ax.plot(iters, mean, color=color, linestyle=ls, linewidth=1.8,
                label=label, marker="o", markersize=3)
        if len(hs) > 1:
            ax.fill_between(iters, mean - std, mean + std,
                            color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel("relearn-attack iteration (continued PPO)")
    ax.set_ylabel("forget effectiveness")
    ax.set_title("Brittle Unlearning: forget behavior under relearning attack")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.3)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left")

    # Panel 2: retain stability over relearning iters
    ax = axes[1]
    for label, (hs, color, ls) in grouped.items():
        iters, mean, std = _stack(hs, "eval/retain_stability")
        if len(iters) == 0: continue
        ax.plot(iters, mean, color=color, linestyle=ls, linewidth=1.8,
                label=label, marker="o", markersize=3)
        if len(hs) > 1:
            ax.fill_between(iters, mean - std, mean + std,
                            color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel("relearn-attack iteration")
    ax.set_ylabel("retain stability")
    ax.set_title("Retain stability over relearning attack")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")

    n_seeds = max(len(v[0]) for v in grouped.values())
    fig.suptitle(
        f"Relearn-resistance attack — {args.env} / avoid_bottom_right "
        f"(mean ± std, {n_seeds} seeds)", fontsize=12,
    )
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
