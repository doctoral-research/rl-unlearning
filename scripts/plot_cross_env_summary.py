"""Cross-env Δforget / Δretain heatmap across all methods.

Headline paper figure: which methods are durable under relearn attack, on
which envs. Each cell is mean Δ across 5 seeds; sign + magnitude reveals
the Brittle Unlearning pattern.

Usage:
    python scripts/plot_cross_env_summary.py \\
        --output docs/images/results/online_unlearn/cross_env_summary.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

ENVS = ["fourrooms", "empty8x8", "lunarlander", "acrobot", "doorkey"]

METHODS = [
    # (display_label, file_suffix)
    ("noop\ndisjoint",        "noop_disjoint"),
    ("is_replay\ndisjoint",   "is_replay_disjoint"),
    ("is_replay\n+ retain",   "is_replay_retain"),
    ("neg_reward\n0.1",       "neg_reward_0.1"),
    ("neg_reward\n1.0",       "neg_reward_1.0"),
    ("NegGrad",               "neggrad"),
    ("KL-NegGrad",            "kl_neggrad"),
    ("NPO",                   "npo"),
    ("Lagrangian",            "lagrangian"),
]

SEEDS = [42, 43, 44, 45, 46]


def delta(env, suffix, key="eval/forget_effectiveness"):
    out_root = ROOT / "experiments" / "outputs"
    deltas = []
    for s in SEEDS:
        f = out_root / f"relearn_attack_{env}_seed{s}_{suffix}" / "history.json"
        if not f.exists():
            continue
        h = json.loads(f.read_text())
        vals = [r[key] for r in h if key in r]
        if len(vals) < 2:
            continue
        deltas.append(vals[-1] - vals[0])
    return np.mean(deltas) if deltas else np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax_i, (key, title, cmap, vmin, vmax) in enumerate([
        ("eval/forget_effectiveness", "Δ forget effectiveness (post-attack − pre-attack)",
         "RdYlGn", -0.6, 0.2),
        ("eval/retain_stability",     "Δ retain stability (post-attack − pre-attack)",
         "RdYlGn", -0.6, 0.2),
    ]):
        M = np.full((len(METHODS), len(ENVS)), np.nan)
        for i, (_, suffix) in enumerate(METHODS):
            for j, env in enumerate(ENVS):
                M[i, j] = delta(env, suffix, key=key)

        ax = axes[ax_i]
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(ENVS))); ax.set_xticklabels(ENVS, rotation=30)
        ax.set_yticks(range(len(METHODS))); ax.set_yticklabels([m[0] for m in METHODS])
        ax.set_title(title, fontsize=11)
        for i in range(len(METHODS)):
            for j in range(len(ENVS)):
                v = M[i, j]
                txt = "n/a" if np.isnan(v) else f"{v:+.2f}"
                color = "white" if not np.isnan(v) and abs(v) > 0.35 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        color=color, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.05)

    fig.suptitle("Relearn-resistance across methods and envs (mean Δ over 5 seeds)",
                 fontsize=12)
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
