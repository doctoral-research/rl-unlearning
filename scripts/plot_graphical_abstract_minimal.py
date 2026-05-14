"""Minimal 2-line graphical abstract using DoorKey (the strongest brittle signal).

Just two lines: cosmetic (penalty) vs structural (disjoint).
Mean across 5 seeds, smoothed with a rolling window for clarity.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "experiments" / "outputs"
SEEDS = [42, 43, 44, 45, 46]
ENV = "empty8x8"

SERIES = [
    ("Reward shaping  (Lagrangian PPO)",    "lagrangian",            "#E63946", "cosmetic"),
    ("Disjoint objective  (is_replay)",     "is_replay_disjoint",    "#2A9D8F", "structural"),
]


def stack(env, suffix):
    rows = []
    for s in SEEDS:
        f = OUT_ROOT / f"relearn_attack_{env}_seed{s}_{suffix}" / "history.json"
        if not f.exists():
            continue
        h = json.loads(f.read_text())
        pts = [(r["iter"], r["eval/forget_effectiveness"]) for r in h
               if "eval/forget_effectiveness" in r]
        if pts:
            rows.append(pts)
    if not rows:
        return None, None, None
    n = min(len(r) for r in rows)
    iters = np.array([r[0] for r in rows[0][:n]])
    Y = np.array([[v for _, v in r[:n]] for r in rows])
    return iters, Y.mean(axis=0), Y.std(axis=0)


def main():
    import matplotlib as mpl
    mpl.rcParams["font.family"] = "Poppins"
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05,
                   rc={"font.family": "Poppins"})

    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    fig.subplots_adjust(left=0.08, right=0.78, top=0.95, bottom=0.14)

    def rolling_mean(arr, window=5):
        arr = np.asarray(arr, dtype=float)
        out = np.empty_like(arr)
        half = window // 2
        for i in range(len(arr)):
            lo = max(0, i - half)
            hi = min(len(arr), i + half + 1)
            out[i] = arr[lo:hi].mean()
        return out

    line_ends = {}
    for label, suffix, color, group in SERIES:
        iters, mean, std = stack(ENV, suffix)
        if iters is None:
            print(f"missing data for {suffix}"); continue
        smooth_mean = rolling_mean(mean, window=5)
        smooth_std = rolling_mean(std, window=5)
        ax.plot(iters, mean, color=color, lw=0, marker="o",
                markersize=4, alpha=0.30, zorder=3)
        ax.plot(iters, smooth_mean, color=color, lw=3.5, zorder=5)
        ax.fill_between(iters, smooth_mean - smooth_std, smooth_mean + smooth_std,
                        color=color, alpha=0.18, linewidth=0, zorder=2)
        line_ends[group] = (iters[-1], smooth_mean[-1], mean[0], mean[-1], color, label)

    ax.set_xlabel("relearn-attack iteration", fontsize=10, color="#555")
    ax.set_ylabel("forget effectiveness", fontsize=10, color="#555")
    ax.set_ylim(0.55, 1.02)
    ax.set_xlim(-2, 205)
    sns.despine(ax=ax)
    ax.tick_params(labelsize=9)

    # Big popping Δ callouts on the right side of the figure.
    # Hierarchy: category (big bold caps), Δ number (huge), method name, pre/post.
    order = [("structural", "STRUCTURAL", "disjoint objective", 0.74),
             ("cosmetic",   "COSMETIC",   "reward shaping",     0.30)]
    for group, category, method, y_fig in order:
        if group not in line_ends: continue
        _, _, pre, post, color, _ = line_ends[group]
        d = post - pre
        sign = "+" if d >= 0 else "−"
        fig.text(0.81, y_fig + 0.13,
                 category,
                 color=color, fontsize=14, fontweight="bold",
                 va="center", ha="left", family="Poppins")
        fig.text(0.81, y_fig + 0.05,
                 r"$\Delta$" + f" {sign}{abs(d):.2f}",
                 color=color, fontsize=26, fontweight="bold",
                 va="center", ha="left", family="Poppins")
        fig.text(0.81, y_fig - 0.02,
                 method,
                 color="#444", fontsize=10, fontweight="bold",
                 va="center", ha="left", family="Poppins")
        fig.text(0.81, y_fig - 0.06,
                 f"{pre:.2f}  to  {post:.2f}",
                 color="#888", fontsize=9, fontweight="normal",
                 va="center", ha="left", family="Poppins")

    out = ROOT / "docs" / "images" / "results" / "online_unlearn" / "graphical_abstract_minimal.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
