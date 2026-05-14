"""Minimal slope plot: failing unlearning methods, pre vs post forget_eff.

Usage:
    python scripts/plot_doorkey_crash.py [--env doorkey|fourrooms]
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "outputs"
SEEDS = [42, 43, 44, 45, 46]

# Poppins for poster typography
mpl.rcParams["font.family"] = "Poppins"
mpl.rcParams["font.weight"] = 400

METHODS = [
    ("neg_reward 0.1",       "neg_reward_0.1"),
    ("neg_reward 1.0",       "neg_reward_1.0"),
    ("Lagrangian PPO",        "lagrangian"),
    ("KL-anchored NegGrad",   "kl_neggrad"),
    ("NPO",                   "npo"),
]


def stack(env, suffix):
    pre, post = [], []
    for s in SEEDS:
        f = OUT / f"relearn_attack_{env}_seed{s}_{suffix}" / "history.json"
        if not f.exists(): continue
        h = json.loads(f.read_text())
        eff = [r["eval/forget_effectiveness"] for r in h if "eval/forget_effectiveness" in r]
        if len(eff) < 2: continue
        pre.append(eff[0]); post.append(eff[-1])
    if not pre: return None
    return np.array(pre), np.array(post)


def gather(env):
    rows = []
    for label, suffix in METHODS:
        r = stack(env, suffix)
        if r is None: continue
        pre, post = r
        rows.append((label, pre.mean(), pre.std(), post.mean(), post.std(),
                     post.mean() - pre.mean()))
    rows.sort(key=lambda r: r[5])
    return rows


def draw_panel(ax, rows, palette, env_label):
    n = len(rows)
    y_positions = np.arange(n)[::-1]
    for y, (label, pre_m, pre_s, post_m, post_s, delta), color in zip(
            y_positions, rows, palette):
        ax.errorbar(pre_m, y, xerr=pre_s, fmt="o", color=color,
                    markersize=8, markerfacecolor="white", markeredgewidth=1.6,
                    capsize=2, lw=0.9, alpha=0.9, zorder=4)
        ax.errorbar(post_m, y, xerr=post_s, fmt="o", color=color,
                    markersize=8, markerfacecolor=color, markeredgewidth=0,
                    capsize=2, lw=0.9, alpha=0.9, zorder=4)
        arrow = FancyArrowPatch((pre_m, y), (post_m, y),
                                  arrowstyle="-|>", mutation_scale=12,
                                  color=color, lw=2.0, alpha=0.85, zorder=3)
        ax.add_patch(arrow)
        sign = "+" if delta >= 0 else "−"
        # Δ values: center-aligned over the Δ column
        ax.text(1.06, y, f"{sign}{abs(delta):.2f}",
                color=color, fontsize=10, fontweight="bold",
                ha="center", va="center")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.5, n - 0.5)
    ax.tick_params(left=False, labelsize=9, length=0)
    sns.despine(ax=ax, left=True, bottom=False)
    # Env label well outside the method-label column. Far enough that
    # even the longest method name "KL-anchored NegGrad" doesn't touch it.
    ax.text(-0.28, 0.5, env_label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", color="#1D3557",
            ha="center", va="center", rotation=90)


def main():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05,
                   rc={"font.family": "Poppins"})

    rows_dk = gather("doorkey")
    rows_fr = gather("fourrooms")

    n_max = max(len(rows_dk), len(rows_fr))
    palette = sns.color_palette("rocket_r", n_colors=n_max)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 5.8), sharex=True,
        gridspec_kw={"hspace": 0.18, "height_ratios": [1, 1]},
    )
    fig.subplots_adjust(left=0.24, right=0.86, top=0.88, bottom=0.10)

    draw_panel(ax_top, rows_dk, palette, "DoorKey-8x8")
    draw_panel(ax_bot, rows_fr, palette, "FourRooms")

    # Thin separator between the panels
    fig.add_artist(plt.Line2D([0.24, 0.86], [0.48, 0.48],
                               transform=fig.transFigure,
                               color="#ddd", lw=0.7))

    # Shared x-axis label at very bottom
    ax_bot.set_xlabel("forget effectiveness", fontsize=10, color="#555")

    # Δ column header — centered over the value column (same x, ha="center")
    ax_top.text(1.06, len(rows_dk) - 0.3, r"$\Delta$",
                color="#444", fontsize=13, fontweight="bold",
                ha="center", va="bottom")

    # Horizontal legend in the figure top margin (above both panels,
    # outside the data area so it never overlaps points)
    from matplotlib.lines import Line2D
    leg_color = "#444"
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="",
               markersize=9, markerfacecolor="white",
               markeredgecolor=leg_color, markeredgewidth=1.6,
               label="before attack"),
        Line2D([0], [0], marker="o", linestyle="",
               markersize=9, markerfacecolor=leg_color,
               markeredgecolor=leg_color,
               label="after 200 PPO steps"),
    ]
    leg = fig.legend(handles=legend_handles,
                      loc="upper center", bbox_to_anchor=(0.535, 0.985),
                      ncol=2, frameon=False, fontsize=10,
                      handletextpad=0.5, columnspacing=1.8,
                      labelcolor=leg_color)
    out = ROOT / "docs" / "images" / "results" / "online_unlearn" / "crash_stacked.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
