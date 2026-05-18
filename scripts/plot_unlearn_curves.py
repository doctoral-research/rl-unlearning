"""§2.1 companion: unlearning-phase forget_eff trajectories on Empty-8x8.

Shows the rise of forget_effectiveness during unlearning for all 5 classic
methods (cosmetic family) and the proposed disjoint methods (structural
family). Same visual register as plot_graphical_abstract_minimal.py:
Poppins, smooth lines + faded scatter + std bands, big family cards on
the right.
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

# (label, suffix, color, family)
SERIES = [
    # Structural (teal shades)
    ("is_replay disjoint",   "is_replay_disjoint",          "#2A9D8F", "structural"),
    ("noop disjoint",        "no_op_disjoint",              "#1D7268", "structural"),
    # Cosmetic (red shades)
    ("Lagrangian PPO",       "no_op_lagrangian",            "#E63946", "cosmetic"),
    ("neg_reward 1.0",       "no_op_neg_reward_1.0",        "#F4A261", "cosmetic"),
    ("neg_reward 0.1",       "no_op_neg_reward_0.1",        "#E76F51", "cosmetic"),
    ("KL-NegGrad",           "gradient_reversal_kl_neggrad","#9B2226", "cosmetic"),
    ("NegGrad",              "gradient_reversal_neggrad",   "#BB3E03", "cosmetic"),
    ("NPO",                  "npo_npo",                     "#AE2012", "cosmetic"),
]


def stack(env, suffix):
    rows = []
    for s in SEEDS:
        f = OUT_ROOT / f"online_unlearn_{env}_seed{s}_{suffix}" / "history.json"
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

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.13)

    def rolling_mean(arr, window=5):
        arr = np.asarray(arr, dtype=float)
        out = np.empty_like(arr)
        half = window // 2
        for i in range(len(arr)):
            lo = max(0, i - half)
            hi = min(len(arr), i + half + 1)
            out[i] = arr[lo:hi].mean()
        return out

    handles_struct, handles_cosm = [], []
    for label, suffix, color, group in SERIES:
        iters, mean, std = stack(ENV, suffix)
        if iters is None:
            print(f"missing data for {suffix}"); continue
        smooth_mean = rolling_mean(mean, window=3)
        smooth_std = rolling_mean(std, window=3)
        lw = 3.4 if group == "structural" else 2.0
        ax.plot(iters, mean, color=color, lw=0, marker="o",
                markersize=3.5, alpha=0.25, zorder=3)
        line, = ax.plot(iters, smooth_mean, color=color, lw=lw,
                        label=label, zorder=5)
        ax.fill_between(iters, smooth_mean - smooth_std, smooth_mean + smooth_std,
                        color=color, alpha=0.12, linewidth=0, zorder=2)
        (handles_struct if group == "structural" else handles_cosm).append(line)

    ax.set_xlabel("felejtési iteráció", fontsize=10, color="#555")
    ax.set_ylabel("forget effectiveness", fontsize=10, color="#555")
    ax.set_ylim(0.30, 1.05)
    ax.set_xlim(-2, 205)
    sns.despine(ax=ax)
    ax.tick_params(labelsize=9)

    # Two compact legends above the plot — STRUCTURAL left, COSMETIC right.
    leg1 = ax.legend(handles=handles_struct, loc="lower center",
                     bbox_to_anchor=(0.18, 1.02),
                     ncol=len(handles_struct), frameon=False,
                     fontsize=9, handlelength=1.6,
                     title="STRUCTURAL", title_fontsize=10,
                     labelcolor="#2B2D42")
    leg1.get_title().set_color("#2A9D8F")
    leg1.get_title().set_fontweight("bold")
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=handles_cosm, loc="lower center",
                     bbox_to_anchor=(0.68, 1.02),
                     ncol=3, frameon=False,
                     fontsize=9, handlelength=1.6,
                     title="COSMETIC", title_fontsize=10,
                     labelcolor="#2B2D42")
    leg2.get_title().set_color("#E63946")
    leg2.get_title().set_fontweight("bold")

    out = ROOT / "docs" / "images" / "results" / "online_unlearn" / "unlearn_curves.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", transparent=True)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
