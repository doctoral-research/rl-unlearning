"""Cross-env relearn-resistance heatmap on validated scenarios only.

Horizontal layout: envs as rows, methods as columns, methods grouped by
family. Cells where unlearning failed pre-attack (forget_eff < 0.65)
flagged as did-not-unlearn. Family-mean Δ column on the right.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

ENV_ROWS = [
    ("fourrooms", "",     "FourRooms"),
    ("empty8x8",  "",     "Empty-8x8"),
    ("doorkey",   "pgc_", "DoorKey"),
]

GROUPS = [
    ("STRUCTURAL", "#2A9D8F", [
        ("noop\ndisjoint",     "noop_disjoint"),
        ("is_replay\ndisjoint","is_replay_disjoint"),
    ]),
    ("COSMETIC", "#E63946", [
        ("neg_reward\n0.1",    "neg_reward_0.1"),
        ("neg_reward\n1.0",    "neg_reward_1.0"),
        ("NegGrad",            "neggrad"),
        ("KL-NegGrad",         "kl_neggrad"),
        ("NPO",                "npo"),
    ]),
]

SEEDS = [42, 43, 44, 45, 46]
UNLEARN_THRESHOLD = 0.65


def get_pre_post(env: str, scen_prefix: str, suffix: str):
    full = scen_prefix + suffix
    pres, posts = [], []
    for s in SEEDS:
        f = ROOT / "experiments" / "outputs" / f"relearn_attack_{env}_seed{s}_{full}" / "history.json"
        if not f.exists():
            continue
        h = json.loads(f.read_text())
        eff = [r["eval/forget_effectiveness"] for r in h if "eval/forget_effectiveness" in r]
        if len(eff) < 2:
            continue
        pres.append(eff[0])
        posts.append(eff[-1])
    if not pres:
        return None, None
    return float(np.mean(pres)), float(np.mean(posts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()

    mpl.rcParams["font.family"] = "Poppins"

    method_cols = []
    for family, color, methods in GROUPS:
        for label, suffix in methods:
            method_cols.append((family, color, label, suffix))

    n_rows = len(ENV_ROWS)
    n_method_cols = len(method_cols)
    n_cols = n_method_cols + 1  # +1 GAP column only

    fig, ax = plt.subplots(figsize=(11.5, 4.0))
    fig.subplots_adjust(left=0.07, right=0.95, top=0.85, bottom=0.16)

    delta = np.full((n_rows, n_cols), np.nan)
    pre_post = [[(None, None) for _ in range(n_cols)] for _ in range(n_rows)]
    for j, (family, color, label, suffix) in enumerate(method_cols):
        for i, (env, prefix, _) in enumerate(ENV_ROWS):
            pre, post = get_pre_post(env, prefix, suffix)
            pre_post[i][j] = (pre, post)
            if pre is not None and post is not None and pre >= UNLEARN_THRESHOLD:
                delta[i, j] = post - pre

    # Last column: STRUCTURAL advantage (structural family mean - cosmetic family mean)
    family_stats = []  # (struct_vals_list, cosm_vals_list, gap_or_None)
    for i in range(n_rows):
        struct_vals = [delta[i, j] for j in range(n_method_cols)
                       if method_cols[j][0] == "STRUCTURAL" and not np.isnan(delta[i, j])]
        cosm_vals = [delta[i, j] for j in range(n_method_cols)
                     if method_cols[j][0] == "COSMETIC" and not np.isnan(delta[i, j])]
        gap = (float(np.mean(struct_vals)) - float(np.mean(cosm_vals))
               if struct_vals and cosm_vals else None)
        family_stats.append((struct_vals, cosm_vals, gap))
        if gap is not None:
            delta[i, n_method_cols] = gap

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "rg", ["#C0392B", "#F4E6C8", "#2A9D8F"]
    )
    im = ax.imshow(delta, cmap=cmap, vmin=-0.30, vmax=0.10, aspect="auto")

    for i in range(n_rows):
        for j in range(n_method_cols):
            pre, post = pre_post[i][j]
            if pre is None:
                ax.text(j, i, "n/a", ha="center", va="center",
                        color="#888", fontsize=9)
            elif pre < UNLEARN_THRESHOLD:
                ax.text(j, i, "no unlearn",
                        ha="center", va="center", color="#777",
                        fontsize=8, fontstyle="italic")
            else:
                d = post - pre
                sign = "+" if d >= 0 else "−"
                ax.text(j, i, r"$\Delta$ " + f"{sign}{abs(d):.2f}",
                        ha="center", va="center",
                        color="#1D3557", fontsize=12, fontweight="bold")
        # STRUCTURAL advantage column: just the gap number
        gap_j = n_method_cols
        s_vals, c_vals, gap = family_stats[i]
        if gap is not None:
            sign = "+" if gap >= 0 else "−"
            ax.text(gap_j, i, f"{sign}{abs(gap):.2f}",
                    ha="center", va="center",
                    color="#1D3557", fontsize=22, fontweight="bold")

    # X tick labels: method names + STRUCTURAL advantage
    xt_labels = ([m[2] for m in method_cols]
                 + ["STRUCTURAL\nadvantage"])
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(xt_labels, fontsize=8.5, color="#2B2D42")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([e[2] for e in ENV_ROWS], fontsize=11,
                       fontweight="bold", color="#1D3557")
    ax.tick_params(left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    # Family color stripes above the method columns
    left, right = 0.08, 0.95
    method_width = (right - left) * (n_method_cols / n_cols)
    col_w = method_width / n_method_cols
    cumulative = 0
    stripe_y = 0.87
    for family, color, methods in GROUPS:
        n = len(methods)
        x_start = left + cumulative * col_w
        x_end = x_start + n * col_w
        fig.add_artist(plt.Rectangle((x_start, stripe_y), x_end - x_start, 0.025,
                                      transform=fig.transFigure,
                                      color=color, zorder=4, clip_on=False))
        fig.text((x_start + x_end) / 2, stripe_y + 0.05, family,
                 transform=fig.transFigure,
                 color=color, fontsize=12, fontweight="bold",
                 ha="center", va="center")
        cumulative += n

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
