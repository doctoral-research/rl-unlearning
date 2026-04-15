"""Generate summary plots for the unlearning experiments.

Uses the same visual style as the MagMA plotting suite
(`magma/results/plotting/plot_mpe_lbf_curves.py` and `plot_chart.py`):
- light-blue facecolor (#f0f6ff)
- spineless axes, white gridlines
- `tsplot` style: mean line with shaded +/- std band across seeds
- shared legend in a blue-background box
- PNG + PDF output with `bbox_inches='tight'`

Aggregates per-step time-series from
`experiments/outputs/*/unlearning_metrics_*.json` across seeds
{42, 43, 44} and writes to `docs/images/results/`.
"""
from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "experiments" / "outputs"
FIG_DIR = ROOT / "docs" / "images" / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ENVS = ["cartpole", "acrobot", "lunarlander"]
METHODS_ALL = [
    "trajectory_selective",
    "strategy_inversion",
    "retain_protection",
    "no_step_filter",
    "random_forget",
]

# MagMA palette
COLORS = {
    "trajectory_selective": "#2196F3",
    "strategy_inversion":   "#4CAF50",
    "retain_protection":    "#FF9800",
    "no_step_filter":       "#9C27B0",
    "random_forget":        "#F44336",
}
LABELS = {
    "trajectory_selective": "Traj Selective",
    "strategy_inversion":   "Strategy Inversion",
    "retain_protection":    "Retain Protection",
    "no_step_filter":       "No Step Filter",
    "random_forget":        "Random Forget",
}
BG = "#f0f6ff"
SEEDS = ["42", "43", "44"]


# ---------------------------------------------------------------- data load
def _json_path(env: str, scen: str, method: str, seed: str) -> Path | None:
    mkey = "trajectory_selective" if method == "no_step_filter" else method
    d = OUT_ROOT / f"{env}_{scen}_{method}_seed{seed}"
    if not d.is_dir():
        return None
    fn = d / f"unlearning_metrics_{mkey}.json"
    if fn.is_file():
        return fn
    for f in os.listdir(d):
        if f.startswith("unlearning_metrics_") and f.endswith(".json"):
            return d / f
    return None


def load_series(env: str, scen: str, method: str, key: str):
    """Returns (steps_array, values_matrix[n_seeds, n_steps]) or None."""
    seed_curves = []
    for seed in SEEDS:
        p = _json_path(env, scen, method, seed)
        if p is None:
            continue
        with open(p) as f:
            lst = json.load(f)
        if not lst:
            continue
        steps = [e["step"] for e in lst]
        vals = [e.get(key) for e in lst]
        if any(v is None for v in vals):
            continue
        seed_curves.append((np.array(steps, dtype=float), np.array(vals, dtype=float)))
    if not seed_curves:
        return None
    common = seed_curves[0][0]
    for s, _ in seed_curves[1:]:
        if len(s) != len(common) or not np.allclose(s, common):
            # align on min length
            L = min(len(c[0]) for c in seed_curves)
            common = seed_curves[0][0][:L]
            arr = np.stack([c[1][:L] for c in seed_curves], axis=0)
            return common, arr
    arr = np.stack([c[1] for c in seed_curves], axis=0)
    return common, arr


# ---------------------------------------------------------------- styling
def style_axis(ax, title=None, xlabel=None, ylabel=None):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor(BG)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=11)
    if title is not None:
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(visible=True, color="white", linewidth=0.9)
    ax.margins(x=0)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.tick_params(labelsize=9)


def tsplot(ax, steps, values, color, label):
    """mean line + shaded +/- std band (values: [n_seeds, n_steps])."""
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)
    ax.plot(steps, mean, color=color, linewidth=1.8, label=label)


def save(fig, stem):
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, format="png", bbox_inches="tight", dpi=200, facecolor="white")
    fig.savefig(pdf, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {png.name} (+ pdf)")


# ---------------------------------------------------------------- figures
def fig_cartpole_left_only_curves():
    """1x3 subplots: FE, RSI, Selectivity vs unlearn step; methods overlaid with std bands."""
    env, scen = "cartpole", "left_only"
    methods = ["trajectory_selective", "strategy_inversion", "retain_protection",
               "no_step_filter", "random_forget"]
    metrics = [
        ("scenario_forget_eff", r"Forget effectiveness $\mathrm{FE}(t)$"),
        ("retain_stability",    r"Retain Stability Index $\mathrm{RSI}(t)$"),
        ("scenario_selectivity", r"Selectivity $S(t) = \mathrm{FE}\cdot\mathrm{RSI}$"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (key, ylabel) in zip(axs, metrics):
        for meth in methods:
            out = load_series(env, scen, meth, key)
            if out is None:
                continue
            steps, vals = out
            tsplot(ax, steps, vals, COLORS[meth], LABELS[meth])
        style_axis(ax, title=None, xlabel="Unlearning step", ylabel=ylabel)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle(r"CartPole $\mathtt{left\_only}$: unlearning curves (mean $\pm$ std, $n{=}3$ seeds)",
                 fontsize=13, fontweight="bold", y=1.02)
    handles, labs = axs[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", bbox_to_anchor=(0.5, -0.06),
               ncol=len(labs), fontsize=11, fancybox=True, shadow=True, facecolor=BG)
    fig.subplots_adjust(wspace=0.28, bottom=0.18)
    save(fig, "cartpole_left_only_curves")


def fig_cross_env_selectivity_curves():
    """2x2: selectivity(t) per non-degenerate cell, 3 principled methods overlaid."""
    cells = [
        ("cartpole",    "left_only",    r"CartPole $\mathtt{left\_only}$"),
        ("acrobot",     "no_clockwise", r"Acrobot $\mathtt{no\_clockwise}$"),
        ("acrobot",     "slow_swing",   r"Acrobot $\mathtt{slow\_swing}$"),
        ("lunarlander", "no_tilt",      r"LunarLander $\mathtt{no\_tilt}$"),
    ]
    methods = ["trajectory_selective", "strategy_inversion", "retain_protection"]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for idx, (env, scen, title) in enumerate(cells):
        ax = axs[idx // 2, idx % 2]
        for meth in methods:
            out = load_series(env, scen, meth, "scenario_selectivity")
            if out is None:
                continue
            steps, vals = out
            tsplot(ax, steps, vals, COLORS[meth], LABELS[meth])
        style_axis(ax, title=title, xlabel="Unlearning step",
                   ylabel=r"Selectivity $S(t)$")
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle(r"Selectivity across envs and non-degenerate scenarios (mean $\pm$ std, $n{=}3$ seeds)",
                 fontsize=13, fontweight="bold", y=1.0)
    handles, labs = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=len(labs), fontsize=11, fancybox=True, shadow=True, facecolor=BG)
    fig.subplots_adjust(hspace=0.38, wspace=0.25, bottom=0.1)
    save(fig, "cross_env_selectivity_curves")


def fig_forget_vs_retain_curves():
    """1x3 subplots: Forget Effectiveness(t) per env (primary scenario), methods overlaid."""
    cells = [
        ("cartpole",    "left_only",    r"CartPole $\mathtt{left\_only}$"),
        ("acrobot",     "no_clockwise", r"Acrobot $\mathtt{no\_clockwise}$"),
        ("lunarlander", "no_tilt",      r"LunarLander $\mathtt{no\_tilt}$"),
    ]
    methods = ["trajectory_selective", "strategy_inversion", "retain_protection"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (env, scen, title) in zip(axs, cells):
        for meth in methods:
            out = load_series(env, scen, meth, "scenario_forget_eff")
            if out is None:
                continue
            steps, vals = out
            tsplot(ax, steps, vals, COLORS[meth], LABELS[meth])
        style_axis(ax, title=title, xlabel="Unlearning step",
                   ylabel=r"Forget effectiveness $\mathrm{FE}(t)$")
        ax.set_ylim(-0.05, 1.05)
    fig.suptitle(r"Forget effectiveness trajectories per env (mean $\pm$ std, $n{=}3$ seeds)",
                 fontsize=13, fontweight="bold", y=1.02)
    handles, labs = axs[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", bbox_to_anchor=(0.5, -0.06),
               ncol=len(labs), fontsize=11, fancybox=True, shadow=True, facecolor=BG)
    fig.subplots_adjust(wspace=0.28, bottom=0.18)
    save(fig, "forget_eff_per_env_curves")


def fig_forget_vs_rsi_scatter():
    """Final (last-checkpoint) FE vs RSI scatter, one point per seed per cell."""
    method_style = {
        "trajectory_selective": ("Traj Selective",     "o", COLORS["trajectory_selective"]),
        "strategy_inversion":   ("Strategy Inversion", "s", COLORS["strategy_inversion"]),
        "retain_protection":    ("Retain Protection",  "^", COLORS["retain_protection"]),
    }
    env_marker_size = {"cartpole": 110, "acrobot": 75, "lunarlander": 150}

    fig, ax = plt.subplots(figsize=(8.5, 6))
    plotted = set()
    for meth, (label, marker, color) in method_style.items():
        for env in ENVS:
            for scen_dir in OUT_ROOT.iterdir():
                if not scen_dir.is_dir():
                    continue
                parts = scen_dir.name.rsplit("_seed", 1)
                if len(parts) != 2:
                    continue
                stem, seed = parts
                if not stem.endswith("_" + meth):
                    continue
                rest = stem[: -len(meth) - 1]
                if not rest.startswith(env + "_"):
                    continue
                p = scen_dir / f"unlearning_metrics_{meth}.json"
                if not p.is_file():
                    continue
                with open(p) as f:
                    lst = json.load(f)
                if not lst:
                    continue
                last = lst[-1]
                fe = last.get("scenario_forget_eff")
                rsi = last.get("retain_stability")
                if fe is None or rsi is None:
                    continue
                legend_label = label if label not in plotted else None
                plotted.add(label)
                ax.scatter(fe, rsi, marker=marker, s=env_marker_size[env],
                           color=color, alpha=0.75, edgecolor="white", linewidth=1.0,
                           label=legend_label)

    ax.axhline(0.9, color="#888", linestyle=":", linewidth=1.2)
    ax.axvline(0.5, color="#888", linestyle=":", linewidth=1.2)
    ax.text(0.02, 0.905, r"RSI target $\geq 0.9$", fontsize=9, color="#555")
    ax.text(0.51,  0.58, r"FE target $\geq 0.5$", fontsize=9, color="#555", rotation=90)
    style_axis(ax,
               title=r"Forget-retain tradeoff: one point per seed; marker size encodes env",
               xlabel=r"Forget effectiveness $\mathrm{FE}$",
               ylabel=r"Retain Stability Index $\mathrm{RSI}$")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.55, 1.05)
    ax.legend(loc="lower right", fontsize=10, title="Method",
              facecolor=BG, edgecolor="white")
    note = ("Size: small=Acrobot, medium=CartPole, large=LunarLander\n"
            "Top-right quadrant is the ideal: high forget, high retain.")
    ax.text(0.98, 0.03, note, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color="#333",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#bbb"))
    plt.tight_layout()
    save(fig, "forget_vs_rsi_scatter")


def _cleanup_old_bar_charts():
    for stem in ["cartpole_left_only_selectivity", "cross_env_selectivity",
                 "cartpole_left_only_dynamics"]:
        for ext in ("png", "pdf"):
            p = FIG_DIR / f"{stem}.{ext}"
            if p.exists():
                p.unlink()
                print(f"removed {p.name}")


def main():
    _cleanup_old_bar_charts()
    fig_cartpole_left_only_curves()
    fig_cross_env_selectivity_curves()
    fig_forget_vs_retain_curves()
    fig_forget_vs_rsi_scatter()


if __name__ == "__main__":
    main()
