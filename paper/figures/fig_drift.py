"""The scope of a label against random label noise.

Each arm is compared with the run that labels matching steps only, seed by
seed. (a) How often the two policies choose the same action on a common
set of states. Random flips of 1% of the labels already move the policy
about as far as unlearning itself did (dashed line), so agreement alone
cannot attribute a difference to the specification. (b, c) The shift in
the forget metric, which random flips leave centred on zero. The two
domains get separate scales because CartPole's seed-to-seed spread is an
order of magnitude larger.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
sys.path.insert(0, str(Path(__file__).parent))
import style  # noqa: E402

style.setup()
DATA = Path(__file__).resolve().parents[1] / "data"
D = json.loads((DATA / "measurements.json").read_text())["drift"]

NAME = {"wt": "scope", "n1": "1%", "n10": "10%"}
COL = {"wt": style.ACCENT, "n1": style.INK, "n10": style.INK}
rng = np.random.default_rng(0)


def ci(v):
    v = np.asarray(v, float)
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def strip(ax, xs, items, key):
    for x, (dom, arm) in zip(xs, items):
        v = D[dom]["arms"][arm][key]
        ax.scatter(x + rng.uniform(-0.15, 0.15, len(v)), v, s=6, color=style.MUTE,
                   alpha=0.55, edgecolor="none", zorder=2)
        m, h = ci(v)
        ax.errorbar(x, m, yerr=h, fmt="o", ms=4.2, color=COL[arm], ecolor=COL[arm],
                    elinewidth=1.2, capsize=2.4, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([NAME[a] for _, a in items], fontsize=7.5)
    ax.tick_params(axis="x", length=0, pad=2)
    style.ygrid(ax)


fig, axes = plt.subplots(1, 3, figsize=(style.TEXTWIDTH, 1.9),
                         gridspec_kw={"width_ratios": [1.62, 0.95, 0.70],
                                      "wspace": 0.80})

ax = axes[0]
items = [("FourRooms", "wt"), ("FourRooms", "n1"), ("FourRooms", "n10"),
         ("CartPole", "wt"), ("CartPole", "n1")]
xs = [0, 1.2, 2.25, 3.95, 5.15]
strip(ax, xs, items, "agreement")
for (lo, hi), dom in (((-0.45, 2.7), "FourRooms"), ((3.5, 5.6), "CartPole")):
    b = np.mean(D[dom]["baseline_agreement"])
    ax.plot([lo, hi], [b, b], color=style.MUTE, lw=0.9, ls="--", zorder=1)
ax.text(-0.42, np.mean(D["FourRooms"]["baseline_agreement"]) - 0.03, "vs. baseline",
        ha="left", va="top", fontsize=6.6, color=style.MUTE)
ax.axvline(3.1, color=style.MUTE, lw=0.5, ls=":", zorder=0)
for x0, dom in ((1.125, "FourRooms"), (4.55, "CartPole")):
    ax.text(x0, 1.015, dom, transform=ax.get_xaxis_transform(), ha="center",
            va="bottom", fontsize=8)
ax.set_xlim(-0.55, 5.7)
ax.set_ylim(0.3, 1.0)
ax.set_ylabel("action agreement with\nmatching-steps run", linespacing=1.2)
style.panel(ax, "a", dx=-0.30, dy=1.13)

for ax, dom, letter in ((axes[1], "FourRooms", "b"), (axes[2], "CartPole", "c")):
    arms = [a for a in ("wt", "n1", "n10") if a in D[dom]["arms"]]
    strip(ax, [1.15 * i for i in range(len(arms))], [(dom, a) for a in arms], "occupancy_shift")
    ax.axhline(0, color="black", lw=0.6, zorder=1)
    ax.set_xlim(-0.6, 1.15 * (len(arms) - 1) + 0.6)
    ax.set_title(dom, fontsize=8, pad=3)
    style.panel(ax, letter, dx=-0.55 if dom == "FourRooms" else -0.72, dy=1.13)
axes[1].set_ylabel("shift in forget-region\noccupancy", linespacing=1.2)

style.save(fig, Path(__file__).parent / "fig_drift.pdf")
