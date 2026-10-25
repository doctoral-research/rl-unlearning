"""Figure 4: a learned view is admitted only when it is measured.

Held-out predictions against ground truth. The full-grid render
determines the agent's position and the probe recovers it; the egocentric
observation does not determine it at all, and no amount of fitting will
change that. The gate refuses the second rather than returning confident
numbers that mean nothing.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import style  # noqa: E402

style.setup()
D = json.loads((Path(__file__).resolve().parents[1] / "data" /
                "probe_predictions.json").read_text())
GATE = 0.9

fig, axes = plt.subplots(1, 2, figsize=(style.TEXTWIDTH * 0.78, 2.05),
                         gridspec_kw={"wspace": 0.30})
rng = np.random.default_rng(0)
panels = [("full_grid", "full-grid render", "a"),
          ("egocentric", "egocentric $7{\\times}7{\\times}3$", "b")]

for ax, (key, title, letter) in zip(axes, panels):
    d = D[key]
    ok = min(d["exact_match"]) >= GATE
    for coord, col, lab in (("x", style.INK, "$x$"), ("y", style.ACCENT, "$y$")):
        t = np.asarray(d[f"true_{coord}"])
        p = np.asarray(d[f"pred_{coord}"])
        j = rng.uniform(-0.28, 0.28, size=len(t))
        ax.scatter(t + j, p, s=1.6, color=col, alpha=0.20, edgecolor="none",
                   zorder=2, rasterized=True, label=lab)
    lo, hi = 0, 18
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.7, ls=(0, (4, 2)),
            zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-4, 22)
    ax.set_xlabel("true coordinate")
    ax.set_title(f"{title}\n{d['width']} input values", fontsize=8,
                 pad=3, linespacing=1.3)
    txt = (f"exact match {d['exact_match'][0]:.2f} / {d['exact_match'][1]:.2f}")
    ax.text(0.04, 0.965, txt, transform=ax.transAxes, fontsize=7.5,
            va="top", ha="left")
    ax.text(0.04, 0.875,
            "admitted" if ok else f"refused (gate {GATE:g})",
            transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
            color=style.GOOD if ok else style.ACCENT)
    style.panel(ax, letter, dx=-0.24, dy=1.20)

axes[0].set_ylabel("predicted coordinate", labelpad=2)
leg = axes[0].legend(loc="lower right", handletextpad=0.2, labelspacing=0.15,
                     borderpad=0.15, fontsize=7.5, markerscale=5,
                     title=None)
for h in leg.legend_handles:
    h.set_alpha(1.0)
style.save(fig, Path(__file__).parent / "fig_probe.pdf")
