"""Fidelity on the fifteen labelled entries, and where disagreement lives.

For each entry, two numbers on one axis: how far the readers of the
description agree with each other, and how far their labels agree with the
formalisation. Reading the two together says which side a disagreement
belongs to. Readers who agree with one another but not with the
formalisation point at the formalisation; readers who disagree with one
another point at the sentence.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, str(Path(__file__).parent))
import style  # noqa: E402

style.setup()
DATA = Path(__file__).resolve().parents[1] / "data"
M = json.loads((DATA / "measurements.json").read_text())
spec = {r["entry"]: r for r in M["corpus"]["fidelity"]}
read = {r["entry"]: r for r in M["readers"]["entries"]}
assert set(spec) == set(read), "fidelity and reader tables cover different entries"

EXACT = 0.999


def group(e):
    rr, rs = read[e]["mean_kappa"], spec[e]["kappa"]
    if rr < EXACT:
        return 2
    return 0 if rs >= EXACT else 1


TITLES = ["readers agree, formalisation agrees",
          "readers agree, formalisation differs",
          "readers disagree with each other"]
COLS = [style.GOOD, style.ACCENT, style.WARN]
partial = {r["entry"] for r in M["corpus"]["fidelity"] if r["verdict"] == "partial"}

rows = []
for g in range(3):
    es = [e for e in spec if group(e) == g]
    es.sort(key=lambda e: (-spec[e]["kappa"], e))
    rows.append(es)

fig, ax = plt.subplots(figsize=(style.TEXTWIDTH, 3.15))
y, yt, ylab, heads = 0.0, [], [], []
for g, es in enumerate(rows):
    heads.append((y, g))
    y -= 1.0
    for e in es:
        rr, rs = read[e]["mean_kappa"], spec[e]["kappa"]
        ax.plot([rs, rr], [y, y], color=style.MUTE, lw=0.8, zorder=1)
        # The ring is drawn larger than the dot so that both stay visible
        # where the two agreements coincide.
        ax.scatter([rr], [y], s=44, facecolor="white", edgecolor=style.INK,
                   lw=1.0, zorder=3)
        ax.scatter([rs], [y], s=17, color=COLS[g], edgecolor="none", zorder=4)
        yt.append(y)
        ylab.append(e + (r"$^{\dagger}$" if e in partial else ""))
        y -= 1.0
    y -= 0.35

for y0, g in heads:
    ax.text(-0.012, y0 - 0.12, TITLES[g], transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=7.4, style="italic", color=COLS[g])

ax.set_yticks(yt)
ax.set_yticklabels(ylab, fontsize=7.2, family="monospace")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlim(-0.08, 1.04)
ax.set_ylim(y + 0.75, 0.45)
ax.set_xlabel(r"Cohen's $\kappa$")
ax.axvline(0, color="black", lw=0.6, zorder=0)
ax.xaxis.grid(True)
ax.set_axisbelow(True)

handles = [Line2D([], [], ls="none", marker="o", ms=6, mfc="white", mec=style.INK,
                  label="reader vs. reader"),
           Line2D([], [], ls="none", marker="o", ms=4, mfc="#555555", mec="none",
                  label="formalisation vs. readers")]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=2,
          fontsize=7.5, handletextpad=0.3, columnspacing=1.6, borderpad=0.2)

style.save(fig, Path(__file__).parent / "fig_fidelity.pdf")
