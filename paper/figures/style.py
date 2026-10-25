"""Shared figure style.

Matched to the ICLR body text. The template loads `times`, which sets the
text in Nimbus Roman and typewriter text in Nimbus Mono, so the figures use
the same two faces, with STIX for mathematics. Everything is vector PDF.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# ICLR text block is 5.5in wide.
TEXTWIDTH = 5.5
HALFWIDTH = 2.68

# Colourblind-safe, low-saturation. INK is for emphasis, MUTE for context.
INK = "#1F3B57"        # deep blue, primary series
ACCENT = "#B3472F"     # brick, the "wrong"/contrast series
MUTE = "#8C8C8C"       # grey, reference lines and context
GOOD = "#3F7F5F"       # green, accepted / passing
WARN = "#C89B3C"       # amber, caution
LIGHT = "#D9E2EA"      # fill


def setup():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "font.monospace": ["Nimbus Mono PS", "DejaVu Sans Mono"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "grid.color": "#B0B0B0",
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def ygrid(ax):
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)


def panel(ax, letter, dx=-0.30, dy=1.16):
    """Bold panel letter, positioned outside the axes."""
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")


def smooth(y, w=5):
    """Centred rolling mean. Evaluation uses 16 episodes, so the raw
    per-checkpoint curve is dominated by evaluation noise rather than by
    the policy; smoothing per seed before aggregating shows the trend
    without pretending the underlying estimate is tighter than it is."""
    import numpy as np
    y = np.asarray(y, dtype=float)
    if w <= 1 or len(y) < w:
        return y
    pad = w // 2
    padded = np.pad(y, (pad, pad), mode="edge")
    ker = np.ones(w) / w
    return np.convolve(padded, ker, mode="valid")[:len(y)]


def band(ax, x, runs, color, label=None, ls="-", w=5):
    """Mean with a 95% CI band over seeds -- the standard way to show a
    learning curve, and honest about how few seeds there are."""
    import numpy as np
    from scipy import stats
    a = np.asarray([smooth(r, w) for r in runs], dtype=float)
    m = a.mean(0)
    n = a.shape[0]
    se = a.std(0, ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    ax.fill_between(x, m - h, m + h, color=color, alpha=0.16, lw=0, zorder=2)
    ax.plot(x, m, color=color, lw=1.3, ls=ls, label=label, zorder=3)
    return m


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")
