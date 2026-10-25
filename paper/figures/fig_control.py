"""Figure 5: without a control, every method looks like a failure.

(a) Forget-region occupancy over online unlearning. Every arm, including
the ones that do nothing to the objective, ends above where it started:
return rises over the same period and the reward-optimal route transits
the region, so occupancy tracks competence. The control arm -- same
scenario, labels computed for metrics only -- is what the others have to
beat, and it is cheap to construct only because the target is a
declarative object rather than inline code.
(b) Endpoint effect against that control, paired by seed.
"""
from __future__ import annotations
import glob, json, re, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
sys.path.insert(0, str(Path(__file__).parent))
import style  # noqa: E402

style.setup()
ROOT = Path(__file__).resolve().parents[2]
KEY = "eval/forget_region_fraction"
CONTROL = "control_nopressure"
ORDER = ["noop_disjoint", "is_replay_disjoint", "npo", "kl_neggrad",
         "neggrad", "neg_reward_1.0", "neg_reward_0.1"]
PRETTY = {"noop_disjoint": "disjoint objective",
          "is_replay_disjoint": "IS-replay + disjoint",
          "npo": "NPO", "kl_neggrad": "NegGrad + KL", "neggrad": "NegGrad",
          "neg_reward_1.0": r"reward penalty $\lambda{=}1.0$",
          "neg_reward_0.1": r"reward penalty $\lambda{=}0.1$"}
SHOW = [(CONTROL, "control (no pressure)", style.MUTE, "-"),
        ("is_replay_disjoint", "IS-replay + disjoint", style.WARN, "-."),
        ("neggrad", "NegGrad", style.GOOD, "--"),
        ("neg_reward_0.1", r"reward penalty $\lambda{=}0.1$", style.INK, "-")]

series, finals = {}, {}
for p in glob.glob(str(ROOT / "experiments/outputs/online_unlearn_fourrooms_seed*_*/history.json")):
    name = Path(p).parent.name
    seed = int(re.search(r"seed(\d+)", name).group(1))
    lab = next((L for L in sorted(ORDER + [CONTROL], key=len, reverse=True)
                if name.endswith("_" + L)), None)
    if not lab:
        continue
    h = json.load(open(p))
    ev = [(x["iter"], x[KEY]) for x in h if KEY in x]
    series.setdefault(lab, []).append(([i for i, _ in ev], [v for _, v in ev]))
    finals.setdefault(lab, {})[seed] = ev[-1][1]

fig, (ax, axf) = plt.subplots(1, 2, figsize=(style.TEXTWIDTH, 2.15),
                              gridspec_kw={"width_ratios": [1.0, 1.05],
                                           "wspace": 1.05})
for lab, pretty, col, ls in SHOW:
    runs = series[lab]
    x = np.asarray(runs[0][0])
    style.band(ax, x, [r[1] for r in runs], col, label=pretty, ls=ls)
ax.set_xlabel("PPO update")
ax.set_ylabel("forget-region\noccupancy", linespacing=1.25)
ax.set_xlim(0, x.max())
style.ygrid(ax)
h, lb = ax.get_legend_handles_labels()
fig.legend(h, lb, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=4,
           fontsize=7, handlelength=1.6, columnspacing=1.1,
           borderpad=0.0, frameon=False)
style.panel(ax, "a", dx=-0.32, dy=1.10)

ctl = finals[CONTROL]
ys = np.arange(len(ORDER))[::-1]
for y, lab in zip(ys, ORDER):
    seeds = sorted(set(finals[lab]) & set(ctl))
    d = np.array([finals[lab][s] - ctl[s] for s in seeds])
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    crit = stats.t.ppf(0.975, len(d) - 1)
    sig = abs(m) - crit * se > 0
    col = style.INK if sig else style.MUTE
    axf.scatter(d, np.full(len(d), y) + 0.22, s=5, color=style.MUTE,
                alpha=0.5, edgecolor="none", zorder=2)
    axf.errorbar(m, y, xerr=crit * se, fmt="o", ms=4.0, color=col,
                 ecolor=col, elinewidth=1.1, capsize=2.0, zorder=3)
axf.axvline(0, color="black", lw=0.7, zorder=1)
axf.set_yticks(ys)
axf.set_yticklabels([PRETTY[l] for l in ORDER], fontsize=7.5)
axf.set_xlabel("change vs control at the final update")
axf.set_ylim(-0.75, len(ORDER) - 0.2)
axf.xaxis.grid(True)
axf.set_axisbelow(True)
axf.spines["left"].set_visible(False)
axf.tick_params(axis="y", length=0)
style.panel(axf, "b", dx=-0.62, dy=1.10)

style.save(fig, Path(__file__).parent / "fig_control.pdf")
