"""Score the re-run online-unlearning grid against a no-pressure control.

The raw final `forget_region_fraction` of a run is not interpretable on
its own. Return climbs throughout the online phase, and in FourRooms the
reward-optimal route transits the forget region, so occupancy rises with
task competence whether or not any unlearning is applied. Every arm in
the grid ends *above* where it started, which reads as "nothing works"
and is an artefact.

The control arm (`no_op`, `disjoint_objective=false`, penalty 0.0) leaves
the objective untouched and computes forget labels for metrics only. It
is what the other arms have to beat. Comparisons are paired by seed,
since each seed shares a baseline checkpoint.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
from collections import defaultdict

from scipy import stats

CONTROL = "control_nopressure"
LABELS = [CONTROL, "noop_disjoint", "is_replay_disjoint", "neg_reward_0.1",
          "neg_reward_1.0", "neggrad", "kl_neggrad", "npo"]
FRAC = "eval/forget_region_fraction"
RET = "eval/mean_return"
SEL = "eval/selectivity"


def load(root: str, env: str):
    """label -> seed -> history. Hydra names dirs <env>_seed<s>_<method>_<label>,
    so the label is matched as a suffix, longest first."""
    runs = defaultdict(dict)
    for path in glob.glob(f"{root}/online_unlearn_{env}_seed*_*/history.json"):
        name = path.split("/")[-2]
        seed = int(re.search(r"seed(\d+)", name).group(1))
        label = next((L for L in sorted(LABELS, key=len, reverse=True)
                      if name.endswith("_" + L)), None)
        if label:
            runs[label][seed] = json.load(open(path))
    return runs


def final(history, key):
    vals = [rec[key] for rec in history if key in rec]
    return vals[-1] if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/outputs")
    ap.add_argument("--env", default="fourrooms")
    args = ap.parse_args()

    runs = load(args.root, args.env)
    if CONTROL not in runs:
        raise SystemExit(
            f"no {CONTROL!r} arm found. Without it the forget fractions are "
            "confounded with task learning and cannot be read.")

    print("FINAL VALUES (mean +- sd over seeds)")
    print(f"{'method':<20} {'forget_frac':<16} {'return':<16} {'selectivity'}")
    print("-" * 72)
    for label in LABELS:
        if label not in runs:
            continue
        fr = [final(h, FRAC) for h in runs[label].values()]
        rt = [final(h, RET) for h in runs[label].values()]
        se = [final(h, SEL) for h in runs[label].values()]
        tag = "  <-- CONTROL" if label == CONTROL else ""
        print(f"{label:<20} {st.mean(fr):.3f}+-{st.stdev(fr):.3f}    "
              f"{st.mean(rt):.3f}+-{st.stdev(rt):.3f}    {st.mean(se):.3f}{tag}")

    ctl = runs[CONTROL]
    n_tests = sum(1 for L in LABELS[1:] if L in runs)
    print(f"\nPAIRED vs CONTROL, final {FRAC}")
    print(f"{'method':<20} {'mean d':>8} {'95% CI':>20} {'t':>7} {'p':>7}   "
          f"{'d return':>9}")
    print("-" * 82)
    for label in LABELS[1:]:
        if label not in runs:
            continue
        seeds = sorted(set(runs[label]) & set(ctl))
        a = [final(runs[label][s], FRAC) for s in seeds]
        b = [final(ctl[s], FRAC) for s in seeds]
        d = [x - y for x, y in zip(a, b)]
        dr = st.mean([final(runs[label][s], RET) - final(ctl[s], RET)
                      for s in seeds])
        t, p = stats.ttest_rel(a, b)
        se = st.stdev(d) / len(d) ** 0.5
        crit = stats.t.ppf(0.975, len(d) - 1)
        lo, hi = st.mean(d) - crit * se, st.mean(d) + crit * se
        star = " *" if p < 0.05 else ""
        print(f"{label:<20} {st.mean(d):+8.3f} [{lo:+7.3f},{hi:+7.3f}] "
              f"{t:7.2f} {p:7.3f}{star}  {dr:+9.3f}")
    print(f"\n* p < 0.05 uncorrected. With {n_tests} comparisons the "
          f"Bonferroni threshold is {0.05 / n_tests:.4f}.")


if __name__ == "__main__":
    main()
