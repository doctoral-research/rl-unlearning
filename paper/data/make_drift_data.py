"""Collect the drift experiment and its label-noise controls.

Writes measurements.json["drift"]. For each domain and each comparison arm
(the other reading of `applies_to`, and random label flips at 1% and, in
FourRooms, 10%), per seed:

  agreement        fraction of shared states on which the arm and the
                   matching-steps run choose the same greedy action
  occupancy_shift  final forget-region occupancy minus that of the
                   matching-steps run

plus the agreement between the matching-steps run and its baseline, which
says how far unlearning itself moved the policy, and the fraction of
rollout steps whose label each perturbation changes.

Policy agreement comes from scripts/drift_policy_divergence.py and label
disagreement from scripts/drift_label_disagreement.py; both write the JSON
files read here. Occupancy is read from the training histories.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

RT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent
KEY = "eval/forget_region_fraction"

DOMAINS = {
    "FourRooms": {"ref": "ab_ms", "prefix": "divergence_fourrooms",
                  "arms": {"wt": ("ab_wt", ""), "n1": ("ab_n1", "_noise1"),
                           "n10": ("ab_n10", "_noise10")}},
    "CartPole": {"ref": "cp_ms", "prefix": "divergence_cartpole",
                 "arms": {"wt": ("cp_wt", ""), "n1": ("cp_n1", "_noise1")}},
}
NOISE = {"wt": None, "n1": 0.01, "n10": 0.10}


def finals(suffix, key=KEY):
    out = {}
    for p in glob.glob(str(RT / f"experiments/outputs/*_{suffix}/history.json")):
        seed = int(re.search(r"seed(\d+)", p).group(1))
        out[seed] = [x[key] for x in json.load(open(p)) if key in x][-1]
    return out


labdis = json.loads((DATA / "drift_label_disagreement.json").read_text())
drift = {}
for dom, d in DOMAINS.items():
    ref_occ, ref_ret = finals(d["ref"]), finals(d["ref"], "eval/mean_return")
    spec_div = json.loads((DATA / f"{d['prefix']}.json").read_text())
    seeds = spec_div["seeds"]
    base = spec_div["arm_baseline_agreement"]
    arms = {}
    for arm, (suffix, tag) in d["arms"].items():
        div = json.loads((DATA / f"{d['prefix']}{tag}.json").read_text())
        assert div["seeds"] == seeds and np.allclose(div["arm_baseline_agreement"], base)
        occ, ret = finals(suffix), finals(suffix, "eval/mean_return")
        shift = [occ[s] - ref_occ[s] for s in seeds]
        arms[arm] = {
            "suffix": suffix, "label_noise": NOISE[arm],
            "agreement": div["arm_arm_agreement"],
            "occupancy_shift": shift,
            "return": [ret[s] for s in seeds],
            "p_occupancy_shift": float(stats.ttest_1samp(shift, 0.0).pvalue),
        }
    spec_ag = np.asarray(arms["wt"]["agreement"])
    for arm in arms:
        if arm == "wt":
            continue
        other = np.asarray(arms[arm]["agreement"])
        arms[arm]["p_agreement_vs_spec"] = float(stats.ttest_rel(spec_ag, other).pvalue)
        arms[arm]["p_agreement_vs_spec_wilcoxon"] = float(stats.wilcoxon(spec_ag, other).pvalue)
        arms[arm]["seeds_spec_lower"] = int((spec_ag < other).sum())
    drift[dom] = {
        "seeds": seeds, "n_states": spec_div["n_states"],
        "n_actions": spec_div["n_actions"],
        "baseline_agreement": base,
        "reference_return": [ref_ret[s] for s in seeds],
        "label_disagreement": {k: labdis[dom][k] for k in
                               ("mean_disagreement", "min", "max",
                                "mean_rate_ms", "mean_rate_wt")},
        "arms": arms,
    }

m = DATA / "measurements.json"
j = json.loads(m.read_text())
j["drift"] = drift
m.write_text(json.dumps(j, indent=2))

for dom, d in drift.items():
    ld = d["label_disagreement"]
    print(f"{dom}: readings disagree on {100 * ld['mean_disagreement']:.1f}% of labels; "
          f"agreement with baseline {np.mean(d['baseline_agreement']):.3f}")
    for arm, a in d["arms"].items():
        ag, sh = np.asarray(a["agreement"]), np.asarray(a["occupancy_shift"])
        extra = (f"  vs spec p={a['p_agreement_vs_spec']:.3f} ({a['seeds_spec_lower']}/15)"
                 if "p_agreement_vs_spec" in a else "")
        print(f"   {arm:<4} agreement {ag.mean():.3f} (sd {ag.std(ddof=1):.3f}){extra}   "
              f"occupancy {sh.mean():+.3f} p={a['p_occupancy_shift']:.3f}")
