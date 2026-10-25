"""Specification drift: does the forget set you TRAIN on equal the one you
MEASURE against?

Unlearning pipelines define the forget set twice. Once to build the label
that drives the update (which transitions get the gradient treatment), and
once to compute the reported metric (what fraction of behaviour was
removed). Those two definitions are usually written in different places,
sometimes in different files, and nothing checks that they agree.

When they disagree, the reported number is not measuring the thing that
was optimised. The gap is invisible: both halves run without error and
produce plausible values.

This module measures the gap. Given the label specification, the metric
specification, and a set of trajectories, it reports how often the two
disagree at step and trajectory level, and — the number that matters —
how much the reported forget-effectiveness moves as a result.

Drift has several ordinary causes, all of which this can quantify:

* different thresholds or regions in the two definitions;
* an **index offset**, e.g. the label computed on the state the agent
  acted from and the metric on the state it landed in (`metric_offset=1`);
* one side evaluated on raw observations, the other on view features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .scenario import ForgetScenario, _feature_seq


@dataclass(frozen=True)
class DriftReport:
    """Agreement between a label specification and a metric specification."""

    n_trajectories: int
    n_steps: int

    # Step-level
    both: int
    only_label: int
    only_metric: int
    neither: int

    # Trajectory-level
    traj_only_label: int
    traj_only_metric: int

    # Derived
    rate_label: float
    rate_metric: float
    metric_offset: int

    @property
    def disagreement(self) -> float:
        """Fraction of steps the two specifications classify differently."""
        return (self.only_label + self.only_metric) / max(self.n_steps, 1)

    @property
    def jaccard(self) -> float:
        """|A n B| / |A u B| over matched steps. 1.0 = identical."""
        union = self.both + self.only_label + self.only_metric
        return self.both / union if union else 1.0

    @property
    def kappa(self) -> float:
        """Cohen's kappa: agreement corrected for chance."""
        n = max(self.n_steps, 1)
        po = (self.both + self.neither) / n
        p_label = (self.both + self.only_label) / n
        p_metric = (self.both + self.only_metric) / n
        pe = p_label * p_metric + (1 - p_label) * (1 - p_metric)
        return 1.0 if pe >= 1.0 else (po - pe) / (1 - pe)

    @property
    def traj_disagreement(self) -> float:
        return (self.traj_only_label + self.traj_only_metric) / max(
            self.n_trajectories, 1)

    @property
    def forget_eff_label(self) -> float:
        """Forget-effectiveness as the LABEL specification would score it."""
        return 1.0 - self.rate_label

    @property
    def forget_eff_metric(self) -> float:
        """Forget-effectiveness as the METRIC specification scores it."""
        return 1.0 - self.rate_metric

    @property
    def metric_delta(self) -> float:
        """Headline number: how far the reported forget-effectiveness sits
        from what the optimised objective actually removed."""
        return abs(self.forget_eff_metric - self.forget_eff_label)

    @property
    def aligned(self) -> bool:
        return self.only_label == 0 and self.only_metric == 0

    def render(self) -> str:
        lines = [
            f"trajectories        {self.n_trajectories}",
            f"steps               {self.n_steps}",
            f"metric offset       {self.metric_offset:+d}",
            "",
            f"label positive rate   {self.rate_label:.4f}",
            f"metric positive rate  {self.rate_metric:.4f}",
            "",
            f"both                {self.both}",
            f"label only          {self.only_label}",
            f"metric only         {self.only_metric}",
            f"neither             {self.neither}",
            "",
            f"step disagreement   {self.disagreement:.4f}",
            f"jaccard             {self.jaccard:.4f}",
            f"cohen kappa         {self.kappa:.4f}",
            f"traj disagreement   {self.traj_disagreement:.4f}",
            "",
            f"forget_eff (label)  {self.forget_eff_label:.4f}",
            f"forget_eff (metric) {self.forget_eff_metric:.4f}",
            f"METRIC DELTA        {self.metric_delta:.4f}",
        ]
        if self.aligned:
            lines.append("")
            lines.append("specifications agree on every step.")
        return "\n".join(lines)


def _step_mask(
    scenario: ForgetScenario, traj: Dict[str, Any], offset: int = 0,
) -> np.ndarray:
    """Per-step match mask for one trajectory.

    `offset` shifts which state the predicate reads: 0 is the state the
    agent acted from, +1 the state it landed in. Steps whose shifted
    index falls off the end are treated as non-matching, which mirrors
    what a rollout loop does at the episode boundary.
    """
    feats = _feature_seq(traj)
    actions = traj.get("actions") or [None] * len(feats)
    rewards = traj.get("rewards") or [None] * len(actions)
    n = len(actions)
    mask = np.zeros(n, dtype=bool)
    for t in range(n):
        src = t + offset
        if src < 0 or src >= len(feats):
            continue
        act = actions[t] if t < len(actions) else None
        rwd = rewards[t] if t < len(rewards) else None
        mask[t] = bool(scenario.matches_features(np.asarray(feats[src]), act, rwd))
    return mask


def compare_masks(
    masks_label: Sequence[np.ndarray],
    masks_metric: Sequence[np.ndarray],
    *,
    traj_only_label: int = 0,
    traj_only_metric: int = 0,
    metric_offset: int = 0,
) -> DriftReport:
    """Agreement statistics between two per-step boolean labellings.

    Split out from `measure_drift` so the same statistics can score a
    specification against *human* labels, which is what corpus fidelity
    measures. Any two labellings of the same steps are comparable here.
    """
    both = only_label = only_metric = neither = n_steps = 0
    for m_l, m_m in zip(masks_label, masks_metric):
        n = min(len(m_l), len(m_m))
        m_l, m_m = np.asarray(m_l[:n], bool), np.asarray(m_m[:n], bool)
        n_steps += n
        both += int(np.sum(m_l & m_m))
        only_label += int(np.sum(m_l & ~m_m))
        only_metric += int(np.sum(~m_l & m_m))
        neither += int(np.sum(~m_l & ~m_m))

    denom = max(n_steps, 1)
    return DriftReport(
        n_trajectories=len(list(masks_label)),
        n_steps=n_steps,
        both=both,
        only_label=only_label,
        only_metric=only_metric,
        neither=neither,
        traj_only_label=traj_only_label,
        traj_only_metric=traj_only_metric,
        rate_label=(both + only_label) / denom,
        rate_metric=(both + only_metric) / denom,
        metric_offset=metric_offset,
    )


def measure_drift(
    label: ForgetScenario,
    metric: ForgetScenario,
    trajectories: Sequence[Dict[str, Any]],
    *,
    metric_offset: int = 0,
) -> DriftReport:
    """Compare a label specification against a metric specification.

    `metric_offset` models the common off-by-one in which the training
    mask is computed on the pre-step state and the evaluation metric on
    the post-step state.
    """
    trajectories = list(trajectories)
    masks_label = [_step_mask(label, t, 0) for t in trajectories]
    masks_metric = [_step_mask(metric, t, metric_offset) for t in trajectories]

    idx_label = set(label.identify_forget_trajectories(trajectories))
    idx_metric = set(metric.identify_forget_trajectories(trajectories))

    return compare_masks(
        masks_label, masks_metric,
        traj_only_label=len(idx_label - idx_metric),
        traj_only_metric=len(idx_metric - idx_label),
        metric_offset=metric_offset,
    )


def scenario_masks(
    scenario: ForgetScenario, trajectories: Sequence[Dict[str, Any]],
    offset: int = 0,
) -> List[np.ndarray]:
    """Per-step masks a scenario assigns to each trajectory."""
    return [_step_mask(scenario, t, offset) for t in trajectories]


def offset_sweep(
    scenario: ForgetScenario,
    trajectories: Sequence[Dict[str, Any]],
    offsets: Sequence[int] = (-1, 0, 1, 2),
) -> List[DriftReport]:
    """Drift induced purely by an index offset, using ONE specification on
    both sides.

    This isolates the off-by-one: any non-zero disagreement here comes
    from reading a different state, not from a different predicate. It is
    the cheapest drift audit available, because it needs no second
    scenario file.
    """
    return [
        measure_drift(scenario, scenario, trajectories, metric_offset=k)
        for k in offsets
    ]
