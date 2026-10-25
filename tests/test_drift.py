"""Specification drift measurement (Stage 3)."""

from __future__ import annotations

import numpy as np
import pytest

from scenarios import ForgetScenario, measure_drift, offset_sweep


def _scen(threshold, mode="any_step", name="s"):
    return ForgetScenario({
        "name": name,
        "description": "Threshold predicate on feature 0 for drift testing.",
        "match_mode": mode,
        "match_threshold": 0.15,
        "groups": [{"conditions": [
            {"dim": 0, "op": ">", "value": threshold}]}],
    })


def _traj(values):
    feats = [np.array([float(v)]) for v in values]
    return {"observations": feats, "actions": [0] * len(feats)}


class TestPerfectAgreement:
    def test_identical_specs_agree_everywhere(self):
        r = measure_drift(_scen(5), _scen(5), [_traj([1, 9, 2, 8])])
        assert r.disagreement == 0.0
        assert r.jaccard == 1.0
        assert r.kappa == 1.0
        assert r.metric_delta == 0.0
        assert r.aligned

    def test_counts_partition_the_steps(self):
        r = measure_drift(_scen(5), _scen(5), [_traj([1, 9, 2, 8])])
        assert r.both + r.only_label + r.only_metric + r.neither == r.n_steps
        assert r.n_steps == 4


class TestDisagreement:
    def test_different_thresholds_drift(self):
        # values 1,3,6,9 -- label>2 catches 3 of them, metric>7 catches 1
        r = measure_drift(_scen(2), _scen(7), [_traj([1, 3, 6, 9])])
        assert r.both == 1
        assert r.only_label == 2
        assert r.only_metric == 0
        assert r.disagreement == pytest.approx(0.5)
        assert r.metric_delta == pytest.approx(0.5)
        assert not r.aligned

    def test_metric_delta_is_the_rate_gap(self):
        r = measure_drift(_scen(2), _scen(7), [_traj([1, 3, 6, 9])])
        assert r.forget_eff_label == pytest.approx(1 - 3 / 4)
        assert r.forget_eff_metric == pytest.approx(1 - 1 / 4)
        assert r.metric_delta == pytest.approx(
            abs(r.forget_eff_metric - r.forget_eff_label))

    def test_disjoint_specs_have_zero_jaccard(self):
        low = _scen(-1)      # matches everything
        traj = {"observations": [np.array([0.0]), np.array([0.0])],
                "actions": [0, 0]}
        never = ForgetScenario({
            "name": "never", "description": "Matches nothing, by construction.",
            "groups": [{"conditions": [{"dim": 0, "op": ">", "value": 1e9}]}],
        })
        r = measure_drift(low, never, [traj])
        assert r.both == 0
        assert r.jaccard == 0.0


class TestOffsets:
    def test_offset_shifts_which_state_is_read(self):
        # 0,0,9,0 -- label sees index t, metric with offset 1 sees t+1
        r = measure_drift(_scen(5), _scen(5), [_traj([0, 0, 9, 0])],
                          metric_offset=1)
        assert r.only_label == 1   # label fires at t=2
        assert r.only_metric == 1  # metric fires at t=1 (reads index 2)

    def test_offset_past_the_end_counts_as_no_match(self):
        # Single step: the label reads index 0 and matches; the metric
        # would read index 1, which is off the end, so it cannot fire.
        r = measure_drift(_scen(5), _scen(5), [_traj([9])], metric_offset=1)
        assert r.n_steps == 1
        assert r.only_label == 1
        assert r.only_metric == 0

    def test_offset_moves_a_match_earlier(self):
        # The match does not vanish, it shifts: the metric sees the final
        # state one step sooner than the label does.
        r = measure_drift(_scen(5), _scen(5), [_traj([0, 0, 0, 9])],
                          metric_offset=1)
        assert (r.only_label, r.only_metric) == (1, 1)
        assert r.rate_label == r.rate_metric

    def test_zero_offset_is_identity(self):
        assert measure_drift(_scen(5), _scen(5), [_traj([1, 9])],
                             metric_offset=0).aligned

    def test_sweep_returns_one_report_per_offset(self):
        reports = offset_sweep(_scen(5), [_traj([1, 9, 2])], offsets=(0, 1, 2))
        assert [r.metric_offset for r in reports] == [0, 1, 2]
        assert reports[0].aligned

    def test_negative_offset_supported(self):
        r = measure_drift(_scen(5), _scen(5), [_traj([9, 0, 0])],
                          metric_offset=-1)
        assert r.only_label == 1
        assert r.only_metric == 1   # metric reads index 0 at t=1


class TestTrajectoryLevel:
    def test_trajectory_selection_disagreement(self):
        label = _scen(5, mode="proportion")
        metric = _scen(5, mode="any_step")
        # one in-region step out of 10 -- any_step selects it, proportion
        # (threshold 0.15) does not
        traj = _traj([0] * 9 + [9])
        r = measure_drift(label, metric, [traj])
        assert r.traj_only_metric == 1
        assert r.traj_disagreement == 1.0

    def test_step_masks_ignore_match_mode(self):
        """Step-level masking bypasses the declared trajectory lifting --
        the drift this whole module exists to surface."""
        traj = _traj([0] * 9 + [9])
        prop = measure_drift(_scen(5, mode="proportion"),
                             _scen(5, mode="any_step"), [traj])
        assert prop.disagreement == 0.0      # identical step predicates
        assert prop.traj_disagreement == 1.0  # different trajectory verdicts


class TestEdgeCases:
    def test_no_trajectories(self):
        r = measure_drift(_scen(5), _scen(5), [])
        assert r.n_steps == 0
        assert r.disagreement == 0.0
        assert r.jaccard == 1.0

    def test_kappa_is_one_when_both_match_everything(self):
        r = measure_drift(_scen(-1), _scen(-1), [_traj([1, 2, 3])])
        assert r.kappa == 1.0

    def test_render_mentions_the_headline(self):
        text = measure_drift(_scen(2), _scen(7), [_traj([1, 9])]).render()
        assert "METRIC DELTA" in text

    def test_render_notes_alignment(self):
        assert "agree on every step" in measure_drift(
            _scen(5), _scen(5), [_traj([1, 9])]).render()

    def test_recorded_features_are_preferred(self):
        traj = {
            "observations": [np.array([0.0]), np.array([0.0])],
            "features": [np.array([9.0]), np.array([9.0])],
            "actions": [0, 0],
        }
        r = measure_drift(_scen(5), _scen(5), [traj])
        assert r.both == 2
