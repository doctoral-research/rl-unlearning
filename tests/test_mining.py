"""Specification mining from labelled demonstrations (stage 5).

Recovering the rule behind a set of flagged episodes, as an ordinary
scenario that can then be validated, rendered and used directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from scenarios import ForgetScenario
from scenarios.mining import (
    MiningError,
    candidate_literals,
    mine_rules,
    mine_scenario,
    rules_to_config,
)
from scenarios.views import FeatureView

VIEW = FeatureView(
    names=("x", "y", "flag"),
    extract_fn=lambda o, i, e: list(o),
    dtypes=("ordinal", "ordinal", "boolean"),
    provenance="declared:test",
)


def _traj(rows):
    feats = [np.asarray(r, dtype=float) for r in rows]
    return {"observations": feats, "actions": [0] * len(feats)}


def _grid(pred, n=12):
    """Split a grid of points by a ground-truth predicate."""
    pos, neg = [], []
    for x in range(n):
        for y in range(n):
            row = [(float(x), float(y), 0.0)]
            (pos if pred(x, y) else neg).append(_traj(row))
    return pos, neg


class TestCandidateLiterals:
    def test_thresholds_come_from_observed_values(self):
        X = np.array([[0.0, 0, 0], [10.0, 0, 0]])
        lits = candidate_literals(X, VIEW)
        assert (0, ">", 5.0) in lits

    def test_categorical_gets_equality_not_ordering(self):
        X = np.array([[0, 0, 0.0], [0, 0, 1.0]])
        ops = {op for d, op, _ in candidate_literals(X, VIEW) if d == 2}
        assert ops == {"==", "!="}

    def test_constant_feature_yields_no_literals(self):
        X = np.array([[1.0, 0, 0], [1.0, 0, 0]])
        assert [l for l in candidate_literals(X, VIEW) if l[0] == 0] == []

    def test_threshold_count_is_capped(self):
        X = np.column_stack([np.arange(200.0), np.zeros(200), np.zeros(200)])
        lits = [l for l in candidate_literals(X, VIEW, max_per_dim=5)
                if l[0] == 0]
        assert len({v for _, _, v in lits}) <= 5


class TestRecovery:
    def test_recovers_a_single_threshold(self):
        pos, neg = _grid(lambda x, y: x > 6)
        s, rep = mine_scenario(pos, neg, view=VIEW)
        assert rep.f1 == pytest.approx(1.0)
        assert rep.n_literals == 1

    def test_recovers_a_conjunction(self):
        pos, neg = _grid(lambda x, y: x > 6 and y > 6)
        s, rep = mine_scenario(pos, neg, view=VIEW)
        assert rep.f1 == pytest.approx(1.0)
        assert rep.n_rules == 1
        assert rep.n_literals == 2

    def test_recovers_a_disjunction_as_two_rules(self):
        pos, neg = _grid(lambda x, y: x > 9 or y > 9)
        s, rep = mine_scenario(pos, neg, view=VIEW, max_rules=3)
        assert rep.recall == pytest.approx(1.0)
        assert rep.n_rules >= 2

    def test_recovered_scenario_reproduces_the_labels(self):
        """The real test: the emitted scenario, re-evaluated, agrees."""
        pos, neg = _grid(lambda x, y: x > 6 and y > 6)
        s, _ = mine_scenario(pos, neg, view=VIEW)
        assert all(s.identify_forget_trajectories([t]) for t in pos)
        assert not any(s.identify_forget_trajectories([t]) for t in neg)

    def test_prefers_few_literals(self):
        pos, neg = _grid(lambda x, y: x > 6)
        _, rep = mine_scenario(pos, neg, view=VIEW, max_literals=3)
        assert rep.n_literals == 1     # not padded out to the budget


class TestReport:
    def test_counts_are_consistent(self):
        pos, neg = _grid(lambda x, y: x > 6)
        _, rep = mine_scenario(pos, neg, view=VIEW)
        assert rep.tp + rep.fn == rep.n_pos
        assert 0.0 <= rep.precision <= 1.0
        assert 0.0 <= rep.recall <= 1.0

    def test_render_mentions_size_and_fit(self):
        pos, neg = _grid(lambda x, y: x > 6)
        _, rep = mine_scenario(pos, neg, view=VIEW)
        text = rep.render()
        assert "precision" in text and "size" in text

    def test_imperfect_separation_is_reported_not_hidden(self):
        """Labels a threshold rule cannot express: fit must be < 1."""
        pos, neg = _grid(lambda x, y: (x + y) % 2 == 0)
        _, rep = mine_scenario(pos, neg, view=VIEW, max_rules=2)
        assert rep.f1 < 1.0


class TestDegenerateInputs:
    def test_no_positives(self):
        X = np.zeros((4, 3)); y = np.zeros(4, bool)
        with pytest.raises(MiningError, match="no positive"):
            mine_rules(X, y, VIEW)

    def test_all_positives(self):
        X = np.zeros((4, 3)); y = np.ones(4, bool)
        with pytest.raises(MiningError, match="no contrast"):
            mine_rules(X, y, VIEW)

    def test_empty_input(self):
        with pytest.raises(MiningError, match="no samples"):
            mine_rules(np.empty((0, 3)), np.zeros(0, bool), VIEW)

    def test_needs_both_classes_of_trajectory(self):
        pos, _ = _grid(lambda x, y: True, n=2)
        with pytest.raises(MiningError, match="both positive and negative"):
            mine_scenario(pos, [], view=VIEW)

    def test_mismatched_feature_width(self):
        a = _traj([(1.0, 1.0, 0.0)])
        b = {"observations": [np.array([1.0, 1.0])], "actions": [0]}
        with pytest.raises(MiningError, match="width differs"):
            mine_scenario([a], [b], view=VIEW)


class TestEmittedConfig:
    def test_uses_feature_names_not_indices(self):
        doc = rules_to_config([[(0, ">", 5.0)]], VIEW)
        cond = doc["groups"][0]["conditions"][0]
        assert cond["dim"] == "x"
        assert cond["dim_name"] == "x"

    def test_falls_back_to_indices_without_a_view(self):
        doc = rules_to_config([[(0, ">", 5.0)]], None)
        assert doc["groups"][0]["conditions"][0]["dim"] == 0

    def test_declares_step_scope(self):
        assert rules_to_config([[(0, ">", 5.0)]], VIEW)["applies_to"]

    def test_emitted_config_is_a_valid_scenario(self):
        doc = rules_to_config([[(0, ">", 5.0), (1, "<=", 2.0)]], VIEW)
        s = ForgetScenario(doc, view=VIEW)
        assert len(s.groups) == 1

    def test_emitted_config_passes_validation(self):
        from scenarios import validate_scenario_dict

        doc = rules_to_config([[(0, ">", 5.0)]], VIEW)
        rep = validate_scenario_dict(doc, view=VIEW)
        assert not rep.errors

    def test_disjunction_becomes_multiple_groups(self):
        doc = rules_to_config([[(0, ">", 5.0)], [(1, ">", 5.0)]], VIEW)
        assert len(doc["groups"]) == 2
