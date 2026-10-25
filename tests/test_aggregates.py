"""Episode aggregates (grammar v1.2).

Named reductions over a whole trajectory, generalising the built-in
`return` and `length`. These closed four of the six recorded corpus gaps.
"""

from __future__ import annotations

import numpy as np
import pytest

from scenarios import ForgetScenario, ScenarioError
from scenarios.aggregates import (
    VALID_OPS,
    AggregateError,
    compile_aggregate_block,
    evaluate_aggregates,
)
from scenarios.views import FeatureView

VIEW = FeatureView(names=("x", "y", "v"),
                   extract_fn=lambda o, i, e: list(o),
                   provenance="declared:test")
NAMES = {"x": 0, "y": 1, "v": 2}


def _resolve(n):
    if n not in NAMES:
        raise AggregateError(f"unknown feature {n!r}")
    return NAMES[n]


def _traj(rows):
    feats = [np.asarray(r, dtype=float) for r in rows]
    return {"observations": feats, "actions": [0] * len(feats)}


def _spec(aggregates, groups, derived=None):
    spec = {
        "name": "agg", "description": "Episode aggregate under test.",
        "match_mode": "trajectory", "applies_to": "whole_trajectory",
        "aggregates": aggregates, "groups": groups,
    }
    if derived:
        spec["derived"] = derived
    return spec


class TestCompilation:
    def test_all_ops_accepted(self):
        for op in VALID_OPS:
            spec = {"op": op}
            if op == "count_distinct":
                spec["of"] = ["x", "y"]
            elif op != "count":
                spec["of"] = "x"
            assert compile_aggregate_block({"a": spec}, _resolve, lambda g: g)

    def test_unknown_op_rejected(self):
        with pytest.raises(AggregateError, match="op must be one of"):
            compile_aggregate_block({"a": {"op": "median", "of": "x"}},
                                    _resolve, lambda g: g)

    def test_count_takes_no_feature(self):
        with pytest.raises(AggregateError, match="takes no 'of'"):
            compile_aggregate_block({"a": {"op": "count", "of": "x"}},
                                    _resolve, lambda g: g)

    def test_count_distinct_needs_features(self):
        with pytest.raises(AggregateError, match="needs 'of'"):
            compile_aggregate_block({"a": {"op": "count_distinct"}},
                                    _resolve, lambda g: g)

    def test_scalar_reducer_needs_exactly_one_feature(self):
        with pytest.raises(AggregateError, match="exactly one"):
            compile_aggregate_block({"a": {"op": "sum", "of": ["x", "y"]}},
                                    _resolve, lambda g: g)

    def test_unknown_feature_fails_at_load(self):
        with pytest.raises(AggregateError, match="unknown feature"):
            compile_aggregate_block({"a": {"op": "sum", "of": "nope"}},
                                    _resolve, lambda g: g)

    def test_where_only_on_counters(self):
        with pytest.raises(AggregateError, match="only applies to"):
            compile_aggregate_block(
                {"a": {"op": "sum", "of": "x", "where": {}}},
                _resolve, lambda g: g)

    def test_block_must_be_mapping(self):
        with pytest.raises(AggregateError, match="must be a mapping"):
            compile_aggregate_block(["a"], _resolve, lambda g: g)

    def test_empty_block(self):
        assert compile_aggregate_block(None, _resolve, lambda g: g) == {}


class TestReductions:
    ROWS = [(1, 1, 2.0), (1, 2, 4.0), (1, 1, -6.0)]

    def _eval(self, aggs):
        compiled = compile_aggregate_block(aggs, _resolve, lambda g: g)
        return evaluate_aggregates(compiled, [np.asarray(r, float)
                                              for r in self.ROWS])

    @pytest.mark.parametrize("op,expected", [
        ("sum", 0.0), ("mean", 0.0), ("min", -6.0), ("max", 4.0),
        ("range", 10.0), ("first", 2.0), ("last", -6.0), ("abs_sum", 12.0),
        ("n_distinct", 3.0),
    ])
    def test_scalar_reducers(self, op, expected):
        assert self._eval({"a": {"op": op, "of": "v"}})["a"] == pytest.approx(expected)

    def test_count_is_step_count(self):
        assert self._eval({"a": {"op": "count"}})["a"] == 3.0

    def test_count_distinct_over_a_tuple(self):
        # (1,1) appears twice, (1,2) once -> 2 distinct cells
        assert self._eval({"a": {"op": "count_distinct", "of": ["x", "y"]}})["a"] == 2.0

    def test_empty_trajectory_is_safe(self):
        compiled = compile_aggregate_block({"a": {"op": "mean", "of": "v"}},
                                           _resolve, lambda g: g)
        assert evaluate_aggregates(compiled, [])["a"] == 0.0

    def test_non_finite_values_ignored(self):
        compiled = compile_aggregate_block({"a": {"op": "sum", "of": "v"}},
                                           _resolve, lambda g: g)
        rows = [np.array([0, 0, 1.0]), np.array([0, 0, np.nan]),
                np.array([0, 0, 2.0])]
        assert evaluate_aggregates(compiled, rows)["a"] == 3.0


class TestScenarioIntegration:
    def test_aggregate_condition_matches(self):
        s = ForgetScenario(_spec(
            {"steps": {"op": "count"}},
            [{"conditions": [{"dim": "steps", "op": ">", "value": 3}]}]),
            view=VIEW)
        assert s.identify_forget_trajectories([_traj([(0,0,0)] * 5)]) == [0]
        assert s.identify_forget_trajectories([_traj([(0,0,0)] * 2)]) == []

    def test_derived_can_combine_aggregates(self):
        s = ForgetScenario(_spec(
            {"cells": {"op": "count_distinct", "of": ["x", "y"]},
             "steps": {"op": "count"}},
            [{"conditions": [{"dim": "revisits", "op": ">", "value": 0}]}],
            derived={"revisits": "steps - cells"}), view=VIEW)
        assert s.identify_forget_trajectories(
            [_traj([(1,1,0), (1,2,0), (1,1,0)])]) == [0]
        assert s.identify_forget_trajectories(
            [_traj([(1,1,0), (1,2,0), (1,3,0)])]) == []

    def test_aggregates_compose_with_boolean_groups(self):
        s = ForgetScenario(_spec(
            {"steps": {"op": "count"}, "peak": {"op": "max", "of": "v"}},
            [{"all_of": [
                {"conditions": [{"dim": "steps", "op": ">", "value": 2}]},
                {"conditions": [{"dim": "peak", "op": ">", "value": 5}]},
            ]}]), view=VIEW)
        assert s.identify_forget_trajectories(
            [_traj([(0,0,9)] * 4)]) == [0]
        assert s.identify_forget_trajectories(
            [_traj([(0,0,1)] * 4)]) == []

    def test_unknown_feature_is_a_scenario_error(self):
        with pytest.raises(ScenarioError):
            ForgetScenario(_spec(
                {"a": {"op": "sum", "of": "missing"}},
                [{"conditions": [{"dim": "a", "op": ">", "value": 0}]}]),
                view=VIEW)

    def test_step_level_use_of_an_aggregate_reports_unknown(self):
        """Aggregates describe an episode, so a step-level consumer cannot
        evaluate one -- it must report unknown, not crash mid-rollout."""
        s = ForgetScenario(_spec(
            {"steps": {"op": "count"}},
            [{"conditions": [{"dim": "ratio", "op": ">", "value": 1}]}],
            derived={"ratio": "steps / 2"}), view=VIEW)
        assert s.matches_features(np.array([0.0, 0.0, 0.0])) is False

    def test_validator_rejects_aggregate_in_step_level_mode(self):
        """A declared aggregate compared at a single step raises during
        evaluation, so the validator must reject it up front."""
        from scenarios.validate import validate_scenario_dict
        doc = dict(_spec({"steps": {"op": "count"}},
                         [{"conditions": [{"dim": "steps", "op": ">", "value": 1}]}]),
                   match_mode="proportion", match_threshold=0.1)
        rep = validate_scenario_dict(doc, view=VIEW)
        assert "aggregate_outside_trajectory" in {i.code for i in rep.errors}
        doc["match_mode"] = "trajectory"
        doc.pop("match_threshold")
        rep = validate_scenario_dict(doc, view=VIEW)
        assert "aggregate_outside_trajectory" not in {i.code for i in rep.errors}

    def test_built_in_aggregates_still_work(self):
        s = ForgetScenario({
            "name": "ret", "description": "Built-in return aggregate.",
            "match_mode": "trajectory",
            "groups": [{"conditions": [
                {"target": "return", "op": ">", "value": 1.0}]}]})
        t = _traj([(0,0,0)] * 3); t["rewards"] = [1.0, 1.0, 0.0]
        assert s.identify_forget_trajectories([t]) == [0]


class TestShippedV12Scenarios:
    """The v1.2 scenarios in benchmark/ must stay loadable."""

    @pytest.mark.parametrize("rel", [
        "fourrooms/no_backtracking", "fourrooms/no_pacing",
        "halfcheetah/no_front_flip",
    ])
    def test_loads_and_has_aggregates(self, rel):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "benchmark" / "scenarios"
        s = ForgetScenario.load(str(root / f"{rel}.yaml"))
        assert s.aggregates
        assert s.applies_to_declared
