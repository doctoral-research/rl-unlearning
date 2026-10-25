"""Typed-dim and range validation (Stage 2).

A bound FeatureView carries a dtype and often a range per feature, which
makes a class of silent scenario bugs catchable at validation time.
"""

from __future__ import annotations

import pytest

from scenarios import validate_scenario_dict
from scenarios.views import (
    BOOLEAN,
    CATEGORICAL,
    CONTINUOUS,
    ORDINAL,
    FeatureView,
)


def _view():
    return FeatureView(
        names=("pos", "heading", "holding", "speed"),
        extract_fn=lambda obs, info, env: list(obs),
        dtypes=(ORDINAL, CATEGORICAL, BOOLEAN, CONTINUOUS),
        ranges=((0.0, 18.0), (0.0, 3.0), (0.0, 1.0), None),
        provenance="declared:test",
    )


def _codes(cond, view=None):
    doc = {
        "name": "t", "description": "typed validation probe",
        "groups": [{"conditions": [cond]}],
    }
    rep = validate_scenario_dict(doc, view=view or _view())
    # Warnings and errors only: the lint layer's informational notices
    # (e.g. missing_version) are orthogonal to type checking.
    return {i.code for i in rep.warnings + rep.errors}


class TestDtypeChecks:
    def test_ordering_a_categorical_warns(self):
        assert "categorical_ordering" in _codes(
            {"dim": "heading", "op": ">", "value": 1})

    def test_equality_on_categorical_is_fine(self):
        assert "categorical_ordering" not in _codes(
            {"dim": "heading", "op": "==", "value": 2})

    def test_ordering_a_boolean_warns(self):
        assert "boolean_ordering" in _codes(
            {"dim": "holding", "op": ">", "value": 0})

    def test_non_binary_boolean_value_warns(self):
        assert "boolean_value" in _codes(
            {"dim": "holding", "op": "==", "value": 7})

    def test_binary_boolean_value_is_fine(self):
        assert "boolean_value" not in _codes(
            {"dim": "holding", "op": "==", "value": 1})

    def test_float_equality_warns(self):
        assert "float_equality" in _codes(
            {"dim": "speed", "op": "==", "value": 1.5})

    def test_ordinal_threshold_is_clean(self):
        assert _codes({"dim": "pos", "op": ">", "value": 9.5}) == set()


class TestRangeChecks:
    @pytest.mark.parametrize("op,value", [
        (">", 100), (">=", 19), ("<", 0), ("<=", -1), ("==", 50),
    ])
    def test_unsatisfiable_conditions_warn(self, op, value):
        assert "condition_never_true" in _codes(
            {"dim": "pos", "op": op, "value": value})

    @pytest.mark.parametrize("op,value", [(">=", 0), (">", -1), ("<=", 18)])
    def test_vacuous_conditions_warn(self, op, value):
        assert "condition_always_true" in _codes(
            {"dim": "pos", "op": op, "value": value})

    def test_in_range_threshold_is_clean(self):
        assert _codes({"dim": "pos", "op": ">", "value": 9.5}) == set()

    def test_abs_ops_skip_range_check(self):
        """abs() folds the interval, so the naive bound test does not apply."""
        assert "condition_never_true" not in _codes(
            {"dim": "pos", "op": "abs>", "value": 100})

    def test_feature_without_range_is_not_range_checked(self):
        assert "condition_never_true" not in _codes(
            {"dim": "speed", "op": ">", "value": 1e9})


class TestViewIsOptional:
    def test_no_view_means_no_type_checks(self):
        doc = {
            "name": "t", "description": "no view bound, so no dtype checks",
            "env_id": "CartPole-v1",
            "groups": [{"conditions": [
                {"dim": 0, "op": "==", "value": 1.5}]}],
        }
        rep = validate_scenario_dict(doc)
        assert "float_equality" not in {i.code for i in rep.issues}


    def test_view_supersedes_the_name_table(self):
        """With a view bound, names resolve against it rather than the
        hand-registered ENV_OBS_DIMS entry for the same env_id."""
        doc = {
            "name": "t", "description": "view overrides table",
            "env_id": "CartPole-v1",
            "groups": [{"conditions": [
                {"dim": "heading", "op": "==", "value": 1}]}],
        }
        rep = validate_scenario_dict(doc, view=_view())
        assert rep.ok
