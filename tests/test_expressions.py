"""Tests for derived features (Stage 2: relational expressiveness)."""

from __future__ import annotations

import numpy as np
import pytest

from scenarios import ForgetScenario, ScenarioError
from scenarios.expressions import (
    ALLOWED_FUNCS,
    ExpressionError,
    compile_derived_block,
    compile_expression,
    evaluate,
    evaluate_all,
)
from scenarios.views import FeatureView

NAMES = {"agent_x": 0, "agent_y": 1, "vx": 2}


def _resolve(name):
    if name not in NAMES:
        raise ExpressionError(f"unknown feature {name!r}")
    return NAMES[name]


def _view(names=("agent_x", "agent_y", "vx")):
    return FeatureView(
        names=tuple(names),
        extract_fn=lambda obs, info, env: list(obs),
        provenance="declared:test",
    )


# ----------------------------------------------------------------------
# Expression compilation
# ----------------------------------------------------------------------


class TestCompileExpression:
    def test_arithmetic(self):
        f = compile_expression("t", "agent_x + agent_y * 2", _resolve)
        assert f([1.0, 3.0, 0.0], {}) == 7.0

    def test_unary_minus(self):
        f = compile_expression("t", "-agent_x", _resolve)
        assert f([4.0, 0.0, 0.0], {}) == -4.0

    def test_precedence_and_parens(self):
        f = compile_expression("t", "(agent_x + agent_y) * 2", _resolve)
        assert f([1.0, 3.0, 0.0], {}) == 8.0

    def test_whitelisted_functions(self):
        f = compile_expression("t", "hypot(agent_x, agent_y)", _resolve)
        assert f([3.0, 4.0, 0.0], {}) == pytest.approx(5.0)

    def test_records_inputs(self):
        f = compile_expression("t", "abs(agent_x) + vx", _resolve)
        assert f.inputs == ("agent_x", "vx")

    def test_division_by_zero_is_nan_not_crash(self):
        f = compile_expression("t", "agent_x / agent_y", _resolve)
        assert np.isnan(f([1.0, 0.0, 0.0], {}))

    @pytest.mark.parametrize("expr", [
        "__import__('os').system('echo pwned')",
        "agent_x.__class__",
        "open('/etc/passwd')",
        "[agent_x for _ in range(3)]",
        "lambda: 1",
        "agent_x if agent_y else 0",
        "agent_x and agent_y",
        "{'a': 1}",
    ])
    def test_rejects_non_arithmetic(self, expr):
        with pytest.raises(ExpressionError):
            compile_expression("t", expr, _resolve)

    def test_rejects_unknown_function(self):
        with pytest.raises(ExpressionError, match="not allowed"):
            compile_expression("t", "eval('1')", _resolve)

    def test_rejects_keyword_arguments(self):
        with pytest.raises(ExpressionError, match="keyword"):
            compile_expression("t", "min(agent_x, key=1)", _resolve)

    def test_rejects_unknown_feature(self):
        with pytest.raises(ExpressionError, match="unknown feature"):
            compile_expression("t", "not_a_feature + 1", _resolve)

    def test_rejects_syntax_error(self):
        with pytest.raises(ExpressionError, match="cannot parse"):
            compile_expression("t", "agent_x +", _resolve)

    def test_rejects_non_string_source(self):
        with pytest.raises(ExpressionError, match="must be a string"):
            compile_expression("t", 42, _resolve)

    def test_rejects_string_literal(self):
        with pytest.raises(ExpressionError, match="numeric literals"):
            compile_expression("t", "'x'", _resolve)

    def test_allowed_funcs_are_all_callable(self):
        assert all(callable(f) for f in ALLOWED_FUNCS.values())


# ----------------------------------------------------------------------
# Blocks and chaining
# ----------------------------------------------------------------------


class TestDerivedBlock:
    def test_chained_references(self):
        d = compile_derived_block({
            "dx": "agent_x - 13",
            "dist": "abs(dx) + abs(agent_y - 3)",
        }, _resolve)
        assert evaluate_all(d, [16.0, 7.0, 0.0]) == {"dx": 3.0, "dist": 7.0}

    def test_three_deep_chain(self):
        d = compile_derived_block({
            "a": "agent_x * 2",
            "b": "a + 1",
            "c": "b * 10",
        }, _resolve)
        assert evaluate_all(d, [3.0, 0.0, 0.0])["c"] == 70.0

    def test_forward_reference_rejected(self):
        with pytest.raises(ExpressionError, match="unknown feature"):
            compile_derived_block({
                "first": "later + 1",
                "later": "agent_x",
            }, _resolve)

    def test_evaluate_single_computes_dependencies(self):
        d = compile_derived_block({
            "dx": "agent_x - 1",
            "twice": "dx * 2",
        }, _resolve)
        assert evaluate(d, "twice", [5.0, 0.0, 0.0]) == 8.0

    def test_evaluate_unknown_name(self):
        d = compile_derived_block({"dx": "agent_x"}, _resolve)
        with pytest.raises(ExpressionError, match="unknown derived"):
            evaluate(d, "nope", [1.0, 0.0, 0.0])

    def test_block_must_be_mapping(self):
        with pytest.raises(ExpressionError, match="must be a mapping"):
            compile_derived_block(["dx"], _resolve)

    def test_empty_block_is_empty_dict(self):
        assert compile_derived_block({}, _resolve) == {}


# ----------------------------------------------------------------------
# Scenario integration
# ----------------------------------------------------------------------


def _spec(**over):
    spec = {
        "name": "relational",
        "description": "Forget wandering far from the goal at (13, 3).",
        "match_mode": "any_step",
        "derived": {
            "dx": "agent_x - 13",
            "dy": "agent_y - 3",
            "manhattan": "abs(dx) + abs(dy)",
        },
        "groups": [{
            "conditions": [{"dim": "manhattan", "op": ">", "value": 12}],
        }],
    }
    spec.update(over)
    return spec


class TestScenarioDerived:
    def test_compiles_on_load(self):
        s = ForgetScenario(_spec(), view=_view())
        assert set(s.derived) == {"dx", "dy", "manhattan"}

    def test_condition_becomes_derived_kind(self):
        s = ForgetScenario(_spec(), view=_view())
        cond = s.groups[0]["conditions"][0]
        assert cond["kind"] == "derived"
        assert cond["name"] == "manhattan"
        assert cond["expr"] == "abs(dx) + abs(dy)"

    def test_matches_relationally(self):
        s = ForgetScenario(_spec(), view=_view())
        assert s.matches_state(np.array([13.0, 3.0, 0.0])) is False   # at goal
        assert s.matches_state(np.array([1.0, 17.0, 0.0])) is True    # far
        assert s.matches_state(np.array([2.0, 2.0, 0.0])) is False    # exactly 12

    def test_bad_expression_is_scenario_error(self):
        with pytest.raises(ScenarioError):
            ForgetScenario(_spec(derived={"bad": "nope + 1"}), view=_view())

    def test_derived_composes_with_boolean_groups(self):
        spec = _spec(groups=[{
            "all_of": [
                {"conditions": [{"dim": "manhattan", "op": ">", "value": 5}]},
                {"conditions": [{"dim": "agent_x", "op": "<", "value": 9.5}]},
            ]
        }])
        s = ForgetScenario(spec, view=_view())
        assert s.matches_state(np.array([1.0, 17.0, 0.0])) is True    # far + left
        assert s.matches_state(np.array([17.0, 17.0, 0.0])) is False  # far + right

    def test_derived_works_with_proportion_mode(self):
        spec = _spec(match_mode="proportion", match_threshold=0.5)
        s = ForgetScenario(spec, view=_view())
        far, near = np.array([1.0, 17.0, 0.0]), np.array([13.0, 3.0, 0.0])
        assert s.identify_forget_trajectories(
            [{"observations": [far, far, near, near]}]) == [0]
        assert s.identify_forget_trajectories(
            [{"observations": [far, near, near, near]}]) == []

    def test_derived_survives_with_view_rebinding(self):
        s = ForgetScenario(_spec(), view=_view())
        swapped = _view(("agent_y", "agent_x", "vx"))
        s2 = s.with_view(swapped)
        # 'agent_x' now lives at index 1, so the same YAML means something
        # different against the new view -- and recompiles rather than
        # silently reusing stale indices.
        assert s2.derived["dx"].inputs == ("agent_x",)
        assert s2.matches_state(np.array([3.0, 13.0, 0.0])) is False

    def test_no_derived_block_is_empty(self):
        spec = {
            "name": "plain", "description": "no derived features",
            "groups": [{"conditions": [{"dim": 0, "op": ">", "value": 1}]}],
        }
        assert ForgetScenario(spec).derived == {}
