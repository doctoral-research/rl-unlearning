"""Indexed families and quantification (grammar v1.3).

Quantifying over a family makes a multi-agent scenario team-size
agnostic; v1.2 required enumerating every pair by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from scenarios import ForgetScenario, ScenarioError
from scenarios.families import (
    BINDING_NAMES,
    REDUCERS,
    FamilyError,
    compile_family_block,
    discover_family,
    expand,
    parse_quantified,
)
from scenarios.views import FeatureView


def _view(n_agents=3):
    names = []
    for i in range(n_agents):
        names += [f"p{i}_row", f"p{i}_col", f"p{i}_score"]
    names += ["n_food"]
    return FeatureView(names=tuple(names), extract_fn=lambda o, i, e: list(o),
                       provenance="declared:test")


class TestDiscovery:
    def test_finds_members_and_attributes(self):
        f = discover_family("agent", "p{i}_", _view(3).names)
        assert f.prefixes == ("p0_", "p1_", "p2_")
        assert f.attributes == ("col", "row", "score")

    def test_member_count_follows_the_view(self):
        assert len(discover_family("agent", "p{i}_", _view(5).names)) == 5

    def test_pattern_needs_an_index_marker(self):
        with pytest.raises(FamilyError, match=r"must contain"):
            discover_family("agent", "p_", _view().names)

    def test_no_match_is_an_error(self):
        with pytest.raises(FamilyError, match="matched no features"):
            discover_family("agent", "zz{i}_", _view().names)

    def test_non_family_features_ignored(self):
        f = discover_family("agent", "p{i}_", _view(2).names)
        assert "n_food" not in f.attributes

    def test_block_must_be_mapping(self):
        with pytest.raises(FamilyError, match="must be a mapping"):
            compile_family_block(["agent"], _view().names)

    def test_empty_block(self):
        assert compile_family_block(None, _view().names) == {}


class TestExpansion:
    SPEC = {"over": "agent", "arity": 2, "reduce": "min",
            "expr": "abs(a_row - b_row) + abs(a_col - b_col)"}

    def test_unordered_distinct_pairs(self):
        f = discover_family("agent", "p{i}_", _view(3).names)
        out = expand(parse_quantified("d", self.SPEC), f)
        assert len(out) == 3                       # C(3,2)
        assert "abs(p0_row - p1_row) + abs(p0_col - p1_col)" in out

    def test_pair_count_is_quadratic_in_team_size(self):
        counts = []
        for n in (2, 3, 4, 5):
            f = discover_family("agent", "p{i}_", _view(n).names)
            counts.append(len(expand(parse_quantified("d", self.SPEC), f)))
        assert counts == [1, 3, 6, 10]

    def test_ordered_pairs(self):
        f = discover_family("agent", "p{i}_", _view(3).names)
        spec = parse_quantified("d", {**self.SPEC, "ordered": True})
        assert len(expand(spec, f)) == 6           # P(3,2)

    def test_non_distinct_includes_self_pairs(self):
        f = discover_family("agent", "p{i}_", _view(3).names)
        spec = parse_quantified("d", {**self.SPEC, "distinct": False})
        assert len(expand(spec, f)) == 9           # 3^2

    def test_arity_one(self):
        f = discover_family("agent", "p{i}_", _view(3).names)
        spec = parse_quantified("d", {"over": "agent", "expr": "a_score",
                                      "reduce": "sum"})
        assert expand(spec, f) == ["p0_score", "p1_score", "p2_score"]

    def test_arity_above_member_count_yields_nothing(self):
        f = discover_family("agent", "p{i}_", _view(2).names)
        spec = parse_quantified("d", {**self.SPEC, "arity": 3})
        assert expand(spec, f) == []

    def test_function_names_are_not_rewritten(self):
        """`abs(` starts with the binding letter `a` and must survive."""
        f = discover_family("agent", "p{i}_", _view(2).names)
        out = expand(parse_quantified("d", self.SPEC), f)
        assert all(o.startswith("abs(") for o in out)


class TestSpecValidation:
    def test_needs_over_and_expr(self):
        with pytest.raises(FamilyError, match="needs 'over'"):
            parse_quantified("d", {"reduce": "min"})

    def test_unknown_reduce_rejected(self):
        with pytest.raises(FamilyError, match="reduce must be one of"):
            parse_quantified("d", {"over": "a", "expr": "x", "reduce": "median"})

    def test_arity_bounds(self):
        with pytest.raises(FamilyError, match="arity must be"):
            parse_quantified("d", {"over": "a", "expr": "x", "arity": 0})
        with pytest.raises(FamilyError, match="arity must be"):
            parse_quantified("d", {"over": "a", "expr": "x",
                                   "arity": len(BINDING_NAMES) + 1})

    def test_all_reducers_callable(self):
        assert all(callable(f) for f in REDUCERS.values())


def _spec(reduce="min", value=3, op="<"):
    return {
        "name": "q", "description": "Quantified predicate under test.",
        "match_mode": "any_step",
        "families": {"agent": "p{i}_"},
        "derived": {"closest": {"over": "agent", "arity": 2,
                                "reduce": reduce,
                                "expr": "abs(a_row - b_row) + abs(a_col - b_col)"}},
        "groups": [{"conditions": [{"dim": "closest", "op": op, "value": value}]}],
    }


def _obs(positions, n=3):
    row = []
    for r, c in positions:
        row += [float(r), float(c), 0.0]
    return np.array(row + [0.0])


class TestScenarioIntegration:
    def test_min_reduce_finds_the_closest_pair(self):
        s = ForgetScenario(_spec(), view=_view(3))
        assert s.matches_features(_obs([(0, 0), (0, 5), (7, 7)])) is False
        assert s.matches_features(_obs([(0, 0), (0, 1), (7, 7)])) is True

    def test_max_reduce_finds_the_furthest_pair(self):
        s = ForgetScenario(_spec(reduce="max", op=">", value=10),
                           view=_view(3))
        assert s.matches_features(_obs([(0, 0), (0, 1), (7, 7)])) is True
        assert s.matches_features(_obs([(0, 0), (0, 1), (0, 2)])) is False

    def test_unknown_family_is_a_scenario_error(self):
        spec = _spec(); spec["derived"]["closest"]["over"] = "nope"
        with pytest.raises(ScenarioError, match="unknown family"):
            ForgetScenario(spec, view=_view(3))

    def test_bad_expression_is_a_scenario_error(self):
        spec = _spec()
        spec["derived"]["closest"]["expr"] = "a_missing + b_missing"
        with pytest.raises(ScenarioError):
            ForgetScenario(spec, view=_view(3))

    @pytest.mark.parametrize("n,pairs", [(2, 1), (3, 3), (4, 6)])
    def test_same_file_adapts_to_team_size(self, n, pairs):
        """The point of the construct: no edits per N."""
        s = ForgetScenario(_spec(), view=_view(n))
        assert len(s.derived) == pairs
        assert len(s.families["agent"]) == n

    def test_plain_string_derived_still_works(self):
        spec = _spec()
        spec["derived"]["plain"] = "p0_row + 1"
        s = ForgetScenario(spec, view=_view(3))
        assert "plain" in s.derived
