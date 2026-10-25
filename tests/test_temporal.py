"""Nestable temporal formulas (grammar v1.4).

v1.3 evaluated a single operator over a whole trajectory, so operands had
to be state predicates. Evaluating at a *position* makes nesting fall out
and gives the operators their standard LTL meanings.
"""

from __future__ import annotations

import numpy as np
import pytest

from scenarios import ForgetScenario, ScenarioError
from scenarios.temporal import TemporalError, holds, matches, parse_formula


def _g(name):
    """A leaf group matching when feature `name` is 1."""
    return {"conditions": [{"dim": {"p": 0, "q": 1}[name], "op": "==",
                            "value": 1}]}


def _parse(node):
    return parse_formula(node, lambda g: g)


def _match(node, rows):
    """rows is a list of (p, q) truth values."""
    f = _parse(node)

    def step(group, t):
        dim = group["conditions"][0]["dim"]
        return bool(rows[t][dim])

    return matches(f, len(rows), step)


class TestParsing:
    def test_leaf_is_a_state_predicate(self):
        assert _parse(_g("p")).kind == "state"

    def test_unary_operators(self):
        for op in ("not", "next", "eventually", "always"):
            assert _parse({op: _g("p")}).kind == op

    def test_nary_needs_a_list(self):
        with pytest.raises(TemporalError, match="list of at least two"):
            _parse({"and": _g("p")})

    def test_until_needs_both_operands(self):
        with pytest.raises(TemporalError, match="needs 'hold' and 'release'"):
            _parse({"until": {"hold": _g("p")}})

    def test_consecutive_needs_k_and_of(self):
        with pytest.raises(TemporalError, match="needs 'k' and 'of'"):
            _parse({"consecutive": {"k": 2}})

    def test_consecutive_rejects_k_below_one(self):
        with pytest.raises(TemporalError, match="k >= 1"):
            _parse({"consecutive": {"k": 0, "of": _g("p")}})

    def test_then_within_rejects_zero_window(self):
        with pytest.raises(TemporalError, match="within >= 1"):
            _parse({"then_within": {"first": _g("p"), "second": _g("q"),
                                    "within": 0}})

    def test_several_operators_in_one_node_rejected(self):
        with pytest.raises(TemporalError, match="several operators"):
            _parse({"eventually": _g("p"), "always": _g("q")})

    def test_depth_reports_nesting(self):
        assert _parse(_g("p")).depth() == 1
        assert _parse({"eventually": _g("p")}).depth() == 2
        assert _parse({"eventually": {"not": {"always": _g("p")}}}).depth() == 4

    def test_describe_is_readable(self):
        f = _parse({"eventually": {"not": _g("p")}})
        assert f.describe() == "eventually(not(state))"


class TestLegacyForm:
    """v1.0 files must keep their meaning."""

    @pytest.mark.parametrize("node,rows,expected", [
        ({"operator": "eventually", "group": _g("p")}, [(0,0),(1,0)], True),
        ({"operator": "eventually", "group": _g("p")}, [(0,0),(0,0)], False),
        ({"operator": "always", "group": _g("p")}, [(1,0),(1,0)], True),
        ({"operator": "always", "group": _g("p")}, [(1,0),(0,0)], False),
    ])
    def test_flat_operators(self, node, rows, expected):
        assert _match(node, rows) is expected

    def test_flat_until(self):
        node = {"operator": "until", "hold": _g("p"), "release": _g("q")}
        assert _match(node, [(1,0),(1,0),(0,1)]) is True
        assert _match(node, [(1,0),(0,0),(0,1)]) is False

    def test_flat_consecutive_is_existential(self):
        """v1.0 consecutive meant 'somewhere in the episode', not 'at the
        start' -- that anchoring must survive."""
        node = {"operator": "consecutive", "k": 2, "group": _g("p")}
        assert _match(node, [(0,0),(1,0),(1,0)]) is True
        assert _match(node, [(1,0),(0,0),(1,0)]) is False

    def test_flat_then_within(self):
        node = {"operator": "then_within", "first": _g("p"),
                "second": _g("q"), "within": 1}
        assert _match(node, [(1,0),(0,1)]) is True
        assert _match(node, [(1,0),(0,0),(0,1)]) is False

    def test_unknown_operator(self):
        with pytest.raises(TemporalError, match="Unknown temporal operator"):
            _parse({"operator": "someday", "group": _g("p")})


class TestSemantics:
    def test_not(self):
        assert _match({"not": {"eventually": _g("p")}}, [(0,0),(0,0)]) is True
        assert _match({"not": {"eventually": _g("p")}}, [(0,0),(1,0)]) is False

    def test_and_or(self):
        rows = [(1, 0)]
        assert _match({"and": [_g("p"), _g("q")]}, rows) is False
        assert _match({"or": [_g("p"), _g("q")]}, rows) is True

    def test_next(self):
        assert _match({"next": _g("p")}, [(0,0),(1,0)]) is True
        assert _match({"next": _g("p")}, [(0,0),(0,0)]) is False

    def test_next_past_the_end_is_false(self):
        assert _match({"next": _g("p")}, [(0,0)]) is False

    def test_eventually_always_nesting(self):
        """eventually(always(p)) -- p holds from some point onward."""
        node = {"eventually": {"always": _g("p")}}
        assert _match(node, [(0,0),(1,0),(1,0)]) is True
        assert _match(node, [(1,0),(0,0),(1,0)]) is True   # from index 2
        assert _match(node, [(1,0),(1,0),(0,0)]) is False

    def test_always_eventually_nesting(self):
        """always(eventually(p)) -- p recurs; on a finite trace this means
        p holds at the final position or later, i.e. at the last step."""
        node = {"always": {"eventually": _g("p")}}
        assert _match(node, [(1,0),(0,0),(1,0)]) is True
        assert _match(node, [(1,0),(1,0),(0,0)]) is False

    def test_empty_trajectory_never_matches(self):
        assert _match({"eventually": _g("p")}, []) is False


class TestKeyHoarding:
    """The corpus entry this version exists for."""

    NODE = {"eventually": {"and": [
        _g("p"),                                   # holds key
        {"not": {"eventually": _g("q")}},          # never opens door after
    ]}}

    @pytest.mark.parametrize("label,rows,expected", [
        ("never picks up",        [(0,0),(0,0),(0,0)], False),
        ("picks up then opens",   [(0,0),(1,0),(1,1)], False),
        ("picks up, never opens", [(0,0),(1,0),(1,0)], True),
    ])
    def test_cases(self, label, rows, expected):
        assert _match(self.NODE, rows) is expected, label

    def test_depth_exceeds_one_operator(self):
        assert _parse(self.NODE).depth() > 2


class TestScenarioIntegration:
    def test_nested_formula_loads(self):
        s = ForgetScenario({
            "name": "n", "description": "Nested temporal formula.",
            "match_mode": "temporal",
            "temporal": {"eventually": {"not": {"conditions": [
                {"dim": 0, "op": ">", "value": 1}]}}},
        })
        assert s.temporal_depth == 3

    def test_malformed_formula_is_a_scenario_error(self):
        with pytest.raises(ScenarioError):
            ForgetScenario({
                "name": "n", "description": "Malformed temporal formula.",
                "match_mode": "temporal",
                "temporal": {"until": {"hold": {"conditions": []}}},
            })

    def test_bare_predicate_under_temporal_warns(self):
        from scenarios import validate_scenario_dict

        rep = validate_scenario_dict({
            "name": "n", "description": "Temporal mode, no operator used.",
            "match_mode": "temporal",
            "temporal": {"conditions": [{"dim": 0, "op": ">", "value": 1}]},
        })
        assert "temporal_without_operator" in {i.code for i in rep.warnings}


class TestPreviousOperator:
    """The past-time mirror of `next`.

    Without it, a rule anchored on a transition INTO a region can only be
    written by anchoring one step early with `next`, which labels the
    wrong transition: the step before entry rather than the entry itself.
    """

    def _formula(self, node):
        return parse_formula(node, lambda g: g)

    def _match(self, seq):
        """Leaf predicate: the step's value equals the leaf's value.

        Leaves must be mappings; a bare string is not a formula node.
        """
        return lambda g, t: seq[t] == g["v"]

    def test_holds_one_step_earlier(self):
        f = self._formula({"previous": {"v": "a"}})
        assert holds(f, 1, 3, self._match(["a", "b", "c"]))

    def test_false_at_the_first_step(self):
        """There is no step before the first one, so it cannot hold."""
        f = self._formula({"previous": {"v": "a"}})
        assert not holds(f, 0, 3, self._match(["a", "b", "c"]))

    def test_mirrors_next(self):
        seq = ["a", "b", "c"]
        nxt = self._formula({"next": {"v": "b"}})
        prv = self._formula({"previous": {"v": "a"}})
        assert holds(nxt, 0, 3, self._match(seq))
        assert holds(prv, 1, 3, self._match(seq))

    def test_nests(self):
        f = self._formula({"previous": {"previous": {"v": "a"}}})
        assert holds(f, 2, 3, self._match(["a", "b", "c"]))
        assert not holds(f, 1, 3, self._match(["a", "b", "c"]))

    def test_combines_with_and(self):
        f = self._formula({"and": [{"previous": {"v": "a"}}, {"v": "b"}]})
        m = self._match(["a", "b", "c"])
        assert holds(f, 1, 3, m)
        assert not holds(f, 2, 3, m)
