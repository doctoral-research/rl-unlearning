"""Authoring with a validator repair loop (stage 4).

Driven by scripted backends, so the loop is tested without a network call
and the failure modes are the ones a model actually produces.
"""

from __future__ import annotations

import numpy as np
import pytest

from scenarios.authoring import (
    FATAL_WARNINGS,
    _fatal_issues,
    author_scenario,
    build_prompt,
    describe_view,
    grammar_reference,
)
from scenarios.views import FeatureView

VIEW = FeatureView(
    names=("agent_x", "agent_y", "direction"),
    extract_fn=lambda o, i, e: list(o),
    dtypes=("ordinal", "ordinal", "categorical"),
    ranges=((0.0, 18.0), (0.0, 18.0), (0.0, 3.0)),
    provenance="auto:minigrid",
)

GOOD = """
name: avoid_br
description: Forget routing through the bottom-right room.
match_mode: proportion
match_threshold: 0.15
applies_to: matching_steps
groups:
  - conditions:
      - dim: agent_x
        op: ">"
        value: 9.5
      - dim: agent_y
        op: ">"
        value: 9.5
"""


def _scripted(*responses):
    it = iter(responses)
    return lambda _prompt: next(it)


def _trajs(points):
    return [{"observations": [np.array([float(x), float(y), 0.0])
                              for x, y in points],
             "actions": [0] * len(points)}]


class TestPrompt:
    def test_lists_feature_names_and_types(self):
        text = describe_view(VIEW)
        assert "agent_x (ordinal, range [0, 18])" in text
        assert "direction (categorical" in text

    def test_says_names_are_exact(self):
        assert "exactly" in describe_view(VIEW)

    def test_no_view_says_use_indices(self):
        assert "integer index" in describe_view(None)

    def test_grammar_is_generated_from_the_live_tables(self):
        """The reference must not drift from the implementation."""
        from scenarios.constants import SCHEMA_VERSION, VALID_MATCH_MODES

        ref = grammar_reference()
        assert SCHEMA_VERSION in ref
        for mode in VALID_MATCH_MODES:
            assert mode in ref

    def test_first_prompt_has_no_feedback_section(self):
        p = build_prompt("Avoid the corner.", view=VIEW)
        assert "previous attempt" not in p

    def test_retry_prompt_includes_attempt_and_reasons(self):
        p = build_prompt("Avoid the corner.", view=VIEW,
                         previous="name: x", feedback=["bad dim 'foo'"])
        assert "name: x" in p
        assert "bad dim 'foo'" in p


class TestAcceptance:
    def test_accepts_a_valid_first_attempt(self):
        r = author_scenario("Avoid the bottom-right room.",
                            _scripted(GOOD), view=VIEW)
        assert r.accepted and r.n_attempts == 1
        assert not r.repaired
        assert r.scenario.name == "avoid_br"

    def test_strips_code_fences(self):
        r = author_scenario("x", _scripted(f"```yaml\n{GOOD}\n```"), view=VIEW)
        assert r.accepted

    def test_handles_bare_fence(self):
        r = author_scenario("x", _scripted(f"```\n{GOOD}\n```"), view=VIEW)
        assert r.accepted


class TestRejectionModes:
    def _reasons(self, response):
        r = author_scenario("x", _scripted(response), view=VIEW,
                            max_attempts=1)
        assert not r.accepted
        return " ".join(r.attempts[0].rejection)

    def test_unparseable_yaml(self):
        assert "not valid YAML" in self._reasons("name: [unclosed")

    def test_non_mapping_top_level(self):
        assert "must be a mapping" in self._reasons("- a\n- b")

    def test_invented_feature_name(self):
        bad = ("name: x\ndescription: An invented feature name.\n"
               "groups:\n  - conditions:\n      - dim: agent_position\n"
               "        op: \">\"\n        value: 1\n")
        assert "not in view" in self._reasons(bad)

    def test_threshold_outside_the_feature_range(self):
        bad = ("name: x\ndescription: A threshold outside the range.\n"
               "groups:\n  - conditions:\n      - dim: agent_x\n"
               "        op: \">\"\n        value: 100\n")
        assert "never hold" in self._reasons(bad)

    def test_undeclared_step_scope(self):
        bad = ("name: x\ndescription: Trajectory mode without a step scope.\n"
               "match_mode: proportion\nmatch_threshold: 0.2\n"
               "groups:\n  - conditions:\n      - dim: agent_x\n"
               "        op: \">\"\n        value: 9.5\n")
        assert "applies_to" in self._reasons(bad)

    def test_ordered_categorical(self):
        bad = ("name: x\ndescription: Ordering a categorical feature.\n"
               "groups:\n  - conditions:\n      - dim: direction\n"
               "        op: \">\"\n        value: 1\n")
        assert "categorical" in self._reasons(bad)

    def test_schema_violation(self):
        bad = "name: x\ndescription: Bad match mode.\nmatch_mode: sometimes\n"
        assert self._reasons(bad)

    def test_fatal_warning_set_is_configurable(self):
        """Relaxing the gate lets a vacuous scenario through -- which is
        why the default does not."""
        bad = ("name: x\ndescription: A threshold outside the range.\n"
               "groups:\n  - conditions:\n      - dim: agent_x\n"
               "        op: \">\"\n        value: 100\n")
        r = author_scenario("x", _scripted(bad), view=VIEW, max_attempts=1,
                            fatal_warnings=frozenset())
        assert r.accepted


class TestRepairLoop:
    def test_repairs_after_feedback(self):
        bad = ("name: x\ndescription: An invented feature name.\n"
               "groups:\n  - conditions:\n      - dim: nope\n"
               "        op: \">\"\n        value: 1\n")
        r = author_scenario("x", _scripted(bad, GOOD), view=VIEW)
        assert r.accepted and r.repaired and r.n_attempts == 2

    def test_feedback_reaches_the_next_prompt(self):
        bad = ("name: x\ndescription: An invented feature name.\n"
               "groups:\n  - conditions:\n      - dim: nope\n"
               "        op: \">\"\n        value: 1\n")
        seen = []

        def backend(prompt):
            seen.append(prompt)
            return GOOD if len(seen) > 1 else bad

        author_scenario("x", backend, view=VIEW)
        assert "not in view" in seen[1]
        assert "dim: nope" in seen[1]

    def test_gives_up_after_max_attempts(self):
        bad = "name: [unclosed"
        r = author_scenario("x", _scripted(bad, bad, bad), view=VIEW,
                            max_attempts=3)
        assert not r.accepted and r.n_attempts == 3
        assert r.scenario is None

    def test_every_attempt_is_retained(self):
        bad = "name: [unclosed"
        r = author_scenario("x", _scripted(bad, bad, GOOD), view=VIEW)
        assert [a.n for a in r.attempts] == [1, 2, 3]
        assert [a.accepted for a in r.attempts] == [False, False, True]


class TestEmpiricalGate:
    """The check that catches a well-formed specification that means
    nothing on real rollouts."""

    def test_rejects_a_scenario_that_fires_on_nothing(self):
        r = author_scenario("x", _scripted(GOOD), view=VIEW, max_attempts=1,
                            trajectories=_trajs([(1, 1), (2, 2)]))
        assert not r.accepted
        assert "matches no steps" in " ".join(r.attempts[0].rejection)

    def test_accepts_when_it_fires(self):
        r = author_scenario("x", _scripted(GOOD), view=VIEW, max_attempts=1,
                            trajectories=_trajs([(1, 1), (12, 12)]))
        assert r.accepted

    def test_rejects_a_scenario_that_matches_everything(self):
        allmatch = ("name: x\ndescription: Matches every step.\n"
                    "groups:\n  - conditions:\n      - dim: agent_x\n"
                    "        op: \">=\"\n        value: 0\n")
        r = author_scenario("x", _scripted(allmatch), view=VIEW,
                            max_attempts=1, trajectories=_trajs([(1, 1)]),
                            fatal_warnings=frozenset())
        assert not r.accepted
        assert "every step" in " ".join(r.attempts[0].rejection)

    def test_gate_is_optional(self):
        r = author_scenario("x", _scripted(GOOD), view=VIEW, max_attempts=1,
                            trajectories=_trajs([(1, 1)]),
                            require_nonvacuous=False)
        assert r.accepted

    def test_match_rate_is_recorded(self):
        r = author_scenario("x", _scripted(GOOD), view=VIEW, max_attempts=1,
                            trajectories=_trajs([(1, 1), (12, 12)]))
        assert r.attempts[0].match_rate == pytest.approx(0.5)


def test_fatal_warning_codes_exist_in_the_validator():
    """Guards against a typo silently disabling a gate."""
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "scenarios"
    text = (src / "validate.py").read_text()
    emitted = set(re.findall(r'ValidationIssue\("(?:warning|error)",\s*"(\w+)"', text))
    missing = FATAL_WARNINGS - emitted
    assert not missing, f"fatal codes never emitted by the validator: {missing}"


class TestRedundantAlwaysTrue:
    """An always-true condition is fatal only when it actually makes the
    group vacuous.

    Found by running the loop against qwen2.5-coder:32b, which writes the
    full bounding box including its trivial faces. Rejecting that was a
    comment on the gate, not on the specification: the 32B model scored
    46.7% against the 7B model's 80% almost entirely on this.
    """

    def _doc(self, conditions, key="conditions"):
        return {"name": "t", "description": "A scenario for the gate test.",
                "groups": [{key: conditions}]}

    def _report(self, doc, view):
        from scenarios.validate import validate_scenario_dict
        return validate_scenario_dict(doc, view=view)

    def test_redundant_conjunct_is_not_fatal(self, minigrid_view):
        doc = self._doc([{"dim": "agent_x", "op": ">=", "value": 0},
                         {"dim": "agent_x", "op": ">=", "value": 14}])
        rep = self._report(doc, minigrid_view)
        assert _fatal_issues(rep, doc, FATAL_WARNINGS) == []

    def test_sole_always_true_condition_is_fatal(self, minigrid_view):
        doc = self._doc([{"dim": "agent_x", "op": ">=", "value": 0}])
        rep = self._report(doc, minigrid_view)
        got = _fatal_issues(rep, doc, FATAL_WARNINGS)
        assert [i.code for i in got] == ["condition_always_true"]

    def test_never_true_stays_fatal_even_beside_others(self, minigrid_view):
        """An unsatisfiable conjunct kills the whole AND group, so unlike
        an always-true one it is never harmless."""
        doc = self._doc([{"dim": "agent_x", "op": ">", "value": 999},
                         {"dim": "agent_x", "op": ">=", "value": 14}])
        rep = self._report(doc, minigrid_view)
        codes = [i.code for i in _fatal_issues(rep, doc, FATAL_WARNINGS)]
        assert "condition_always_true" not in codes
        assert codes                      # something still rejects it

    def test_other_fatal_warnings_are_unaffected(self, minigrid_view):
        doc = self._doc([{"dim": "agent_x", "op": ">=", "value": 0},
                         {"dim": "agent_x", "op": "==", "value": 3.5}])
        rep = self._report(doc, minigrid_view)
        codes = [i.code for i in _fatal_issues(rep, doc, FATAL_WARNINGS)]
        assert "condition_always_true" not in codes


@pytest.fixture
def minigrid_view():
    """A view with real ranges, so always-true/never-true can be decided."""
    return FeatureView(
        names=("agent_x", "agent_y", "direction"),
        extract_fn=lambda obs, info, env: [0.0, 0.0, 0.0],
        dtypes=("ordinal", "ordinal", "categorical"),
        ranges=((0.0, 18.0), (0.0, 18.0), (0.0, 3.0)),
        provenance="auto:minigrid",
        env_id="MiniGrid-FourRooms-v0",
    )


class TestRegionRedundancy:
    """A box drawn to the edge of a feature's range has trivial faces.

    qwen2.5-coder emits regions rather than bare conditions, and regions
    are desugared into conditions before validation, so the fatal-warning
    filter has to reason about the group as a whole rather than the
    single condition the warning points at.
    """

    def _report_and_doc(self, groups, view):
        from scenarios.validate import validate_scenario_dict
        doc = {"name": "t", "description": "A box for the gate test.",
               "groups": groups}
        return validate_scenario_dict(doc, view=view), doc

    def _box(self, lo, hi):
        return {"region": {"type": "box", "dims": ["agent_x", "agent_y"],
                           "lo": lo, "hi": hi}}

    def test_box_with_one_trivial_face_is_accepted(self, minigrid_view):
        rep, doc = self._report_and_doc(
            [self._box([13, 0], [15, 3])], minigrid_view)
        assert _fatal_issues(rep, doc, FATAL_WARNINGS) == []

    def test_box_covering_the_whole_range_stays_fatal(self, minigrid_view):
        rep, doc = self._report_and_doc(
            [self._box([0, 0], [18, 18])], minigrid_view)
        codes = {i.code for i in _fatal_issues(rep, doc, FATAL_WARNINGS)}
        assert codes == {"condition_always_true"}

    def test_each_group_is_judged_separately(self, minigrid_view):
        """A real box next to a vacuous one must not launder it."""
        rep, doc = self._report_and_doc(
            [self._box([13, 0], [15, 3]), self._box([0, 0], [18, 18])],
            minigrid_view)
        fatal = _fatal_issues(rep, doc, FATAL_WARNINGS)
        assert fatal and all(i.path.startswith("groups[1]") for i in fatal)
