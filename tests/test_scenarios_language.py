"""Tests for the ForgetScenario v1.0 language: schema, validator, new
expressivity constructs (recursive composition, regions, temporal,
macros), and introspection (natural-language / predicate-logic /
coverage).

These tests are intentionally hermetic — every scenario is constructed
in-memory via dict literals, then either passed to `ForgetScenario(...)`
or written to a tmp YAML file and validated. No dependence on any of
the on-disk `configs/scenarios/*.yaml` files, so the tests survive
edits to those examples.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest
import yaml

from scenarios import (
    ForgetScenario,
    ScenarioError,
    coverage,
    to_natural_language,
    to_predicate,
    validate_scenario,
    validate_scenario_dict,
)


# ---------------------------------------------------------------------------
# Fixtures: tiny helpers that build common scenario dicts.
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, content: dict) -> Path:
    p = tmp_path / "scenario.yaml"
    p.write_text(yaml.safe_dump(content, sort_keys=False))
    return p


def _cartpole_left_only() -> dict:
    return {
        "name": "left_only_test",
        "description": "right-side trajectories — test fixture",
        "env_id": "CartPole-v1",
        "match_mode": "proportion",
        "match_threshold": 0.3,
        "groups": [
            {
                "name": "right_side",
                "conditions": [
                    {"dim": 0, "op": ">", "value": 0.0, "target": "state"}
                ],
            }
        ],
    }


# ---------------------------------------------------------------------------
# Legacy back-compat: every old-style scenario still loads + matches as before.
# ---------------------------------------------------------------------------


class TestLegacyBackwardCompat:
    def test_basic_load(self):
        s = ForgetScenario(_cartpole_left_only())
        assert s.name == "left_only_test"
        assert s.match_mode == "proportion"
        assert s.match_threshold == 0.3

    def test_matches_state_legacy(self):
        s = ForgetScenario(_cartpole_left_only())
        assert s.matches_state(np.array([1.0, 0, 0, 0])) is True
        assert s.matches_state(np.array([-1.0, 0, 0, 0])) is False

    def test_proportion_mode_trajectory(self):
        s = ForgetScenario(_cartpole_left_only())
        traj_mostly_right = {
            "observations": [np.array([1.0, 0, 0, 0])] * 5
                             + [np.array([-1.0, 0, 0, 0])] * 5,
            "actions": [0] * 10,
            "rewards": [1.0] * 10,
        }
        assert s.identify_forget_trajectories([traj_mostly_right]) == [0]

    def test_top_percentile_load(self):
        cfg = {
            "name": "high_reward",
            "description": "top 10% by return",
            "env_id": "CartPole-v1",
            "match_mode": "top_percentile",
            "match_threshold": 0.1,
            "groups": [{"rank_by": "return", "rank_order": "descending"}],
        }
        s = ForgetScenario(cfg)
        trajs = [
            {"observations": [np.zeros(4)] * 10,
             "actions": [0] * 10,
             "rewards": [float(i)] * 10}
            for i in range(10)
        ]
        idx = s.identify_forget_trajectories(trajs)
        assert idx == [9]  # top 10% of 10 = 1, descending → highest return


# ---------------------------------------------------------------------------
# New expressivity: boolean composition.
# ---------------------------------------------------------------------------


class TestBooleanComposition:
    def test_all_of_requires_both(self):
        cfg = {
            "name": "all_of_test",
            "description": "x>0 AND y<0",
            "env_id": "LunarLander-v3",
            "match_mode": "any_step",
            "groups": [{
                "all_of": [
                    {"conditions": [{"dim": 0, "op": ">", "value": 0.0}]},
                    {"conditions": [{"dim": 1, "op": "<", "value": 0.0}]},
                ],
            }],
        }
        s = ForgetScenario(cfg)
        obs = np.zeros(8)
        obs[0] = 1.0
        obs[1] = -1.0
        assert s.matches_state(obs) is True
        obs[1] = 1.0
        assert s.matches_state(obs) is False

    def test_any_of_inside_top_level(self):
        cfg = {
            "name": "any_of_test",
            "description": "x>0 OR x<-5",
            "env_id": "LunarLander-v3",
            "match_mode": "any_step",
            "groups": [{
                "any_of": [
                    {"conditions": [{"dim": 0, "op": ">", "value": 0.0}]},
                    {"conditions": [{"dim": 0, "op": "<", "value": -5.0}]},
                ],
            }],
        }
        s = ForgetScenario(cfg)
        obs = np.zeros(8)
        obs[0] = 1.0
        assert s.matches_state(obs) is True
        obs[0] = -6.0
        assert s.matches_state(obs) is True
        obs[0] = -1.0
        assert s.matches_state(obs) is False

    def test_not_inverts_match(self):
        cfg = {
            "name": "not_test",
            "description": "NOT (x>0)",
            "env_id": "LunarLander-v3",
            "match_mode": "any_step",
            "groups": [{
                "not": {"conditions": [{"dim": 0, "op": ">", "value": 0.0}]},
            }],
        }
        s = ForgetScenario(cfg)
        obs = np.zeros(8)
        obs[0] = -1.0
        assert s.matches_state(obs) is True
        obs[0] = 1.0
        assert s.matches_state(obs) is False

    def test_nested_combinators(self):
        # (x>0 AND y<0) OR NOT(z>0)
        cfg = {
            "name": "nested",
            "description": "deeply nested boolean composition",
            "env_id": "LunarLander-v3",
            "match_mode": "any_step",
            "groups": [{
                "any_of": [
                    {"all_of": [
                        {"conditions": [{"dim": 0, "op": ">", "value": 0.0}]},
                        {"conditions": [{"dim": 1, "op": "<", "value": 0.0}]},
                    ]},
                    {"not": {"conditions": [{"dim": 2, "op": ">", "value": 0.0}]}},
                ],
            }],
        }
        s = ForgetScenario(cfg)
        obs = np.zeros(8)
        obs[2] = -1.0  # NOT(z>0) is true
        assert s.matches_state(obs) is True

        obs = np.zeros(8)
        obs[0] = 1.0
        obs[1] = -1.0
        obs[2] = 1.0  # x>0 AND y<0, NOT(z>0) false
        assert s.matches_state(obs) is True

        obs = np.zeros(8)
        obs[2] = 1.0
        obs[0] = -1.0
        assert s.matches_state(obs) is False


# ---------------------------------------------------------------------------
# Region shorthand: box, complement_of_box, dim-name resolution.
# ---------------------------------------------------------------------------


class TestRegionShorthand:
    def test_box_with_dim_indices(self):
        cfg = {
            "name": "box_test",
            "description": "3x3 box around (4,4)",
            "env_id": "MiniGrid-FourRooms-v0",
            "match_mode": "any_step",
            "groups": [{
                "name": "tl_box",
                "region": {
                    "type": "box",
                    "dims": [0, 1],
                    "lo": [3, 3],
                    "hi": [5, 5],
                },
            }],
        }
        s = ForgetScenario(cfg)
        assert s.matches_state(np.array([4, 4, 0])) is True
        assert s.matches_state(np.array([3, 3, 0])) is True
        assert s.matches_state(np.array([6, 4, 0])) is False
        assert s.matches_state(np.array([2, 4, 0])) is False

    def test_box_with_dim_names(self):
        cfg = {
            "name": "box_named",
            "description": "named dims via env_id lookup",
            "env_id": "CartPole-v1",
            "match_mode": "any_step",
            "groups": [{
                "region": {
                    "type": "box",
                    "dims": ["cart_position", "pole_angle"],
                    "lo": [-0.5, -0.1],
                    "hi": [0.5, 0.1],
                },
            }],
        }
        s = ForgetScenario(cfg)
        assert s.matches_state(np.array([0.0, 0.0, 0.05, 0.0])) is True
        assert s.matches_state(np.array([1.0, 0.0, 0.0, 0.0])) is False

    def test_complement_of_box(self):
        cfg = {
            "name": "outside",
            "description": "anything outside the central box",
            "env_id": "CartPole-v1",
            "match_mode": "any_step",
            "groups": [{
                "region": {
                    "type": "complement_of_box",
                    "dims": [0],
                    "lo": [-0.5],
                    "hi": [0.5],
                },
            }],
        }
        s = ForgetScenario(cfg)
        assert s.matches_state(np.array([0.0, 0, 0, 0])) is False
        assert s.matches_state(np.array([0.6, 0, 0, 0])) is True
        assert s.matches_state(np.array([-1.0, 0, 0, 0])) is True

    def test_box_with_null_bound(self):
        cfg = {
            "name": "half_box",
            "description": "half-bounded box (effectively a half-space)",
            "env_id": "CartPole-v1",
            "match_mode": "any_step",
            "groups": [{
                "region": {
                    "type": "box",
                    "dims": [0],
                    "lo": [0.5],
                    "hi": [None],
                },
            }],
        }
        s = ForgetScenario(cfg)
        assert s.matches_state(np.array([0.6, 0, 0, 0])) is True
        assert s.matches_state(np.array([0.4, 0, 0, 0])) is False
        assert s.matches_state(np.array([100.0, 0, 0, 0])) is True


# ---------------------------------------------------------------------------
# Macros: defines + $ref expansion.
# ---------------------------------------------------------------------------


class TestMacros:
    def test_ref_to_defined_group(self):
        cfg = {
            "name": "ref_test",
            "description": "use $ref to a named macro",
            "env_id": "CartPole-v1",
            "match_mode": "any_step",
            "defines": {
                "right_side": {
                    "conditions": [{"dim": 0, "op": ">", "value": 0.0}],
                },
            },
            "groups": [{"$ref": "right_side"}],
        }
        s = ForgetScenario(cfg)
        assert s.matches_state(np.array([1.0, 0, 0, 0])) is True
        assert s.matches_state(np.array([-1.0, 0, 0, 0])) is False

    def test_unknown_ref_raises(self):
        cfg = {
            "name": "bad_ref",
            "description": "ref to a macro that does not exist",
            "env_id": "CartPole-v1",
            "match_mode": "any_step",
            "defines": {"foo": {"conditions": [
                {"dim": 0, "op": ">", "value": 0}]}},
            "groups": [{"$ref": "missing"}],
        }
        with pytest.raises(ScenarioError, match="Unknown \\$ref"):
            ForgetScenario(cfg)

    def test_unused_macro_flagged_by_validator(self, tmp_path):
        cfg = {
            "name": "unused_macro",
            "description": "a macro that is defined but never used",
            "env_id": "CartPole-v1",
            "match_mode": "any_step",
            "defines": {
                "unused": {"conditions": [
                    {"dim": 0, "op": ">", "value": 0}]},
                "used": {"conditions": [
                    {"dim": 0, "op": ">", "value": 0}]},
            },
            "groups": [{"$ref": "used"}],
        }
        report = validate_scenario(_write_yaml(tmp_path, cfg))
        unused_codes = [i.code for i in report.warnings]
        assert "unused_define" in unused_codes
        assert report.ok  # warnings only


# ---------------------------------------------------------------------------
# Temporal patterns.
# ---------------------------------------------------------------------------


class TestTemporal:
    def _make_temporal(self, operator: str, **kwargs):
        cfg = {
            "name": f"temporal_{operator}",
            "description": "temporal pattern test",
            "env_id": "CartPole-v1",
            "match_mode": "temporal",
            "temporal": {"operator": operator, **kwargs},
        }
        return ForgetScenario(cfg)

    def test_consecutive(self):
        s = self._make_temporal(
            "consecutive", k=3,
            group={"conditions": [{"dim": 0, "op": ">", "value": 0}]},
        )
        # Two runs of length 2 and 3: must catch the second.
        traj = {
            "observations": [
                np.array([1.0, 0, 0, 0]), np.array([1.0, 0, 0, 0]),
                np.array([-1.0, 0, 0, 0]),
                np.array([1.0, 0, 0, 0]), np.array([1.0, 0, 0, 0]),
                np.array([1.0, 0, 0, 0]),
            ],
            "actions": [0] * 6,
            "rewards": [1.0] * 6,
        }
        assert s.identify_forget_trajectories([traj]) == [0]

    def test_consecutive_below_threshold(self):
        s = self._make_temporal(
            "consecutive", k=5,
            group={"conditions": [{"dim": 0, "op": ">", "value": 0}]},
        )
        traj = {
            "observations": [np.array([1.0, 0, 0, 0])] * 4
                             + [np.array([-1.0, 0, 0, 0])],
            "actions": [0] * 5,
            "rewards": [1.0] * 5,
        }
        assert s.identify_forget_trajectories([traj]) == []

    def test_eventually(self):
        s = self._make_temporal(
            "eventually",
            group={"conditions": [{"dim": 0, "op": ">", "value": 0.5}]},
        )
        traj_hit = {
            "observations": [np.array([0.0, 0, 0, 0])] * 9
                             + [np.array([1.0, 0, 0, 0])],
            "actions": [0] * 10, "rewards": [1.0] * 10,
        }
        traj_miss = {
            "observations": [np.array([0.0, 0, 0, 0])] * 10,
            "actions": [0] * 10, "rewards": [1.0] * 10,
        }
        assert s.identify_forget_trajectories([traj_hit, traj_miss]) == [0]

    def test_always(self):
        s = self._make_temporal(
            "always",
            group={"conditions": [{"dim": 0, "op": ">", "value": -10}]},
        )
        traj = {
            "observations": [np.array([0.0, 0, 0, 0])] * 5,
            "actions": [0] * 5, "rewards": [1.0] * 5,
        }
        assert s.identify_forget_trajectories([traj]) == [0]

    def test_then_within(self):
        s = self._make_temporal(
            "then_within", within=2,
            first={"conditions": [{"dim": 0, "op": ">", "value": 0.5}]},
            second={"conditions": [{"dim": 0, "op": "<", "value": -0.5}]},
        )
        traj = {
            "observations": [
                np.array([1.0, 0, 0, 0]),
                np.array([0.0, 0, 0, 0]),
                np.array([-1.0, 0, 0, 0]),
            ],
            "actions": [0, 0, 0], "rewards": [1.0, 1.0, 1.0],
        }
        assert s.identify_forget_trajectories([traj]) == [0]

    def test_then_within_too_far(self):
        s = self._make_temporal(
            "then_within", within=1,
            first={"conditions": [{"dim": 0, "op": ">", "value": 0.5}]},
            second={"conditions": [{"dim": 0, "op": "<", "value": -0.5}]},
        )
        traj = {
            "observations": [
                np.array([1.0, 0, 0, 0]),
                np.array([0.0, 0, 0, 0]),
                np.array([-1.0, 0, 0, 0]),
            ],
            "actions": [0, 0, 0], "rewards": [1.0, 1.0, 1.0],
        }
        assert s.identify_forget_trajectories([traj]) == []


# ---------------------------------------------------------------------------
# Validator: schema, semantic, lint.
# ---------------------------------------------------------------------------


class TestValidator:
    def test_clean_scenario_passes(self, tmp_path):
        report = validate_scenario(_write_yaml(tmp_path, _cartpole_left_only()))
        assert report.ok
        # version-info hint expected
        assert any(i.code == "missing_version" for i in report.infos)

    def test_typo_in_match_mode_caught(self, tmp_path):
        cfg = _cartpole_left_only()
        cfg["match_mode"] = "any_steps"  # plural typo
        report = validate_scenario(_write_yaml(tmp_path, cfg))
        assert not report.ok
        assert any(i.code == "schema" for i in report.errors)

    def test_dim_out_of_range_for_env(self, tmp_path):
        cfg = _cartpole_left_only()
        cfg["groups"][0]["conditions"][0]["dim"] = 99  # cartpole only has 4
        report = validate_scenario(_write_yaml(tmp_path, cfg))
        codes = [i.code for i in report.errors]
        assert "dim_out_of_range" in codes

    def test_dim_name_mismatch_warns(self, tmp_path):
        cfg = _cartpole_left_only()
        cfg["groups"][0]["conditions"][0]["dim_name"] = "wrong_label"
        report = validate_scenario(_write_yaml(tmp_path, cfg))
        # Still ok overall, but warning is present.
        assert report.ok
        assert any(i.code == "dim_name_mismatch" for i in report.warnings)

    def test_unknown_op_caught(self, tmp_path):
        cfg = _cartpole_left_only()
        cfg["groups"][0]["conditions"][0]["op"] = "approximately"
        report = validate_scenario(_write_yaml(tmp_path, cfg))
        assert not report.ok
        # schema layer rejects unknown enum
        assert any(i.code == "schema" for i in report.errors)

    def test_contradictory_conditions_caught(self, tmp_path):
        cfg = _cartpole_left_only()
        cfg["groups"][0]["conditions"] = [
            {"dim": 0, "op": ">", "value": 5.0, "target": "state"},
            {"dim": 0, "op": "<", "value": 0.0, "target": "state"},
        ]
        report = validate_scenario(_write_yaml(tmp_path, cfg))
        assert not report.ok
        assert any(i.code == "contradictory_conditions" for i in report.errors)

    def test_short_description_lint(self, tmp_path):
        cfg = _cartpole_left_only()
        cfg["description"] = "x"
        report = validate_scenario(_write_yaml(tmp_path, cfg))
        assert any(i.code == "short_description" for i in report.warnings)

    def test_unknown_env_lint(self, tmp_path):
        cfg = _cartpole_left_only()
        cfg["env_id"] = "Atari-Mystery-v0"
        report = validate_scenario(_write_yaml(tmp_path, cfg))
        assert any(i.code == "unknown_env" for i in report.warnings)

    def test_temporal_requires_temporal_block(self, tmp_path):
        cfg = _cartpole_left_only()
        cfg["match_mode"] = "temporal"
        report = validate_scenario(_write_yaml(tmp_path, cfg))
        assert not report.ok

    def test_in_memory_dict_validation(self):
        # Smoke test the programmatic API: same checks, no tmp file.
        cfg = _cartpole_left_only()
        cfg["groups"][0]["conditions"][0]["op"] = "approximately"
        report = validate_scenario_dict(cfg)
        assert not report.ok


# ---------------------------------------------------------------------------
# Introspection: prose + LaTeX + coverage.
# ---------------------------------------------------------------------------


class TestIntrospection:
    def test_natural_language_contains_threshold_and_dim_name(self):
        s = ForgetScenario(_cartpole_left_only())
        prose = to_natural_language(s)
        assert "left_only_test" in prose
        assert "CartPole-v1" in prose
        assert "30%" in prose
        # dim_name resolved from env_id even though not declared
        assert "cart_position" in prose
        assert "greater than" in prose

    def test_predicate_has_quantifier_and_threshold(self):
        s = ForgetScenario(_cartpole_left_only())
        latex = to_predicate(s)
        assert r"\mathrm{forget}(\tau)" in latex
        assert r"\iff" in latex
        assert "0.30" in latex
        assert r"\geq" in latex

    def test_predicate_for_any_step(self):
        cfg = _cartpole_left_only()
        cfg["match_mode"] = "any_step"
        cfg.pop("match_threshold", None)
        s = ForgetScenario(cfg)
        latex = to_predicate(s)
        assert r"\exists" in latex
        assert r"\forall" not in latex

    def test_predicate_for_all_steps(self):
        cfg = _cartpole_left_only()
        cfg["match_mode"] = "all_steps"
        cfg.pop("match_threshold", None)
        s = ForgetScenario(cfg)
        latex = to_predicate(s)
        assert r"\forall" in latex

    def test_predicate_for_top_percentile(self):
        cfg = {
            "name": "tp", "description": "top 10% by return",
            "env_id": "CartPole-v1",
            "match_mode": "top_percentile", "match_threshold": 0.1,
            "groups": [{"rank_by": "return", "rank_order": "descending"}],
        }
        s = ForgetScenario(cfg)
        latex = to_predicate(s)
        assert r"\mathrm{top}" in latex
        assert "0.10" in latex

    def test_natural_language_for_boolean_composition(self):
        cfg = {
            "name": "comp", "description": "x>0 AND y<0",
            "env_id": "LunarLander-v3",
            "match_mode": "any_step",
            "groups": [{
                "all_of": [
                    {"conditions": [{"dim": 0, "op": ">", "value": 0.0}]},
                    {"conditions": [{"dim": 1, "op": "<", "value": 0.0}]},
                ],
            }],
        }
        s = ForgetScenario(cfg)
        prose = to_natural_language(s)
        # "and" in conjunction connector
        assert " and " in prose

    def test_coverage_empty(self):
        s = ForgetScenario(_cartpole_left_only())
        rep = coverage(s, [])
        assert rep.num_trajectories == 0

    def test_coverage_matches_step_and_traj_fractions(self):
        s = ForgetScenario(_cartpole_left_only())
        # Two trajectories: one 50% right (matches @ thresh 0.3),
        # one 10% right (does not match @ thresh 0.3).
        trajs = [
            {"observations": [np.array([1.0, 0, 0, 0])] * 5
                             + [np.array([-1.0, 0, 0, 0])] * 5,
             "actions": [0] * 10, "rewards": [1.0] * 10},
            {"observations": [np.array([1.0, 0, 0, 0])] * 1
                             + [np.array([-1.0, 0, 0, 0])] * 9,
             "actions": [0] * 10, "rewards": [1.0] * 10},
        ]
        rep = coverage(s, trajs)
        assert rep.num_trajectories == 2
        assert rep.matching_trajectories == 1
        assert rep.forget_traj_fraction == 0.5
        # 5 + 1 = 6 matching steps out of 20.
        assert rep.matching_steps == 6
        assert abs(rep.forget_region_fraction - 0.3) < 1e-9


# ---------------------------------------------------------------------------
# All bundled scenarios round-trip through the validator.
# ---------------------------------------------------------------------------


class TestBundledScenarios:
    """The bundled configs/scenarios/*.yaml are part of the project's
    public surface. Any change that makes one fail validation is a real
    regression."""

    @pytest.fixture(scope="class")
    def bundled_scenario_files(self):
        root = Path(__file__).resolve().parent.parent / "configs" / "scenarios"
        return sorted(root.rglob("*.yaml"))

    def test_all_bundled_pass_validation(self, bundled_scenario_files):
        for p in bundled_scenario_files:
            report = validate_scenario(p)
            assert report.ok, (
                f"Bundled scenario {p.relative_to(p.parents[2])} failed:"
                + "\n" + report.render()
            )

    def test_all_bundled_load_and_render(self, bundled_scenario_files):
        for p in bundled_scenario_files:
            s = ForgetScenario.load(str(p))
            # Both renderers must produce non-empty strings.
            assert to_natural_language(s).strip()
            assert to_predicate(s).strip()
