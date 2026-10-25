"""Multi-agent view via the contrib plugin path (Stage 2).

Separate module so a machine without lbforaging skips only these tests.
"""

from __future__ import annotations

import pytest

from scenarios import ForgetScenario, build_view

pytest.importorskip("lbforaging", reason="lbforaging not installed")

import gymnasium as gym  # noqa: E402
import lbforaging  # noqa: F401,E402

from scenarios.contrib.lbf import lbf_view, register  # noqa: E402

ENV_ID = "Foraging-8x8-3p-2f-v3"


@pytest.fixture()
def env():
    e = gym.make(ENV_ID)
    e.reset(seed=0)
    yield e
    e.close()


class TestLBFView:
    def test_declines_on_non_lbf(self):
        assert lbf_view(object()) is None

    def test_one_block_of_features_per_agent(self, env):
        v = lbf_view(env)
        n = len(env.unwrapped.players)
        assert len(v) == 4 * n + 2
        for i in range(n):
            for suffix in ("row", "col", "level", "score"):
                assert f"p{i}_{suffix}" in v.names

    def test_provenance_and_requirements(self, env):
        v = lbf_view(env)
        assert v.provenance == "auto:lbf"
        assert v.requires == ("env",)

    def test_reads_joint_ground_truth(self, env):
        v = lbf_view(env)
        out = v.extract(env=env)
        for i, p in enumerate(env.unwrapped.players):
            assert out[v.index(f"p{i}_row")] == float(p.position[0])
            assert out[v.index(f"p{i}_col")] == float(p.position[1])

    def test_grid_bounds_become_ranges(self, env):
        v = lbf_view(env)
        rows, cols = env.unwrapped.field.shape
        assert v.ranges[v.index("p0_row")] == (0.0, float(rows - 1))
        assert v.ranges[v.index("p0_col")] == (0.0, float(cols - 1))

    def test_food_aggregates(self, env):
        import numpy as np

        v = lbf_view(env)
        out = v.extract(env=env)
        fld = np.asarray(env.unwrapped.field)
        assert out[v.index("n_food")] == float((fld > 0).sum())
        assert out[v.index("food_level_total")] == float(fld.sum())


class TestPluginRegistration:
    def test_register_puts_lbf_in_the_cascade(self, env):
        from scenarios import views as _v

        before = list(_v._PROVIDERS)
        try:
            register()
            assert build_view(env=env).provenance == "auto:lbf"
        finally:
            _v._PROVIDERS[:] = before

    def test_without_registration_lbf_is_not_used(self, env):
        from scenarios import views as _v

        assert "lbf" not in [p[0] for p in _v._PROVIDERS]


class TestMultiAgentRelationalScenario:
    """Indexed names + derived features give relational multi-agent
    predicates without extending the grammar."""

    SPEC = {
        "name": "no_crowding",
        "description": "Forget letting agents 0 and 1 crowd together.",
        "match_mode": "any_step",
        "derived": {
            "separation_01": "abs(p0_row - p1_row) + abs(p0_col - p1_col)",
        },
        "groups": [{"conditions": [
            {"dim": "separation_01", "op": "<", "value": 3}]}],
    }

    def test_matches_when_agents_converge(self, env):
        s = ForgetScenario(self.SPEC, view=lbf_view(env))
        u = env.unwrapped
        u.players[0].position = (4, 4)
        u.players[1].position = (0, 0)
        assert s.matches_state(None, env=env) is False
        u.players[1].position = (4, 5)
        assert s.matches_state(None, env=env) is True

    def test_three_way_predicate(self, env):
        spec = dict(self.SPEC)
        spec["derived"] = {
            "sep_01": "abs(p0_row - p1_row) + abs(p0_col - p1_col)",
            "sep_02": "abs(p0_row - p2_row) + abs(p0_col - p2_col)",
            "min_sep": "min(sep_01, sep_02)",
        }
        spec["groups"] = [{"conditions": [
            {"dim": "min_sep", "op": "<", "value": 2}]}]
        s = ForgetScenario(spec, view=lbf_view(env))
        u = env.unwrapped
        u.players[0].position = (4, 4)
        u.players[1].position = (0, 0)
        u.players[2].position = (7, 7)
        assert s.matches_state(None, env=env) is False
        u.players[2].position = (4, 5)
        assert s.matches_state(None, env=env) is True

    def test_typo_in_agent_index_fails_at_load(self, env):
        from scenarios import ScenarioError

        spec = dict(self.SPEC)
        spec["derived"] = {"bad": "p9_row - p0_row"}
        spec["groups"] = [{"conditions": [
            {"dim": "bad", "op": "<", "value": 1}]}]
        with pytest.raises(ScenarioError):
            ForgetScenario(spec, view=lbf_view(env))
