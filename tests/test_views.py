"""Tests for the FeatureView layer (Stage 0: view/observation decoupling)."""

from __future__ import annotations

import numpy as np
import pytest

from scenarios import ForgetScenario, ScenarioError
from scenarios.views import (
    BOOLEAN,
    CONTINUOUS,
    FeatureView,
    ViewError,
    build_view,
    identity_view,
    minigrid_view,
    table_view,
)


# ----------------------------------------------------------------------
# FeatureView construction and validation
# ----------------------------------------------------------------------


def _const_view(names, values, **kw):
    return FeatureView(
        names=tuple(names),
        extract_fn=lambda obs, info, env: list(values),
        **kw,
    )


class TestFeatureViewConstruction:
    def test_defaults_are_filled(self):
        v = _const_view(["a", "b"], [1.0, 2.0])
        assert len(v) == 2
        assert v.dtypes == (CONTINUOUS, CONTINUOUS)
        assert v.ranges == (None, None)
        assert v.provenance == "declared:identity"

    def test_duplicate_names_rejected(self):
        with pytest.raises(ViewError, match="duplicate"):
            _const_view(["a", "a"], [1.0, 2.0])

    def test_dtype_length_mismatch_rejected(self):
        with pytest.raises(ViewError, match="dtypes"):
            _const_view(["a", "b"], [1.0, 2.0], dtypes=(CONTINUOUS,))

    def test_unknown_dtype_rejected(self):
        with pytest.raises(ViewError, match="unknown dtype"):
            _const_view(["a"], [1.0], dtypes=("colour",))

    def test_range_length_mismatch_rejected(self):
        with pytest.raises(ViewError, match="ranges"):
            _const_view(["a", "b"], [1.0, 2.0], ranges=((0.0, 1.0),))

    def test_bad_requires_rejected(self):
        with pytest.raises(ViewError, match="requires"):
            _const_view(["a"], [1.0], requires=("obs", "moon"))

    def test_index_and_name_lookup(self):
        v = _const_view(["x", "y"], [1.0, 2.0])
        assert v.index("y") == 1
        assert v.name_of(0) == "x"
        assert v.name_of(99) is None
        with pytest.raises(ViewError, match="not in view"):
            v.index("z")

    def test_as_dim_table_matches_constants_shape(self):
        v = _const_view(["x", "y"], [1.0, 2.0])
        assert v.as_dim_table() == {0: "x", 1: "y"}

    def test_describe_mentions_provenance_and_names(self):
        v = _const_view(["x"], [1.0], provenance="auto:test")
        text = v.describe()
        assert "auto:test" in text and "x" in text


class TestFeatureViewExtraction:
    def test_extract_returns_float_vector(self):
        v = _const_view(["a", "b"], [1, 2])
        out = v.extract(obs=np.zeros(3))
        assert out.dtype == np.float32
        assert out.tolist() == [1.0, 2.0]

    def test_missing_requirement_raises(self):
        v = _const_view(["a"], [1.0], requires=("env",))
        with pytest.raises(ViewError, match="requires"):
            v.extract(obs=np.zeros(2))

    def test_wrong_arity_raises(self):
        v = FeatureView(
            names=("a", "b"),
            extract_fn=lambda obs, info, env: [1.0],
        )
        with pytest.raises(ViewError, match="produced 1 features"):
            v.extract(obs=np.zeros(2))


# ----------------------------------------------------------------------
# Built-in providers
# ----------------------------------------------------------------------


class TestIdentityView:
    def test_reproduces_raw_observation(self):
        v = identity_view(4)
        obs = np.array([3.0, 1.0, 4.0, 1.0])
        assert v.extract(obs=obs).tolist() == obs.tolist()

    def test_names_are_anonymous(self):
        assert identity_view(3).names == ("dim_0", "dim_1", "dim_2")

    def test_provenance(self):
        assert identity_view(2).provenance == "declared:identity"


class TestTableView:
    def test_known_env_gets_names(self):
        v = table_view("CartPole-v1")
        assert v is not None
        assert v.names[0] == "cart_position"
        assert v.provenance == "declared:table"

    def test_unknown_env_returns_none(self):
        assert table_view("NotAnEnv-v9") is None

    def test_extract_truncates_to_named_prefix(self):
        v = table_view("MiniGrid-FourRooms-v0")
        assert len(v) == 3  # agent_x, agent_y, direction
        out = v.extract(obs=np.arange(100, dtype=np.float32))
        assert out.tolist() == [0.0, 1.0, 2.0]


class TestMiniGridView:
    def _env(self):
        from minigrid.envs.fourrooms import FourRoomsEnv

        env = FourRoomsEnv(agent_pos=(2, 2), goal_pos=(13, 3))
        env.reset(seed=0)
        return env

    def test_declines_on_non_minigrid(self):
        assert minigrid_view(object()) is None

    def test_reads_env_not_observation(self):
        env = self._env()
        v = minigrid_view(env)
        assert v is not None
        assert v.requires == ("env",)
        env.unwrapped.agent_pos = (11, 12)
        # obs deliberately garbage: the view must not consult it
        out = v.extract(obs={"nonsense": 1}, env=env)
        assert out[v.index("agent_x")] == 11.0
        assert out[v.index("agent_y")] == 12.0

    def test_provenance_and_dtypes(self):
        v = minigrid_view(self._env())
        assert v.provenance == "auto:minigrid"
        assert v.dtype_of(v.index("carrying_key")) == BOOLEAN


class TestBuildViewCascade:
    def test_env_provider_wins_over_table(self):
        from minigrid.envs.fourrooms import FourRoomsEnv

        env = FourRoomsEnv(agent_pos=(2, 2), goal_pos=(13, 3))
        env.reset(seed=0)
        v = build_view(env=env, env_id="MiniGrid-FourRooms-v0")
        assert v.provenance == "auto:minigrid"

    def test_falls_back_to_table(self):
        v = build_view(env_id="CartPole-v1")
        assert v.provenance == "declared:table"

    def test_falls_back_to_identity(self):
        v = build_view(env_id="NotAnEnv-v9", n_dims=5)
        assert v.provenance == "declared:identity"
        assert len(v) == 5

    def test_no_fallback_available_raises(self):
        with pytest.raises(ViewError, match="cannot build a view"):
            build_view(env_id="NotAnEnv-v9")

    def test_prefer_identity(self):
        v = build_view(env_id="CartPole-v1", n_dims=4, prefer="identity")
        assert v.provenance == "declared:identity"

    def test_prefer_unknown_provider_raises(self):
        with pytest.raises(ViewError, match="unknown provider"):
            build_view(env_id="CartPole-v1", prefer="telepathy")

    def test_prefer_declining_provider_raises(self):
        with pytest.raises(ViewError, match="declined"):
            build_view(env=object(), prefer="minigrid")


# ----------------------------------------------------------------------
# Scenario integration
# ----------------------------------------------------------------------


SPEC = {
    "name": "view_bound",
    "description": "Scenario written against feature names, not raw indices.",
    "match_mode": "any_step",
    "groups": [
        {
            "name": "right_half",
            "conditions": [{"dim": "agent_x", "op": ">", "value": 9.5}],
        }
    ],
}


class TestScenarioWithView:
    def _view(self):
        return _const_view(["agent_x", "agent_y"], [0.0, 0.0])

    def test_names_resolve_through_view(self):
        s = ForgetScenario(SPEC, view=self._view())
        assert s.groups[0]["conditions"][0]["dim"] == 0

    def test_unknown_name_is_scenario_error(self):
        bad = dict(SPEC)
        bad["groups"] = [{"conditions": [{"dim": "nope", "op": ">", "value": 1}]}]
        with pytest.raises(ScenarioError, match="not in view"):
            ForgetScenario(bad, view=self._view())

    def test_matches_state_goes_through_view(self):
        view = FeatureView(
            names=("agent_x", "agent_y"),
            extract_fn=lambda obs, info, env: [env["x"], env["y"]],
            requires=("env",),
            provenance="auto:test",
        )
        s = ForgetScenario(SPEC, view=view)
        assert s.matches_state(obs=None, env={"x": 12.0, "y": 0.0}) is True
        assert s.matches_state(obs=None, env={"x": 3.0, "y": 0.0}) is False

    def test_without_view_observation_is_used_directly(self):
        """v1.0 behaviour: raw integer dims against the observation."""
        spec = dict(SPEC)
        spec["groups"] = [{"conditions": [{"dim": 0, "op": ">", "value": 9.5}]}]
        s = ForgetScenario(spec)
        assert s.view is None
        assert s.matches_state(np.array([12.0, 0.0])) is True
        assert s.matches_state(np.array([3.0, 0.0])) is False

    def test_with_view_rebinds(self):
        s = ForgetScenario(SPEC, view=self._view())
        other = _const_view(["agent_y", "agent_x"], [0.0, 0.0])  # swapped order
        s2 = s.with_view(other)
        assert s.groups[0]["conditions"][0]["dim"] == 0
        assert s2.groups[0]["conditions"][0]["dim"] == 1
        assert s is not s2

    def test_features_helper_without_view_is_identity(self):
        spec = dict(SPEC)
        spec["groups"] = [{"conditions": [{"dim": 0, "op": ">", "value": 1}]}]
        s = ForgetScenario(spec)
        obs = np.array([1.0, 2.0])
        assert s.features(obs) is obs

    def test_trajectory_prefers_recorded_features(self):
        spec = dict(SPEC)
        spec["match_mode"] = "proportion"
        spec["match_threshold"] = 0.5
        spec["groups"] = [{"conditions": [{"dim": 0, "op": ">", "value": 9.5}]}]
        s = ForgetScenario(spec)
        traj = {
            "observations": [np.array([0.0]), np.array([0.0])],  # would not match
            "features": [np.array([20.0]), np.array([20.0])],    # matches
        }
        assert s.identify_forget_trajectories([traj]) == [0]


# ----------------------------------------------------------------------
# Stage 1: generic providers, tiering, portability reporting
# ----------------------------------------------------------------------

from scenarios.views import (  # noqa: E402
    TIER_ENV,
    TIER_GENERIC,
    describe_domain,
    domain_report,
    info_view,
    provider_cost,
    register_view_provider,
)


class TestInfoView:
    def test_declines_without_probe(self):
        assert info_view(object(), None) is None

    def test_declines_on_empty_info(self):
        assert info_view(object(), {"info": {}}) is None

    def test_declines_when_no_numeric_keys(self):
        assert info_view(object(), {"info": {"mission": "go home"}}) is None

    def test_picks_up_numeric_keys_sorted(self):
        v = info_view(object(), {"info": {"score": 3, "lives": 2, "msg": "x"}})
        assert v is not None
        assert v.names == ("lives", "score")
        assert v.provenance == "auto:info_dict"
        assert v.requires == ("info",)

    def test_extracts_from_info(self):
        v = info_view(object(), {"info": {"a": 1, "b": 2}})
        assert v.extract(info={"a": 7.0, "b": 8.0}).tolist() == [7.0, 8.0]


class TestTiering:
    def test_name_table_beats_generic_info(self):
        """Regression: info_dict used to win on Taxi and produce a useless
        single `prob` feature, shadowing the hand-written names."""
        class FakeEnv:
            pass

        v = build_view(env=FakeEnv(), env_id="CartPole-v1",
                       probe={"info": {"prob": 1.0}})
        assert v.provenance == "declared:table"

    def test_generic_used_when_no_table(self):
        class FakeEnv:
            pass

        v = build_view(env=FakeEnv(), env_id="Unregistered-v0",
                       probe={"info": {"prob": 1.0}})
        assert v.provenance == "auto:info_dict"

    def test_registered_env_provider_beats_table(self):
        marker = FeatureView(
            names=("custom",),
            extract_fn=lambda obs, info, env: [1.0],
            provenance="auto:pytest_probe",
        )
        register_view_provider(
            "pytest_probe", lambda env, probe=None: marker, tier=TIER_ENV,
            first=True,
        )
        try:
            v = build_view(env=object(), env_id="CartPole-v1")
            assert v.provenance == "auto:pytest_probe"
        finally:
            from scenarios import views as _v
            _v._PROVIDERS[:] = [p for p in _v._PROVIDERS
                                if p[0] != "pytest_probe"]


class TestPortabilityReport:
    def test_provider_cost_counts_adapter_lines(self):
        loc = provider_cost("auto:minigrid")
        assert isinstance(loc, int) and loc > 5

    def test_provider_cost_none_for_identity(self):
        assert provider_cost("declared:identity") is None

    def test_describe_domain_known_env(self):
        r = describe_domain("CartPole-v1")
        assert r["error"] is None
        assert r["provenance"] == "declared:table"
        assert r["n_features"] == 4
        assert r["names"][0] == "cart_position"

    def test_describe_domain_minigrid_is_automatic(self):
        r = describe_domain("MiniGrid-FourRooms-v0")
        assert r["error"] is None
        assert r["provenance"] == "auto:minigrid"
        assert "agent_x" in r["names"]

    def test_describe_domain_captures_failure(self):
        r = describe_domain("DefinitelyNotAnEnv-v0")
        assert r["error"] is not None
        assert r["provenance"] is None

    def test_domain_report_is_one_row_per_env(self):
        rows = domain_report(["CartPole-v1", "Pendulum-v1"])
        assert [r["env_id"] for r in rows] == ["CartPole-v1", "Pendulum-v1"]

