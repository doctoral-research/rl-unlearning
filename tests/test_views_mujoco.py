"""MuJoCo FeatureView tests. Split from test_views.py because a
module-level importorskip would otherwise skip the entire view suite on a
machine without mujoco (e.g. CI)."""

from __future__ import annotations

import numpy as np
import pytest

from scenarios import ForgetScenario
from scenarios.views import FeatureView  # noqa: F401

pytest.importorskip("mujoco", reason="mujoco not installed")

from scenarios.views import mujoco_view  # noqa: E402



class TestMuJoCoView:
    def _env(self, eid="HalfCheetah-v5"):
        import gymnasium as gym

        env = gym.make(eid)
        env.reset(seed=0)
        return env

    def test_declines_on_non_mujoco(self):
        assert mujoco_view(object()) is None

    def test_uses_real_joint_names(self):
        env = self._env()
        try:
            v = mujoco_view(env)
            assert v is not None
            assert v.provenance == "auto:mujoco"
            for expected in ("rootx", "rootz", "rooty", "bthigh", "ffoot"):
                assert expected in v.names
            assert "v_rootx" in v.names
        finally:
            env.close()

    def test_covers_qpos_and_qvel(self):
        env = self._env()
        try:
            m = env.unwrapped.model
            v = mujoco_view(env)
            assert len(v) == int(m.nq) + int(m.nv)
        finally:
            env.close()

    def test_recovers_coordinate_absent_from_observation(self):
        """rootx is dropped from the Gymnasium MuJoCo observation, so it is
        not expressible against obs at any index. Reading the model
        recovers it."""
        env = self._env()
        try:
            obs, _ = env.reset(seed=0)
            v = mujoco_view(env)
            assert len(v) > obs.shape[0]          # 18 features vs 17 obs dims
            out = v.extract(env=env)
            assert np.isfinite(out[v.index("rootx")])
        finally:
            env.close()

    def test_walker_joint_names_differ_from_cheetah(self):
        env = self._env("Walker2d-v5")
        try:
            v = mujoco_view(env)
            assert "thigh_left_joint" in v.names
            assert "bthigh" not in v.names
        finally:
            env.close()

    def test_scenario_can_target_a_model_only_feature(self):
        """A forget scenario over absolute position: impossible in v1.0."""
        env = self._env()
        try:
            v = mujoco_view(env)
            spec = {
                "name": "no_far_forward",
                "description": "Forget running past x = 10.",
                "match_mode": "any_step",
                "groups": [{"conditions": [
                    {"dim": "rootx", "op": ">", "value": 10.0}
                ]}],
            }
            s = ForgetScenario(spec, view=v)
            assert s.matches_state(obs=None, env=env) is False
            env.unwrapped.data.qpos[0] = 42.0
            assert s.matches_state(obs=None, env=env) is True
        finally:
            env.close()
