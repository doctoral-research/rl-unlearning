"""Gym env wrappers used by the RL-unlearning pipeline.

MiniGrid envs give a 7x7x3 RGB partial-view image + a separate `direction`
scalar + `mission` string. Our PPO + scenario stack expects a flat float
observation whose first dimensions are interpretable (so scenario YAMLs
can reference them by index). The `MiniGridFlatPos` wrapper exposes the
agent's absolute (x, y, direction) as the first 3 dimensions, then
appends the flattened partial-view image.

This lets `configs/scenarios/fourrooms/*.yaml` define forget regions in
terms of grid coordinates (`dim: 0` = x, `dim: 1` = y) while the policy
still sees the local view it would normally consume.
"""

from typing import Tuple

import gymnasium as gym
import numpy as np


class MiniGridFlatPos(gym.ObservationWrapper):
    """Flatten MiniGrid dict-obs and prepend [x, y, dir] in absolute coords."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        sample = env.observation_space["image"]
        img_size = int(np.prod(sample.shape))
        # 3 pose dims (x, y, dir) + flat image.
        flat_dim = 3 + img_size
        self._img_size = img_size
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32,
        )

    def observation(self, obs):
        x, y = self.env.unwrapped.agent_pos
        d = obs.get("direction", self.env.unwrapped.agent_dir)
        img = np.asarray(obs["image"], dtype=np.float32).reshape(-1)
        return np.concatenate(
            [np.array([x, y, d], dtype=np.float32), img], axis=0,
        ).astype(np.float32)


def make_minigrid_env(
    env_id: str, seed: int | None = None, **gym_kwargs,
) -> gym.Env:
    """Construct a MiniGrid env with the standard flat-pos wrapper."""
    import minigrid  # noqa: F401  (registers MiniGrid envs)
    env = gym.make(env_id, **gym_kwargs)
    env = MiniGridFlatPos(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def is_minigrid_env(env_id: str) -> bool:
    return env_id.startswith("MiniGrid-")


def make_env(env_id: str, seed: int | None = None, **gym_kwargs) -> gym.Env:
    """Single entry point used across train/unlearn/evaluate.

    Routes MiniGrid envs through the flat-pos wrapper; everything else
    goes through plain `gym.make(env_id, **gym_kwargs)` to preserve
    existing behaviour. Extra kwargs (e.g. ``render_mode="rgb_array"``)
    pass through to `gym.make`.
    """
    if is_minigrid_env(env_id):
        return make_minigrid_env(env_id, seed=seed, **gym_kwargs)
    env = gym.make(env_id, **gym_kwargs)
    if seed is not None:
        env.reset(seed=seed)
    return env
