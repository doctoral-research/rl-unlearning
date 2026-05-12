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


class CountBasedExploration(gym.Wrapper):
    """Add a per-pose visit-count exploration bonus to the env reward.

    Bonus added at step t:    beta / sqrt(N(s_t) + 1)
    where N(s) is the number of times pose s has been visited across the
    wrapper's lifetime. Pose is hashed from the first ``hash_dims`` obs
    components — for MiniGrid wrapped with MiniGridFlatPos these are
    (agent_x, agent_y, direction).

    Defeats vanilla PPO's exploration ceiling on sparse-reward gridworlds
    like FourRooms. Opt-in only — applied via ``make_env(..., exploration_bonus=...)``.
    """

    def __init__(self, env: gym.Env, beta: float = 0.05, hash_dims: int = 3):
        super().__init__(env)
        self._beta = float(beta)
        self._hash_dims = int(hash_dims)
        self._counts: dict[tuple, int] = {}

    def _key(self, obs: np.ndarray) -> tuple:
        flat = np.asarray(obs).reshape(-1)
        return tuple(int(round(v)) for v in flat[: self._hash_dims])

    def reset(self, **kwargs):
        # Visit counts persist across episodes (intrinsic memory).
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        key = self._key(obs)
        self._counts[key] = self._counts.get(key, 0) + 1
        bonus = self._beta / np.sqrt(self._counts[key] + 1)
        info.setdefault("intrinsic_reward", 0.0)
        info["intrinsic_reward"] = float(bonus)
        return obs, float(reward) + bonus, term, trunc, info


def make_minigrid_env(
    env_id: str, seed: int | None = None,
    exploration_bonus: str | None = None,
    exploration_beta: float = 0.05,
    **gym_kwargs,
) -> gym.Env:
    """Construct a MiniGrid env with the flat-pos wrapper.

    Optional ``exploration_bonus="count_based"`` adds a count-based
    intrinsic reward to every step (see CountBasedExploration).
    """
    import minigrid  # noqa: F401  (registers MiniGrid envs)
    env = gym.make(env_id, **gym_kwargs)
    env = MiniGridFlatPos(env)
    if exploration_bonus == "count_based":
        env = CountBasedExploration(env, beta=exploration_beta)
    if seed is not None:
        env.reset(seed=seed)
    return env


def is_minigrid_env(env_id: str) -> bool:
    return env_id.startswith("MiniGrid-")


def make_env(
    env_id: str, seed: int | None = None,
    exploration_bonus: str | None = None,
    exploration_beta: float = 0.05,
    **gym_kwargs,
) -> gym.Env:
    """Single entry point used across train/unlearn/evaluate.

    Routes MiniGrid envs through the flat-pos wrapper; everything else
    goes through plain `gym.make(env_id, **gym_kwargs)` to preserve
    existing behaviour. Extra kwargs (e.g. ``render_mode="rgb_array"``)
    pass through to `gym.make`.

    ``exploration_bonus="count_based"`` (MiniGrid only) adds a per-pose
    count-based intrinsic reward — useful for sparse-reward envs like
    FourRooms where vanilla PPO can't crack exploration.
    """
    if is_minigrid_env(env_id):
        return make_minigrid_env(
            env_id, seed=seed,
            exploration_bonus=exploration_bonus,
            exploration_beta=exploration_beta,
            **gym_kwargs,
        )
    env = gym.make(env_id, **gym_kwargs)
    if seed is not None:
        env.reset(seed=seed)
    return env
