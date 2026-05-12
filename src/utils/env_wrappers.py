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
    """Flat obs whose first 3 dims are [agent_x, agent_y, direction] in
    absolute grid coords, followed by either:

    - ``obs_encoding="symbolic"`` (default, recommended): MiniGrid's
      ``FlatObsWrapper`` one-hot symbolic encoding of the partial view
      (object_type x color x state x mission). This is what SB3-zoo /
      Farama use to actually solve FourRooms with vanilla PPO. The raw
      7x7x3 RGB image is hostile to flat MLPs.
    - ``obs_encoding="image"``: flattened raw partial RGB image
      (legacy; kept so existing Empty-8x8 checkpoints still load).

    The (x, y, dir) prefix is preserved so scenario YAMLs can define
    forget regions in terms of grid coordinates (``dim: 0`` = x, etc.).
    """

    def __init__(self, env: gym.Env, obs_encoding: str = "image"):
        super().__init__(env)
        self._encoding = obs_encoding
        if obs_encoding == "symbolic":
            from minigrid.wrappers import FlatObsWrapper
            # Wrap env to get FlatObs's one-hot output. We don't keep
            # FlatObsWrapper as a parent — we use it as a helper.
            self._flat = FlatObsWrapper(self.env)
            inner_dim = int(np.prod(self._flat.observation_space.shape))
        else:
            sample = env.observation_space["image"]
            inner_dim = int(np.prod(sample.shape))
            self._flat = None
        self._inner_dim = inner_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3 + inner_dim,), dtype=np.float32,
        )

    def observation(self, obs):
        x, y = self.env.unwrapped.agent_pos
        d = obs.get("direction", self.env.unwrapped.agent_dir)
        if self._encoding == "symbolic":
            inner = self._flat.observation(obs).astype(np.float32)
        else:
            inner = np.asarray(obs["image"], dtype=np.float32).reshape(-1)
        return np.concatenate(
            [np.array([x, y, d], dtype=np.float32), inner], axis=0,
        ).astype(np.float32)


class CountBasedExploration(gym.Wrapper):
    """Add a per-pose visit-count exploration bonus to the env reward.

    Bonus at step t:    beta(t) / sqrt(N(s_t) + 1)
    where N(s) is the visit count for pose s across the wrapper's
    lifetime and beta(t) optionally anneals linearly from ``beta`` to 0
    over ``anneal_steps`` global steps. Annealing matters because
    count-based bonuses help early exploration but actively destabilise
    converged policies if they keep firing — observed empirically as
    eval-reward decay from 0.85 → 0.05 over 5M FourRooms steps.

    Defeats vanilla PPO's exploration ceiling on sparse-reward gridworlds
    like FourRooms. Opt-in via ``make_env(exploration_bonus="count_based")``.
    """

    def __init__(
        self, env: gym.Env, beta: float = 0.05, hash_dims: int = 3,
        anneal_steps: int = 0,
    ):
        super().__init__(env)
        self._beta = float(beta)
        self._hash_dims = int(hash_dims)
        self._anneal_steps = int(anneal_steps)
        self._global_step = 0
        self._counts: dict[tuple, int] = {}

    def _key(self, obs: np.ndarray) -> tuple:
        flat = np.asarray(obs).reshape(-1)
        return tuple(int(round(v)) for v in flat[: self._hash_dims])

    def _current_beta(self) -> float:
        if self._anneal_steps <= 0:
            return self._beta
        frac = max(0.0, 1.0 - self._global_step / self._anneal_steps)
        return self._beta * frac

    def reset(self, **kwargs):
        # Visit counts persist across episodes (intrinsic memory).
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self._global_step += 1
        key = self._key(obs)
        self._counts[key] = self._counts.get(key, 0) + 1
        bonus = self._current_beta() / np.sqrt(self._counts[key] + 1)
        info["intrinsic_reward"] = float(bonus)
        return obs, float(reward) + bonus, term, trunc, info


def make_minigrid_env(
    env_id: str, seed: int | None = None,
    exploration_bonus: str | None = None,
    exploration_beta: float = 0.05,
    exploration_anneal_steps: int = 0,
    obs_encoding: str = "image",
    **gym_kwargs,
) -> gym.Env:
    """Construct a MiniGrid env with the flat-pos wrapper."""
    import minigrid  # noqa: F401  (registers MiniGrid envs)
    env = gym.make(env_id, **gym_kwargs)
    env = MiniGridFlatPos(env, obs_encoding=obs_encoding)
    if exploration_bonus == "count_based":
        env = CountBasedExploration(
            env, beta=exploration_beta, anneal_steps=exploration_anneal_steps,
        )
    if seed is not None:
        env.reset(seed=seed)
    return env


def is_minigrid_env(env_id: str) -> bool:
    return env_id.startswith("MiniGrid-")


def make_env(
    env_id: str, seed: int | None = None,
    exploration_bonus: str | None = None,
    exploration_beta: float = 0.05,
    exploration_anneal_steps: int = 0,
    obs_encoding: str = "image",
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
            exploration_anneal_steps=exploration_anneal_steps,
            obs_encoding=obs_encoding,
            **gym_kwargs,
        )
    env = gym.make(env_id, **gym_kwargs)
    if seed is not None:
        env.reset(seed=seed)
    return env
