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
        anneal_steps: int = 0, hash_scale: float = 1.0,
    ):
        super().__init__(env)
        self._beta = float(beta)
        self._hash_dims = int(hash_dims)
        self._anneal_steps = int(anneal_steps)
        self._hash_scale = float(hash_scale)
        self._global_step = 0
        self._counts: dict[tuple, int] = {}

    def _key(self, obs: np.ndarray) -> tuple:
        flat = np.asarray(obs).reshape(-1)
        return tuple(
            int(round(v * self._hash_scale)) for v in flat[: self._hash_dims]
        )

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
    fixed_goal_pos: tuple | None = None,
    **gym_kwargs,
) -> gym.Env:
    """Construct a MiniGrid env with the flat-pos wrapper.

    ``fixed_goal_pos=(x, y)`` pins the goal cell for envs that normally
    randomise it (e.g. FourRooms). Important so that "avoid region X"
    scenarios remain well-defined across episodes — without pinning,
    ~19% of FourRooms seeds put the goal inside the forget region.
    """
    import minigrid  # noqa: F401  (registers MiniGrid envs)
    if fixed_goal_pos is not None and "FourRooms" in env_id:
        from minigrid.envs.fourrooms import FourRoomsEnv
        env = FourRoomsEnv(goal_pos=tuple(fixed_goal_pos), **gym_kwargs)
    else:
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


class ProcgenAdapter:
    """Make procgen's ProcgenEnv look like a gymnasium VectorEnv.

    procgen.ProcgenEnv is natively vectorized (num_envs param) but uses
    the old gym 4-tuple step API and returns a Dict obs space. This
    adapter:
      - exposes single_observation_space / single_action_space attrs
      - returns obs as a flat (N, 64, 64, 3) float32 tensor in [0, 1]
      - converts 4-tuple step output to gymnasium 5-tuple
    """

    def __init__(self, env_name: str = "coinrun", num_envs: int = 8,
                 num_levels: int = 0, start_level: int = 0,
                 distribution_mode: str = "easy"):
        import procgen
        self._env = procgen.ProcgenEnv(
            num_envs=num_envs, env_name=env_name,
            num_levels=num_levels, start_level=start_level,
            distribution_mode=distribution_mode,
        )
        self.num_envs = num_envs
        self.single_observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(64, 64, 3), dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(num_envs, 64, 64, 3), dtype=np.float32,
        )
        self.single_action_space = gym.spaces.Discrete(15)
        self.action_space = gym.spaces.MultiDiscrete([15] * num_envs)

    def reset(self, seed=None):
        obs = self._env.reset()
        rgb = obs["rgb"].astype(np.float32) / 255.0
        return rgb, {}

    def step(self, action):
        obs, reward, done, info = self._env.step(np.asarray(action, dtype=np.int32))
        rgb = obs["rgb"].astype(np.float32) / 255.0
        # Old gym `done` covers terminated+truncated; procgen episodes end
        # on level completion or death so we treat all as terminations.
        terminated = done.astype(bool)
        truncated = np.zeros_like(done, dtype=bool)
        return rgb, reward.astype(np.float32), terminated, truncated, info

    def close(self):
        self._env.close()


class RunningMeanStd:
    """Welford running mean/variance, vectorized across one obs vector."""

    def __init__(self, shape: tuple, epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0) if x.ndim > 1 else x
        batch_var = x.var(axis=0) if x.ndim > 1 else np.zeros_like(x)
        batch_count = float(x.shape[0]) if x.ndim > 1 else 1.0
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot
        self.mean = new_mean
        self.var = M2 / tot
        self.count = tot


class NormalizeObservation(gym.ObservationWrapper):
    """Standardize obs to zero mean / unit variance using a running estimate.

    Standard recipe for envs whose obs dims live on wildly different
    scales (Pendulum: cos_theta in [-1, 1], theta_dot in [-8, 8]). Without
    normalization the network is dominated by the high-magnitude dim.
    SB3-zoo and the original PPO paper both use this on continuous-control
    benchmarks.

    State is shared across env.reset() so the running estimate keeps
    accumulating across episodes (the natural choice for on-policy RL).
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        super().__init__(env)
        self.epsilon = float(epsilon)
        self.obs_rms = RunningMeanStd(self.observation_space.shape)

    def observation(self, obs):
        obs = np.asarray(obs, dtype=np.float64)
        self.obs_rms.update(obs)
        return ((obs - self.obs_rms.mean) /
                np.sqrt(self.obs_rms.var + self.epsilon)).astype(np.float32)


class ForgetMaskWrapper(gym.Wrapper):
    """Hard-constraint env wrapper for the full-retrain oracle baseline.

    Whenever the agent enters a forget-scenario state, this wrapper either
    terminates the episode with a large penalty (``mode="terminate"``) or
    zeros the reward there (``mode="zero_reward"``). Training PPO under
    this wrapper from a random init yields a policy that has provably
    *never* been reinforced in the forget region — the ground-truth
    "fresh policy that didn't learn the forget behavior."

    Used to compare against online unlearning methods on axes 1+4 (forget
    effectiveness and durability) while paying the full axis-3 cost
    (training from scratch). Standard machine-unlearning oracle.
    """

    def __init__(self, env: gym.Env, scenario, mode: str = "terminate",
                 penalty: float = 10.0):
        super().__init__(env)
        self.scenario = scenario
        self.mode = mode
        self.penalty = float(penalty)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        if self.scenario.matches_state(np.asarray(obs)):
            info["forget_violation"] = True
            if self.mode == "terminate":
                return obs, reward - self.penalty, True, trunc, info
            if self.mode == "zero_reward":
                return obs, 0.0, term, trunc, info
        return obs, reward, term, trunc, info


class TaxiDecode(gym.ObservationWrapper):
    """Expose Taxi-v3's `Discrete(500)` state as `[row, col, pass_loc, dest]`.

    Taxi-v3 encodes the state as
        s = ((taxi_row * 5 + taxi_col) * 5 + pass_loc) * 4 + dest_idx.
    Returning the 4-tuple as the observation lets the PPO MLP learn a
    Q function over a tractable input AND lets dim-based ForgetScenarios
    reference (row, col, pass_loc, dest) directly. The integer-state form
    would force the network through an embedding and force scenarios to
    decode bit-fiddled ints themselves.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([4, 4, 4, 3], dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, obs):
        s = int(obs)
        dest = s % 4
        pass_loc = (s // 4) % 5
        col = (s // 20) % 5
        row = (s // 100) % 5
        return np.array([row, col, pass_loc, dest], dtype=np.float32)


def make_env(
    env_id: str, seed: int | None = None,
    exploration_bonus: str | None = None,
    exploration_beta: float = 0.05,
    exploration_anneal_steps: int = 0,
    exploration_hash_scale: float = 1.0,
    obs_encoding: str = "image",
    fixed_goal_pos: tuple | None = None,
    normalize_obs: bool = False,
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
            fixed_goal_pos=fixed_goal_pos,
            **gym_kwargs,
        )
    env = gym.make(env_id, **gym_kwargs)
    if env_id.startswith("Taxi"):
        env = TaxiDecode(env)
    if normalize_obs:
        env = NormalizeObservation(env)
    if exploration_bonus == "count_based":
        env = CountBasedExploration(
            env, beta=exploration_beta, anneal_steps=exploration_anneal_steps,
            hash_scale=exploration_hash_scale,
        )
    if seed is not None:
        env.reset(seed=seed)
    return env
