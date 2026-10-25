"""Semantic feature views — the domain interface of the scenario language.

A `FeatureView` is the map

    Sigma : (obs, info, env) -> R^n

together with the names, types and ranges of those n features. It is the
*only* thing the scenario language needs to know about a domain.

Why this exists
---------------
v1.0 of the language evaluated conditions directly against the policy's
observation (`obs[dim]`). That has two costs:

1. It only works when the policy observation happens to contain
   interpretable scalars at fixed indices. For pixel, dict, token or
   graph observations there is no such index, so the language does not
   apply at all.
2. It forces semantics *into* the policy's input. `MiniGridFlatPos`
   prepends absolute (x, y, direction) to the agent's observation purely
   so scenarios can reference grid coordinates — which hands a
   partially-observable agent privileged state it should not have.

Decoupling the view from the observation removes both. `extract` may read
the `info` dict and the environment object, so a scenario can be written
over semantics the *agent never sees*: "avoid the bottom-right room" is
expressible for an agent whose observation is raw pixels.

Provenance
----------
Every view records how its semantics were obtained:

    declared:identity   fallback, dims are anonymous (dim_0, dim_1, ...)
    declared:table      hand-registered name table (constants.ENV_OBS_DIMS)
    auto:minigrid       introspected from the MiniGrid env object
    auto:dict_space     derived from a gymnasium Dict observation space
    auto:mujoco         derived from MuJoCo model joint/body names
    learned:<kind>      fitted from data; carries a fidelity score

Reporting provenance per domain is what lets portability be measured
instead of asserted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .constants import ENV_OBS_DIMS

# Feature types. Used by the validator to reject nonsense comparisons
# (`>` on a categorical, `==` on a float) and by the renderer to choose
# sensible prose.
CONTINUOUS = "continuous"
ORDINAL = "ordinal"
CATEGORICAL = "categorical"
BOOLEAN = "boolean"
VALID_DTYPES = {CONTINUOUS, ORDINAL, CATEGORICAL, BOOLEAN}


class ViewError(Exception):
    """Raised when a view is malformed or cannot extract."""


@dataclass(frozen=True)
class FeatureView:
    """Named semantic features extracted from (obs, info, env).

    `extract_fn` receives all three and returns a 1-D float array of
    length `len(names)`. It may ignore any of them; `requires` declares
    which ones it actually needs so callers can fail loudly rather than
    silently extracting garbage.
    """

    names: tuple[str, ...]
    extract_fn: Callable[[Any, Any, Any], Sequence[float]]
    dtypes: tuple[str, ...] = ()
    ranges: tuple[Optional[tuple[float, float]], ...] = ()
    provenance: str = "declared:identity"
    requires: tuple[str, ...] = ("obs",)
    fidelity: Optional[float] = None
    env_id: Optional[str] = None

    def __post_init__(self) -> None:
        names = tuple(self.names)
        object.__setattr__(self, "names", names)
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ViewError(f"duplicate feature names: {dupes}")
        # Fill optional metadata so downstream code can index freely.
        if not self.dtypes:
            object.__setattr__(self, "dtypes", (CONTINUOUS,) * len(names))
        elif len(self.dtypes) != len(names):
            raise ViewError(
                f"dtypes has {len(self.dtypes)} entries, names has {len(names)}"
            )
        bad = set(self.dtypes) - VALID_DTYPES
        if bad:
            raise ViewError(f"unknown dtype(s): {sorted(bad)}")
        if not self.ranges:
            object.__setattr__(self, "ranges", (None,) * len(names))
        elif len(self.ranges) != len(names):
            raise ViewError(
                f"ranges has {len(self.ranges)} entries, names has {len(names)}"
            )
        bad_req = set(self.requires) - {"obs", "info", "env"}
        if bad_req:
            raise ViewError(
                f"requires must be obs/info/env; got {sorted(bad_req)}")

    # -- introspection -------------------------------------------------

    def __len__(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise ViewError(
                f"feature {name!r} not in view (provenance={self.provenance}); "
                f"available: {list(self.names)}"
            ) from None

    def name_of(self, dim: int) -> Optional[str]:
        return self.names[dim] if 0 <= dim < len(self.names) else None

    def dtype_of(self, dim: int) -> Optional[str]:
        return self.dtypes[dim] if 0 <= dim < len(self.dtypes) else None

    def as_dim_table(self) -> dict[int, str]:
        """Same shape as constants.ENV_OBS_DIMS, so existing validator and
        renderer paths can consume a view without changes."""
        return {i: n for i, n in enumerate(self.names)}

    # -- extraction ----------------------------------------------------

    def extract(self, obs=None, info=None, env=None) -> np.ndarray:
        """Extract the feature vector."""
        missing = [
            r for r in self.requires
            if {"obs": obs, "info": info, "env": env}[r] is None
        ]
        if missing:
            raise ViewError(
                f"view {self.provenance!r} requires {missing} but they were "
                "not supplied"
            )
        vec = np.asarray(self.extract_fn(obs, info, env), dtype=np.float32).reshape(-1)
        if vec.shape[0] != len(self.names):
            raise ViewError(
                f"view {self.provenance!r} produced {vec.shape[0]} features, "
                f"expected {len(self.names)}"
            )
        return vec

    def describe(self) -> str:
        lines = [f"FeatureView({self.provenance}, n={len(self)})"]
        if self.env_id:
            lines.append(f"  env_id: {self.env_id}")
        if self.fidelity is not None:
            lines.append(f"  fidelity: {self.fidelity:.3f}")
        lines.append(f"  requires: {', '.join(self.requires)}")
        for i, (n, d) in enumerate(zip(self.names, self.dtypes)):
            rng = self.ranges[i]
            rng_s = f"  range={rng}" if rng else ""
            lines.append(f"  [{i}] {n}  ({d}){rng_s}")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Built-in providers
# ----------------------------------------------------------------------


def identity_view(n_dims: int, env_id: Optional[str] = None) -> FeatureView:
    """Fallback: the observation itself, dims anonymous. This reproduces
    v1.0 behaviour exactly, so scenarios written against raw indices keep
    working."""
    return FeatureView(
        names=tuple(f"dim_{i}" for i in range(n_dims)),
        extract_fn=lambda obs, info, env: np.asarray(obs, dtype=np.float32).reshape(-1),
        provenance="declared:identity",
        requires=("obs",),
        env_id=env_id,
    )


def table_view(env_id: str) -> Optional[FeatureView]:
    """Hand-registered names from constants.ENV_OBS_DIMS, applied to the
    leading dims of the observation. This is v1.0's named-dim mechanism,
    demoted to just one provider among several."""
    table = ENV_OBS_DIMS.get(env_id)
    if not table:
        return None
    n = max(table) + 1
    names = tuple(table.get(i, f"dim_{i}") for i in range(n))
    return FeatureView(
        names=names,
        extract_fn=lambda obs, info, env: np.asarray(
            obs, dtype=np.float32
        ).reshape(-1)[: len(names)],
        provenance="declared:table",
        requires=("obs",),
        env_id=env_id,
    )


def minigrid_view(env, probe=None) -> Optional[FeatureView]:
    """Introspect a MiniGrid env for agent pose and inventory.

    Reads the *environment*, not the observation, so this works whether
    the policy sees symbolic features, raw pixels, or anything else. This
    is the provider that makes the position dims unnecessary inside
    `MiniGridFlatPos`.
    """
    try:
        inner = env.unwrapped
    except AttributeError:
        return None
    if not (hasattr(inner, "agent_pos") and hasattr(inner, "agent_dir")):
        return None
    if not hasattr(inner, "grid"):
        return None

    def _extract(obs, info, env):
        u = env.unwrapped
        x, y = u.agent_pos
        carrying = getattr(u, "carrying", None)
        ctype = getattr(carrying, "type", None)
        return [
            float(x),
            float(y),
            float(u.agent_dir),
            1.0 if ctype == "key" else 0.0,
            1.0 if carrying is not None else 0.0,
        ]

    width = getattr(inner, "width", None)
    height = getattr(inner, "height", None)
    return FeatureView(
        names=("agent_x", "agent_y", "direction", "carrying_key", "carrying_any"),
        extract_fn=_extract,
        dtypes=(ORDINAL, ORDINAL, CATEGORICAL, BOOLEAN, BOOLEAN),
        ranges=(
            (0.0, float(width - 1)) if width else None,
            (0.0, float(height - 1)) if height else None,
            (0.0, 3.0),
            (0.0, 1.0),
            (0.0, 1.0),
        ),
        provenance="auto:minigrid",
        requires=("env",),
        env_id=getattr(getattr(env, "spec", None), "id", None),
    )


def dict_space_view(env, probe=None) -> Optional[FeatureView]:
    """Derive names from a gymnasium Dict observation space. Scalar keys
    keep their name; vector keys are expanded to `key[i]`."""
    try:
        import gymnasium as gym
    except ImportError:
        return None
    space = getattr(env, "observation_space", None)
    if not isinstance(space, gym.spaces.Dict):
        return None

    layout: list[tuple[str, int]] = []
    names: list[str] = []
    for key in sorted(space.spaces):
        sub = space.spaces[key]
        size = int(np.prod(getattr(sub, "shape", ()) or (1,)))
        layout.append((key, size))
        if size == 1:
            names.append(key)
        else:
            names.extend(f"{key}[{i}]" for i in range(size))

    def _extract(obs, info, env):
        out: list[float] = []
        for key, size in layout:
            v = np.asarray(obs[key], dtype=np.float32).reshape(-1)
            if v.shape[0] != size:
                raise ViewError(
                    f"dict key {key!r} gave {v.shape[0]} values, expected {size}"
                )
            out.extend(v.tolist())
        return out

    return FeatureView(
        names=tuple(names),
        extract_fn=_extract,
        provenance="auto:dict_space",
        requires=("obs",),
        env_id=getattr(getattr(env, "spec", None), "id", None),
    )


def mujoco_view(env, probe=None) -> Optional[FeatureView]:
    """Named generalised coordinates from a MuJoCo model.

    MuJoCo models carry joint names, so a domain whose observation is an
    anonymous Box can be named automatically by reading the model rather
    than the observation. Features are all of qpos followed by all of
    qvel, with velocities prefixed `v_`.

    This deliberately reads the simulator state, not the observation.
    Gymnasium's MuJoCo envs drop the root slide coordinate from the
    observation (so policies are translation invariant), which means the
    agent's absolute position is NOT expressible against `obs` at all.
    Reading the model recovers it, exactly as the MiniGrid view recovers
    grid position for a pixel-observing agent.
    """
    inner = getattr(env, "unwrapped", env)
    model = getattr(inner, "model", None)
    data = getattr(inner, "data", None)
    if model is None or data is None:
        return None
    try:
        import mujoco
    except ImportError:
        return None
    try:
        nq, nv, njnt = int(model.nq), int(model.nv), int(model.njnt)
    except Exception:
        return None

    # qpos/qvel slot counts per joint type: free, ball, slide, hinge.
    SLOTS = {0: (7, 6), 1: (4, 3), 2: (1, 1), 3: (1, 1)}
    # Suffixes for multi-slot joints so every feature gets a distinct name.
    FREE_Q = ("x", "y", "z", "qw", "qx", "qy", "qz")
    FREE_V = ("vx", "vy", "vz", "wx", "wy", "wz")
    BALL_Q = ("qw", "qx", "qy", "qz")
    BALL_V = ("wx", "wy", "wz")

    qnames: list[Optional[str]] = [None] * nq
    vnames: list[Optional[str]] = [None] * nv
    qlimits: list[Optional[tuple[float, float]]] = [None] * nq

    for j in range(njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"jnt{j}"
        jtype = int(model.jnt_type[j])
        nq_j, nv_j = SLOTS.get(jtype, (1, 1))
        qadr, vadr = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
        if jtype == 0:
            qsuf, vsuf = FREE_Q, FREE_V
        elif jtype == 1:
            qsuf, vsuf = BALL_Q, BALL_V
        else:
            qsuf, vsuf = (None,), (None,)
        limited = bool(model.jnt_limited[j]) if hasattr(model, "jnt_limited") else False
        lo, hi = (float(model.jnt_range[j][0]), float(model.jnt_range[j][1])) \
            if limited else (None, None)
        for k in range(nq_j):
            if qadr + k < nq:
                suf = qsuf[k] if k < len(qsuf) and qsuf[k] else None
                qnames[qadr + k] = f"{jname}_{suf}" if suf else jname
                if lo is not None and nq_j == 1:
                    qlimits[qadr + k] = (lo, hi)
        for k in range(nv_j):
            if vadr + k < nv:
                suf = vsuf[k] if k < len(vsuf) and vsuf[k] else None
                vnames[vadr + k] = f"v_{jname}_{suf}" if suf else f"v_{jname}"

    names = tuple(
        [n or f"qpos_{i}" for i, n in enumerate(qnames)]
        + [n or f"qvel_{i}" for i, n in enumerate(vnames)]
    )
    ranges = tuple(list(qlimits) + [None] * nv)

    def _extract(obs, info, env):
        u = getattr(env, "unwrapped", env)
        return np.concatenate(
            [np.asarray(u.data.qpos).reshape(-1),
             np.asarray(u.data.qvel).reshape(-1)]
        )

    return FeatureView(
        names=names,
        extract_fn=_extract,
        ranges=ranges,
        provenance="auto:mujoco",
        requires=("env",),
        env_id=getattr(getattr(env, "spec", None), "id", None),
    )


def info_view(env, probe=None) -> Optional[FeatureView]:
    """Derive features from scalar numeric entries in the env's `info` dict.

    The most domain-agnostic provider there is: it needs no knowledge of
    the environment at all, only that the author exposed something useful
    in `info`. Requires a probe sample (obs, info) from a reset or step,
    because info keys cannot be discovered from the spaces.
    """
    info = (probe or {}).get("info")
    if not isinstance(info, dict) or not info:
        return None
    keys = sorted(
        k for k, v in info.items()
        if isinstance(v, (int, float, bool)) and not isinstance(v, str)
    )
    if not keys:
        return None

    def _extract(obs, info, env):
        return [float(info[k]) for k in keys]

    return FeatureView(
        names=tuple(keys),
        extract_fn=_extract,
        provenance="auto:info_dict",
        requires=("info",),
        env_id=getattr(getattr(env, "spec", None), "id", None),
    )


# ----------------------------------------------------------------------
# Provider registry and cascade
# ----------------------------------------------------------------------

# Providers are tried in tier order. Tiers exist because not all
# semantics are equal: an env-specific adapter that knows what a feature
# MEANS beats a hand-written name table, which in turn beats scraping
# whatever scalars happen to sit in the info dict.
TIER_ENV = 0        # env-specific introspection (minigrid, mujoco, ...)
TIER_TABLE = 1      # hand-registered name table
TIER_GENERIC = 2    # structural / opportunistic (info dict)
TIER_FALLBACK = 3   # anonymous identity

_PROVIDERS: list[tuple[str, Callable, int]] = [
    ("minigrid", minigrid_view, TIER_ENV),
    ("mujoco", mujoco_view, TIER_ENV),
    ("dict_space", dict_space_view, TIER_ENV),
    ("info_dict", info_view, TIER_GENERIC),
]


def register_view_provider(
    name: str, provider: Callable, tier: int = TIER_ENV, first: bool = False
) -> None:
    """Add a provider to the cascade. New domains plug in here without
    touching the language core."""
    entry = (name, provider, int(tier))
    if first:
        _PROVIDERS.insert(0, entry)
    else:
        _PROVIDERS.append(entry)


def available_providers() -> list[str]:
    return [n for n, _, _ in _PROVIDERS] + ["table", "identity"]


def _try_tier(tier: int, env, probe) -> Optional[FeatureView]:
    if env is None:
        return None
    for _name, provider, t in _PROVIDERS:
        if t != tier:
            continue
        try:
            v = provider(env, probe)
        except Exception:
            v = None
        if v is not None:
            return v
    return None


def build_view(
    env=None,
    env_id: Optional[str] = None,
    n_dims: Optional[int] = None,
    prefer: Optional[str] = None,
    probe: Optional[dict] = None,
) -> FeatureView:
    """Resolve the best available view for a domain.

    Cascade: env-specific providers -> hand-registered name table ->
    generic providers -> anonymous identity. Always returns a view;
    `provenance` records which tier answered.

    `prefer` forces a specific provider and raises if it declines, which
    is what experiments should use so the view cannot silently degrade
    between runs.
    """
    if prefer:
        if prefer == "identity":
            if n_dims is None:
                raise ViewError("identity view needs n_dims")
            return identity_view(n_dims, env_id)
        if prefer == "table":
            v = table_view(env_id or "")
            if v is None:
                raise ViewError(f"no name table for env_id={env_id!r}")
            return v
        for name, provider, _tier in _PROVIDERS:
            if name == prefer:
                v = provider(env, probe)
                if v is None:
                    raise ViewError(
                        f"provider {prefer!r} declined for this environment"
                    )
                return v
        raise ViewError(
            f"unknown provider {prefer!r}; available: {available_providers()}"
        )

    v = _try_tier(TIER_ENV, env, probe)
    if v is not None:
        return v

    if env_id:
        v = table_view(env_id)
        if v is not None:
            return v

    v = _try_tier(TIER_GENERIC, env, probe)
    if v is not None:
        return v

    if n_dims is None:
        raise ViewError(
            "cannot build a view: no provider matched, no name table for "
            f"env_id={env_id!r}, and n_dims was not given for the identity "
            "fallback"
        )
    return identity_view(n_dims, env_id)


def view_from_cfg(cfg, env=None) -> FeatureView:
    """Build the feature view an experiment should use, from a Hydra config.

    `scenario_view` in the config selects the provider. When it is unset
    the cascade is deliberately run WITHOUT the environment, so it can
    only reach the name table or the identity fallback. Env-introspecting
    views are opt-in because they reinterpret what integer `dim:` indices
    mean: a scenario written against `observation[0]` is not the same
    predicate as one written against `qpos[0]`.
    """
    prefer = cfg.get("scenario_view", None)
    env_id = cfg.get("env_id", None)
    n_dims = cfg.get("observation_dim", None)
    if prefer in (None, "", "none"):
        return build_view(env_id=env_id, n_dims=n_dims)
    return build_view(env=env, env_id=env_id, n_dims=n_dims, prefer=prefer)


# ----------------------------------------------------------------------
# Portability reporting
# ----------------------------------------------------------------------


def provider_cost(provenance: str) -> Optional[int]:
    """Source lines of the adapter that produced a view.

    This is the portability metric: the cost of teaching the language a
    new domain, measured rather than asserted. `declared:identity` costs
    nothing (no semantics); `declared:table` is counted as the size of
    that environment's name-table entry, filled in by the caller.
    """
    import inspect

    fn = {
        "auto:minigrid": minigrid_view,
        "auto:mujoco": mujoco_view,
        "auto:dict_space": dict_space_view,
        "auto:info_dict": info_view,
        "declared:identity": None,
    }.get(provenance)
    if fn is None:
        return None
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return None
    return sum(
        1 for ln in src.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    )


def describe_domain(
    env_id: str, make_env_fn: Optional[Callable[[str], Any]] = None,
    prefer: Optional[str] = None,
) -> dict:
    """Resolve a view for one environment and report how it was obtained.

    Returns a record with provenance, feature count, names and adapter
    cost. Failures are captured rather than raised so a sweep over many
    domains produces a complete table.
    """
    record: dict = {
        "env_id": env_id, "provenance": None, "n_features": None,
        "names": [], "requires": [], "adapter_loc": None, "error": None,
    }
    env = None
    try:
        if make_env_fn is not None:
            env = make_env_fn(env_id)
        else:
            import gymnasium as gym

            if env_id.startswith("MiniGrid-"):
                import minigrid  # noqa: F401  (registers MiniGrid envs)
            env = gym.make(env_id)
        obs, info = env.reset(seed=0)
        probe = {"obs": obs, "info": info}
        n_dims = None
        shape = getattr(getattr(env, "observation_space", None), "shape", None)
        if shape and len(shape) == 1:
            n_dims = int(shape[0])
        view = build_view(env=env, env_id=env_id, n_dims=n_dims,
                          prefer=prefer, probe=probe)
        record.update(
            provenance=view.provenance,
            n_features=len(view),
            names=list(view.names),
            requires=list(view.requires),
            adapter_loc=(
                provider_cost(view.provenance)
                if view.provenance != "declared:table"
                else len(ENV_OBS_DIMS.get(env_id, {}))
            ),
        )
    except Exception as exc:  # noqa: BLE001 - report, don't abort the sweep
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return record


def domain_report(
    env_ids: Sequence[str],
    make_env_fn: Optional[Callable[[str], Any]] = None,
) -> list[dict]:
    """describe_domain over many environments."""
    return [describe_domain(e, make_env_fn=make_env_fn) for e in env_ids]
