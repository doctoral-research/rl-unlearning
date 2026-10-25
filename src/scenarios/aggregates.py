"""Episode aggregates: named reductions over a whole trajectory.

v1.1 could compare a feature at one step, and could compare exactly two
episode-level quantities: `return` and `length`. Everything else about an
episode as a whole was unsayable, which is what made four of the six
recorded corpus gaps gaps:

    no_backtracking   needs "how many distinct cells did it visit"
    no_pacing         needs "how many steps per distinct cell"
    no_front_flip     needs cumulative rotation, not instantaneous pitch
    no_freeloading    needs each agent's contribution over the episode

An `aggregates:` block generalises the two hardcoded quantities into
arbitrary named reductions:

    aggregates:
      cells_visited:  {op: count_distinct, of: [agent_x, agent_y]}
      total_rotation: {op: sum, of: v_rooty}
      steps_in_corner: {op: count, where: {conditions: [...]}}

The resulting names are trajectory-level scalars, addressable from a
condition exactly like `return` and `length` are, and usable inside
`derived:` expressions so relational comparisons between aggregates work
without new syntax.

Aggregates describe the episode, so they are only meaningful under
`match_mode: trajectory` — the validator enforces that, as it already
does for `target: return`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


class AggregateError(ValueError):
    """Raised for a malformed aggregate specification."""


def _finite(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    return arr[np.isfinite(arr)] if arr.size else arr


def _safe(fn: Callable[[np.ndarray], float], empty: float = 0.0):
    def _inner(arr: np.ndarray) -> float:
        arr = _finite(arr)
        return float(fn(arr)) if arr.size else empty
    return _inner


# Reductions over a single feature's per-step values.
_REDUCERS: Dict[str, Callable[[np.ndarray], float]] = {
    "sum": _safe(np.sum),
    "mean": _safe(np.mean),
    "min": _safe(np.min, float("inf")),
    "max": _safe(np.max, float("-inf")),
    "std": _safe(np.std),
    "range": _safe(lambda a: float(np.max(a) - np.min(a))),
    "first": _safe(lambda a: float(a[0])),
    "last": _safe(lambda a: float(a[-1])),
    "abs_sum": _safe(lambda a: float(np.abs(a).sum())),
    "n_distinct": _safe(lambda a: float(len(np.unique(a)))),
}

# Reductions that do not read a feature at all.
_COUNTERS = ("count", "count_distinct")

VALID_OPS = tuple(sorted(_REDUCERS)) + _COUNTERS


@dataclass(frozen=True)
class Aggregate:
    """One named episode-level reduction."""

    name: str
    op: str
    dims: tuple[int, ...] = ()
    dim_names: tuple[str, ...] = ()
    where: Optional[Dict[str, Any]] = None

    @property
    def needs_predicate(self) -> bool:
        return self.where is not None


def compile_aggregate(
    name: str, spec: Any, resolve: Callable[[str], int],
    normalise_group: Callable[[Any], Dict[str, Any]],
) -> Aggregate:
    """Compile one `aggregates:` entry.

    `resolve` maps a feature name to its index (so a typo fails at load
    time) and `normalise_group` canonicalises an optional `where:`
    predicate using the scenario's own group machinery, so `where` gets
    the full grammar rather than a second, weaker one.
    """
    if not isinstance(spec, dict):
        raise AggregateError(
            f"aggregate {name!r} must be a mapping; got {type(spec).__name__}")
    op = spec.get("op")
    if op not in VALID_OPS:
        raise AggregateError(
            f"aggregate {name!r}: op must be one of {list(VALID_OPS)}; "
            f"got {op!r}")

    raw_of = spec.get("of")
    if raw_of is None:
        of: List[Any] = []
    elif isinstance(raw_of, (list, tuple)):
        of = list(raw_of)
    else:
        of = [raw_of]

    if op == "count":
        if of:
            raise AggregateError(
                f"aggregate {name!r}: 'count' counts steps and takes no 'of'")
    elif op == "count_distinct":
        if not of:
            raise AggregateError(
                f"aggregate {name!r}: 'count_distinct' needs 'of' (the tuple "
                "of features whose distinct combinations are counted)")
    elif len(of) != 1:
        raise AggregateError(
            f"aggregate {name!r}: op {op!r} reduces a single feature, so "
            f"'of' must name exactly one; got {len(of)}")

    dims: List[int] = []
    names: List[str] = []
    for item in of:
        if isinstance(item, int):
            dims.append(item)
            names.append(f"dim_{item}")
        elif isinstance(item, str):
            dims.append(resolve(item))
            names.append(item)
        else:
            raise AggregateError(
                f"aggregate {name!r}: 'of' entries must be names or indices; "
                f"got {item!r}")

    where = spec.get("where")
    if where is not None:
        if op not in ("count", "count_distinct"):
            raise AggregateError(
                f"aggregate {name!r}: 'where' filters which steps are "
                f"counted, so it only applies to {list(_COUNTERS)}")
        where = normalise_group(where)

    return Aggregate(name=name, op=op, dims=tuple(dims),
                     dim_names=tuple(names), where=where)


def compile_aggregate_block(
    block: Any, resolve: Callable[[str], int],
    normalise_group: Callable[[Any], Dict[str, Any]],
) -> Dict[str, Aggregate]:
    if block in (None, {}):
        return {}
    if not isinstance(block, dict):
        raise AggregateError(
            f"'aggregates' must be a mapping of name -> spec; "
            f"got {type(block).__name__}")
    return {
        name: compile_aggregate(name, spec, resolve, normalise_group)
        for name, spec in block.items()
    }


def evaluate_aggregates(
    aggregates: Dict[str, Aggregate],
    features: Sequence[Sequence[float]],
    step_match: Optional[Callable[[Dict[str, Any], int], bool]] = None,
) -> Dict[str, float]:
    """Evaluate every aggregate over one trajectory.

    `step_match(group, t)` decides whether step `t` satisfies a `where:`
    predicate; it is supplied by the scenario so `where` reuses the same
    matcher as everything else.
    """
    out: Dict[str, float] = {}
    n = len(features)
    for name, agg in aggregates.items():
        if agg.op == "count":
            if agg.where is None or step_match is None:
                out[name] = float(n)
            else:
                out[name] = float(sum(
                    1 for t in range(n) if step_match(agg.where, t)))
            continue

        if agg.op == "count_distinct":
            seen = set()
            for t in range(n):
                if agg.where is not None and step_match is not None:
                    if not step_match(agg.where, t):
                        continue
                seen.add(tuple(float(features[t][d]) for d in agg.dims))
            out[name] = float(len(seen))
            continue

        d = agg.dims[0]
        vals = np.asarray([float(features[t][d]) for t in range(n)],
                          dtype=np.float64)
        out[name] = _REDUCERS[agg.op](vals)
    return out
