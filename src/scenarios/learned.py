"""Learned feature views: semantics fitted from data rather than declared.

The view cascade has three tiers that all rest on someone already knowing
what the features mean — an env-specific adapter, a hand-written name
table, or scraped `info` keys. A pixel observation has none of that. No
index of a 608x608 RGB frame means `agent_x`, so the whole language is
inapplicable to the domain by default.

A probe closes that: fit a small model from raw observations to the
semantic features you want, and expose its predictions as a view. The
scenario grammar does not change at all, because a view is a view.

Fidelity is the part that matters
---------------------------------
A fitted view can be wrong, and a wrong view is worse than a missing one:
it produces confident numbers that mean nothing, and every scenario
written against it inherits the error silently. So a probe is fitted on
one split and **scored on another**, per feature, and the view carries
that score. `min_fidelity` then refuses a view that cannot be trusted
rather than returning it with a caveat nobody reads.

This is what makes "learned" an honest tier rather than an escape hatch.
Some things genuinely are not recoverable — a partially-observable
egocentric view does not determine absolute position, and no amount of
fitting will change that. The correct behaviour there is to fail loudly,
which is what the gate does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .views import CONTINUOUS, ORDINAL, FeatureView

# Features whose predictions are rounded to integers and scored by exact
# match; everything else is scored by R^2 on the held-out split.
DISCRETE_DTYPES = (ORDINAL,)


class ProbeError(ValueError):
    """Raised when a probe cannot be fitted or does not meet its gate."""


@dataclass
class ProbeReport:
    """How well the fitted view reproduces its targets, out of sample."""

    names: Tuple[str, ...]
    scores: Tuple[float, ...]
    n_train: int
    n_test: int
    input_dim: int
    ridge_alpha: float

    @property
    def fidelity(self) -> float:
        """The weakest feature. A view is only as trustworthy as its worst
        component, because a scenario may depend on any one of them."""
        return float(min(self.scores)) if self.scores else 0.0

    @property
    def per_feature(self) -> Dict[str, float]:
        return dict(zip(self.names, self.scores))

    def render(self) -> str:
        lines = [f"probe over {self.input_dim} raw inputs",
                 f"  train/test : {self.n_train}/{self.n_test}",
                 f"  ridge alpha: {self.ridge_alpha:g}",
                 "  held-out score per feature:"]
        for n, sc in zip(self.names, self.scores):
            flag = "" if sc >= 0.9 else "   <-- weak"
            lines.append(f"    {n:<20s} {sc:6.3f}{flag}")
        lines.append(f"  fidelity (min): {self.fidelity:.3f}")
        return "\n".join(lines)


def _ridge(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge regression with an intercept column.

    Written out rather than pulled from sklearn because it is four lines
    and keeps the language core free of another dependency.
    """
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    A = Xb.T @ Xb
    # Regularise the weights but not the intercept.
    reg = alpha * np.eye(d + 1)
    reg[-1, -1] = 0.0
    return np.linalg.solve(A + reg, Xb.T @ Y)


def _score(y_true: np.ndarray, y_pred: np.ndarray, discrete: bool) -> float:
    if discrete:
        return float((np.rint(y_pred) == np.rint(y_true)).mean())
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot <= 1e-12:                      # constant target
        return 1.0 if ss_res <= 1e-12 else 0.0
    return 1.0 - ss_res / ss_tot


def fit_probe_view(
    raw: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    names: Sequence[str],
    *,
    dtypes: Optional[Sequence[str]] = None,
    ranges: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
    test_frac: float = 0.3,
    ridge_alpha: float = 1.0,
    min_fidelity: Optional[float] = None,
    env_id: Optional[str] = None,
    seed: int = 0,
    extract_raw: Optional[Callable[[Any, Any, Any], Sequence[float]]] = None,
) -> Tuple[FeatureView, ProbeReport]:
    """Fit a probe from raw observations to named semantic features.

    `extract_raw` maps (obs, info, env) to the same raw vector the probe
    was fitted on; without it the observation is used directly, which is
    the common case for a flat pixel or latent vector.

    Raises `ProbeError` when `min_fidelity` is set and the held-out score
    of the weakest feature falls below it.
    """
    X = np.asarray(raw, dtype=np.float64)
    Y = np.asarray(targets, dtype=np.float64)
    if X.ndim != 2:
        raise ProbeError(f"raw inputs must be 2-D; got shape {X.shape}")
    if Y.ndim == 1:
        Y = Y[:, None]
    if len(X) != len(Y):
        raise ProbeError(f"{len(X)} inputs but {len(Y)} target rows")
    if Y.shape[1] != len(names):
        raise ProbeError(
            f"{Y.shape[1]} target columns but {len(names)} names")
    if len(X) < 4:
        raise ProbeError("need at least 4 samples to fit and score a probe")

    dtypes = tuple(dtypes or (CONTINUOUS,) * len(names))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    n_test = max(1, round(test_frac * len(X)))
    test_idx, train_idx = order[:n_test], order[n_test:]
    if len(train_idx) < 2:
        raise ProbeError("too few training samples after the split")

    # Standardise inputs so one ridge penalty is sensible across features.
    mu = X[train_idx].mean(axis=0)
    sd = X[train_idx].std(axis=0)
    sd[sd < 1e-8] = 1.0

    W = _ridge((X[train_idx] - mu) / sd, Y[train_idx], ridge_alpha)
    pred_test = np.hstack([(X[test_idx] - mu) / sd,
                           np.ones((len(test_idx), 1))]) @ W

    scores = tuple(
        _score(Y[test_idx][:, j], pred_test[:, j], dtypes[j] in DISCRETE_DTYPES)
        for j in range(len(names))
    )
    report = ProbeReport(
        names=tuple(names), scores=scores, n_train=len(train_idx),
        n_test=len(test_idx), input_dim=X.shape[1], ridge_alpha=ridge_alpha,
    )

    if min_fidelity is not None and report.fidelity < min_fidelity:
        weak = [f"{n}={s:.3f}" for n, s in report.per_feature.items()
                if s < min_fidelity]
        raise ProbeError(
            f"probe fidelity {report.fidelity:.3f} is below the required "
            f"{min_fidelity:.3f}; weak features: {', '.join(weak)}. "
            "The raw observation may not determine these features at all — "
            "check that before lowering the gate."
        )

    discrete = tuple(dt in DISCRETE_DTYPES for dt in dtypes)

    def _extract(obs, info, env):
        vec = (np.asarray(extract_raw(obs, info, env), dtype=np.float64)
               if extract_raw is not None
               else np.asarray(obs, dtype=np.float64).reshape(-1))
        z = (vec - mu) / sd
        out = np.concatenate([z, [1.0]]) @ W
        return [float(np.rint(v)) if d else float(v)
                for v, d in zip(out, discrete)]

    view = FeatureView(
        names=tuple(names),
        extract_fn=_extract,
        dtypes=dtypes,
        ranges=tuple(ranges) if ranges else (),
        provenance="learned:probe",
        requires=("obs",) if extract_raw is None else ("obs", "info", "env"),
        fidelity=report.fidelity,
        env_id=env_id,
    )
    return view, report
