"""Synthesising a scenario from labelled demonstrations.

Writing a specification by hand assumes you can already say what the
behaviour is. Often you cannot: what exists is a pile of episodes someone
flagged — a regulator's complaint, a batch of incident reports, a set of
rollouts an operator marked as "this, stop doing this" — and the rule
behind them has to be recovered.

This module recovers it. Given trajectories labelled positive and
negative, it searches the grammar for the smallest predicate that
separates them and emits an ordinary scenario: validatable, renderable to
prose and LaTeX, and usable directly as a forget specification.

Why the search is tractable
---------------------------
The grammar is small and the target form is restricted. A scenario's
top-level `groups` are OR'd and a group's `conditions` are AND'd, so a
scenario *is* a DNF over threshold literals. Learning small DNFs from
labelled data is a solved shape — sequential covering — and the
restriction is a feature rather than a limitation: the output has to stay
readable, because a specification nobody can check is not a specification.

The algorithm is:

    1. propose candidate literals (feature, operator, threshold) from
       values actually observed in the data;
    2. beam-search a conjunction that is precise on the positives it
       covers;
    3. keep it as one group, drop the positives it covers, and repeat;
    4. stop when recall is adequate, no rule improves things, or the
       size budget is spent.

Preferring precision per rule and stopping early is deliberate. A longer
rule set can always fit better, and a specification that fits perfectly
but cannot be read has bought nothing — `MiningReport` carries both the
fit and the size so that trade is visible rather than implicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .scenario import ForgetScenario, _feature_seq
from .views import BOOLEAN, CATEGORICAL, FeatureView

# Literal = (dim index, operator, threshold)
Literal = Tuple[int, str, float]

# Operators the miner proposes. Ordering operators for numeric features,
# equality for categorical and boolean ones -- proposing `>` on a
# categorical would produce exactly the mistake the typed validator
# exists to reject.
ORDERING = (">", "<=")
EQUALITY = ("==", "!=")


class MiningError(ValueError):
    """Raised when there is nothing to mine from."""


@dataclass
class MiningReport:
    """Fit and size of a mined specification."""

    n_pos: int
    n_neg: int
    tp: int = 0
    fp: int = 0
    fn: int = 0
    rules: List[List[Literal]] = field(default_factory=list)
    candidates_considered: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def n_rules(self) -> int:
        return len(self.rules)

    @property
    def n_literals(self) -> int:
        return sum(len(r) for r in self.rules)

    def render(self) -> str:
        return "\n".join([
            f"positives   {self.n_pos}",
            f"negatives   {self.n_neg}",
            f"tp/fp/fn    {self.tp}/{self.fp}/{self.fn}",
            f"precision   {self.precision:.4f}",
            f"recall      {self.recall:.4f}",
            f"f1          {self.f1:.4f}",
            f"size        {self.n_rules} rules, {self.n_literals} literals",
            f"considered  {self.candidates_considered} candidate literals",
        ])


def _thresholds(values: np.ndarray, max_per_dim: int) -> List[float]:
    """Split points between observed values.

    Midpoints between consecutive distinct values are the only thresholds
    that can separate the data differently, so there is no reason to
    consider any others.
    """
    uniq = np.unique(values[np.isfinite(values)])
    if uniq.size < 2:
        return []
    mids = (uniq[:-1] + uniq[1:]) / 2.0
    if mids.size <= max_per_dim:
        return [float(m) for m in mids]
    idx = np.linspace(0, mids.size - 1, max_per_dim).round().astype(int)
    return [float(mids[i]) for i in np.unique(idx)]


def candidate_literals(
    X: np.ndarray, view: Optional[FeatureView] = None,
    max_per_dim: int = 12,
) -> List[Literal]:
    """Propose literals from the values present in the data."""
    out: List[Literal] = []
    for d in range(X.shape[1]):
        col = X[:, d]
        dtype = view.dtype_of(d) if view is not None else None
        if dtype in (CATEGORICAL, BOOLEAN):
            for v in np.unique(col[np.isfinite(col)]):
                for op in EQUALITY:
                    out.append((d, op, float(v)))
        else:
            for thr in _thresholds(col, max_per_dim):
                for op in ORDERING:
                    out.append((d, op, thr))
    return out


def _apply(X: np.ndarray, lit: Literal) -> np.ndarray:
    d, op, v = lit
    col = X[:, d]
    if op == ">":
        return col > v
    if op == "<=":
        return col <= v
    if op == "==":
        return col == v
    if op == "!=":
        return col != v
    raise MiningError(f"unknown operator {op!r}")


def _rule_mask(X: np.ndarray, rule: Sequence[Literal]) -> np.ndarray:
    mask = np.ones(len(X), dtype=bool)
    for lit in rule:
        mask &= _apply(X, lit)
    return mask


def _simplify_rule(
    X: np.ndarray, y: np.ndarray, rule: List[Literal],
    literals: Sequence[Literal],
) -> List[Literal]:
    """Generalise a grown rule without letting negatives in.

    Beam search commits early: a tighter literal can win at depth 1 and
    the rule that survives covers less than it could, which then needs a
    second rule to pick up the remainder. Two rules where one would do is
    exactly what this module is supposed to avoid, so after growing, the
    rule is relaxed as far as it will go.

    Both passes are strictly safe -- a change is kept only when it adds no
    false positive -- so simplification never costs precision.
    """
    def fps(r: Sequence[Literal]) -> int:
        return int((_rule_mask(X, r) & ~y).sum())

    def cov(r: Sequence[Literal]) -> int:
        return int(_rule_mask(X, r).sum())

    rule = list(rule)
    base_fp = fps(rule)

    # 1. Drop literals that are not carrying their weight.
    changed = True
    while changed and len(rule) > 1:
        changed = False
        for i in range(len(rule)):
            trial = rule[:i] + rule[i + 1:]
            if fps(trial) <= base_fp and cov(trial) > cov(rule):
                rule = trial
                changed = True
                break

    # 2. Loosen each threshold as far as it will go.
    for i, (d, op, _v) in enumerate(list(rule)):
        alts = sorted(
            (lit for lit in literals if lit[0] == d and lit[1] == op),
            key=lambda lit: lit[2], reverse=(op in (">",)))
        for alt in alts:
            trial = rule[:i] + [alt] + rule[i + 1:]
            if fps(trial) <= base_fp and cov(trial) > cov(rule):
                rule = trial
    return rule


def _grow_rule(
    X: np.ndarray, y: np.ndarray, literals: Sequence[Literal],
    max_literals: int, beam_width: int, min_coverage: int,
) -> Optional[List[Literal]]:
    """Beam-search one conjunction, maximising precision on what it covers.

    A rule that covers almost nothing can be trivially precise, so
    `min_coverage` keeps the search honest.
    """
    def score(mask: np.ndarray) -> float:
        cov = int(mask.sum())
        if cov < min_coverage:
            return -1.0
        tp = int((mask & y).sum())
        # Laplace-corrected precision: prefers a rule that is precise AND
        # covers enough to be worth stating.
        return (tp + 1.0) / (cov + 2.0)

    beam: List[Tuple[float, List[Literal], np.ndarray]] = [
        (score(np.ones(len(X), dtype=bool)), [], np.ones(len(X), dtype=bool))
    ]
    best: Optional[Tuple[float, List[Literal]]] = None

    for _depth in range(max_literals):
        nxt: List[Tuple[float, List[Literal], np.ndarray]] = []
        for _s, rule, mask in beam:
            used = {(d, op) for d, op, _ in rule}
            for lit in literals:
                if (lit[0], lit[1]) in used:
                    continue
                m = mask & _apply(X, lit)
                sc = score(m)
                if sc < 0:
                    continue
                nxt.append((sc, rule + [lit], m))
        if not nxt:
            break
        nxt.sort(key=lambda t: (-t[0], len(t[1])))
        beam = nxt[:beam_width]
        top_score, top_rule, _ = beam[0]
        if best is None or top_score > best[0] + 1e-12:
            best = (top_score, list(top_rule))

    return best[1] if best else None


def mine_rules(
    X: np.ndarray, y: np.ndarray, view: Optional[FeatureView] = None,
    *, max_rules: int = 3, max_literals: int = 3, beam_width: int = 8,
    min_coverage: int = 1, target_recall: float = 0.95,
    max_per_dim: int = 12,
) -> Tuple[List[List[Literal]], MiningReport]:
    """Sequential covering: grow a rule, remove what it covers, repeat."""
    if len(X) == 0:
        raise MiningError("no samples to mine from")
    if not y.any():
        raise MiningError("no positive samples: nothing to characterise")
    if y.all():
        raise MiningError(
            "every sample is positive: no contrast to learn a rule from")

    literals = candidate_literals(X, view, max_per_dim)
    report = MiningReport(n_pos=int(y.sum()), n_neg=int((~y).sum()),
                          candidates_considered=len(literals))
    if not literals:
        return [], report

    remaining = y.copy()
    rules: List[List[Literal]] = []
    for _ in range(max_rules):
        if not remaining.any():
            break
        rule = _grow_rule(X, remaining, literals, max_literals, beam_width,
                          min_coverage)
        if rule is None:
            break
        # Generalise against the FULL positive set, not the remaining one:
        # a rule may safely re-cover positives an earlier rule already took.
        rule = _simplify_rule(X, y, rule, literals)
        covered = _rule_mask(X, rule)
        if not (covered & remaining).any():
            break                       # no progress; stop rather than loop
        rules.append(rule)
        remaining = remaining & ~covered
        recall = 1.0 - remaining.sum() / max(y.sum(), 1)
        if recall >= target_recall:
            break

    if rules:
        pred = np.zeros(len(X), dtype=bool)
        for r in rules:
            pred |= _rule_mask(X, r)
        report.tp = int((pred & y).sum())
        report.fp = int((pred & ~y).sum())
        report.fn = int((~pred & y).sum())
    else:
        report.fn = int(y.sum())
    report.rules = rules
    return rules, report


def rules_to_config(
    rules: Sequence[Sequence[Literal]],
    view: Optional[FeatureView] = None,
    *, name: str = "mined", description: Optional[str] = None,
    match_mode: str = "any_step", applies_to: str = "matching_steps",
    match_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Turn mined rules into a scenario config.

    Groups are OR'd and conditions AND'd, which is exactly the DNF the
    miner produces, so no translation is lost.
    """
    groups = []
    for i, rule in enumerate(rules):
        conditions = []
        for d, op, v in rule:
            cond: Dict[str, Any] = {
                "dim": view.name_of(d) if view is not None else d,
                "op": op,
                "value": round(float(v), 6),
            }
            if view is not None:
                cond["dim_name"] = view.name_of(d)
            conditions.append(cond)
        groups.append({"name": f"rule_{i}", "conditions": conditions})

    doc: Dict[str, Any] = {
        "name": name,
        "description": description or (
            "Mined from labelled demonstrations: the smallest predicate in "
            "the grammar separating the flagged episodes from the rest."),
        "match_mode": match_mode,
        "applies_to": applies_to,
        "groups": groups,
    }
    if match_threshold is not None:
        doc["match_threshold"] = match_threshold
    if view is not None and view.env_id:
        doc["env_id"] = view.env_id
    return doc


def _stack(trajectories: Sequence[Dict[str, Any]]) -> np.ndarray:
    rows = []
    for t in trajectories:
        for f in _feature_seq(t):
            rows.append(np.asarray(f, dtype=np.float64))
    if not rows:
        raise MiningError("trajectories contain no steps")
    return np.vstack(rows)


def mine_scenario(
    positive: Sequence[Dict[str, Any]],
    negative: Sequence[Dict[str, Any]],
    view: Optional[FeatureView] = None,
    **kwargs: Any,
) -> Tuple[ForgetScenario, MiningReport]:
    """Mine a scenario from flagged and unflagged trajectories.

    Steps of positive trajectories are the positive class. That is the
    weaker labelling — only the episode is flagged, not the moment — and
    it is what an incident report actually gives you.
    """
    name = kwargs.pop("name", "mined")
    description = kwargs.pop("description", None)
    match_mode = kwargs.pop("match_mode", "any_step")
    applies_to = kwargs.pop("applies_to", "matching_steps")
    match_threshold = kwargs.pop("match_threshold", None)

    Xp = _stack(positive) if positive else np.empty((0, 0))
    Xn = _stack(negative) if negative else np.empty((0, 0))
    if Xp.size == 0 or Xn.size == 0:
        raise MiningError("need both positive and negative trajectories")
    if Xp.shape[1] != Xn.shape[1]:
        raise MiningError(
            f"feature width differs: positives {Xp.shape[1]}, "
            f"negatives {Xn.shape[1]}")

    X = np.vstack([Xp, Xn])
    y = np.concatenate([np.ones(len(Xp), bool), np.zeros(len(Xn), bool)])

    rules, report = mine_rules(X, y, view, **kwargs)
    doc = rules_to_config(rules, view, name=name, description=description,
                          match_mode=match_mode, applies_to=applies_to,
                          match_threshold=match_threshold)
    return ForgetScenario(doc, view=view), report
