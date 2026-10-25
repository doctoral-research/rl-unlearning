"""Temporal formulas: a nestable fragment of LTL over state predicates.

v1.3's temporal layer was one operator deep. The operands of
`eventually`, `always`, `until` and friends were state predicates, never
other formulas, so anything of the shape "eventually P and then never Q"
was unsayable — which is why `doorkey.no_key_hoarding` sat in the corpus
as inexpressible.

The fix is to evaluate formulas **at a position** rather than over the
whole trajectory at once. Once `holds(formula, t)` is the primitive,
nesting follows for free and the operators get their standard meanings:

    state(g)          g matches the step at t
    not F             F does not hold at t
    and / or [F...]   all / any hold at t
    next F            F holds at t+1
    previous F        F holds at t-1, and t is not the first step
    eventually F      F holds at some t' >= t
    always F          F holds at every t' >= t
    until(H, R)       R holds at some t' >= t, and H holds throughout [t, t')

A trajectory matches when the formula holds at position 0.

Two operators keep an existential flavour so v1.0 files keep their
meaning. `consecutive` and `then_within` are anchored as "there exists a
later position where ...", which is what they meant when the only
evaluation point was the start of the episode:

    consecutive(F, k)        some t' >= t starts k steps of F
    then_within(F, G, w)     some t' >= t has F, and G follows within w

Leaves are ordinary group expressions, so the whole state-level grammar
(regions, boolean combinators, `$ref`, derived features) is available
inside a temporal formula without restatement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

# Keys that mark a node as a temporal operator rather than a state
# predicate. A dict carrying exactly one of these is a formula; anything
# else is a group expression and becomes a `state` leaf.
UNARY_OPS = ("not", "next", "previous", "eventually", "always")
NARY_OPS = ("and", "or")
STRUCTURED_OPS = ("until", "consecutive", "then_within")
TEMPORAL_KEYS = frozenset(UNARY_OPS + NARY_OPS + STRUCTURED_OPS)

# Legacy flat form: {operator: <name>, ...}
LEGACY_OPERATORS = frozenset(
    {"eventually", "always", "until", "consecutive", "then_within"}
)


class TemporalError(ValueError):
    """Raised for a malformed temporal formula."""


@dataclass(frozen=True)
class Formula:
    """A node of the temporal AST.

    `kind` is the operator name (or "state" for a leaf). `children` holds
    sub-formulas; `group` holds the normalised group expression of a leaf;
    `k` / `within` carry the integer parameters.
    """

    kind: str
    children: tuple = ()
    group: Any = None
    k: int = 0
    within: int = 0

    def depth(self) -> int:
        """Nesting depth. 1 for a leaf; used to report that the fragment
        is no longer one operator deep."""
        return 1 + max((c.depth() for c in self.children), default=0)

    def describe(self) -> str:
        if self.kind == "state":
            return "state"
        if self.kind in ("consecutive",):
            return f"consecutive[{self.k}]({self.children[0].describe()})"
        if self.kind == "then_within":
            return (f"then_within[{self.within}]("
                    f"{self.children[0].describe()}, "
                    f"{self.children[1].describe()})")
        inner = ", ".join(c.describe() for c in self.children)
        return f"{self.kind}({inner})"


def _as_formula(node: Any, normalise_group: Callable[[Any], Any]) -> Formula:
    """Parse one node, which is either a temporal operator or a leaf."""
    if not isinstance(node, dict):
        raise TemporalError(
            f"temporal formula node must be a mapping; got {type(node).__name__}")

    # Legacy flat form kept working: {operator: eventually, group: {...}}
    if "operator" in node:
        return _from_legacy(node, normalise_group)

    keys = [k for k in node if k in TEMPORAL_KEYS]
    if len(keys) > 1:
        raise TemporalError(
            f"temporal node has several operators {keys}; wrap them in "
            "'and'/'or' to combine")
    if not keys:
        # Not an operator -> a state predicate over the step at position t.
        return Formula(kind="state", group=normalise_group(node))

    op = keys[0]
    body = node[op]

    if op in UNARY_OPS:
        return Formula(kind=op, children=(_as_formula(body, normalise_group),))

    if op in NARY_OPS:
        if not isinstance(body, (list, tuple)) or len(body) < 2:
            raise TemporalError(f"'{op}' needs a list of at least two formulas")
        return Formula(
            kind=op,
            children=tuple(_as_formula(b, normalise_group) for b in body))

    if op == "until":
        if not isinstance(body, dict) or not {"hold", "release"} <= set(body):
            raise TemporalError("'until' needs 'hold' and 'release'")
        return Formula(kind="until", children=(
            _as_formula(body["hold"], normalise_group),
            _as_formula(body["release"], normalise_group)))

    if op == "consecutive":
        if not isinstance(body, dict) or "k" not in body or "of" not in body:
            raise TemporalError("'consecutive' needs 'k' and 'of'")
        k = int(body["k"])
        if k < 1:
            raise TemporalError("'consecutive' needs k >= 1")
        return Formula(kind="consecutive", k=k,
                       children=(_as_formula(body["of"], normalise_group),))

    if op == "then_within":
        need = {"first", "second", "within"}
        if not isinstance(body, dict) or not need <= set(body):
            raise TemporalError("'then_within' needs 'first', 'second', 'within'")
        w = int(body["within"])
        if w < 1:
            raise TemporalError("'then_within' needs within >= 1")
        return Formula(kind="then_within", within=w, children=(
            _as_formula(body["first"], normalise_group),
            _as_formula(body["second"], normalise_group)))

    raise TemporalError(f"unhandled temporal operator {op!r}")


def _from_legacy(node: Dict[str, Any],
                 normalise_group: Callable[[Any], Any]) -> Formula:
    """Parse the v1.0 flat form, whose operands are always groups."""
    op = node.get("operator")
    if op not in LEGACY_OPERATORS:
        raise TemporalError(
            f"Unknown temporal operator: {op!r}; "
            f"valid: {sorted(LEGACY_OPERATORS)}")
    leaf = lambda key: Formula(kind="state", group=normalise_group(node[key]))  # noqa: E731
    if op in ("eventually", "always"):
        if "group" not in node:
            raise TemporalError(f"'{op}' needs 'group'")
        return Formula(kind=op, children=(leaf("group"),))
    if op == "until":
        if not {"hold", "release"} <= set(node):
            raise TemporalError("'until' needs 'hold' and 'release'")
        return Formula(kind="until",
                       children=(leaf("hold"), leaf("release")))
    if op == "consecutive":
        if "k" not in node or "group" not in node:
            raise TemporalError("'consecutive' needs 'k' and 'group'")
        return Formula(kind="consecutive", k=int(node["k"]),
                       children=(leaf("group"),))
    # then_within
    if not {"first", "second", "within"} <= set(node):
        raise TemporalError("'then_within' needs 'first', 'second', 'within'")
    return Formula(kind="then_within", within=int(node["within"]),
                   children=(leaf("first"), leaf("second")))


def parse_formula(node: Any,
                  normalise_group: Callable[[Any], Any]) -> Formula:
    """Parse a `temporal:` block into a formula tree."""
    return _as_formula(node, normalise_group)


def holds(formula: Formula, t: int, n: int,
          step_match: Callable[[Any, int], bool]) -> bool:
    """Whether `formula` holds at position `t` of an n-step trajectory."""
    if t < 0 or t >= n:
        return False
    k = formula.kind

    if k == "state":
        return bool(step_match(formula.group, t))
    if k == "not":
        return not holds(formula.children[0], t, n, step_match)
    if k == "and":
        return all(holds(c, t, n, step_match) for c in formula.children)
    if k == "or":
        return any(holds(c, t, n, step_match) for c in formula.children)
    if k == "next":
        return holds(formula.children[0], t + 1, n, step_match)
    if k == "previous":
        # The past-time mirror of `next`. Without it a rule anchored on a
        # transition INTO a region can only be written by anchoring one
        # step early, which labels the wrong transition: the step before
        # entry rather than the entry itself.
        return holds(formula.children[0], t - 1, n, step_match)
    if k == "eventually":
        return any(holds(formula.children[0], u, n, step_match)
                   for u in range(t, n))
    if k == "always":
        return all(holds(formula.children[0], u, n, step_match)
                   for u in range(t, n))
    if k == "until":
        hold_f, release_f = formula.children
        for u in range(t, n):
            if holds(release_f, u, n, step_match):
                return all(holds(hold_f, v, n, step_match)
                           for v in range(t, u))
        return False
    if k == "consecutive":
        child = formula.children[0]
        for u in range(t, n):
            if u + formula.k <= n and all(
                    holds(child, u + i, n, step_match)
                    for i in range(formula.k)):
                return True
        return False
    if k == "then_within":
        first_f, second_f = formula.children
        for u in range(t, n):
            if not holds(first_f, u, n, step_match):
                continue
            hi = min(u + formula.within, n - 1)
            if any(holds(second_f, v, n, step_match)
                   for v in range(u + 1, hi + 1)):
                return True
        return False

    raise TemporalError(f"cannot evaluate temporal node {k!r}")


def matches(formula: Formula, n: int,
            step_match: Callable[[Any, int], bool]) -> bool:
    """A trajectory matches when the formula holds at its first position."""
    if n <= 0:
        return False
    return holds(formula, 0, n, step_match)
