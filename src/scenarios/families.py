"""Indexed families and quantification over them.

Multi-agent views expose one block of features per agent under indexed
names (`p0_row`, `p0_col`, `p1_row`, ...). v1.2 could read those names,
but had no way to say "over all agents", so a predicate about *the team*
had to enumerate every pair by hand:

    derived:
      sep_01: "abs(p0_row - p1_row) + abs(p0_col - p1_col)"
      sep_02: "abs(p0_row - p2_row) + abs(p0_col - p2_col)"
      sep_12: "abs(p1_row - p2_row) + abs(p1_col - p2_col)"
      closest: "min(sep_01, min(sep_02, sep_12))"

That is quadratic in team size and has to be rewritten for every N, which
is why `lbf.no_crowding` was recorded as only partially expressible.

A `families:` block names an indexed group, and a quantified `derived:`
entry binds members of it and reduces over the bindings:

    families:
      agent: "p{i}_"

    derived:
      closest_pair:
        over: agent
        arity: 2
        reduce: min
        expr: "abs(a_row - b_row) + abs(a_col - b_col)"

Bindings are named `a`, `b`, `c`, ... in order, and an attribute of a
binding is written `<binding>_<attr>`. The template is instantiated once
per combination and each instantiation is compiled by the ordinary
expression compiler, so quantification adds no new evaluation machinery
and inherits the same whitelist.

Members are discovered from the bound feature view, so the same scenario
file works for a 3-agent and an 8-agent team without edits.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Reductions over the instantiated values. `any`/`all` treat a value as
# true when it is non-zero, so a template that reads as a comparison can
# be quantified existentially or universally.
REDUCERS: Dict[str, Callable[[Sequence[float]], float]] = {
    "min": lambda v: float(min(v)) if v else float("inf"),
    "max": lambda v: float(max(v)) if v else float("-inf"),
    "sum": lambda v: float(sum(v)),
    "mean": lambda v: float(np.mean(v)) if v else 0.0,
    "count": lambda v: float(sum(1 for x in v if x)),
    "any": lambda v: 1.0 if any(bool(x) for x in v) else 0.0,
    "all": lambda v: 1.0 if v and all(bool(x) for x in v) else 0.0,
}

BINDING_NAMES = "abcdefgh"


class FamilyError(ValueError):
    """Raised for a malformed family or quantified expression."""


@dataclass(frozen=True)
class Family:
    """An indexed group of features discovered from a view."""

    name: str
    pattern: str
    prefixes: Tuple[str, ...]      # e.g. ("p0_", "p1_", "p2_")
    attributes: Tuple[str, ...]    # e.g. ("row", "col", "level", "score")

    def __len__(self) -> int:
        return len(self.prefixes)


def discover_family(name: str, pattern: str,
                    feature_names: Sequence[str]) -> Family:
    """Find the members of a family in a view's feature names.

    `pattern` is a prefix template containing `{i}`, e.g. `p{i}_`. Every
    feature whose name starts with an instantiation of it belongs to that
    member, and the remainder is the attribute.
    """
    if "{i}" not in pattern:
        raise FamilyError(
            f"family {name!r}: pattern {pattern!r} must contain '{{i}}' "
            "to mark the member index")

    regex = re.compile("^" + re.escape(pattern).replace(r"\{i\}", r"(\d+)")
                       + r"(.+)$")
    members: Dict[int, set] = {}
    for fname in feature_names:
        m = regex.match(fname)
        if m:
            members.setdefault(int(m.group(1)), set()).add(m.group(2))

    if not members:
        raise FamilyError(
            f"family {name!r}: pattern {pattern!r} matched no features. "
            f"Available: {list(feature_names)[:12]}")

    idx = sorted(members)
    common = set.intersection(*members.values())
    if not common:
        raise FamilyError(
            f"family {name!r}: members {idx} share no common attribute")

    prefixes = tuple(pattern.replace("{i}", str(i)) for i in idx)
    return Family(name=name, pattern=pattern, prefixes=prefixes,
                  attributes=tuple(sorted(common)))


def compile_family_block(
    block: Optional[Dict[str, str]], feature_names: Sequence[str],
) -> Dict[str, Family]:
    if not block:
        return {}
    if not isinstance(block, dict):
        raise FamilyError(
            f"'families' must be a mapping of name -> pattern; "
            f"got {type(block).__name__}")
    return {
        name: discover_family(name, pattern, feature_names)
        for name, pattern in block.items()
    }


@dataclass(frozen=True)
class QuantifiedSpec:
    """A `derived:` entry that quantifies over a family."""

    name: str
    family: str
    arity: int
    reduce: str
    template: str
    distinct: bool
    ordered: bool


def parse_quantified(name: str, spec: Dict) -> QuantifiedSpec:
    if "over" not in spec or "expr" not in spec:
        raise FamilyError(
            f"quantified feature {name!r} needs 'over' (a family) and 'expr'")
    arity = int(spec.get("arity", 1))
    if not 1 <= arity <= len(BINDING_NAMES):
        raise FamilyError(
            f"quantified feature {name!r}: arity must be 1..{len(BINDING_NAMES)}")
    reduce = spec.get("reduce", "min")
    if reduce not in REDUCERS:
        raise FamilyError(
            f"quantified feature {name!r}: reduce must be one of "
            f"{sorted(REDUCERS)}; got {reduce!r}")
    return QuantifiedSpec(
        name=name, family=spec["over"], arity=arity, reduce=reduce,
        template=spec["expr"],
        distinct=bool(spec.get("distinct", True)),
        ordered=bool(spec.get("ordered", False)),
    )


def _substitute(template: str, bindings: Sequence[str]) -> str:
    """Replace `<binding>_<attr>` with `<member-prefix><attr>`.

    Longest binding names first is unnecessary here (they are single
    letters) but the word boundary matters: a template mentioning
    `abs(...)` must not have its leading `a` rewritten.
    """
    out = template
    for letter, prefix in zip(BINDING_NAMES, bindings):
        out = re.sub(rf"\b{letter}_(\w+)", lambda m: f"{prefix}{m.group(1)}", out)
    return out


def expand(spec: QuantifiedSpec, family: Family) -> List[str]:
    """Instantiate the template once per binding combination."""
    if spec.arity > len(family):
        return []
    combos = (itertools.permutations(family.prefixes, spec.arity)
              if spec.ordered else
              itertools.combinations(family.prefixes, spec.arity))
    if not spec.distinct:
        combos = itertools.product(family.prefixes, repeat=spec.arity)
    return [_substitute(spec.template, list(c)) for c in combos]


def reduce_values(spec: QuantifiedSpec, values: Sequence[float]) -> float:
    return REDUCERS[spec.reduce](list(values))
