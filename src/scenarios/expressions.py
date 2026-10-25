"""Safe arithmetic expressions over named features.

v1.0 conditions could only compare one feature against a constant, so
anything *relational* was inexpressible: "closer to the goal than to the
hazard", "leading the opponent", "torso pitched further than the legs".
Every such predicate had to be pre-computed by the domain adapter, which
pushed scenario logic out of the scenario and into Python.

A `derived:` block adds named features computed from existing ones:

    derived:
      dist_to_goal: "abs(agent_x - 13) + abs(agent_y - 3)"
      lead:         "agent_x - opponent_x"

    groups:
      - conditions:
          - dim: dist_to_goal
            op: ">"
            value: 10

Expressions are parsed with `ast` and evaluated by a whitelisted walker,
never by `eval`. Only arithmetic over feature names, numeric literals and
a fixed function set is reachable, so a scenario file cannot execute
code.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Set, Tuple


class ExpressionError(ValueError):
    """Raised for malformed or disallowed expressions."""


# Whitelisted callables. Deliberately small: enough for distances and
# comparisons, nothing with side effects or attribute access.
ALLOWED_FUNCS: Dict[str, Callable] = {
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "hypot": math.hypot,
    "floor": math.floor,
    "ceil": math.ceil,
    "sign": lambda v: (v > 0) - (v < 0),
}

_BINOPS: Dict[type, Callable[[float, float], float]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b if b != 0 else float("nan"),
    ast.Pow: lambda a, b: a ** b,
    ast.Mod: lambda a, b: a % b if b != 0 else float("nan"),
}


@dataclass(frozen=True)
class DerivedFeature:
    """A named feature computed from other features."""

    name: str
    source: str
    fn: Callable[[Sequence[float], Dict[str, float]], float]
    inputs: Tuple[str, ...]

    def __call__(self, features: Sequence[float],
                 env: Dict[str, float] | None = None) -> float:
        return self.fn(features, env or {})


def _build(node: ast.AST, resolve: Callable[[str], int], inputs: Set[str],
           known_derived: Set[str]) -> Callable[[Sequence[float], Dict[str, float]], float]:
    """Recursively turn a whitelisted AST into a closure over
    (feature vector, already-computed derived values)."""
    if isinstance(node, ast.Expression):
        return _build(node.body, resolve, inputs, known_derived)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionError(
                f"only numeric literals are allowed; got {node.value!r}"
            )
        value = float(node.value)
        return lambda _f, _e: value

    if isinstance(node, ast.Name):
        inputs.add(node.id)
        if node.id in known_derived:
            token = node.id
            return lambda _f, e: float(e[token])
        idx = resolve(node.id)
        return lambda f, _e: float(f[idx])

    if isinstance(node, ast.UnaryOp):
        operand = _build(node.operand, resolve, inputs, known_derived)
        if isinstance(node.op, ast.USub):
            return lambda f, e: -operand(f, e)
        if isinstance(node.op, ast.UAdd):
            return operand
        raise ExpressionError(f"unary operator {type(node.op).__name__} not allowed")

    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ExpressionError(
                f"operator {type(node.op).__name__} not allowed; "
                f"allowed: {sorted(o.__name__ for o in _BINOPS)}"
            )
        left = _build(node.left, resolve, inputs, known_derived)
        right = _build(node.right, resolve, inputs, known_derived)
        return lambda f, e: op(left(f, e), right(f, e))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("only direct calls to allowed functions")
        fname = node.func.id
        fn = ALLOWED_FUNCS.get(fname)
        if fn is None:
            raise ExpressionError(
                f"function {fname!r} not allowed; "
                f"allowed: {sorted(ALLOWED_FUNCS)}"
            )
        if node.keywords:
            raise ExpressionError(f"{fname}(): keyword arguments not allowed")
        args = [_build(a, resolve, inputs, known_derived) for a in node.args]
        return lambda f, e: float(fn(*[a(f, e) for a in args]))

    raise ExpressionError(
        f"expression node {type(node).__name__} not allowed "
        "(only arithmetic over feature names and numbers)"
    )


def compile_expression(
    name: str, source: str, resolve: Callable[[str], int],
    known_derived: Set[str] | None = None,
) -> DerivedFeature:
    """Compile one `derived:` entry.

    `resolve` maps a feature name to its index in the feature vector and
    should raise if the name is unknown, so a typo in a derived
    expression fails at load time rather than silently at match time.
    """
    if not isinstance(source, str):
        raise ExpressionError(
            f"derived feature {name!r} must be a string expression; "
            f"got {type(source).__name__}"
        )
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(
            f"derived feature {name!r}: cannot parse {source!r} ({exc.msg})"
        ) from None
    inputs: Set[str] = set()
    fn = _build(tree, resolve, inputs, known_derived or set())
    return DerivedFeature(
        name=name, source=source, fn=fn, inputs=tuple(sorted(inputs)),
    )


def compile_derived_block(
    block: Dict[str, str], resolve: Callable[[str], int],
    external: Optional[Set[str]] = None,
) -> Dict[str, DerivedFeature]:
    """Compile a whole `derived:` mapping.

    Entries are compiled in declaration order and may reference features
    defined earlier in the same block, so chains like

        derived:
          dx:   "agent_x - 13"
          dist: "abs(dx) + abs(agent_y - 3)"

    work without the caller flattening them by hand. Forward references
    are rejected at compile time.
    """
    if not isinstance(block, dict):
        raise ExpressionError(
            f"'derived' must be a mapping of name -> expression; "
            f"got {type(block).__name__}"
        )
    compiled: Dict[str, DerivedFeature] = {}
    # `external` names (episode aggregates) resolve from the evaluation
    # environment rather than the per-step feature vector, exactly like a
    # reference to an earlier derived feature.
    known = set(external or ())
    for fname, source in block.items():
        if not isinstance(fname, str) or not fname:
            raise ExpressionError(f"derived feature name must be a string: {fname!r}")
        compiled[fname] = compile_expression(
            fname, source, resolve, known_derived=known | set(compiled),
        )
    return compiled


def evaluate_all(
    derived: Dict[str, DerivedFeature], features: Sequence[float],
    env0: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Evaluate every derived feature, in declaration order.

    `env0` seeds the environment with externally-computed values such as
    episode aggregates.
    """
    env: Dict[str, float] = dict(env0 or {})
    for name, feat in derived.items():
        env[name] = feat.fn(features, env)
    return env


def evaluate(
    derived: Dict[str, DerivedFeature], name: str, features: Sequence[float],
) -> float:
    """Evaluate one derived feature, computing its dependencies first."""
    env: Dict[str, float] = {}
    for other, feat in derived.items():
        env[other] = feat.fn(features, env)
        if other == name:
            return env[other]
    raise ExpressionError(f"unknown derived feature {name!r}")
