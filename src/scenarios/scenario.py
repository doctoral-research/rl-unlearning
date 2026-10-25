"""Core ForgetScenario implementation.

Loads YAML scenarios, normalises them into a canonical AST (resolving
macros, desugaring region shorthands), and provides the matching
predicates used by the unlearning pipeline (`identify_forget_trajectories`,
`matches_state`, `step_labels`, ...).

The class is wire-compatible with the legacy `from scenarios import
ForgetScenario` import — all v1.0 constructs are additive. Scenarios that
only use the original grammar continue to work without changes.
"""
from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .constants import (
    ENV_OBS_DIMS,
    ENV_ACTION_DIMS,
    _OPS,
    SCHEMA_VERSION,
)
from .views import FeatureView, ViewError
from .expressions import (
    DerivedFeature,
    ExpressionError,
    compile_derived_block,
    evaluate_all,
)
from .families import (
    Family,
    FamilyError,
    QuantifiedSpec,
    compile_family_block,
    expand as expand_quantified,
    parse_quantified,
    reduce_values,
)
from .temporal import (
    Formula,
    TemporalError,
    holds as temporal_holds,
    matches as temporal_matches,
    parse_formula,
)
from .aggregates import (
    Aggregate,
    AggregateError,
    compile_aggregate_block,
    evaluate_aggregates,
)


def _feature_seq(trajectory: Dict) -> Sequence:
    """Per-step features for a recorded trajectory.

    Prefers a pre-extracted `features` sequence (written at collection
    time by a FeatureView) and falls back to raw `observations`, which is
    what v1.0 trajectories carry. Recording features at collection time
    is what makes env-reading views usable for offline analysis, since
    the environment object is long gone by then.
    """
    feats = trajectory.get("features")
    return feats if feats is not None else trajectory["observations"]


# Sentinel returned by group matchers when not enough information is
# available (e.g. an action-condition is evaluated with action=None).
# Treated as False by both AND and OR fold below, matching the original
# behaviour.
_UNKNOWN = object()

# `applies_to` — which STEPS carry the forget label, given that
# `match_mode` has already decided which TRAJECTORIES match.
#
# These are genuinely different sets and both are legitimate, but the
# grammar used to name only the second. A step-level consumer therefore
# ignored `match_mode` entirely and silently disagreed with the
# trajectory-level reading of the same file. Measured on trained
# FourRooms policies, the two readings of one scenario differ by 17.4
# percentage points of reported forget-effectiveness.
MATCHING_STEPS = "matching_steps"        # only steps satisfying the predicate
WHOLE_TRAJECTORY = "whole_trajectory"    # every step of a matched trajectory
VALID_STEP_SCOPES = (MATCHING_STEPS, WHOLE_TRAJECTORY)

# Modes that select trajectories by an aggregate, so a step-level reading
# necessarily throws information away unless `applies_to` says otherwise.
TRAJECTORY_LEVEL_MODES = frozenset(
    {"proportion", "top_percentile", "trajectory", "temporal"}
)


class ScenarioError(ValueError):
    """Raised for structural problems detected at load time that prevent
    the scenario from being usable (missing required fields, unresolved
    $refs, region/dim mismatches). Validator emits richer diagnostics —
    this is the last-line safety net for direct .load() callers."""


class ForgetScenario:
    """Declarative description of behavior to unlearn.

    The grammar (v1.0):

      scenario := { name, description, env_id?, version?, match_mode?,
                    match_threshold?, defines?, groups | temporal }
      group_expr :=
          { conditions: [cond+], name?, rank_by?, rank_order? }   # leaf
        | { all_of: [group_expr+], name? }
        | { any_of: [group_expr+], name? }
        | { not: group_expr, name? }
        | { region: <region_box | complement_of_box>, name? }
        | { $ref: <defines-key> }
      cond := { dim, dim_name?, op, value, target? }
            | { action_sequence: [a+] }
            | { $ref: <defines-key> }
      temporal := { operator: consecutive|then_within|eventually|always,
                    group | first/second, k | within }
    """

    def __init__(self, config: dict, view: Optional[FeatureView] = None):
        if not isinstance(config, dict):
            raise ScenarioError(
                f"Scenario root must be a mapping; got {type(config).__name__}"
            )

        # The feature view is the scenario's domain interface. It must be
        # set before groups are normalised, because dim names resolve to
        # indices eagerly at construction time. The raw config is kept so
        # `with_view` can re-resolve against a different view.
        self._config: dict = config
        self.view: Optional[FeatureView] = view

        # ---- Required fields ----
        try:
            self.name: str = config["name"]
            self.description: str = config["description"]
        except KeyError as missing:
            raise ScenarioError(f"Scenario missing required field {missing}") from None

        # ---- Optional metadata ----
        self.version: str = str(config.get("version", SCHEMA_VERSION))
        self.env_id: Optional[str] = config.get("env_id")
        self.match_mode: str = config.get("match_mode", "any_step")
        self.match_threshold: float = float(config.get("match_threshold", 0.5))

        # Declared step scope. Defaults to matching_steps, which is what
        # every step-level consumer already did implicitly; declaring it
        # makes the choice visible and checkable rather than accidental.
        self.applies_to: str = config.get("applies_to", MATCHING_STEPS)
        if self.applies_to not in VALID_STEP_SCOPES:
            raise ScenarioError(
                f"applies_to must be one of {VALID_STEP_SCOPES}; "
                f"got {self.applies_to!r}"
            )
        self.applies_to_declared: bool = "applies_to" in config

        # ---- Indexed families ----
        # Discovered from the bound view's feature names, so the same file
        # works for any team size. Without a view there are no names to
        # discover from, and a quantified expression will say so.
        try:
            self.families: Dict[str, Family] = compile_family_block(
                config.get("families"),
                self.view.names if self.view is not None else (),
            )
        except FamilyError as exc:
            raise ScenarioError(str(exc)) from None

        # ---- Episode aggregates ----
        # Compiled before derived features so a `derived:` expression can
        # combine them (e.g. steps_per_cell = path_length / cells_visited).
        # `where:` predicates are normalised lazily, after `_defines` is
        # populated, because they may contain $ref.
        self._raw_aggregates = config.get("aggregates") or {}
        self.aggregates: Dict[str, Aggregate] = {}

        # ---- Derived features ----
        # Compiled before groups so conditions can reference the names.
        # `_resolve_base_dim` is the plain view/table lookup, so a typo in
        # an expression fails here rather than silently at match time.
        raw_derived = config.get("derived") or {}
        # Quantified entries expand into one plain expression per binding
        # combination; the reduction is applied over their values. The
        # expansion happens here so everything downstream sees ordinary
        # derived features and no new evaluation path exists.
        self.quantified: Dict[str, QuantifiedSpec] = {}
        expanded: Dict[str, str] = {}
        try:
            for fname, body in raw_derived.items():
                if not isinstance(body, dict):
                    expanded[fname] = body
                    continue
                qs = parse_quantified(fname, body)
                if qs.family not in self.families:
                    raise FamilyError(
                        f"quantified feature {fname!r} refers to unknown "
                        f"family {qs.family!r}; declared: "
                        f"{sorted(self.families)}")
                self.quantified[fname] = qs
                for k, expr in enumerate(expand_quantified(
                        qs, self.families[qs.family])):
                    expanded[f"__{fname}__{k}"] = expr
        except FamilyError as exc:
            raise ScenarioError(str(exc)) from None

        try:
            self.derived: Dict[str, DerivedFeature] = compile_derived_block(
                expanded, self._resolve_base_dim,
                external=set(self._raw_aggregates),
            )
        except ExpressionError as exc:
            raise ScenarioError(str(exc)) from None

        # ---- Macro table (used to expand $ref) ----
        # Defines are resolved EAGERLY at load time. A $ref to an unknown
        # name is a load-time error so eval is fast and predictable.
        self._defines: Dict[str, Any] = dict(config.get("defines", {}) or {})

        # Aggregates are compiled here: `_defines` is now populated, so a
        # `where:` predicate can use $ref like any other group.
        try:
            self.aggregates = compile_aggregate_block(
                self._raw_aggregates, self._resolve_base_dim,
                self._normalise_group_expr,
            )
        except (AggregateError, ExpressionError) as exc:
            # _resolve_base_dim raises ExpressionError (it is shared with the
            # derived-feature compiler), so both surface as ScenarioError to
            # keep scenario construction's error contract single-typed.
            raise ScenarioError(str(exc)) from None

        # ---- Groups / temporal expression ----
        raw_groups = config.get("groups")
        raw_temporal = config.get("temporal")

        if self.match_mode == "temporal":
            if raw_temporal is None:
                raise ScenarioError(
                    "match_mode=temporal requires a top-level 'temporal:' block"
                )
            self.temporal: Optional[Formula] = self._normalise_temporal(
                raw_temporal
            )
            # Groups can still be present and used as legacy state predicates,
            # but at temporal eval time they are not consulted by default.
            self.groups: List[Dict[str, Any]] = self._normalise_groups(
                raw_groups or []
            )
        else:
            if not raw_groups:
                raise ScenarioError(
                    f"Scenario '{self.name}' has no groups and match_mode is "
                    f"'{self.match_mode}'. Add at least one group or switch "
                    "to match_mode=temporal."
                )
            self.temporal = None
            self.groups = self._normalise_groups(raw_groups)

    # ------------------------------------------------------------------
    # Public class methods (unchanged signatures)
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str, view: Optional[FeatureView] = None) -> "ForgetScenario":
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Scenario file not found: {path_obj}")
        with open(path_obj) as f:
            config = yaml.safe_load(f)
        return cls(config, view=view)

    def with_view(self, view: Optional[FeatureView]) -> "ForgetScenario":
        """Re-bind this scenario to a different feature view.

        Returns a new scenario rather than mutating in place, because dim
        names were already resolved to indices against the old view and
        those indices would otherwise silently change meaning.
        """
        return type(self)(self._config, view=view)

    @classmethod
    def from_config(cls, cfg, view: Optional[FeatureView] = None) -> Optional["ForgetScenario"]:
        """Load scenario from a Hydra config that has a 'scenario' field.

        Returns None if no scenario is configured. The scenario field is a
        relative path like 'cartpole/left_only' (without .yaml).
        """
        scenario_name = cfg.get("scenario", None)
        if not scenario_name:
            return None

        search_paths = [
            Path("configs/scenarios") / f"{scenario_name}.yaml",
            Path(__file__).resolve().parent.parent.parent
            / "configs"
            / "scenarios"
            / f"{scenario_name}.yaml",
        ]
        for p in search_paths:
            if p.exists():
                return cls.load(str(p), view=view)

        raise FileNotFoundError(
            f"Scenario '{scenario_name}' not found. "
            f"Searched: {[str(p) for p in search_paths]}"
        )

    # ------------------------------------------------------------------
    # Normalisation (load-time canonicalisation)
    # ------------------------------------------------------------------

    def _normalise_groups(
        self, raw_groups: Sequence[Any]
    ) -> List[Dict[str, Any]]:
        return [self._normalise_group_expr(g) for g in raw_groups]

    def _normalise_group_expr(self, expr: Any) -> Dict[str, Any]:
        """Convert a raw YAML group expression into a canonical AST node.

        Canonical forms (internal):
          {"kind": "leaf",   "name": str?, "conditions": [...],
           "rank_by"?, "rank_order"?}
          {"kind": "all_of", "name": str?, "children": [...]}
          {"kind": "any_of", "name": str?, "children": [...]}
          {"kind": "not",    "name": str?, "child": ...}

        Region shorthands and $refs are eagerly expanded into combinations
        of the above so the matcher only deals with three node kinds.
        """
        if not isinstance(expr, dict):
            raise ScenarioError(f"Group expression must be a mapping; got {expr!r}")

        # 1. Macro reference — expand inline then re-normalise.
        if "$ref" in expr:
            ref_name = expr["$ref"]
            if ref_name not in self._defines:
                raise ScenarioError(
                    f"Unknown $ref '{ref_name}'. Available defines: "
                    f"{sorted(self._defines)}"
                )
            return self._normalise_group_expr(self._defines[ref_name])

        # 2. Boolean combinators.
        if "all_of" in expr:
            return {
                "kind": "all_of",
                "name": expr.get("name"),
                "children": [self._normalise_group_expr(c) for c in expr["all_of"]],
            }
        if "any_of" in expr:
            return {
                "kind": "any_of",
                "name": expr.get("name"),
                "children": [self._normalise_group_expr(c) for c in expr["any_of"]],
            }
        if "not" in expr:
            return {
                "kind": "not",
                "name": expr.get("name"),
                "child": self._normalise_group_expr(expr["not"]),
            }

        # 3. Region shorthand: desugar to a leaf with AND'd dim conditions
        #    (box) or an any_of of complementary half-spaces (complement_of_box).
        if "region" in expr:
            return self._desugar_region(expr.get("name"), expr["region"])

        # 4. Leaf group: conditions list + optional rank_by/rank_order.
        if "conditions" in expr:
            conditions = [self._normalise_condition(c) for c in expr["conditions"]]
            node = {"kind": "leaf", "name": expr.get("name"), "conditions": conditions}
            if "rank_by" in expr:
                node["rank_by"] = expr["rank_by"]
            if "rank_order" in expr:
                node["rank_order"] = expr["rank_order"]
            return node

        # 5. top_percentile groups have no conditions, only rank_by / rank_order.
        if "rank_by" in expr or "rank_order" in expr:
            return {
                "kind": "leaf",
                "name": expr.get("name"),
                "conditions": [],
                "rank_by": expr.get("rank_by", "return"),
                "rank_order": expr.get("rank_order", "descending"),
            }

        raise ScenarioError(
            f"Unrecognised group expression keys: {sorted(expr)}. "
            "Expected one of: conditions, all_of, any_of, not, region, $ref, "
            "rank_by/rank_order (for top_percentile)."
        )

    def _normalise_condition(self, cond: Any) -> Dict[str, Any]:
        if not isinstance(cond, dict):
            raise ScenarioError(f"Condition must be a mapping; got {cond!r}")

        if "$ref" in cond:
            ref_name = cond["$ref"]
            if ref_name not in self._defines:
                raise ScenarioError(f"Unknown condition $ref '{ref_name}'")
            return self._normalise_condition(self._defines[ref_name])

        if "action_sequence" in cond:
            return {
                "kind": "action_sequence",
                "sequence": list(cond["action_sequence"]),
            }

        if "target" in cond and cond["target"] in ("return", "length"):
            # Trajectory-aggregate condition (no dim required).
            return {
                "kind": "trajectory_aggregate",
                "target": cond["target"],
                "op": cond["op"],
                "value": float(cond["value"]),
            }

        if "target" in cond and cond["target"] == "reward":
            # Per-step scalar reward condition. No 'dim' because rewards
            # are scalars per step. Evaluated only when the step matcher
            # is supplied with the reward (i.e. during trajectory walks).
            if "op" not in cond or "value" not in cond:
                raise ScenarioError(f"reward condition missing op/value: {cond!r}")
            return {
                "kind": "reward",
                "op": cond["op"],
                "value": float(cond["value"]),
            }

        # Default: dim condition.
        if "op" not in cond or "value" not in cond:
            raise ScenarioError(f"Condition missing op/value: {cond!r}")

        dim_raw = cond.get("dim")
        if dim_raw is None:
            raise ScenarioError(f"Condition missing 'dim': {cond!r}")

        # A condition may target an episode aggregate by name.
        if (isinstance(dim_raw, str) and dim_raw in self._raw_aggregates
                and cond.get("target", "state") == "state"):
            return {
                "kind": "aggregate",
                "name": dim_raw,
                "op": cond["op"],
                "value": float(cond["value"]),
                "target": "state",
            }

        # A condition may target a derived feature by name. Derived names
        # shadow base features, which is checked when the block compiles.
        if (isinstance(dim_raw, str)
                and (dim_raw in self.derived or dim_raw in self.quantified)
                and cond.get("target", "state") == "state"):
            return {
                "kind": "derived",
                "name": dim_raw,
                "expr": (self.derived[dim_raw].source
                         if dim_raw in self.derived
                         else f"{self.quantified[dim_raw].reduce} over "
                              f"{self.quantified[dim_raw].family}: "
                              f"{self.quantified[dim_raw].template}"),
                "op": cond["op"],
                "value": float(cond["value"]),
                "target": "state",
            }

        dim_index = self._resolve_dim(dim_raw, target=cond.get("target", "state"))
        return {
            "kind": "dim",
            "dim": dim_index,
            "dim_name": cond.get("dim_name") or self._lookup_dim_name(
                dim_index, target=cond.get("target", "state"),
            ),
            "op": cond["op"],
            "value": float(cond["value"]),
            "target": cond.get("target", "state"),
        }

    def _resolve_base_dim(self, name: str) -> int:
        """Resolve a BASE feature name to an index, for expression
        compilation. Deliberately excludes derived names: that lookup is
        handled by the expression compiler itself so chains resolve in
        declaration order and forward references are rejected."""
        try:
            return self._resolve_dim(name, target="state")
        except ScenarioError as exc:
            raise ExpressionError(str(exc)) from None

    def _resolve_dim(self, dim_raw: Any, target: str = "state") -> int:
        """Allow scenarios to reference dims by name OR index. Names are
        resolved against ENV_OBS_DIMS / ENV_ACTION_DIMS for the scenario's
        env_id. Falling back to an unresolved string is a load-time error
        — strings only work when env_id is set and known to us."""
        if isinstance(dim_raw, int):
            return dim_raw
        if isinstance(dim_raw, str):
            # A bound feature view is the actual semantic interface the
            # scenario will be evaluated against, so it wins over the
            # hand-registered name table.
            if target == "state" and self.view is not None:
                try:
                    return self.view.index(dim_raw)
                except ViewError as exc:
                    raise ScenarioError(str(exc)) from None
            table = ENV_OBS_DIMS if target == "state" else ENV_ACTION_DIMS
            env_dims = table.get(self.env_id or "")
            if env_dims is None:
                raise ScenarioError(
                    f"Cannot resolve dim name '{dim_raw}': env_id "
                    f"'{self.env_id}' has no registered {target} dim table. "
                    "Use an integer index or set env_id."
                )
            for idx, name in env_dims.items():
                if name == dim_raw:
                    return idx
            raise ScenarioError(
                f"Dim name '{dim_raw}' not in {target} dims for "
                f"env_id={self.env_id}: {sorted(env_dims.values())}"
            )
        raise ScenarioError(f"dim must be int or str; got {dim_raw!r}")

    def _lookup_dim_name(self, dim: int, target: str = "state") -> Optional[str]:
        """Best-effort reverse lookup so introspection can render a name
        even if the YAML didn't supply dim_name."""
        if target == "state" and self.view is not None:
            return self.view.name_of(dim)
        table = ENV_OBS_DIMS if target == "state" else ENV_ACTION_DIMS
        return table.get(self.env_id or "", {}).get(dim)

    def _desugar_region(
        self, name: Optional[str], region: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Convert a region shorthand into a canonical group AST node.

        box                conjunction of half-spaces (AND'd dim conditions)
        complement_of_box  disjunction of complementary half-spaces (any_of)
        ellipsoid          one 'ellipsoid' condition with center + radii
        polygon            one 'polygon' condition with ordered 2D vertices
        """
        rtype = region.get("type")

        if rtype == "ellipsoid":
            dims_raw = region.get("dims", [])
            center = list(region.get("center", []))
            radii = list(region.get("radii", []))
            if not (len(dims_raw) == len(center) == len(radii)):
                raise ScenarioError(
                    f"region '{name or rtype}': dims, center, radii must have equal length"
                )
            if any(float(r) <= 0 for r in radii):
                raise ScenarioError(
                    f"region '{name or rtype}': radii must be strictly positive"
                )
            dim_indices = [self._resolve_dim(d) for d in dims_raw]
            return {
                "kind": "leaf",
                "name": name,
                "conditions": [{
                    "kind": "ellipsoid",
                    "dims": dim_indices,
                    "center": [float(c) for c in center],
                    "radii": [float(r) for r in radii],
                }],
            }

        if rtype == "polygon":
            dims_raw = region.get("dims", [])
            vertices = region.get("vertices", [])
            if len(dims_raw) != 2:
                raise ScenarioError(
                    f"region '{name or rtype}': polygon requires exactly 2 dims"
                )
            if len(vertices) < 3:
                raise ScenarioError(
                    f"region '{name or rtype}': polygon requires at least 3 vertices"
                )
            dim_indices = [self._resolve_dim(d) for d in dims_raw]
            return {
                "kind": "leaf",
                "name": name,
                "conditions": [{
                    "kind": "polygon",
                    "dims": dim_indices,
                    "vertices": [(float(v[0]), float(v[1])) for v in vertices],
                }],
            }

        dims_raw = region.get("dims", [])
        lo = region.get("lo", [])
        hi = region.get("hi", [])
        if len(dims_raw) != len(lo) or len(dims_raw) != len(hi):
            raise ScenarioError(
                f"region '{name or rtype}': dims, lo, hi must have equal length"
            )
        dim_indices = [self._resolve_dim(d) for d in dims_raw]

        conditions_in_box: List[Dict[str, Any]] = []
        for idx, d_lo, d_hi in zip(dim_indices, lo, hi):
            if d_lo is not None:
                conditions_in_box.append({
                    "kind": "dim",
                    "dim": idx,
                    "dim_name": self._lookup_dim_name(idx),
                    "op": ">=",
                    "value": float(d_lo),
                    "target": "state",
                })
            if d_hi is not None:
                conditions_in_box.append({
                    "kind": "dim",
                    "dim": idx,
                    "dim_name": self._lookup_dim_name(idx),
                    "op": "<=",
                    "value": float(d_hi),
                    "target": "state",
                })

        if rtype == "box":
            return {"kind": "leaf", "name": name, "conditions": conditions_in_box}

        if rtype == "complement_of_box":
            # Complement: any dim outside its [lo, hi]. Build a disjunction.
            outside_groups: List[Dict[str, Any]] = []
            for idx, d_lo, d_hi in zip(dim_indices, lo, hi):
                if d_lo is not None:
                    outside_groups.append({
                        "kind": "leaf", "name": None,
                        "conditions": [{
                            "kind": "dim", "dim": idx,
                            "dim_name": self._lookup_dim_name(idx),
                            "op": "<", "value": float(d_lo), "target": "state",
                        }],
                    })
                if d_hi is not None:
                    outside_groups.append({
                        "kind": "leaf", "name": None,
                        "conditions": [{
                            "kind": "dim", "dim": idx,
                            "dim_name": self._lookup_dim_name(idx),
                            "op": ">", "value": float(d_hi), "target": "state",
                        }],
                    })
            if not outside_groups:
                raise ScenarioError(
                    f"region '{name}': complement_of_box has no bounded sides"
                )
            return {"kind": "any_of", "name": name, "children": outside_groups}

        raise ScenarioError(f"region '{name}': unknown type '{rtype}'")

    def _normalise_temporal(self, raw: Dict[str, Any]) -> Any:
        """Parse the `temporal:` block into a nestable formula tree.

        The v1.0 flat form ({operator: ..., group: ...}) still parses; its
        operands simply become leaf state predicates.
        """
        try:
            return parse_formula(raw, self._normalise_group_expr)
        except TemporalError as exc:
            raise ScenarioError(str(exc)) from None

    # ------------------------------------------------------------------
    # Public matching API (unchanged from legacy)
    # ------------------------------------------------------------------

    def identify_forget_trajectories(
        self, trajectories: List[Dict]
    ) -> List[int]:
        """Return indices of trajectories matching the forget criteria."""
        if self.match_mode == "top_percentile":
            return self._top_percentile_match(trajectories)
        if self.match_mode == "temporal":
            return [
                i for i, t in enumerate(trajectories) if self._temporal_match(t)
            ]

        return [i for i, t in enumerate(trajectories)
                if self._trajectory_matches(t)]

    def matches_state(
        self, obs: np.ndarray, action=None, reward: Optional[float] = None,
        info=None, env=None,
    ) -> bool:
        """True if this step matches ANY top-level group.

        When a feature view is bound, `obs`/`info`/`env` are passed through
        `view.extract` first and conditions are evaluated against the
        resulting feature vector. With no view bound the observation is
        used directly, which is exactly v1.0 behaviour.

        `reward` is optional for backwards-compat. Scenarios that include
        a `target: reward` condition will report _UNKNOWN (treated as False)
        when reward is not supplied at the call site, matching how missing
        action handling already works.
        """
        return self._step_matches(self.features(obs, info, env), action, reward)

    @property
    def step_scope_is_ambiguous(self) -> bool:
        """True when this scenario selects trajectories by an aggregate but
        does not say which of their steps carry the label.

        Such a scenario has two defensible readings that a step-level
        consumer cannot distinguish. `applies_to` resolves it.
        """
        return (self.match_mode in TRAJECTORY_LEVEL_MODES
                and not self.applies_to_declared)

    def step_labels(self, trajectory: Dict) -> np.ndarray:
        """Per-step forget labels for one trajectory, honouring `applies_to`.

        This is the canonical way to build a training mask. Calling
        `matches_features` per step instead gives the matching_steps
        reading regardless of what the scenario declares, which is exactly
        the drift this method exists to prevent.
        """
        feats = _feature_seq(trajectory)
        actions = trajectory.get("actions") or [None] * len(feats)
        rewards = trajectory.get("rewards") or [None] * len(actions)
        n = len(actions)

        if self.applies_to == WHOLE_TRAJECTORY:
            # A rollout buffer is many episodes concatenated. Selecting over
            # the whole buffer asks "did 15% of these 2048 steps match",
            # which is almost always true and labels everything -- a
            # different specification from the one the file declares.
            # When episode boundaries are supplied, decide per episode.
            dones = trajectory.get("dones")
            if dones is not None and len(dones) == n and n > 0:
                out = np.zeros(n, dtype=bool)
                start = 0
                for i in range(n):
                    if bool(dones[i]) or i == n - 1:
                        ep = {k: v[start:i + 1] for k, v in trajectory.items()
                              if hasattr(v, "__len__") and len(v) == n}
                        out[start:i + 1] = bool(
                            self.identify_forget_trajectories([ep]))
                        start = i + 1
                return out
            selected = bool(self.identify_forget_trajectories([trajectory]))
            return np.full(n, selected, dtype=bool)

        if self.match_mode == "temporal" and self.temporal is not None:
            # A temporal formula is evaluated AT a position, so a step
            # label is simply whether it holds there. Falling back to the
            # state predicate instead -- which is what this did before --
            # silently drops the temporal structure: "turn left
            # immediately on entering the room" becomes "turn left
            # anywhere in the room", a strictly larger set.
            step_match, n_obs = self._temporal_step_match(trajectory)
            return np.array(
                [temporal_holds(self.temporal, t, n_obs, step_match)
                 for t in range(n_obs)], dtype=bool)

        mask = np.zeros(n, dtype=bool)
        for t in range(min(n, len(feats))):
            act = actions[t] if t < len(actions) else None
            rwd = rewards[t] if t < len(rewards) else None
            mask[t] = bool(self.matches_features(np.asarray(feats[t]), act, rwd))
        return mask

    def _derived_env(self, features, env0=None) -> Dict[str, float]:
        """Evaluate derived features and reduce quantified ones.

        Quantified entries were expanded at load time into one hidden
        feature per binding combination; this folds them back into a
        single value under the declared name.
        """
        env = evaluate_all(self.derived, features, env0)
        for name, qs in self.quantified.items():
            vals = [v for k, v in env.items()
                    if k.startswith(f"__{name}__")]
            env[name] = reduce_values(qs, vals)
        return env

    def matches_features(
        self, features, action=None, reward: Optional[float] = None,
    ) -> bool:
        """Match against an ALREADY-EXTRACTED feature vector.

        Rollout loops must extract features at collection time (the env
        has moved on by the time a rollout is masked), so they call
        `features()` per step and this afterwards. Calling `matches_state`
        with a feature vector would double-extract when a view is bound.
        """
        return self._step_matches(features, action, reward)

    def features(self, obs=None, info=None, env=None) -> np.ndarray:
        """Extract the feature vector this scenario evaluates over.

        Exposed so callers can record features alongside observations at
        collection time (see `_feature_seq`), which is what lets
        env-reading views be replayed offline.
        """
        if self.view is None:
            return obs
        return self.view.extract(obs, info, env)

    # ------------------------------------------------------------------
    # Internal: group matching
    # ------------------------------------------------------------------

    def _step_matches(
        self, obs: np.ndarray, action=None, reward: Optional[float] = None,
    ) -> bool:
        """OR across top-level groups (current semantics)."""
        for g in self.groups:
            r = self._group_matches(g, obs, action, reward)
            if r is True:
                return True
        return False

    def _group_matches(
        self, group: Dict[str, Any], obs: np.ndarray, action=None,
        reward: Optional[float] = None,
    ) -> Any:
        kind = group["kind"]
        if kind == "leaf":
            return self._leaf_matches(group, obs, action, reward)
        if kind == "all_of":
            for child in group["children"]:
                r = self._group_matches(child, obs, action, reward)
                if r is False:
                    return False
                if r is _UNKNOWN:
                    return _UNKNOWN
            return True
        if kind == "any_of":
            saw_unknown = False
            for child in group["children"]:
                r = self._group_matches(child, obs, action, reward)
                if r is True:
                    return True
                if r is _UNKNOWN:
                    saw_unknown = True
            return _UNKNOWN if saw_unknown else False
        if kind == "not":
            r = self._group_matches(group["child"], obs, action, reward)
            if r is _UNKNOWN:
                return _UNKNOWN
            return not r
        raise ScenarioError(f"Unknown group kind: {kind!r}")

    def _leaf_matches(
        self, group: Dict[str, Any], obs: np.ndarray, action=None,
        reward: Optional[float] = None,
    ) -> Any:
        # Empty leaf (top_percentile) matches everything as a placeholder.
        conditions = group.get("conditions", [])
        if not conditions:
            return True

        for cond in conditions:
            ckind = cond["kind"]
            if ckind == "dim":
                target = cond["target"]
                if target == "state":
                    val = float(obs[cond["dim"]])
                elif target == "action":
                    if action is None:
                        return _UNKNOWN
                    val = float(action) if np.isscalar(action) else float(action[cond["dim"]])
                else:
                    # Trajectory-aggregate dim condition would have been
                    # routed to 'trajectory_aggregate' kind during
                    # normalisation; arriving here is a bug.
                    raise ScenarioError(
                        f"Unexpected target '{target}' in step-level evaluation"
                    )
                op_fn = _OPS.get(cond["op"])
                if op_fn is None:
                    raise ScenarioError(f"Unknown operator: {cond['op']!r}")
                if not op_fn(val, cond["value"]):
                    return False
            elif ckind == "derived":
                # Derived features are recomputed per step. The whole block
                # is evaluated because later entries may depend on earlier
                # ones; the block is small so this stays cheap.
                # Aggregates are episode-level and unavailable here, so a
                # derived feature that needs one reports unknown rather
                # than crashing mid-rollout.
                try:
                    env = self._derived_env(obs)
                except KeyError:
                    return _UNKNOWN
                op_fn = _OPS.get(cond["op"])
                if op_fn is None:
                    raise ScenarioError(f"Unknown operator: {cond['op']!r}")
                if not op_fn(env[cond["name"]], cond["value"]):
                    return False
            elif ckind == "reward":
                # Per-step reward predicate. Reward is only available when
                # the trajectory walk supplies it. External callers using
                # matches_state(obs, action) with no reward will see this
                # as _UNKNOWN, mirroring missing-action handling above.
                if reward is None:
                    return _UNKNOWN
                op_fn = _OPS.get(cond["op"])
                if op_fn is None:
                    raise ScenarioError(f"Unknown operator: {cond['op']!r}")
                if not op_fn(float(reward), cond["value"]):
                    return False
            elif ckind == "ellipsoid":
                # sum_i ((x_i - c_i) / r_i)^2 < 1
                total = 0.0
                for d, c, r in zip(cond["dims"], cond["center"], cond["radii"]):
                    total += ((float(obs[d]) - c) / r) ** 2
                if not (total < 1.0):
                    return False
            elif ckind == "polygon":
                # Ray-casting point-in-polygon. Works for arbitrary simple
                # polygons (convex or concave). Horizontal ray to +inf;
                # count edge crossings.
                x = float(obs[cond["dims"][0]])
                y = float(obs[cond["dims"][1]])
                if not _point_in_polygon(x, y, cond["vertices"]):
                    return False
            elif ckind == "trajectory_aggregate":
                # Not valid at step level — treat as unknown so AND/OR
                # short-circuit correctly without raising during scans.
                return _UNKNOWN
            elif ckind == "action_sequence":
                return _UNKNOWN
            else:
                raise ScenarioError(f"Unknown condition kind: {ckind!r}")
        return True

    # ------------------------------------------------------------------
    # Internal: trajectory-level matching
    # ------------------------------------------------------------------

    def _trajectory_matches(self, trajectory: Dict) -> bool:
        obs = _feature_seq(trajectory)
        actions = trajectory.get("actions", [None] * len(obs))
        rewards = trajectory.get("rewards", [None] * len(obs))
        if self.match_mode == "trajectory":
            agg = {
                "return": float(sum(r for r in rewards if r is not None)),
                "length": int(len(obs)),
            }
            agg.update(self._episode_aggregates(trajectory))
            return self._trajectory_level_groups_match(trajectory, agg)

        if self.match_mode == "any_step":
            for t in range(len(obs)):
                act = actions[t] if t < len(actions) else None
                rwd = rewards[t] if t < len(rewards) else None
                if self._step_matches(obs[t], act, rwd):
                    return True
            return False
        if self.match_mode == "all_steps":
            for t in range(len(obs)):
                act = actions[t] if t < len(actions) else None
                rwd = rewards[t] if t < len(rewards) else None
                if not self._step_matches(obs[t], act, rwd):
                    return False
            return True
        if self.match_mode == "proportion":
            if len(obs) == 0:
                return False
            count = sum(
                1 for t in range(len(obs))
                if self._step_matches(
                    obs[t],
                    actions[t] if t < len(actions) else None,
                    rewards[t] if t < len(rewards) else None,
                )
            )
            return (count / len(obs)) >= self.match_threshold

        raise ScenarioError(f"Unknown match_mode: {self.match_mode!r}")

    def _episode_aggregates(self, trajectory: Dict) -> Dict[str, float]:
        """Evaluate this scenario's `aggregates:` over one trajectory."""
        if not self.aggregates:
            return {}
        feats = _feature_seq(trajectory)
        actions = trajectory.get("actions") or [None] * len(feats)
        rewards = trajectory.get("rewards") or [None] * len(actions)

        def _step_match(group: Dict[str, Any], t: int) -> bool:
            if t >= len(feats):
                return False
            act = actions[t] if t < len(actions) else None
            rwd = rewards[t] if t < len(rewards) else None
            return self._group_matches(
                group, np.asarray(feats[t]), act, rwd) is True

        return evaluate_aggregates(self.aggregates, feats, _step_match)

    def _trajectory_level_groups_match(
        self, trajectory: Dict, agg: Dict[str, float],
    ) -> bool:
        for g in self.groups:
            if self._trajectory_group_matches(g, trajectory, agg):
                return True
        return False

    def _trajectory_group_matches(
        self, group: Dict[str, Any], trajectory: Dict, agg: Dict[str, float],
    ) -> bool:
        kind = group["kind"]
        if kind == "leaf":
            return self._trajectory_leaf_matches(group, trajectory, agg)
        if kind == "all_of":
            return all(self._trajectory_group_matches(c, trajectory, agg)
                       for c in group["children"])
        if kind == "any_of":
            return any(self._trajectory_group_matches(c, trajectory, agg)
                       for c in group["children"])
        if kind == "not":
            return not self._trajectory_group_matches(
                group["child"], trajectory, agg,
            )
        raise ScenarioError(f"Unknown group kind: {kind!r}")

    def _trajectory_leaf_matches(
        self, group: Dict[str, Any], trajectory: Dict, agg: Dict[str, float],
    ) -> bool:
        for cond in group.get("conditions", []):
            ckind = cond["kind"]
            if ckind == "derived":
                feats = _feature_seq(trajectory)
                env = self._derived_env(feats[0] if len(feats) else [], agg)
                op_fn = _OPS.get(cond["op"])
                if not op_fn(env[cond["name"]], cond["value"]):
                    return False
            elif ckind == "aggregate":
                val = agg.get(cond["name"])
                if val is None:
                    return False
                op_fn = _OPS.get(cond["op"])
                if not op_fn(val, cond["value"]):
                    return False
            elif ckind == "trajectory_aggregate":
                val = agg.get(cond["target"])
                if val is None:
                    return False
                op_fn = _OPS.get(cond["op"])
                if not op_fn(val, cond["value"]):
                    return False
            elif ckind == "action_sequence":
                if not self._action_sequence_matches(
                    trajectory.get("actions", []), cond["sequence"],
                ):
                    return False
            elif ckind == "dim":
                # Dim conditions at trajectory level are interpreted as
                # "ANY step in the trajectory satisfies this dim condition" —
                # consistent with the any_step interpretation at scope mismatch.
                obs = _feature_seq(trajectory)
                target = cond["target"]
                if target == "state":
                    vals = [float(obs[t][cond["dim"]]) for t in range(len(obs))]
                else:
                    actions = trajectory.get("actions", [])
                    vals = [
                        float(actions[t]) if np.isscalar(actions[t])
                        else float(actions[t][cond["dim"]])
                        for t in range(len(actions))
                    ]
                op_fn = _OPS.get(cond["op"])
                if not any(op_fn(v, cond["value"]) for v in vals):
                    return False
            else:
                raise ScenarioError(f"Unknown trajectory-leaf condition kind: {ckind!r}")
        return True

    @staticmethod
    def _action_sequence_matches(actions: Sequence, target: Sequence) -> bool:
        n = len(actions)
        m = len(target)
        if m == 0 or n < m:
            return False
        for i in range(n - m + 1):
            if all(actions[i + j] == target[j] for j in range(m)):
                return True
        return False

    # ------------------------------------------------------------------
    # Temporal evaluation
    # ------------------------------------------------------------------

    def _temporal_step_match(self, trajectory: Dict):
        """A predicate `(group, t) -> bool` over one trajectory, plus its
        length. Shared by trajectory-level temporal matching and by
        per-step temporal labelling so the two cannot disagree about what
        a leaf of the formula means."""
        obs = _feature_seq(trajectory)
        actions = trajectory.get("actions", [None] * len(obs))
        rewards = trajectory.get("rewards", [None] * len(obs))

        def step_match(g, t):
            return self._group_matches(
                g,
                obs[t],
                actions[t] if t < len(actions) else None,
                rewards[t] if t < len(rewards) else None,
            ) is True

        return step_match, len(obs)

    def _temporal_match(self, trajectory: Dict) -> bool:
        if self.temporal is None:
            return False
        step_match, n = self._temporal_step_match(trajectory)
        return temporal_matches(self.temporal, n, step_match)

    @property
    def temporal_depth(self) -> int:
        """Nesting depth of the temporal formula (1 = a bare predicate)."""
        return self.temporal.depth() if self.temporal is not None else 0

    # ------------------------------------------------------------------
    # top_percentile
    # ------------------------------------------------------------------

    def _top_percentile_match(self, trajectories: List[Dict]) -> List[int]:
        first = self.groups[0] if self.groups else {}
        rank_by = first.get("rank_by", "return")
        rank_order = first.get("rank_order", "descending")
        if rank_by == "return":
            values = [sum(t["rewards"]) for t in trajectories]
        elif rank_by == "length":
            values = [len(_feature_seq(t)) for t in trajectories]
        else:
            raise ScenarioError(f"Unknown rank_by: {rank_by}")
        n = len(trajectories)
        num_forget = max(1, int(n * self.match_threshold))
        sorted_idx = np.argsort(values)
        if rank_order == "descending":
            return list(sorted_idx[-num_forget:])
        return list(sorted_idx[:num_forget])

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary (preserved API). Use to_natural_language()
        or to_predicate() in introspect.py for paper-quality output."""
        lines = [f"Scenario: {self.name}", f"  {self.description.strip()}"]
        lines.append(f"  match_mode: {self.match_mode}")
        if self.match_mode in ("proportion", "top_percentile"):
            lines.append(f"  match_threshold: {self.match_threshold}")
        if self.match_mode == "temporal" and self.temporal:
            lines.append(f"  temporal: {self.temporal['operator']}")
        for g in self.groups:
            lines.extend(self._summarise_group(g, indent="  "))
        return "\n".join(lines)

    def _summarise_group(self, g: Dict[str, Any], indent: str) -> List[str]:
        kind = g["kind"]
        gname = g.get("name") or kind
        if kind == "leaf":
            lines = [f"{indent}group '{gname}':"]
            if self.match_mode == "top_percentile":
                lines.append(
                    f"{indent}  rank_by: {g.get('rank_by', 'return')} "
                    f"({g.get('rank_order', 'descending')})"
                )
            for c in g.get("conditions", []):
                lines.append(f"{indent}  {self._summarise_condition(c)}")
            return lines
        if kind in ("all_of", "any_of"):
            label = "ALL of" if kind == "all_of" else "ANY of"
            lines = [f"{indent}{label} '{gname}':"]
            for c in g["children"]:
                lines.extend(self._summarise_group(c, indent + "  "))
            return lines
        if kind == "not":
            lines = [f"{indent}NOT '{gname}':"]
            lines.extend(self._summarise_group(g["child"], indent + "  "))
            return lines
        return [f"{indent}<unknown group kind {kind}>"]

    def _summarise_condition(self, c: Dict[str, Any]) -> str:
        ck = c["kind"]
        if ck == "dim":
            label = c.get("dim_name") or f"dim[{c['dim']}]"
            return f"{c['target']}.{label} {c['op']} {c['value']}"
        if ck == "trajectory_aggregate":
            return f"{c['target']} {c['op']} {c['value']}"
        if ck == "reward":
            return f"reward(s_t) {c['op']} {c['value']}"
        if ck == "ellipsoid":
            dims = ",".join(str(d) for d in c["dims"])
            return f"ellipsoid(dims=({dims}), center={c['center']}, radii={c['radii']})"
        if ck == "polygon":
            return f"polygon(dims={tuple(c['dims'])}, vertices={len(c['vertices'])} pts)"
        if ck == "action_sequence":
            return f"action_sequence == {c['sequence']}"
        return f"<unknown condition {ck}>"

    def __repr__(self) -> str:
        return f"ForgetScenario(name={self.name!r}, groups={len(self.groups)})"


def _point_in_polygon(
    x: float, y: float, vertices: Sequence[Tuple[float, float]],
) -> bool:
    """Standard ray-casting point-in-polygon test.

    Casts a horizontal ray from (x, y) to +infinity and counts how many
    polygon edges it crosses. Odd = inside, even = outside. Works for
    arbitrary simple polygons (convex or concave). Edge cases where the
    point lies exactly on an edge are reported as outside, which is
    consistent with strict-inclusion semantics of the rest of the language
    (e.g. ellipsoid uses `< 1`, not `<= 1`).
    """
    n = len(vertices)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        # Edge (xj, yj) -> (xi, yi). Crossing test: edge straddles y and
        # the intersection x-coordinate lies to the right of the test point.
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi
        ):
            inside = not inside
        j = i
    return inside


def list_scenarios(base_dir: str = "configs/scenarios") -> List[str]:
    """List all available scenario names (relative paths without .yaml)."""
    base = Path(base_dir)
    if not base.exists():
        return []
    scenarios = []
    for p in sorted(base.rglob("*.yaml")):
        rel = p.relative_to(base).with_suffix("")
        scenarios.append(str(rel))
    return scenarios
