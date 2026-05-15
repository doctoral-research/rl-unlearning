"""Core ForgetScenario implementation.

Loads YAML scenarios, normalises them into a canonical AST (resolving
macros, desugaring region shorthands), and provides the matching
predicates used by the unlearning pipeline (`identify_forget_trajectories`,
`matches_state`, `get_forget_states_from_trajectories`, ...).

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


# Sentinel returned by group matchers when not enough information is
# available (e.g. an action-condition is evaluated with action=None).
# Treated as False by both AND and OR fold below, matching the original
# behaviour.
_UNKNOWN = object()


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

    def __init__(self, config: dict):
        if not isinstance(config, dict):
            raise ScenarioError(
                f"Scenario root must be a mapping; got {type(config).__name__}"
            )

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

        # ---- Macro table (used to expand $ref) ----
        # Defines are resolved EAGERLY at load time. A $ref to an unknown
        # name is a load-time error so eval is fast and predictable.
        self._defines: Dict[str, Any] = dict(config.get("defines", {}) or {})

        # ---- Groups / temporal expression ----
        raw_groups = config.get("groups")
        raw_temporal = config.get("temporal")

        if self.match_mode == "temporal":
            if raw_temporal is None:
                raise ScenarioError(
                    "match_mode=temporal requires a top-level 'temporal:' block"
                )
            self.temporal: Optional[Dict[str, Any]] = self._normalise_temporal(
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
    def load(cls, path: str) -> "ForgetScenario":
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Scenario file not found: {path_obj}")
        with open(path_obj) as f:
            config = yaml.safe_load(f)
        return cls(config)

    @classmethod
    def from_config(cls, cfg) -> Optional["ForgetScenario"]:
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
                return cls.load(str(p))

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

        # Default: dim condition.
        if "op" not in cond or "value" not in cond:
            raise ScenarioError(f"Condition missing op/value: {cond!r}")

        dim_raw = cond.get("dim")
        if dim_raw is None:
            raise ScenarioError(f"Condition missing 'dim': {cond!r}")
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

    def _resolve_dim(self, dim_raw: Any, target: str = "state") -> int:
        """Allow scenarios to reference dims by name OR index. Names are
        resolved against ENV_OBS_DIMS / ENV_ACTION_DIMS for the scenario's
        env_id. Falling back to an unresolved string is a load-time error
        — strings only work when env_id is set and known to us."""
        if isinstance(dim_raw, int):
            return dim_raw
        if isinstance(dim_raw, str):
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
        table = ENV_OBS_DIMS if target == "state" else ENV_ACTION_DIMS
        return table.get(self.env_id or "", {}).get(dim)

    def _desugar_region(
        self, name: Optional[str], region: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Convert a region shorthand into a canonical group AST node.

        box                — conjunction of half-spaces (AND'd dim conditions)
        complement_of_box  — disjunction of complementary half-spaces (any_of)
        """
        rtype = region.get("type")
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

    def _normalise_temporal(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        op = raw.get("operator")
        if op == "consecutive":
            return {
                "operator": "consecutive",
                "k": int(raw["k"]),
                "group": self._normalise_group_expr(raw["group"]),
            }
        if op == "then_within":
            return {
                "operator": "then_within",
                "within": int(raw["within"]),
                "first": self._normalise_group_expr(raw["first"]),
                "second": self._normalise_group_expr(raw["second"]),
            }
        if op == "eventually":
            return {
                "operator": "eventually",
                "group": self._normalise_group_expr(raw["group"]),
            }
        if op == "always":
            return {
                "operator": "always",
                "group": self._normalise_group_expr(raw["group"]),
            }
        raise ScenarioError(f"Unknown temporal operator: {op!r}")

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

    def matches_state(self, obs: np.ndarray, action=None) -> bool:
        """True if the (obs, action) tuple matches ANY top-level group."""
        return self._step_matches(obs, action)

    def get_forget_states_from_trajectories(
        self, trajectories: List[Dict], forget_indices: List[int],
    ) -> List[np.ndarray]:
        """Extract per-step matching observations from forget trajectories.

        Useful for strategy_inversion / TRIAD: gives on-distribution seed
        states that lie in the forget region. For trajectory-level modes
        (top_percentile / trajectory / temporal) returns all observations.
        """
        states: List[np.ndarray] = []
        has_step_conditions = self.match_mode in ("any_step", "all_steps", "proportion")
        for idx in forget_indices:
            traj = trajectories[idx]
            obs = traj["observations"]
            actions = traj.get("actions", [None] * len(obs))
            for t in range(len(obs)):
                if has_step_conditions:
                    act = actions[t] if t < len(actions) else None
                    if self._step_matches(obs[t], act):
                        states.append(np.array(obs[t]))
                else:
                    states.append(np.array(obs[t]))
        return states

    # ------------------------------------------------------------------
    # Internal: group matching
    # ------------------------------------------------------------------

    def _step_matches(self, obs: np.ndarray, action=None) -> bool:
        """OR across top-level groups (current semantics)."""
        for g in self.groups:
            r = self._group_matches(g, obs, action)
            if r is True:
                return True
        return False

    def _group_matches(
        self, group: Dict[str, Any], obs: np.ndarray, action=None,
    ) -> Any:
        kind = group["kind"]
        if kind == "leaf":
            return self._leaf_matches(group, obs, action)
        if kind == "all_of":
            for child in group["children"]:
                r = self._group_matches(child, obs, action)
                if r is False:
                    return False
                if r is _UNKNOWN:
                    return _UNKNOWN
            return True
        if kind == "any_of":
            saw_unknown = False
            for child in group["children"]:
                r = self._group_matches(child, obs, action)
                if r is True:
                    return True
                if r is _UNKNOWN:
                    saw_unknown = True
            return _UNKNOWN if saw_unknown else False
        if kind == "not":
            r = self._group_matches(group["child"], obs, action)
            if r is _UNKNOWN:
                return _UNKNOWN
            return not r
        raise ScenarioError(f"Unknown group kind: {kind!r}")

    def _leaf_matches(
        self, group: Dict[str, Any], obs: np.ndarray, action=None,
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
        obs = trajectory["observations"]
        actions = trajectory.get("actions", [None] * len(obs))
        if self.match_mode == "trajectory":
            agg = {
                "return": float(sum(trajectory.get("rewards", []))),
                "length": int(len(obs)),
            }
            return self._trajectory_level_groups_match(trajectory, agg)

        if self.match_mode == "any_step":
            for t in range(len(obs)):
                act = actions[t] if t < len(actions) else None
                if self._step_matches(obs[t], act):
                    return True
            return False
        if self.match_mode == "all_steps":
            for t in range(len(obs)):
                act = actions[t] if t < len(actions) else None
                if not self._step_matches(obs[t], act):
                    return False
            return True
        if self.match_mode == "proportion":
            if len(obs) == 0:
                return False
            count = sum(
                1 for t in range(len(obs))
                if self._step_matches(
                    obs[t], actions[t] if t < len(actions) else None,
                )
            )
            return (count / len(obs)) >= self.match_threshold

        raise ScenarioError(f"Unknown match_mode: {self.match_mode!r}")

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
            if ckind == "trajectory_aggregate":
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
                obs = trajectory["observations"]
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

    def _temporal_match(self, trajectory: Dict) -> bool:
        if self.temporal is None:
            return False
        op = self.temporal["operator"]
        obs = trajectory["observations"]
        actions = trajectory.get("actions", [None] * len(obs))

        def step_match(g, t):
            return self._group_matches(
                g, obs[t], actions[t] if t < len(actions) else None,
            ) is True

        if op == "consecutive":
            k = self.temporal["k"]
            g = self.temporal["group"]
            run = 0
            for t in range(len(obs)):
                run = run + 1 if step_match(g, t) else 0
                if run >= k:
                    return True
            return False
        if op == "always":
            g = self.temporal["group"]
            return all(step_match(g, t) for t in range(len(obs)))
        if op == "eventually":
            g = self.temporal["group"]
            return any(step_match(g, t) for t in range(len(obs)))
        if op == "then_within":
            within = self.temporal["within"]
            first = self.temporal["first"]
            second = self.temporal["second"]
            for t in range(len(obs)):
                if not step_match(first, t):
                    continue
                upper = min(len(obs), t + 1 + within)
                if any(step_match(second, s) for s in range(t + 1, upper)):
                    return True
            return False
        raise ScenarioError(f"Unknown temporal operator: {op!r}")

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
            values = [len(t["observations"]) for t in trajectories]
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
        if ck == "action_sequence":
            return f"action_sequence == {c['sequence']}"
        return f"<unknown condition {ck}>"

    def __repr__(self) -> str:
        return f"ForgetScenario(name={self.name!r}, groups={len(self.groups)})"


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
