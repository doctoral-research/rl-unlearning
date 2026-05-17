"""Multi-layer validation for ForgetScenario YAML files.

The validator runs three layers, in order:

  1. **Schema**          — JSON Schema (Draft 2020-12) check via the
                           `jsonschema` library. Catches typos, missing
                           required fields, wrong types, illegal enums.
  2. **Semantic**        — checks the schema cannot express: env-specific
                           dim bounds, dim_name vs canonical name match,
                           op/target compatibility, contradictions inside
                           AND'd condition groups, unresolved $refs, region
                           dim/lo/hi length mismatches, etc.
  3. **Lint**            — non-fatal style hints: empty description,
                           unusual thresholds, redundant conditions.

Issues are returned as a structured `ValidationReport`. The CLI maps
errors → exit code 1, lints-only → exit code 0 (but reports them).

Run from the command line:

    python -m scenarios validate configs/scenarios/cartpole/left_only.yaml
    python -m scenarios validate 'configs/scenarios/**/*.yaml'
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml

from .constants import (
    ENV_ACTION_DIMS,
    ENV_OBS_DIMS,
    SCHEMA_VERSION,
    VALID_MATCH_MODES,
    _OPS,
)
from .scenario import ForgetScenario, ScenarioError


_SCHEMA_PATH = Path(__file__).parent / "schemas" / "scenario.schema.v1.json"


# ---------------------------------------------------------------------------
# Issue / report types
# ---------------------------------------------------------------------------


Severity = str  # "error" | "warning" | "info"


@dataclass
class ValidationIssue:
    """A single problem found in a scenario.

    `path` is a JSON-pointer-style location like `groups[0].conditions[1].op`
    so authors can find the offending entry in the YAML quickly. `file` is
    set for filesystem-backed validations; in-memory dict validation leaves
    it as None.
    """

    severity: Severity
    code: str          # short machine identifier, e.g. "unknown_op"
    message: str       # human-readable explanation
    path: str = ""     # location within the document
    file: Optional[str] = None

    def render(self) -> str:
        loc = f"{self.file}:{self.path}" if self.file else self.path
        loc = loc.strip(":")
        prefix = f"[{self.severity.upper():7}] {self.code}"
        if loc:
            return f"{prefix:34} at {loc}\n         {self.message}"
        return f"{prefix:34}\n         {self.message}"


@dataclass
class ValidationReport:
    """Aggregate result of validating one or more scenarios."""

    issues: List[ValidationIssue] = field(default_factory=list)

    # Sugar -----------------------------------------------------------------

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def infos(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "info"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def merge(self, other: "ValidationReport") -> "ValidationReport":
        return ValidationReport(issues=self.issues + other.issues)

    def render(self) -> str:
        if not self.issues:
            return "OK"
        return "\n".join(i.render() for i in self.issues)


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------


def validate_scenario(path: Union[str, Path]) -> ValidationReport:
    """Validate a YAML scenario file. Returns a ValidationReport — does
    not raise on validation errors; reserve exceptions for true I/O or
    schema-loading failures."""
    p = Path(path)
    if not p.exists():
        return _single_error("file_not_found", f"Scenario file not found: {p}", file=str(p))
    try:
        with open(p) as f:
            doc = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return _single_error("yaml_parse_error", f"YAML parse error: {e}", file=str(p))
    if doc is None:
        return _single_error("empty_file", "YAML file is empty", file=str(p))

    return validate_scenario_dict(doc, file=str(p))


def validate_scenario_dict(
    doc: Any, *, file: Optional[str] = None,
) -> ValidationReport:
    """Validate an in-memory scenario dict.

    Useful for programmatic checks (tests, IDE integrations) and for
    validating scenarios constructed at runtime.
    """
    report = ValidationReport()
    _check_schema(doc, report, file=file)
    if report.errors:
        # Don't run semantic checks if the schema is broken — they would
        # be confusing follow-on errors rather than independent findings.
        return report

    _check_semantic(doc, report, file=file)
    _check_lint(doc, report, file=file)
    return report


# ---------------------------------------------------------------------------
# Layer 1 — JSON Schema
# ---------------------------------------------------------------------------


def _check_schema(
    doc: Any, report: ValidationReport, *, file: Optional[str],
) -> None:
    """Run draft 2020-12 schema validation if jsonschema is installed.

    We tolerate jsonschema being absent (the dependency is optional in
    minimal installs) — fall back to a softer check that catches the most
    common typos by reading the schema's `enum` and `required` lists.
    """
    try:
        import jsonschema  # type: ignore
    except ImportError:
        _soft_schema_check(doc, report, file=file)
        return

    with open(_SCHEMA_PATH) as f:
        schema = json.load(f)

    validator = jsonschema.Draft202012Validator(schema)
    for err in sorted(validator.iter_errors(doc), key=lambda e: list(e.path)):
        report.add(ValidationIssue(
            severity="error",
            code="schema",
            message=err.message,
            path=_jpath(err.path),
            file=file,
        ))


def _soft_schema_check(
    doc: Any, report: ValidationReport, *, file: Optional[str],
) -> None:
    """Minimal validation when jsonschema is unavailable. Catches obvious
    typos so the user doesn't get a misleading semantic-layer error."""
    if not isinstance(doc, dict):
        report.add(ValidationIssue("error", "schema",
            "Scenario root must be a mapping", "", file))
        return
    for required in ("name", "description"):
        if required not in doc:
            report.add(ValidationIssue("error", "schema_missing_required",
                f"Missing required field: {required}", "", file))
    mm = doc.get("match_mode", "any_step")
    if mm not in VALID_MATCH_MODES:
        report.add(ValidationIssue("error", "schema_unknown_match_mode",
            f"Unknown match_mode {mm!r}; valid: {sorted(VALID_MATCH_MODES)}",
            "match_mode", file))


# ---------------------------------------------------------------------------
# Layer 2 — Semantic
# ---------------------------------------------------------------------------


def _check_semantic(
    doc: Dict[str, Any], report: ValidationReport, *, file: Optional[str],
) -> None:
    """Env-aware and cross-field checks.

    A whole-document instantiation is attempted via `ForgetScenario(doc)` —
    if it raises, the diagnostic is surfaced as a structured error.
    Otherwise we walk the canonical AST and perform deeper checks.
    """
    try:
        scenario = ForgetScenario(doc)
    except ScenarioError as e:
        report.add(ValidationIssue("error", "scenario_load",
            str(e), "", file))
        return

    env_id = scenario.env_id
    obs_dims = ENV_OBS_DIMS.get(env_id or "", {})
    act_dims = ENV_ACTION_DIMS.get(env_id or "", {})

    if env_id and not obs_dims:
        report.add(ValidationIssue("warning", "unknown_env",
            f"env_id={env_id!r} is not in the ENV_OBS_DIMS table; "
            "dim names cannot be validated. Add it to "
            "src/scenarios/constants.py for full validation.",
            "env_id", file))

    # ---- Walk each top-level group ----
    for gi, group in enumerate(scenario.groups):
        _walk_group(group, scenario, obs_dims, act_dims, report,
                    path=f"groups[{gi}]", file=file)

    # ---- Temporal sanity ----
    if scenario.match_mode == "temporal" and scenario.temporal:
        t = scenario.temporal
        op = t["operator"]
        if op == "consecutive" and t["k"] < 1:
            report.add(ValidationIssue("error", "temporal_bad_k",
                "consecutive.k must be >= 1", "temporal.k", file))
        if op == "then_within" and t["within"] < 1:
            report.add(ValidationIssue("error", "temporal_bad_within",
                "then_within.within must be >= 1", "temporal.within", file))
        if op == "until":
            # Both arguments are required by schema, but be defensive in
            # case a hand-built dict bypasses the schema layer.
            for key in ("hold", "release"):
                if key not in t:
                    report.add(ValidationIssue("error", "until_missing_arg",
                        f"until.{key} is required", f"temporal.{key}", file))

    # ---- Defines: warn about unused macros ----
    # Walk both 'groups' AND 'temporal' (macros can be referenced from
    # either). Skip the 'defines' section itself so a macro that only
    # references another macro doesn't count its own keys as "used".
    used_refs: set[str] = set()
    for key in ("groups", "temporal"):
        _collect_refs(doc.get(key, []), used_refs)
    defines = doc.get("defines") or {}
    for k in defines:
        if k not in used_refs:
            report.add(ValidationIssue("warning", "unused_define",
                f"defines['{k}'] is declared but never $ref'd",
                f"defines.{k}", file))


def _walk_group(
    group: Dict[str, Any],
    scenario: ForgetScenario,
    obs_dims: Dict[int, str],
    act_dims: Dict[int, str],
    report: ValidationReport,
    *,
    path: str,
    file: Optional[str],
) -> None:
    kind = group["kind"]
    if kind == "leaf":
        _check_leaf(group, scenario, obs_dims, act_dims, report,
                    path=path, file=file)
        return
    if kind in ("all_of", "any_of"):
        for i, child in enumerate(group["children"]):
            _walk_group(child, scenario, obs_dims, act_dims, report,
                        path=f"{path}.{kind}[{i}]", file=file)
        # Empty combinator is meaningless — surface as a warning since the
        # schema already enforces minItems but defensive duplication is cheap.
        if not group["children"]:
            report.add(ValidationIssue("warning", "empty_combinator",
                f"{kind} has no children — group will never match",
                path, file))
        return
    if kind == "not":
        _walk_group(group["child"], scenario, obs_dims, act_dims, report,
                    path=f"{path}.not", file=file)
        return


def _check_leaf(
    group: Dict[str, Any],
    scenario: ForgetScenario,
    obs_dims: Dict[int, str],
    act_dims: Dict[int, str],
    report: ValidationReport,
    *,
    path: str,
    file: Optional[str],
) -> None:
    conditions = group.get("conditions", [])

    # Detect intra-AND contradictions on the same (target, dim).
    seen: Dict[tuple, List[Dict[str, Any]]] = {}
    for ci, cond in enumerate(conditions):
        cpath = f"{path}.conditions[{ci}]"
        ckind = cond["kind"]
        if ckind == "dim":
            _check_dim_condition(cond, scenario, obs_dims, act_dims,
                                 report, path=cpath, file=file)
            key = (cond["target"], cond["dim"])
            seen.setdefault(key, []).append(cond)
        elif ckind == "trajectory_aggregate":
            if scenario.match_mode != "trajectory":
                report.add(ValidationIssue("error", "aggregate_outside_trajectory",
                    f"condition with target={cond['target']!r} only valid "
                    "in match_mode=trajectory; saw "
                    f"match_mode={scenario.match_mode!r}",
                    cpath, file))
        elif ckind == "reward":
            # Per-step reward conditions are valid in every step-level mode
            # AND inside temporal patterns. Only invalid in 'trajectory'
            # mode where there is no per-step iteration.
            if scenario.match_mode == "trajectory":
                report.add(ValidationIssue("warning", "reward_outside_step_scope",
                    "target=reward is a per-step predicate; consider "
                    "any_step / all_steps / proportion / temporal "
                    "instead of match_mode=trajectory",
                    cpath, file))
        elif ckind == "ellipsoid":
            _check_ellipsoid(cond, obs_dims, scenario, report,
                             path=cpath, file=file)
        elif ckind == "polygon":
            _check_polygon(cond, obs_dims, scenario, report,
                           path=cpath, file=file)
        elif ckind == "action_sequence":
            if scenario.match_mode not in ("trajectory",):
                report.add(ValidationIssue("warning", "action_sequence_scope",
                    "action_sequence conditions are only evaluated at "
                    "trajectory scope; consider match_mode=trajectory",
                    cpath, file))

    # Per-(target, dim) interval feasibility check.
    for (target, dim), conds in seen.items():
        if not _interval_satisfiable(conds):
            report.add(ValidationIssue("error", "contradictory_conditions",
                f"Conditions on {target} dim {dim} are mutually unsatisfiable "
                f"(e.g. x > 5 AND x < 0)",
                path, file))


def _check_ellipsoid(
    cond: Dict[str, Any],
    obs_dims: Dict[int, str],
    scenario: ForgetScenario,
    report: ValidationReport,
    *,
    path: str,
    file: Optional[str],
) -> None:
    dims = cond.get("dims", [])
    radii = cond.get("radii", [])
    if any(float(r) <= 0 for r in radii):
        report.add(ValidationIssue("error", "ellipsoid_bad_radii",
            "ellipsoid radii must be strictly positive",
            f"{path}.radii", file))
    if scenario.env_id and obs_dims:
        for d in dims:
            if d not in obs_dims:
                report.add(ValidationIssue("error", "dim_out_of_range",
                    f"ellipsoid dim {d} not registered for "
                    f"env_id={scenario.env_id!r}",
                    f"{path}.dims", file))


def _check_polygon(
    cond: Dict[str, Any],
    obs_dims: Dict[int, str],
    scenario: ForgetScenario,
    report: ValidationReport,
    *,
    path: str,
    file: Optional[str],
) -> None:
    dims = cond.get("dims", [])
    vertices = cond.get("vertices", [])
    if len(dims) != 2:
        report.add(ValidationIssue("error", "polygon_bad_dims",
            "polygon must reference exactly 2 dims",
            f"{path}.dims", file))
    if len(vertices) < 3:
        report.add(ValidationIssue("error", "polygon_few_vertices",
            f"polygon needs at least 3 vertices, got {len(vertices)}",
            f"{path}.vertices", file))
    if scenario.env_id and obs_dims:
        for d in dims:
            if d not in obs_dims:
                report.add(ValidationIssue("error", "dim_out_of_range",
                    f"polygon dim {d} not registered for "
                    f"env_id={scenario.env_id!r}",
                    f"{path}.dims", file))


def _check_dim_condition(
    cond: Dict[str, Any],
    scenario: ForgetScenario,
    obs_dims: Dict[int, str],
    act_dims: Dict[int, str],
    report: ValidationReport,
    *,
    path: str,
    file: Optional[str],
) -> None:
    target = cond["target"]
    dim = cond["dim"]
    table = obs_dims if target == "state" else act_dims
    label = "obs" if target == "state" else "action"

    if scenario.env_id and table:
        if dim not in table:
            report.add(ValidationIssue("error", "dim_out_of_range",
                f"{label} dim {dim} not registered for "
                f"env_id={scenario.env_id!r} (valid: "
                f"{sorted(table)})",
                f"{path}.dim", file))
        # dim_name mismatch
        declared = cond.get("dim_name")
        canonical = table.get(dim)
        if declared and canonical and declared != canonical:
            report.add(ValidationIssue("warning", "dim_name_mismatch",
                f"dim_name={declared!r} does not match canonical name "
                f"{canonical!r} for {label} dim {dim} on "
                f"env_id={scenario.env_id!r}",
                f"{path}.dim_name", file))

    if cond["op"] not in _OPS:
        report.add(ValidationIssue("error", "unknown_op",
            f"Unknown operator {cond['op']!r}; valid: {sorted(_OPS)}",
            f"{path}.op", file))


def _interval_satisfiable(conds: Sequence[Dict[str, Any]]) -> bool:
    """Check whether a list of conditions on the same (target, dim) is
    jointly satisfiable. Handles >, >=, <, <=, ==, !=. Treats abs-prefixed
    operators as always-satisfiable (full analysis would require a sign
    case-split; conservative fallback prevents false positives).
    """
    lo = -math.inf
    lo_strict = False
    hi = math.inf
    hi_strict = False
    eqs: List[float] = []
    neqs: List[float] = []
    for c in conds:
        op = c["op"]
        v = c["value"]
        if op == ">":
            if v >= lo:
                lo, lo_strict = v, True
        elif op == ">=":
            if v > lo:
                lo, lo_strict = v, False
        elif op == "<":
            if v <= hi:
                hi, hi_strict = v, True
        elif op == "<=":
            if v < hi:
                hi, hi_strict = v, False
        elif op == "==":
            eqs.append(v)
        elif op == "!=":
            neqs.append(v)
        elif op.startswith("abs"):
            return True  # conservative
    if eqs:
        # All == values must agree, and the chosen value must satisfy the
        # rest. Two distinct == values are immediately unsat.
        if len(set(eqs)) > 1:
            return False
        x = eqs[0]
        if (x < lo or (x == lo and lo_strict)
                or x > hi or (x == hi and hi_strict)):
            return False
        if x in neqs:
            return False
        return True
    # No equalities: just check lo < hi (or <= depending on strictness).
    if lo > hi:
        return False
    if lo == hi and (lo_strict or hi_strict):
        return False
    return True


def _collect_refs(obj: Any, out: set) -> None:
    if isinstance(obj, dict):
        if "$ref" in obj and isinstance(obj["$ref"], str):
            out.add(obj["$ref"])
        for v in obj.values():
            _collect_refs(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_refs(v, out)


# ---------------------------------------------------------------------------
# Layer 3 — Lint
# ---------------------------------------------------------------------------


def _check_lint(
    doc: Dict[str, Any], report: ValidationReport, *, file: Optional[str],
) -> None:
    if "version" not in doc:
        report.add(ValidationIssue("info", "missing_version",
            f"No 'version' field — assuming {SCHEMA_VERSION}. Add "
            f"`version: \"{SCHEMA_VERSION}\"` for forward-compat.",
            "version", file))

    desc = (doc.get("description") or "").strip()
    if len(desc) < 20:
        report.add(ValidationIssue("warning", "short_description",
            "Description is < 20 chars; please document what behavior is "
            "being forgotten and why (this text is used in paper figures).",
            "description", file))

    if doc.get("match_mode") == "proportion":
        thr = doc.get("match_threshold", 0.5)
        if thr <= 0.0 or thr >= 1.0:
            report.add(ValidationIssue("warning", "extreme_threshold",
                f"match_threshold={thr} is on the boundary; 0 matches all, "
                "1 matches nothing. Did you mean a value inside (0, 1)?",
                "match_threshold", file))

    if doc.get("match_mode") == "top_percentile":
        thr = doc.get("match_threshold", 0.1)
        if thr > 0.5:
            report.add(ValidationIssue("info", "large_top_percentile",
                f"top_percentile threshold={thr} forgets more than half the "
                "trajectories — likely indiscriminate forgetting rather than "
                "targeted.", "match_threshold", file))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _jpath(path) -> str:
    """Render a jsonschema deque path as `groups[0].conditions[1].op`."""
    parts: List[str] = []
    for p in path:
        if isinstance(p, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{p}]"
            else:
                parts.append(f"[{p}]")
        else:
            parts.append(str(p))
    return ".".join(parts)


def _single_error(code: str, msg: str, *, file: Optional[str]) -> ValidationReport:
    return ValidationReport(issues=[
        ValidationIssue("error", code, msg, "", file),
    ])
