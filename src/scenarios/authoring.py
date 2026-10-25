"""Authoring a scenario from a natural-language description.

Turning "stop the agent cutting through the bottom-right room" into a
valid scenario is a reasonable job for a language model, and a poor job to
trust one with unchecked: a plausible-looking YAML can reference features
that do not exist, threshold outside a feature's range so it matches
nothing, or declare a trajectory mode while meaning a step-level label.
All three read fine and none of them fail loudly.

What makes this tractable here is that the checking already exists. A
candidate is accepted only if it

    1. parses as YAML,
    2. passes the JSON Schema layer,
    3. constructs against the domain's feature view (so every name it
       mentions is real), and
    4. fires on actual rollouts, when they are supplied -- neither
       matching nothing nor matching everything.

A rejected candidate is not discarded: the validator's own diagnostics go
back into the next prompt, so the model is repairing against machine
feedback rather than guessing again. `AuthoringResult` keeps every
attempt, which is what makes the loop measurable: how often it converges,
in how many rounds, and how much the feedback is worth compared with
resampling blind.

The model backend is any `Callable[[str], str]` mapping a prompt to YAML
text, so this module has no dependency on a particular provider -- and
the tests drive it with scripted responses rather than a network call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import yaml

from .constants import _OPS, SCHEMA_VERSION, VALID_MATCH_MODES
from .introspect import coverage
from .scenario import (
    MATCHING_STEPS,
    VALID_STEP_SCOPES,
    WHOLE_TRAJECTORY,
    ForgetScenario,
    ScenarioError,
)
from .validate import ValidationReport, validate_scenario_dict
from .views import FeatureView

# A backend maps a prompt to candidate YAML. Anything callable will do.
Backend = Callable[[str], str]

# Warnings that are fatal when authoring. Interactively these are advice;
# for a generated candidate they mean the scenario is wrong in a way that
# still parses -- a threshold outside the feature's range matches nothing,
# an ordered categorical compares codes as magnitudes, an undeclared step
# scope leaves the label ambiguous. Rejecting them is the whole point of
# checking rather than trusting.
FATAL_WARNINGS = frozenset({
    "condition_never_true",
    "condition_always_true",
    "categorical_ordering",
    "boolean_ordering",
    "boolean_value",
    "float_equality",
    "step_scope_ambiguous",
    "temporal_without_operator",
    "dim_out_of_range",
})


_COND_PATH = re.compile(r"^groups\[(\d+)\]\.conditions\[(\d+)\]")


def _group_conjunct_count(group) -> Optional[int]:
    """How many AND'd conditions a group contributes after desugaring.

    Regions are expanded into conditions before validation, so a box over
    k dims contributes 2k of them. Returns None for a group whose parts
    are OR'd, where one always-true member really does make it vacuous.
    """
    if not isinstance(group, dict):
        return None
    if "any_of" in group:
        return None
    if isinstance(group.get("conditions"), list):
        return len(group["conditions"])
    if isinstance(group.get("all_of"), list):
        return len(group["all_of"])
    region = group.get("region")
    if isinstance(region, dict):
        dims = region.get("dims")
        if region.get("type") == "box" and isinstance(dims, list):
            return 2 * len(dims)
        return 1
    return None


def _is_harmless_redundancy(issue, doc, report) -> bool:
    """True for an always-true condition that is merely redundant.

    Conjuncts are AND'd, so `agent_x >= 0` beside a real constraint
    narrows nothing but changes nothing either, and a box region drawn to
    the edge of a feature's range has trivial faces for the same reason.
    Either is fatal only when EVERY conjunct in the group is trivial,
    because then the group really does match everything.

    Found by running the loop against qwen2.5-coder, which draws full
    bounding boxes; rejecting those was a comment on the gate rather than
    on the specification.
    """
    if issue.code != "condition_always_true":
        return False
    m = _COND_PATH.match(issue.path or "")
    if not m:
        return False
    gi = int(m.group(1))
    try:
        group = doc["groups"][gi]
    except (KeyError, IndexError, TypeError):
        return False
    total = _group_conjunct_count(group)
    if not total:
        return False
    prefix = f"groups[{gi}].conditions["
    trivial = sum(1 for i in report.warnings
                  if i.code == "condition_always_true"
                  and (i.path or "").startswith(prefix))
    return trivial < total


def _fatal_issues(report, doc, fatal_warnings):
    return [i for i in report.warnings
            if i.code in fatal_warnings
            and not _is_harmless_redundancy(i, doc, report)]


def _fence_strip(text: str) -> str:
    """Pull YAML out of a fenced code block if the model used one."""
    t = text.strip()
    if "```" not in t:
        return t
    parts = t.split("```")
    for chunk in parts[1:]:
        body = chunk.split("\n", 1)
        if len(body) == 2:
            lang, rest = body
            if lang.strip().lower() in ("", "yaml", "yml"):
                return rest.strip()
    return t


def describe_view(view: Optional[FeatureView]) -> str:
    """The feature vocabulary a scenario may reference.

    This is the part that stops a model inventing `agent_position` when
    the domain calls it `agent_x`: the names come from the same view the
    scenario will be evaluated against.
    """
    if view is None:
        return ("No feature view is bound, so conditions must address "
                "dimensions by integer index.")
    lines = [f"Available features (view provenance: {view.provenance}):"]
    for i, name in enumerate(view.names):
        dtype = view.dtypes[i] if i < len(view.dtypes) else "continuous"
        rng = view.ranges[i] if i < len(view.ranges) else None
        rng_s = f", range [{rng[0]:g}, {rng[1]:g}]" if rng else ""
        lines.append(f"  - {name} ({dtype}{rng_s})")
    lines.append("Use these names exactly. Any other name is an error.")
    return "\n".join(lines)


def grammar_reference() -> str:
    """A compact grammar summary, generated from the same tables the
    validator uses so the prompt cannot drift from the implementation."""
    return "\n".join([
        f"Scenario grammar (version {SCHEMA_VERSION}). Emit YAML only.",
        "",
        "Required: name, description.",
        '  name must match ^[A-Za-z0-9_-]+$ : a slug like avoid_bottom_right,',
        "  not a title. Put the human-readable phrasing in description.",
        f"match_mode: one of {sorted(VALID_MATCH_MODES)} (default any_step).",
        "  proportion and top_percentile also need match_threshold.",
        f"applies_to: one of {list(VALID_STEP_SCOPES)}. Declare it whenever",
        f"  match_mode selects whole trajectories. {MATCHING_STEPS} labels",
        f"  only the steps that satisfy the predicate; {WHOLE_TRAJECTORY}",
        "  labels every step of a selected trajectory.",
        "",
        "groups: a list, OR'd. Each entry is one of:",
        "  {conditions: [...]}            conditions are AND'd",
        "  {all_of: [...]} {any_of: [...]} {not: ...}",
        "  {region: {type: box, dims: [...], lo: [...], hi: [...]}}",
        "    also ellipsoid (center/radii) and polygon (vertices, 2-D)",
        "  {$ref: <name from defines>}",
        "",
        'condition: {dim: <feature name or index>, op: ">=", value: <number>,',
        "            target: state|action|reward}",
        f"  ops: {sorted(_OPS)}",
        "  value must be a NUMBER. To compare two features, define the",
        "  difference in `derived:` and threshold that instead, e.g.",
        '  derived: {gap: "p0_row - p1_row"} then dim: gap, op: "==", value: 0.',
        '  QUOTE THE OPERATOR. Write op: ">=" not op: >=. An unquoted > or <',
        "  opens a YAML block scalar and the document will not parse.",
        "",
        "Optional blocks:",
        "  defines:    reusable named groups, referenced with $ref",
        "  derived:    name -> arithmetic expression over feature names,",
        "              e.g. dist: \"abs(agent_x - 13) + abs(agent_y - 3)\"",
        "              functions: abs min max sqrt hypot floor ceil sign",
        "  aggregates: name -> {op: sum|mean|min|max|std|range|first|last|",
        "              abs_sum|n_distinct|count|count_distinct, of: <feature>}",
        "              episode-level; needs match_mode: trajectory",
        "  families:   name -> prefix pattern with {i}, e.g. \"p{i}_\"",
        "              then a derived entry may quantify:",
        "              {over: <family>, arity: 2, reduce: min, expr: ...}",
        "              bindings are named a, b, ...; attributes are a_<attr>",
        "  temporal:   with match_mode: temporal. Formulas nest:",
        "              {eventually: F} {always: F} {next: F} {not: F}",
        "              {and: [F,...]} {or: [F,...]}",
        "              {until: {hold: F, release: F}}",
        "              {consecutive: {k: n, of: F}}",
        "              {then_within: {first: F, second: F, within: w}}",
        "              a non-operator mapping is a state predicate",
    ])


def build_prompt(
    description: str,
    view: Optional[FeatureView] = None,
    env_id: Optional[str] = None,
    previous: Optional[str] = None,
    feedback: Optional[Sequence[str]] = None,
) -> str:
    """Assemble the prompt. On a retry, the previous attempt and the
    validator's complaints about it are included verbatim."""
    parts = [
        "Write a forget-scenario that captures this behaviour:",
        f"  {description.strip()}",
        "",
    ]
    if env_id:
        parts += [f"Environment: {env_id}", ""]
    parts += [describe_view(view), "", grammar_reference()]
    if previous is not None:
        parts += [
            "",
            "Your previous attempt was rejected:",
            "```yaml",
            previous.strip(),
            "```",
            "",
            "Problems found:",
        ]
        parts += [f"  - {f}" for f in (feedback or [])]
        parts += ["", "Emit a corrected scenario. YAML only, no commentary."]
    else:
        parts += ["", "Emit the scenario as YAML only, no commentary."]
    return "\n".join(parts)


@dataclass
class Attempt:
    """One pass through the loop."""

    n: int
    prompt: str
    raw: str
    doc: Optional[Dict[str, Any]] = None
    report: Optional[ValidationReport] = None
    accepted: bool = False
    rejection: List[str] = field(default_factory=list)
    match_rate: Optional[float] = None


@dataclass
class AuthoringResult:
    description: str
    attempts: List[Attempt] = field(default_factory=list)
    scenario: Optional[ForgetScenario] = None

    @property
    def accepted(self) -> bool:
        return self.scenario is not None

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)

    @property
    def repaired(self) -> bool:
        """Accepted, but not on the first try -- the feedback did work."""
        return self.accepted and self.n_attempts > 1

    def render(self) -> str:
        lines = [f"description : {self.description}",
                 f"attempts    : {self.n_attempts}",
                 f"accepted    : {self.accepted}"]
        for a in self.attempts:
            status = "accepted" if a.accepted else "rejected"
            lines.append(f"  [{a.n}] {status}")
            for r in a.rejection:
                lines.append(f"      {r}")
        return "\n".join(lines)


def _evaluate_candidate(
    raw: str,
    view: Optional[FeatureView],
    trajectories: Optional[Sequence[Dict[str, Any]]],
    require_nonvacuous: bool,
    fatal_warnings: frozenset = FATAL_WARNINGS,
) -> tuple[Optional[ForgetScenario], Optional[Dict], Optional[ValidationReport],
           List[str], Optional[float]]:
    """Run a candidate through every gate. Returns the scenario when it
    passes, plus the reasons when it does not."""
    text = _fence_strip(raw)
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return None, None, None, [f"not valid YAML: {exc}"], None
    if not isinstance(doc, dict):
        return None, None, None, ["top level must be a mapping"], None

    report = validate_scenario_dict(doc, view=view)
    if report.errors:
        return None, doc, report, [i.message for i in report.errors], None
    fatal = _fatal_issues(report, doc, fatal_warnings)
    if fatal:
        return None, doc, report, [i.message for i in fatal], None

    try:
        scenario = ForgetScenario(doc, view=view)
    except ScenarioError as exc:
        return None, doc, report, [str(exc)], None

    match_rate = None
    if trajectories:
        cov = coverage(scenario, list(trajectories))
        frac = cov.forget_region_fraction
        match_rate = None if frac != frac else frac    # NaN check
        if require_nonvacuous and match_rate is not None:
            if match_rate == 0.0:
                return (None, doc, report,
                        ["matches no steps in the supplied rollouts, so it "
                         "would produce an empty forget set"], match_rate)
            if match_rate == 1.0:
                return (None, doc, report,
                        ["matches every step in the supplied rollouts, so it "
                         "does not single out any behaviour"], match_rate)
    return scenario, doc, report, [], match_rate


def author_scenario(
    description: str,
    backend: Backend,
    *,
    view: Optional[FeatureView] = None,
    env_id: Optional[str] = None,
    trajectories: Optional[Sequence[Dict[str, Any]]] = None,
    max_attempts: int = 3,
    require_nonvacuous: bool = True,
    fatal_warnings: frozenset = FATAL_WARNINGS,
) -> AuthoringResult:
    """Author a scenario, repairing against validator feedback.

    Supplying `trajectories` turns on the empirical gate, which is the one
    that catches a specification that is well-formed and means nothing.
    """
    result = AuthoringResult(description=description)
    previous: Optional[str] = None
    feedback: List[str] = []

    for n in range(1, max_attempts + 1):
        prompt = build_prompt(description, view=view, env_id=env_id,
                              previous=previous, feedback=feedback)
        raw = backend(prompt)
        scenario, doc, report, reasons, rate = _evaluate_candidate(
            raw, view, trajectories, require_nonvacuous, fatal_warnings)
        attempt = Attempt(n=n, prompt=prompt, raw=raw, doc=doc, report=report,
                          accepted=scenario is not None, rejection=reasons,
                          match_rate=rate)
        result.attempts.append(attempt)
        if scenario is not None:
            result.scenario = scenario
            return result
        previous, feedback = raw, reasons

    return result
