"""Forget Scenario language — declarative description of *what* to unlearn.

Public surface is intentionally identical to the original single-file
`scenarios.py` so that existing callers (`from scenarios import
ForgetScenario, _OPS, ENV_OBS_DIMS, list_scenarios`) keep working
unmodified after the package split.

New entry points exposed by the v1.0 language:

- `validate_scenario(...)` — schema + semantic validation
- `to_natural_language(scenario)` — paper-prose rendering
- `to_predicate(scenario)` — LaTeX predicate-logic rendering
- `coverage(scenario, trajectories)` — empirical match statistics

These are also exposed via the `python -m scenarios` CLI.
"""
from __future__ import annotations

from .constants import (
    ENV_OBS_DIMS,
    ENV_ACTION_DIMS,
    OP_RENDER,
    SCHEMA_VERSION,
    VALID_MATCH_MODES,
    VALID_REGION_TYPES,
    VALID_TARGETS,
    VALID_TEMPORAL_OPERATORS,
    _OPS,
)
from .scenario import (
    ForgetScenario,
    ScenarioError,
    list_scenarios,
)
from .validate import (
    ValidationIssue,
    ValidationReport,
    validate_scenario,
    validate_scenario_dict,
)
from .introspect import (
    coverage,
    to_natural_language,
    to_predicate,
)

__all__ = [
    "ForgetScenario",
    "ScenarioError",
    "ENV_OBS_DIMS",
    "ENV_ACTION_DIMS",
    "OP_RENDER",
    "SCHEMA_VERSION",
    "VALID_MATCH_MODES",
    "VALID_REGION_TYPES",
    "VALID_TARGETS",
    "VALID_TEMPORAL_OPERATORS",
    "_OPS",
    "list_scenarios",
    "ValidationIssue",
    "ValidationReport",
    "validate_scenario",
    "validate_scenario_dict",
    "coverage",
    "to_natural_language",
    "to_predicate",
]
