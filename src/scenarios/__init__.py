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
from .authoring import (
    FATAL_WARNINGS,
    Attempt,
    AuthoringResult,
    author_scenario,
    build_prompt,
    describe_view,
    grammar_reference,
)
from .learned import (
    ProbeError,
    ProbeReport,
    fit_probe_view,
)
from .mining import (
    MiningError,
    MiningReport,
    candidate_literals,
    mine_rules,
    mine_scenario,
    rules_to_config,
)
from .corpus import (
    CorpusEntry,
    CorpusError,
    load_corpus,
    resolve_views,
    score_corpus,
)
from .drift import (
    DriftReport,
    compare_masks,
    scenario_masks,
    measure_drift,
    offset_sweep,
)
from .introspect import (
    coverage,
    to_natural_language,
    to_predicate,
)
from .views import (
    FeatureView,
    ViewError,
    available_providers,
    build_view,
    identity_view,
    minigrid_view,
    register_view_provider,
    table_view,
    view_from_cfg,
    describe_domain,
    domain_report,
    provider_cost,
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
    "FATAL_WARNINGS",
    "AuthoringResult",
    "Attempt",
    "author_scenario",
    "build_prompt",
    "describe_view",
    "grammar_reference",
    "ProbeError",
    "ProbeReport",
    "fit_probe_view",
    "MiningError",
    "MiningReport",
    "mine_scenario",
    "mine_rules",
    "rules_to_config",
    "candidate_literals",
    "CorpusEntry",
    "CorpusError",
    "load_corpus",
    "resolve_views",
    "score_corpus",
    "DriftReport",
    "compare_masks",
    "scenario_masks",
    "measure_drift",
    "offset_sweep",
    "to_natural_language",
    "to_predicate",
    "FeatureView",
    "ViewError",
    "build_view",
    "identity_view",
    "table_view",
    "minigrid_view",
    "register_view_provider",
    "available_providers",
    "view_from_cfg",
    "describe_domain",
    "domain_report",
    "provider_cost",
]
