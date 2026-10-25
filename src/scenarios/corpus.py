"""Behaviour corpus: the benchmark the language is evaluated on.

A specification language is only as good as the range of behaviours it can
state, and the only honest way to show that range is a fixed corpus of
behaviours written down *before* the formalisation is attempted.

Each entry pairs a natural-language description of a behaviour someone
might want removed with an attempt to formalise it, and records the
verdict:

    expressible     the grammar states it exactly
    partial         the grammar states an approximation; `gap` says what
                    is lost
    inexpressible   the grammar cannot state it; `gap` says why

Recording failures is the point. A corpus of only expressible behaviours
measures nothing, and the inexpressible entries are the roadmap for the
next version of the grammar.

Three things are scored:

* **expressibility** — what fraction of behaviours the grammar can state;
* **coverage** — whether a formalised behaviour actually fires on real
  rollouts, catching specifications that are vacuous or match everything;
* **fidelity** — agreement between the specification and reference labels,
  where those exist. Fidelity reuses the drift statistics, because
  comparing a spec to human labels is the same computation as comparing
  two specs.

Fidelity is only reported for entries that carry labels. An entry without
them is not scored as perfect; it is scored as unmeasured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import re

import numpy as np
import yaml

from .drift import DriftReport, compare_masks, scenario_masks
from .introspect import coverage
from .scenario import ForgetScenario, ScenarioError
from .views import FeatureView

EXPRESSIBLE = "expressible"
PARTIAL = "partial"
INEXPRESSIBLE = "inexpressible"
VERDICTS = (EXPRESSIBLE, PARTIAL, INEXPRESSIBLE)


class CorpusError(ValueError):
    """Raised for a malformed corpus entry."""


@dataclass
class CorpusEntry:
    """One behaviour someone might want an agent to forget."""

    id: str
    domain: str
    description: str
    verdict: str
    scenario_ref: Optional[str] = None
    gap: Optional[str] = None
    tags: tuple[str, ...] = ()
    labels_ref: Optional[str] = None
    source: Optional[str] = None
    path: Optional[Path] = None

    @property
    def is_formalised(self) -> bool:
        return self.verdict in (EXPRESSIBLE, PARTIAL) and bool(self.scenario_ref)

    def load_scenario(
        self, root: Path, view: Optional[FeatureView] = None,
    ) -> ForgetScenario:
        if not self.scenario_ref:
            raise CorpusError(f"entry {self.id!r} has no scenario to load")
        # The corpus's own formalisations come first. A training scenario
        # under configs/ may share a name with one written for an entry,
        # and must not shadow it.
        candidates = [
            root / "benchmark" / "scenarios" / f"{self.scenario_ref}.yaml",
            root / "configs" / "scenarios" / f"{self.scenario_ref}.yaml",
            root / self.scenario_ref,
            Path(self.scenario_ref),
        ]
        for c in candidates:
            if c.exists():
                return ForgetScenario.load(str(c), view=view)
        raise CorpusError(
            f"entry {self.id!r}: scenario {self.scenario_ref!r} not found "
            f"(looked in {[str(c) for c in candidates]})"
        )


def _require(doc: Dict[str, Any], key: str, path: Path) -> Any:
    if key not in doc:
        raise CorpusError(f"{path}: missing required field {key!r}")
    return doc[key]


def load_entry(path: Path) -> CorpusEntry:
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    if not isinstance(doc, dict):
        raise CorpusError(f"{path}: entry must be a mapping")

    verdict = _require(doc, "verdict", path)
    if verdict not in VERDICTS:
        raise CorpusError(
            f"{path}: verdict must be one of {VERDICTS}; got {verdict!r}")

    entry = CorpusEntry(
        id=_require(doc, "id", path),
        domain=_require(doc, "domain", path),
        description=_require(doc, "description", path),
        verdict=verdict,
        scenario_ref=doc.get("scenario"),
        gap=doc.get("gap"),
        tags=tuple(doc.get("tags") or ()),
        labels_ref=doc.get("labels"),
        source=doc.get("source"),
        path=path,
    )

    if verdict == EXPRESSIBLE and not entry.scenario_ref:
        raise CorpusError(
            f"{path}: verdict 'expressible' requires a 'scenario' reference")
    if verdict in (PARTIAL, INEXPRESSIBLE) and not entry.gap:
        raise CorpusError(
            f"{path}: verdict {verdict!r} requires a 'gap' explaining what "
            "the grammar cannot state")
    return entry


def load_corpus(directory: Path | str) -> List[CorpusEntry]:
    """Load every entry under `directory`, sorted by id."""
    root = Path(directory)
    if not root.exists():
        raise CorpusError(f"corpus directory not found: {root}")
    entries = [load_entry(p) for p in sorted(root.rglob("*.yaml"))]
    seen: Dict[str, Path] = {}
    for e in entries:
        if e.id in seen:
            raise CorpusError(
                f"duplicate entry id {e.id!r} in {e.path} and {seen[e.id]}")
        seen[e.id] = e.path
    return sorted(entries, key=lambda e: e.id)


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------


@dataclass
class EntryScore:
    """What could be measured about one entry."""

    entry: CorpusEntry
    scenario_ok: bool = False
    error: Optional[str] = None
    match_rate: Optional[float] = None       # fraction of steps matched
    traj_rate: Optional[float] = None        # fraction of trajectories matched
    fidelity: Optional[DriftReport] = None

    @property
    def vacuous(self) -> bool:
        """Matches nothing on the rollouts it was checked against."""
        return self.match_rate is not None and self.match_rate == 0.0

    @property
    def saturated(self) -> bool:
        """Matches every step -- almost always a mis-specification."""
        return self.match_rate is not None and self.match_rate == 1.0


# A behaviour description names what to REMOVE. A sentence in any other
# voice ("Balance only while ...", "Approach the goal along ...") names
# what to KEEP, and a reader then has to guess which set of steps the
# label marks. Measured on three independent readings of one such entry,
# all three marked the complement of what the formalisation marked:
# agreement 0.00, with no threshold involved at all. The check is a lint
# on the prose, not on the grammar, which is precisely where that defect
# lives.
PROHIBITION_VOICE = re.compile(
    r"^\s*(stop|never|don'?t|do not|avoid|no longer|forget|prevent|"
    r"cease|refrain)\b", re.IGNORECASE)


def description_voice_ok(description: str) -> bool:
    """True when the description is phrased as a prohibition."""
    return bool(PROHIBITION_VOICE.match(description or ""))


def audit_description_voice(entries) -> List[Tuple[str, str]]:
    """Entries whose description does not name the behaviour to remove.

    Returns (id, description) pairs for the offenders, so a corpus can be
    checked the way a linter checks code.
    """
    return [(e.id, " ".join((e.description or "").split()))
            for e in entries if not description_voice_ok(e.description)]


def score_entry(
    entry: CorpusEntry,
    root: Path,
    trajectories: Optional[Sequence[Dict[str, Any]]] = None,
    view: Optional[FeatureView] = None,
    labels: Optional[Sequence[np.ndarray]] = None,
) -> EntryScore:
    """Score one entry: does it load, does it fire, does it agree with
    reference labels."""
    score = EntryScore(entry=entry)
    if not entry.is_formalised:
        return score
    try:
        scenario = entry.load_scenario(root, view=view)
        score.scenario_ok = True
    except (CorpusError, ScenarioError, FileNotFoundError) as exc:
        score.error = f"{type(exc).__name__}: {exc}"
        return score

    if trajectories:
        rep = coverage(scenario, list(trajectories))
        score.traj_rate = rep.forget_traj_fraction
        frac = rep.forget_region_fraction
        score.match_rate = None if frac != frac else frac  # NaN check
        if labels is not None:
            score.fidelity = compare_masks(
                labels, scenario_masks(scenario, list(trajectories)))
    return score


@dataclass
class CorpusReport:
    scores: List[EntryScore] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.scores)

    def count(self, verdict: str) -> int:
        return sum(1 for s in self.scores if s.entry.verdict == verdict)

    @property
    def expressibility(self) -> float:
        """Fraction the grammar states exactly. Partial entries count as
        half, since they state an approximation."""
        if not self.scores:
            return 0.0
        return (self.count(EXPRESSIBLE) + 0.5 * self.count(PARTIAL)) / self.n

    @property
    def broken(self) -> List[EntryScore]:
        return [s for s in self.scores if s.error]

    @property
    def vacuous(self) -> List[EntryScore]:
        return [s for s in self.scores if s.vacuous]

    @property
    def measured_fidelity(self) -> List[EntryScore]:
        return [s for s in self.scores if s.fidelity is not None]

    def by_domain(self) -> Dict[str, List[EntryScore]]:
        out: Dict[str, List[EntryScore]] = {}
        for s in self.scores:
            out.setdefault(s.entry.domain, []).append(s)
        return out

    def render(self) -> str:
        lines = [
            f"entries            {self.n}",
            f"  expressible      {self.count(EXPRESSIBLE)}",
            f"  partial          {self.count(PARTIAL)}",
            f"  inexpressible    {self.count(INEXPRESSIBLE)}",
            f"expressibility     {self.expressibility:.3f}",
        ]
        if self.broken:
            lines.append(f"BROKEN             {len(self.broken)}")
            for s in self.broken:
                lines.append(f"    {s.entry.id}: {s.error}")
        if self.vacuous:
            lines.append(f"vacuous (match 0)  {len(self.vacuous)}")
            for s in self.vacuous:
                lines.append(f"    {s.entry.id}")
        measured = self.measured_fidelity
        lines.append(
            f"fidelity measured  {len(measured)}/{self.n}"
            + ("" if measured else "   (no reference labels yet)")
        )
        return "\n".join(lines)


def score_corpus(
    entries: Sequence[CorpusEntry],
    root: Path,
    trajectories_by_domain: Optional[Dict[str, Sequence[Dict[str, Any]]]] = None,
    views_by_domain: Optional[Dict[str, FeatureView]] = None,
) -> CorpusReport:
    trajectories_by_domain = trajectories_by_domain or {}
    views_by_domain = views_by_domain or {}

    def labels_for(entry):
        """Reference labels and the steps they were assigned to.

        A label file records the per-step feature values the annotator
        saw, so the scenario is replayed on exactly those steps. Scoring
        labels against separately collected rollouts would compare two
        different trajectories and call the difference drift.

        A missing or unfinished label file is not an error: most entries
        have none yet, and fidelity is simply not reported for them.
        """
        if not entry.labels_ref:
            return None, None
        from .labels import LabelError, resolve_labels
        try:
            ls = resolve_labels(entry, root)
        except LabelError:
            return None, None
        if ls is None:
            return None, None
        doc = yaml.safe_load((Path(root) / entry.labels_ref).read_text())
        names = doc.get("features") or []
        trajs = [{"features": [[float(st[n]) for n in names]
                               for st in ep["steps"]],
                  "actions": [st.get("action") for st in ep["steps"]]}
                 for ep in doc["episodes"]]
        return ls.as_masks(), trajs

    scores = []
    for e in entries:
        labels, label_trajs = labels_for(e)
        scores.append(score_entry(
            e, root,
            trajectories=label_trajs or trajectories_by_domain.get(e.domain),
            view=views_by_domain.get(e.domain),
            labels=labels,
        ))
    return CorpusReport(scores=scores)


# ----------------------------------------------------------------------
# View resolution for a corpus
# ----------------------------------------------------------------------

# Contrib providers are registered on demand: the core must not import a
# simulator just because a corpus entry mentions one.
_CONTRIB_MODULES = ("scenarios.contrib.lbf",)


def register_contrib_providers() -> List[str]:
    """Best-effort registration of optional domain adapters."""
    import importlib

    registered: List[str] = []
    for mod_name in _CONTRIB_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            mod.register()
            registered.append(mod_name)
        except Exception:
            continue
    return registered


def resolve_views(domains: Sequence[str]) -> Dict[str, Optional[FeatureView]]:
    """Build a feature view per domain, skipping domains whose simulator
    is unavailable.

    An entry whose view cannot be resolved is not silently downgraded: it
    is scored with `view=None`, which surfaces as a load error if the
    scenario references names only that view supplies.
    """
    from .views import build_view

    register_contrib_providers()
    out: Dict[str, Optional[FeatureView]] = {}
    for domain in sorted(set(domains)):
        env = None
        try:
            import gymnasium as gym

            if domain.startswith("MiniGrid-"):
                import minigrid  # noqa: F401
            elif domain.startswith("Foraging-"):
                import lbforaging  # noqa: F401
            env = gym.make(domain)
            obs, info = env.reset(seed=0)
            shape = getattr(env.observation_space, "shape", None)
            n_dims = int(shape[0]) if shape and len(shape) == 1 else None
            out[domain] = build_view(env=env, env_id=domain, n_dims=n_dims,
                                     probe={"obs": obs, "info": info})
        except Exception:
            out[domain] = None
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
    return out
