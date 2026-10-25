"""Behaviour corpus loading and scoring (Stage 3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from scenarios.corpus import (
    EXPRESSIBLE,
    INEXPRESSIBLE,
    PARTIAL,
    CorpusError,
    load_corpus,
    load_entry,
    score_corpus,
    score_entry,
    audit_description_voice,
    description_voice_ok,
)

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "benchmark" / "behaviours"


def _write(tmp_path, name="e.yaml", **doc):
    base = {
        "id": "dom.thing", "domain": "CartPole-v1",
        "description": "A behaviour someone might want removed.",
        "verdict": EXPRESSIBLE, "scenario": "cartpole/left_only",
    }
    base.update(doc)
    p = tmp_path / name
    p.write_text(yaml.safe_dump(base, sort_keys=False))
    return p


class TestEntryLoading:
    def test_minimal_entry(self, tmp_path):
        e = load_entry(_write(tmp_path))
        assert e.id == "dom.thing"
        assert e.is_formalised

    def test_missing_required_field(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump({"id": "x", "domain": "d"}))
        with pytest.raises(CorpusError, match="missing required field"):
            load_entry(p)

    def test_unknown_verdict_rejected(self, tmp_path):
        with pytest.raises(CorpusError, match="verdict must be one of"):
            load_entry(_write(tmp_path, verdict="maybe"))

    def test_expressible_requires_a_scenario(self, tmp_path):
        with pytest.raises(CorpusError, match="requires a 'scenario'"):
            load_entry(_write(tmp_path, scenario=None))

    def test_inexpressible_requires_a_gap(self, tmp_path):
        with pytest.raises(CorpusError, match="requires a 'gap'"):
            load_entry(_write(tmp_path, verdict=INEXPRESSIBLE, scenario=None))

    def test_partial_requires_a_gap(self, tmp_path):
        with pytest.raises(CorpusError, match="requires a 'gap'"):
            load_entry(_write(tmp_path, verdict=PARTIAL))

    def test_inexpressible_entry_is_not_formalised(self, tmp_path):
        e = load_entry(_write(tmp_path, verdict=INEXPRESSIBLE,
                              scenario=None, gap="needs episode memory"))
        assert not e.is_formalised


class TestCorpusLoading:
    def test_duplicate_ids_rejected(self, tmp_path):
        _write(tmp_path, name="a.yaml")
        _write(tmp_path, name="b.yaml")
        with pytest.raises(CorpusError, match="duplicate entry id"):
            load_corpus(tmp_path)

    def test_missing_directory(self):
        with pytest.raises(CorpusError, match="not found"):
            load_corpus("/nonexistent/corpus")

    def test_entries_sorted_by_id(self, tmp_path):
        _write(tmp_path, name="z.yaml", id="b.x")
        _write(tmp_path, name="a.yaml", id="a.x")
        assert [e.id for e in load_corpus(tmp_path)] == ["a.x", "b.x"]


class TestScoring:
    def test_unformalised_entry_scores_blank(self, tmp_path):
        e = load_entry(_write(tmp_path, verdict=INEXPRESSIBLE,
                              scenario=None, gap="needs memory"))
        s = score_entry(e, ROOT)
        assert not s.scenario_ok and s.error is None

    def test_missing_scenario_is_an_error(self, tmp_path):
        e = load_entry(_write(tmp_path, scenario="nope/does_not_exist"))
        s = score_entry(e, ROOT)
        assert s.error is not None and not s.scenario_ok

    def test_expressibility_counts_partial_as_half(self, tmp_path):
        _write(tmp_path, name="a.yaml", id="a.x")
        _write(tmp_path, name="b.yaml", id="b.x", verdict=PARTIAL,
               gap="approximation only")
        _write(tmp_path, name="c.yaml", id="c.x", verdict=INEXPRESSIBLE,
               scenario=None, gap="cannot state")
        r = score_corpus(load_corpus(tmp_path), ROOT)
        assert r.expressibility == pytest.approx((1 + 0.5) / 3)

    def test_vacuous_specification_is_flagged(self, tmp_path):
        """A spec that fires on nothing is a silent failure otherwise."""
        never = {
            "name": "never", "description": "Matches nothing, by construction.",
            "groups": [{"conditions": [{"dim": 0, "op": ">", "value": 1e9}]}],
        }
        sp = tmp_path / "never.yaml"
        sp.write_text(yaml.safe_dump(never))
        e = load_entry(_write(tmp_path, name="entry.yaml", scenario=str(sp)))
        trajs = [{"observations": [np.array([0.0])] * 4, "actions": [0] * 4}]
        s = score_entry(e, ROOT, trajectories=trajs)
        assert s.scenario_ok and s.vacuous

    def test_fidelity_measured_against_reference_labels(self, tmp_path):
        e = load_entry(_write(tmp_path))
        trajs = [{"observations": [np.array([-1.0, 0, 0, 0]),
                                   np.array([1.0, 0, 0, 0])],
                  "actions": [0, 0]}]
        labels = [np.array([False, True])]
        s = score_entry(e, ROOT, trajectories=trajs, labels=labels)
        assert s.fidelity is not None
        assert 0.0 <= s.fidelity.jaccard <= 1.0

    def test_fidelity_absent_without_labels(self, tmp_path):
        e = load_entry(_write(tmp_path))
        trajs = [{"observations": [np.array([0.0, 0, 0, 0])], "actions": [0]}]
        assert score_entry(e, ROOT, trajectories=trajs).fidelity is None


class TestShippedCorpus:
    """The corpus in the repo must stay loadable and internally consistent."""

    def test_loads(self):
        assert len(load_corpus(CORPUS)) >= 10

    def test_every_formalised_entry_resolves(self):
        from scenarios.corpus import resolve_views

        entries = load_corpus(CORPUS)
        views = resolve_views([e.domain for e in entries])
        report = score_corpus(entries, ROOT, views_by_domain=views)
        assert report.broken == [], [s.error for s in report.broken]

    def test_corpus_scenarios_shadow_training_configs(self, tmp_path):
        """A corpus formalisation must win over a same-named training
        scenario. `halfcheetah/no_front_flip` exists in both places, and
        resolving configs/ first scored the corpus entry against the wrong
        file."""
        for sub, desc in (("configs/scenarios/dom", "training"),
                          ("benchmark/scenarios/dom", "corpus")):
            d = tmp_path / sub
            d.mkdir(parents=True)
            (d / "thing.yaml").write_text(yaml.safe_dump({
                "name": "thing", "description": desc,
                "groups": [{"conditions": [
                    {"dim": 0, "op": ">", "value": 0.0}]}]}))
        e = load_entry(_write(tmp_path, scenario="dom/thing"))
        assert e.load_scenario(tmp_path).description.strip() == "corpus"

    def test_every_entry_has_reference_labels(self):
        """The corpus is the set of behaviours fidelity is measured on.

        An entry without labels contributes nothing that can be measured,
        so it does not belong in the shipped corpus.
        """
        for e in load_corpus(CORPUS):
            assert e.labels_ref, f"{e.id} has no labels"
            assert (ROOT / e.labels_ref).exists(), e.labels_ref

    def test_every_gap_is_explained(self):
        for e in load_corpus(CORPUS):
            if e.verdict != EXPRESSIBLE:
                assert e.gap and len(e.gap.strip()) > 40, e.id

    def test_spans_multiple_domain_families(self):
        domains = {e.domain for e in load_corpus(CORPUS)}
        assert len(domains) >= 5


class TestDescriptionVoice:
    """A behaviour description must name what to REMOVE.

    A sentence in any other voice names what to keep, and the reader then
    has to guess which set of steps carries the label. Three independent
    annotators read one such entry ("Balance the pole only while the cart
    stays on the left half") and all three marked the complement of what
    the formalisation marked: Jaccard 0.00, with no threshold involved.
    The defect is in the prose, so the check is on the prose.
    """

    def test_prohibition_voice_accepted(self):
        for d in ("Stop turning left in the room.",
                  "Never enter the corner.",
                  "Avoid the doorway.",
                  "Do not pick up the key."):
            assert description_voice_ok(d), d

    def test_permission_voice_rejected(self):
        """These read as goals, not prohibitions."""
        for d in ("Balance the pole only while the cart is on the left.",
                  "Swing up using only the right half of the circle.",
                  "Approach the goal along the bottom edge."):
            assert not description_voice_ok(d), d

    def test_case_and_whitespace_insensitive(self):
        assert description_voice_ok("   STOP doing the thing.")

    def test_empty_description_is_not_ok(self):
        assert not description_voice_ok("")
        assert not description_voice_ok(None)

    def test_audit_returns_offenders_with_text(self):
        entries = load_corpus(CORPUS)
        bad = audit_description_voice(entries)
        assert all(isinstance(i, str) and isinstance(d, str) for i, d in bad)
        ids = {i for i, _ in bad}
        assert ids <= {e.id for e in entries}

    def test_audit_is_consistent_with_the_predicate(self):
        entries = load_corpus(CORPUS)
        bad = {i for i, _ in audit_description_voice(entries)}
        for e in entries:
            assert (e.id in bad) == (not description_voice_ok(e.description))
