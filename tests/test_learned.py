"""Learned feature views fitted from data (stage 6).

The tier that makes the language applicable to a domain nobody has
written an adapter for -- and the fidelity gate that stops it being an
escape hatch.
"""

from __future__ import annotations

import numpy as np
import pytest

from scenarios import ForgetScenario
from scenarios.learned import ProbeError, fit_probe_view
from scenarios.views import ORDINAL


def _linear_task(n=300, d=12, seed=0, noise=0.0):
    """Targets that ARE a linear function of the inputs, so a correct
    probe must recover them."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    W = rng.normal(size=(d, 2))
    Y = X @ W
    if noise:
        Y = Y + rng.normal(scale=noise, size=Y.shape)
    return X, Y


def _unrecoverable(n=300, d=6, seed=0):
    """Targets independent of the inputs: no probe can do better than
    chance, which is exactly what the gate must catch."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)), rng.normal(size=(n, 2))


class TestFitting:
    def test_recovers_a_linear_map(self):
        X, Y = _linear_task()
        _, rep = fit_probe_view(X, Y, ["a", "b"], ridge_alpha=1e-6)
        assert rep.fidelity > 0.99

    def test_provenance_marks_it_learned(self):
        X, Y = _linear_task()
        view, _ = fit_probe_view(X, Y, ["a", "b"])
        assert view.provenance == "learned:probe"

    def test_view_carries_its_fidelity(self):
        X, Y = _linear_task()
        view, rep = fit_probe_view(X, Y, ["a", "b"], ridge_alpha=1e-6)
        assert view.fidelity == pytest.approx(rep.fidelity)

    def test_extract_produces_the_declared_width(self):
        X, Y = _linear_task()
        view, _ = fit_probe_view(X, Y, ["a", "b"])
        assert view.extract(obs=X[0]).shape == (2,)

    def test_ordinal_targets_are_rounded(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 5))
        Y = np.rint(X[:, :2] * 3)
        view, _ = fit_probe_view(X, Y, ["a", "b"], dtypes=(ORDINAL, ORDINAL),
                                 ridge_alpha=1e-6)
        out = view.extract(obs=X[0])
        assert np.allclose(out, np.rint(out))

    def test_scored_out_of_sample(self):
        """A probe that memorises must not score well on held-out data."""
        X, Y = _unrecoverable(n=40, d=60)      # more inputs than samples
        _, rep = fit_probe_view(X, Y, ["a", "b"], ridge_alpha=1e-8)
        assert rep.fidelity < 0.5

    def test_ranges_are_carried_through(self):
        X, Y = _linear_task()
        view, _ = fit_probe_view(X, Y, ["a", "b"], ranges=((0., 1.), (0., 1.)))
        assert view.ranges[0] == (0.0, 1.0)


class TestFidelityGate:
    def test_refuses_an_unrecoverable_target(self):
        X, Y = _unrecoverable()
        with pytest.raises(ProbeError, match="below the required"):
            fit_probe_view(X, Y, ["a", "b"], min_fidelity=0.9)

    def test_error_names_the_weak_features(self):
        X, Y = _unrecoverable()
        with pytest.raises(ProbeError, match="weak features"):
            fit_probe_view(X, Y, ["a", "b"], min_fidelity=0.9)

    def test_accepts_when_the_target_is_recoverable(self):
        X, Y = _linear_task()
        view, _ = fit_probe_view(X, Y, ["a", "b"], min_fidelity=0.9,
                                 ridge_alpha=1e-6)
        assert view.fidelity > 0.9

    def test_no_gate_means_no_refusal(self):
        X, Y = _unrecoverable()
        view, _ = fit_probe_view(X, Y, ["a", "b"])
        assert view.fidelity < 0.5          # returned, but honestly labelled

    def test_fidelity_is_the_weakest_feature(self):
        """One good feature must not mask a useless one."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 8))
        Y = np.column_stack([X[:, 0] * 2.0, rng.normal(size=300)])
        _, rep = fit_probe_view(X, Y, ["good", "noise"], ridge_alpha=1e-6)
        assert rep.per_feature["good"] > 0.9
        assert rep.per_feature["noise"] < 0.5
        assert rep.fidelity == pytest.approx(rep.per_feature["noise"])


class TestValidation:
    def test_rejects_mismatched_row_counts(self):
        with pytest.raises(ProbeError, match="target rows"):
            fit_probe_view(np.zeros((10, 3)), np.zeros((9, 2)), ["a", "b"])

    def test_rejects_wrong_number_of_names(self):
        with pytest.raises(ProbeError, match="target columns"):
            fit_probe_view(np.zeros((10, 3)), np.zeros((10, 2)), ["a"])

    def test_rejects_too_few_samples(self):
        with pytest.raises(ProbeError, match="at least 4"):
            fit_probe_view(np.zeros((3, 2)), np.zeros((3, 1)), ["a"])

    def test_rejects_non_2d_inputs(self):
        with pytest.raises(ProbeError, match="2-D"):
            fit_probe_view(np.zeros(10), np.zeros((10, 1)), ["a"])

    def test_constant_target_is_scored_as_recovered(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 4))
        Y = np.ones((50, 1))
        _, rep = fit_probe_view(X, Y, ["c"], ridge_alpha=1e-6)
        assert rep.fidelity > 0.9


class TestScenarioOnLearnedView:
    def test_scenario_binds_to_a_probe_view(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, 6))
        Y = np.column_stack([X[:, 0] * 5.0, X[:, 1] * 5.0])
        view, _ = fit_probe_view(X, Y, ["agent_x", "agent_y"],
                                 ridge_alpha=1e-6, min_fidelity=0.9)
        spec = {
            "name": "learned", "description": "Spatial rule on a learned view.",
            "groups": [{"conditions": [
                {"dim": "agent_x", "op": ">", "value": 0.0}]}],
        }
        s = ForgetScenario(spec, view=view)
        assert s.groups[0]["conditions"][0]["dim"] == 0

    def test_predictions_track_the_ground_truth(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(600, 6))
        Y = np.column_stack([X[:, 0] * 5.0, X[:, 1] * 5.0])
        view, _ = fit_probe_view(X, Y, ["agent_x", "agent_y"],
                                 ridge_alpha=1e-6, min_fidelity=0.9)
        spec = {
            "name": "learned", "description": "Spatial rule on a learned view.",
            "groups": [{"conditions": [
                {"dim": "agent_x", "op": ">", "value": 2.0}]}],
        }
        s = ForgetScenario(spec, view=view)
        agree = sum(bool(s.matches_state(x)) == bool(y[0] > 2.0)
                    for x, y in zip(X[:200], Y[:200]))
        assert agree / 200 > 0.95
