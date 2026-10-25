"""`applies_to`: which steps of a matched trajectory carry the forget label.

The ambiguity this resolves was measured at 17.4 percentage points of
reported forget-effectiveness on trained FourRooms policies -- larger than
the effect sizes the surrounding study reports.
"""

from __future__ import annotations

import numpy as np
import pytest

from scenarios import ForgetScenario, ScenarioError, validate_scenario_dict
from scenarios.scenario import (
    MATCHING_STEPS,
    TRAJECTORY_LEVEL_MODES,
    WHOLE_TRAJECTORY,
)


def _spec(**over):
    spec = {
        "name": "scope", "description": "Step-scope behaviour under test.",
        "match_mode": "proportion", "match_threshold": 0.15,
        "groups": [{"conditions": [{"dim": 0, "op": ">", "value": 5}]}],
    }
    spec.update(over)
    return spec


def _traj(values):
    feats = [np.array([float(v)]) for v in values]
    return {"observations": feats, "actions": [0] * len(feats)}


class TestDeclaration:
    def test_defaults_to_matching_steps(self):
        assert ForgetScenario(_spec()).applies_to == MATCHING_STEPS

    def test_default_is_not_recorded_as_declared(self):
        assert ForgetScenario(_spec()).applies_to_declared is False

    def test_explicit_value_is_recorded(self):
        s = ForgetScenario(_spec(applies_to=WHOLE_TRAJECTORY))
        assert s.applies_to == WHOLE_TRAJECTORY
        assert s.applies_to_declared is True

    def test_unknown_scope_rejected(self):
        with pytest.raises(ScenarioError, match="applies_to must be one of"):
            ForgetScenario(_spec(applies_to="sometimes"))


class TestAmbiguityDetection:
    @pytest.mark.parametrize("mode", sorted(TRAJECTORY_LEVEL_MODES - {"temporal"}))
    def test_trajectory_modes_without_declaration_are_ambiguous(self, mode):
        spec = _spec(match_mode=mode)
        if mode == "trajectory":
            spec["groups"] = [{"conditions": [
                {"target": "return", "op": ">", "value": 1}]}]
        assert ForgetScenario(spec).step_scope_is_ambiguous

    def test_declaring_resolves_ambiguity(self):
        assert not ForgetScenario(
            _spec(applies_to=MATCHING_STEPS)).step_scope_is_ambiguous

    def test_step_level_mode_is_never_ambiguous(self):
        assert not ForgetScenario(
            _spec(match_mode="any_step")).step_scope_is_ambiguous

    def test_validator_warns_on_ambiguity(self):
        rep = validate_scenario_dict(_spec())
        assert "step_scope_ambiguous" in {i.code for i in rep.warnings}

    def test_validator_quiet_once_declared(self):
        rep = validate_scenario_dict(_spec(applies_to=MATCHING_STEPS))
        assert "step_scope_ambiguous" not in {i.code for i in rep.warnings}


class TestStepLabels:
    # 2 of 10 steps in region -> 0.2 >= threshold 0.15, so the trajectory
    # is selected. The two readings then disagree on 8 steps.
    TRAJ = _traj([0, 0, 9, 0, 0, 9, 0, 0, 0, 0])

    def test_matching_steps_labels_only_the_predicate(self):
        s = ForgetScenario(_spec(applies_to=MATCHING_STEPS))
        assert s.step_labels(self.TRAJ).sum() == 2

    def test_whole_trajectory_labels_every_step(self):
        s = ForgetScenario(_spec(applies_to=WHOLE_TRAJECTORY))
        assert s.step_labels(self.TRAJ).sum() == 10

    def test_the_two_readings_disagree(self):
        a = ForgetScenario(_spec(applies_to=MATCHING_STEPS)).step_labels(self.TRAJ)
        b = ForgetScenario(_spec(applies_to=WHOLE_TRAJECTORY)).step_labels(self.TRAJ)
        assert int((a != b).sum()) == 8

    def test_whole_trajectory_labels_nothing_when_unselected(self):
        s = ForgetScenario(_spec(applies_to=WHOLE_TRAJECTORY))
        assert s.step_labels(_traj([0] * 20 + [9])).sum() == 0

    def test_matching_steps_still_labels_when_unselected(self):
        """Step scope and trajectory selection are independent axes."""
        s = ForgetScenario(_spec(applies_to=MATCHING_STEPS))
        assert s.step_labels(_traj([0] * 20 + [9])).sum() == 1

    def test_label_length_matches_action_count(self):
        s = ForgetScenario(_spec())
        assert len(s.step_labels(self.TRAJ)) == len(self.TRAJ["actions"])

    def test_recorded_features_are_preferred(self):
        traj = {"observations": [np.array([0.0])] * 3,
                "features": [np.array([9.0])] * 3,
                "actions": [0, 0, 0]}
        assert ForgetScenario(_spec()).step_labels(traj).sum() == 3


class TestShippedScenarios:
    def test_no_bundled_scenario_is_ambiguous(self):
        """Regression: 18 of 21 were, before applies_to existed."""
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "configs" / "scenarios"
        bad = [p.name for p in sorted(root.rglob("*.yaml"))
               if ForgetScenario.load(str(p)).step_scope_is_ambiguous]
        assert bad == []


class TestEpisodeSegmentation:
    """A rollout buffer is many episodes concatenated. A trajectory-level
    mode must be decided per episode, not across the whole buffer.

    Without this, `whole_trajectory` on a 2048-step buffer asks "did 15%
    of these 2048 steps match", which is nearly always true, and labels
    every step -- silently a different specification from the declared one.
    """

    def _scen(self, scope):
        return ForgetScenario({
            "name": "seg", "description": "Proportion mode, two readings.",
            "match_mode": "proportion", "match_threshold": 0.5,
            "applies_to": scope,
            "groups": [{"conditions": [{"dim": 0, "op": ">", "value": 0.5}]}],
        })

    def _traj(self):
        # Episode A: 3 of 4 steps match (0.75 >= 0.5) -> selected.
        # Episode B: 1 of 4 steps match (0.25 <  0.5) -> not selected.
        feats = [[1.0], [1.0], [1.0], [0.0],
                 [1.0], [0.0], [0.0], [0.0]]
        dones = [False, False, False, True,
                 False, False, False, True]
        return {"features": feats, "actions": [None] * 8, "dones": dones}

    def test_segments_on_done_flags(self):
        got = self._scen(WHOLE_TRAJECTORY).step_labels(self._traj())
        assert list(got) == [True] * 4 + [False] * 4

    def test_without_dones_the_whole_buffer_is_one_trajectory(self):
        traj = self._traj()
        del traj["dones"]
        got = self._scen(WHOLE_TRAJECTORY).step_labels(traj)
        # 4/8 = 0.5 >= 0.5, so the entire buffer is selected: the reading
        # that motivated segmentation.
        assert got.all()

    def test_matching_steps_is_unaffected_by_segmentation(self):
        traj = self._traj()
        with_d = self._scen(MATCHING_STEPS).step_labels(traj)
        del traj["dones"]
        without_d = self._scen(MATCHING_STEPS).step_labels(traj)
        assert list(with_d) == list(without_d)
        assert list(with_d) == [True, True, True, False,
                                True, False, False, False]

    def test_final_episode_without_a_done_flag_still_closes(self):
        traj = self._traj()
        traj["dones"][-1] = False          # buffer cut mid-episode
        got = self._scen(WHOLE_TRAJECTORY).step_labels(traj)
        assert list(got[:4]) == [True] * 4
        assert list(got[4:]) == [False] * 4

    def test_segmentation_changes_the_label_count(self):
        traj = self._traj()
        seg = self._scen(WHOLE_TRAJECTORY).step_labels(traj).mean()
        del traj["dones"]
        unseg = self._scen(WHOLE_TRAJECTORY).step_labels(traj).mean()
        assert seg == 0.5 and unseg == 1.0


class TestCoverageUsesFeatures:
    """`coverage` receives already-extracted features.

    Re-running the view on them is a no-op for an identity view, which is
    why this went unnoticed, and an error for a view that reads the
    environment -- the env has moved on by the time a rollout is scored.
    """

    def _env_reading_view(self):
        from scenarios.views import FeatureView

        def extract(obs, info, env):
            if env is None:
                raise AssertionError("view was called without an env")
            return [float(obs[0])]

        return FeatureView(
            names=("x",), extract_fn=extract, dtypes=("continuous",),
            ranges=((0.0, 10.0),), provenance="auto:test",
            requires=("obs", "env"))

    def test_does_not_re_extract(self):
        from scenarios.introspect import coverage

        scen = ForgetScenario({
            "name": "c", "description": "Coverage over a view needing env.",
            "groups": [{"conditions": [{"dim": "x", "op": ">", "value": 1.0}]}],
        }, view=self._env_reading_view())
        traj = {"features": [[0.0], [2.0], [3.0]],
                "actions": [None, None, None]}
        rep = coverage(scen, [traj])
        assert rep.forget_region_fraction == pytest.approx(2 / 3)


class TestTemporalStepLabels:
    """A temporal formula is evaluated AT a position, so a step label is
    whether it holds there.

    Before this was wired up, `match_mode: temporal` with
    `applies_to: matching_steps` fell back to the state-level `groups`
    predicate, silently discarding the temporal structure. The corpus
    entry `no_corner_turn_reflex` read "turn left immediately on entering
    the room" and labelled turning left ANYWHERE in the room: about
    twenty times as many steps. Nothing in the suite covered this path,
    which is how it survived.
    """

    def _scenario(self, temporal):
        return ForgetScenario({
            "name": "t", "description": "Turn left on entering the region.",
            "match_mode": "temporal", "applies_to": MATCHING_STEPS,
            "temporal": temporal,
            # deliberately broad: if the fallback ever returns, this
            # labels every in-region step and the tests below fail
            "groups": [{"conditions": [{"dim": 0, "op": ">", "value": 9.5}]}],
        })

    def _traj(self):
        # outside, outside, ENTER (turn left), inside (turn left), inside
        return {"features": [[5.0], [8.0], [12.0], [13.0], [14.0]],
                "actions": [2, 2, 0, 0, 2]}

    ENTER_AND_TURN = {"and": [
        {"conditions": [{"dim": 0, "op": ">", "value": 9.5},
                        {"dim": 0, "op": "==", "value": 0, "target": "action"}]},
        {"previous": {"conditions": [{"dim": 0, "op": "<", "value": 9.5}]}}]}

    def test_labels_only_the_entry_step(self):
        got = self._scenario(self.ENTER_AND_TURN).step_labels(self._traj())
        assert list(got) == [False, False, True, False, False]

    def test_does_not_fall_back_to_the_state_predicate(self):
        """The broad `groups` predicate would label steps 2, 3 and 4."""
        got = self._scenario(self.ENTER_AND_TURN).step_labels(self._traj())
        assert int(got.sum()) == 1

    def test_next_anchors_one_step_earlier(self):
        """Shows why `previous` was needed: the same rule written with
        `next` fires on the step before the entry, which is a different
        transition."""
        f = {"and": [
            {"conditions": [{"dim": 0, "op": "<", "value": 9.5}]},
            {"next": {"conditions": [
                {"dim": 0, "op": ">", "value": 9.5},
                {"dim": 0, "op": "==", "value": 0, "target": "action"}]}}]}
        got = self._scenario(f).step_labels(self._traj())
        assert list(got) == [False, True, False, False, False]

    def test_whole_trajectory_still_labels_everything(self):
        """Trajectory-level matching anchors at position 0, so a pattern
        occurring mid-episode must be wrapped in `eventually`. Step-level
        labelling needs no such wrapper because it asks the formula at
        every position; the asymmetry is worth knowing about."""
        s = ForgetScenario({
            "name": "t", "description": "Turn left on entering the region.",
            "match_mode": "temporal", "applies_to": WHOLE_TRAJECTORY,
            "temporal": {"eventually": self.ENTER_AND_TURN},
            "groups": [{"conditions": [{"dim": 0, "op": ">", "value": 9.5}]}],
        })
        assert s.step_labels(self._traj()).all()

    def test_unanchored_pattern_does_not_match_a_trajectory(self):
        """Same formula without `eventually`: it does not hold at step 0,
        so the trajectory is not selected at all."""
        s = ForgetScenario({
            "name": "t", "description": "Turn left on entering the region.",
            "match_mode": "temporal", "applies_to": WHOLE_TRAJECTORY,
            "temporal": self.ENTER_AND_TURN,
            "groups": [{"conditions": [{"dim": 0, "op": ">", "value": 9.5}]}],
        })
        assert not s.step_labels(self._traj()).any()

    def test_no_match_labels_nothing(self):
        traj = self._traj()
        traj["actions"] = [2, 2, 2, 2, 2]          # never turns left
        got = self._scenario(self.ENTER_AND_TURN).step_labels(traj)
        assert not got.any()
