"""
Comprehensive test suite for unlearning metrics.

Covers:
- AUFC: monotonic scores, edge cases, normalization
- RSI: positive/negative/near-zero baselines, edge cases
- Selectivity: boundary values, multiplicative property
- Forget effectiveness: relative to baseline, threshold sensitivity,
  collapse vs selective forgetting
- Privacy AUC: perfect/random/reversed attacks
- Trajectory similarity: shape, range, sensitivity to weight changes
- Membership inference attack: output structure, label correctness
- compute_all_metrics: completeness, optional params
"""
import pytest
import numpy as np
import torch

from agents import PPOAgent
from metrics import UnlearningMetrics, evaluate_trajectory_similarity, membership_inference_attack


# ---------------------------------------------------------------------------
# AUFC
# ---------------------------------------------------------------------------

class TestAUFC:
    def test_decreasing_scores(self):
        m = UnlearningMetrics()
        scores = [0.9, 0.7, 0.5, 0.3, 0.1]
        aufc = m.compute_aufc(scores)
        assert 0 < aufc < 1

    def test_constant_scores(self):
        m = UnlearningMetrics()
        scores = [0.5, 0.5, 0.5, 0.5]
        aufc = m.compute_aufc(scores)
        assert aufc == pytest.approx(0.5, abs=0.01)

    def test_increasing_scores_higher_than_decreasing(self):
        """Increasing scores (high area) should have higher AUFC than
        decreasing scores (low area), or they can be equal if symmetric."""
        m = UnlearningMetrics()
        increasing = [0.1, 0.3, 0.5, 0.7, 0.9]
        decreasing = [0.9, 0.7, 0.5, 0.3, 0.1]
        aufc_inc = m.compute_aufc(increasing)
        aufc_dec = m.compute_aufc(decreasing)
        # With uniform timesteps and symmetric scores, AUFCs may be equal
        assert aufc_inc >= aufc_dec - 0.01

    def test_single_score_returns_zero(self):
        m = UnlearningMetrics()
        assert m.compute_aufc([0.5]) == 0.0

    def test_empty_returns_zero(self):
        m = UnlearningMetrics()
        assert m.compute_aufc([]) == 0.0

    def test_custom_timesteps(self):
        m = UnlearningMetrics()
        scores = [0.8, 0.6, 0.4, 0.2]
        timesteps = [0, 100, 200, 300]
        aufc = m.compute_aufc(scores, timesteps)
        assert 0 < aufc < 1

    def test_non_uniform_timesteps(self):
        """Non-uniform timesteps should produce different AUFC than uniform."""
        m = UnlearningMetrics()
        scores = [0.8, 0.2, 0.1, 0.05]
        uniform = m.compute_aufc(scores, [0, 1, 2, 3])
        # Early drop: most time spent at low scores
        front_loaded = m.compute_aufc(scores, [0, 1, 100, 101])
        # These should differ (early drop means more area under low scores)
        assert uniform != pytest.approx(front_loaded, abs=0.01)

    def test_all_zeros(self):
        m = UnlearningMetrics()
        scores = [0.0, 0.0, 0.0, 0.0]
        aufc = m.compute_aufc(scores)
        assert aufc == pytest.approx(0.0, abs=1e-6)

    def test_all_ones(self):
        m = UnlearningMetrics()
        scores = [1.0, 1.0, 1.0, 1.0]
        aufc = m.compute_aufc(scores)
        assert aufc == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class TestRSI:
    def test_perfect_preservation(self):
        m = UnlearningMetrics()
        rsi = m.compute_retain_stability_index([100, 100, 100], [100, 100, 100])
        assert rsi == pytest.approx(1.0)

    def test_complete_degradation(self):
        m = UnlearningMetrics()
        # Current is very far from baseline
        rsi = m.compute_retain_stability_index([0, 0, 0], [500, 500, 500])
        assert rsi == pytest.approx(0.0)

    def test_partial_degradation(self):
        m = UnlearningMetrics()
        rsi = m.compute_retain_stability_index([90, 90, 90], [100, 100, 100])
        assert 0 < rsi < 1

    def test_negative_baseline_acrobot(self):
        """Acrobot has negative returns (~-80). RSI should still work."""
        m = UnlearningMetrics()
        baseline = [-80, -85, -75]
        current = [-80, -85, -75]
        rsi = m.compute_retain_stability_index(current, baseline)
        assert rsi == pytest.approx(1.0, abs=0.01)

    def test_negative_baseline_degradation(self):
        m = UnlearningMetrics()
        baseline = [-80, -80, -80]
        current = [-120, -120, -120]  # worse (more negative)
        rsi = m.compute_retain_stability_index(current, baseline)
        assert rsi < 1.0
        assert rsi >= 0.0

    def test_near_zero_baseline(self):
        """LunarLander early training ~0. Scale should default to max(std, 1.0)."""
        m = UnlearningMetrics()
        baseline = [0.1, -0.1, 0.05]
        current = [0.5, 0.3, 0.4]
        rsi = m.compute_retain_stability_index(current, baseline)
        assert 0 <= rsi <= 1

    def test_improvement_counts_as_instability(self):
        """RSI uses |current - baseline|, so improvement also reduces RSI."""
        m = UnlearningMetrics()
        rsi = m.compute_retain_stability_index([200, 200], [100, 100])
        assert rsi < 1.0

    def test_set_baseline_scalar(self):
        m = UnlearningMetrics()
        m.set_baseline([100, 100, 100])
        rsi = m.compute_retain_stability_index([95, 95, 95])
        assert 0 < rsi < 1

    def test_baseline_none_returns_zero(self):
        m = UnlearningMetrics()
        rsi = m.compute_retain_stability_index([100, 100])
        assert rsi == 0.0

    def test_symmetric(self):
        """RSI should be the same whether performance goes up or down by X."""
        m = UnlearningMetrics()
        baseline = [100, 100, 100]
        rsi_down = m.compute_retain_stability_index([80, 80, 80], baseline)
        rsi_up = m.compute_retain_stability_index([120, 120, 120], baseline)
        assert rsi_down == pytest.approx(rsi_up, abs=1e-5)


# ---------------------------------------------------------------------------
# Selectivity
# ---------------------------------------------------------------------------

class TestSelectivity:
    def test_perfect_selectivity(self):
        m = UnlearningMetrics()
        assert m.compute_selectivity(1.0, 1.0) == pytest.approx(1.0)

    def test_zero_selectivity(self):
        m = UnlearningMetrics()
        assert m.compute_selectivity(0.0, 1.0) == pytest.approx(0.0)
        assert m.compute_selectivity(1.0, 0.0) == pytest.approx(0.0)

    def test_multiplicative(self):
        m = UnlearningMetrics()
        assert m.compute_selectivity(0.8, 0.9) == pytest.approx(0.72)

    def test_commutative(self):
        m = UnlearningMetrics()
        assert m.compute_selectivity(0.3, 0.7) == pytest.approx(
            m.compute_selectivity(0.7, 0.3)
        )


# ---------------------------------------------------------------------------
# Forget Effectiveness
# ---------------------------------------------------------------------------

class TestForgetEffectiveness:
    def test_all_below_threshold(self):
        m = UnlearningMetrics()
        scores = [0.01, 0.05, 0.02]
        eff = m.compute_forget_effectiveness(scores, threshold=0.3)
        assert eff > 0.8

    def test_all_above_threshold(self):
        m = UnlearningMetrics()
        scores = [0.9, 0.95, 0.88]
        eff = m.compute_forget_effectiveness(scores, threshold=0.3)
        assert eff < 0.2

    def test_empty_returns_zero(self):
        m = UnlearningMetrics()
        assert m.compute_forget_effectiveness([]) == 0.0

    def test_lower_scores_higher_effectiveness(self):
        m = UnlearningMetrics()
        high = m.compute_forget_effectiveness([0.9, 0.8, 0.85])
        low = m.compute_forget_effectiveness([0.1, 0.2, 0.15])
        assert low > high

    def test_relative_to_baseline_no_change(self):
        """If scores didn't change from baseline, effectiveness should be low."""
        m = UnlearningMetrics()
        baseline = [0.9, 0.85, 0.92]
        current = [0.9, 0.85, 0.92]
        eff = m.compute_forget_effectiveness(current, baseline_forget_scores=baseline)
        # relative_reduction = 0
        assert eff < 0.5

    def test_relative_to_baseline_halved(self):
        m = UnlearningMetrics()
        baseline = [0.8, 0.8, 0.8]
        current = [0.4, 0.4, 0.4]
        eff = m.compute_forget_effectiveness(
            current, baseline_forget_scores=baseline, threshold=0.5
        )
        assert eff > 0.2

    def test_selective_beats_collapse(self):
        """True selective forgetting should score higher than random collapse."""
        m = UnlearningMetrics()
        baseline = [0.9, 0.9, 0.9]
        selective = [0.05, 0.03, 0.04]
        collapse = [0.5, 0.5, 0.5]
        eff_sel = m.compute_forget_effectiveness(selective, baseline_forget_scores=baseline)
        eff_col = m.compute_forget_effectiveness(collapse, baseline_forget_scores=baseline)
        assert eff_sel > eff_col

    def test_threshold_sensitivity(self):
        """Lower threshold = harder to count as forgotten."""
        m = UnlearningMetrics()
        scores = [0.25, 0.25, 0.25]
        eff_easy = m.compute_forget_effectiveness(scores, threshold=0.3)
        eff_hard = m.compute_forget_effectiveness(scores, threshold=0.1)
        assert eff_easy >= eff_hard

    def test_without_baseline_uses_absolute(self):
        m = UnlearningMetrics()
        scores = [0.1, 0.2, 0.15]
        eff = m.compute_forget_effectiveness(scores, baseline_forget_scores=None)
        assert eff > 0


# ---------------------------------------------------------------------------
# Privacy AUC
# ---------------------------------------------------------------------------

class TestPrivacyAUC:
    def test_perfect_attack(self):
        """If attack perfectly separates members/non-members, privacy_auc=0."""
        m = UnlearningMetrics()
        scores = [0.9, 0.8, 0.7, 0.1, 0.2, 0.3]
        labels = [True, True, True, False, False, False]
        pauc = m.compute_privacy_auc(scores, labels)
        assert pauc == pytest.approx(0.0, abs=0.05)

    def test_random_attack(self):
        """If attack is random, privacy_auc should be ~0.5."""
        m = UnlearningMetrics()
        np.random.seed(42)
        scores = np.random.rand(100).tolist()
        labels = ([True] * 50 + [False] * 50)
        pauc = m.compute_privacy_auc(scores, labels)
        assert 0.3 < pauc < 0.7

    def test_reversed_attack(self):
        """If members have LOWER scores (reversed), privacy_auc=1."""
        m = UnlearningMetrics()
        scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        labels = [True, True, True, False, False, False]
        pauc = m.compute_privacy_auc(scores, labels)
        assert pauc == pytest.approx(1.0, abs=0.05)

    def test_mismatched_lengths(self):
        m = UnlearningMetrics()
        assert m.compute_privacy_auc([0.5, 0.5], [True]) == 0.0

    def test_empty(self):
        """Empty input hits the ValueError/TypeError except → returns 0.5."""
        m = UnlearningMetrics()
        assert m.compute_privacy_auc([], []) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Trajectory Similarity
# ---------------------------------------------------------------------------

class TestTrajectorySimilarity:
    @pytest.fixture
    def agent(self):
        return PPOAgent(
            observation_dim=4, action_dim=2,
            action_type="discrete", device="cpu"
        )

    def test_returns_float_in_zero_one(self, agent):
        traj = {
            "observations": [np.random.randn(4) for _ in range(10)],
            "actions": [np.random.randint(0, 2) for _ in range(10)],
        }
        score = evaluate_trajectory_similarity(agent, traj, "cpu")
        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_single_step_trajectory(self, agent):
        traj = {
            "observations": [np.zeros(4)],
            "actions": [0],
        }
        score = evaluate_trajectory_similarity(agent, traj, "cpu")
        assert 0 <= score <= 1

    def test_consistent_for_same_input(self, agent):
        traj = {
            "observations": [np.array([0.1, -0.2, 0.3, -0.4])],
            "actions": [1],
        }
        s1 = evaluate_trajectory_similarity(agent, traj, "cpu")
        s2 = evaluate_trajectory_similarity(agent, traj, "cpu")
        assert s1 == pytest.approx(s2)

    def test_changes_after_weight_perturbation(self, agent):
        traj = {
            "observations": [np.random.randn(4) for _ in range(20)],
            "actions": [np.random.randint(0, 2) for _ in range(20)],
        }
        before = evaluate_trajectory_similarity(agent, traj, "cpu")
        with torch.no_grad():
            for p in agent.network.parameters():
                p.add_(torch.randn_like(p) * 2.0)
        after = evaluate_trajectory_similarity(agent, traj, "cpu")
        assert before != pytest.approx(after, abs=0.01)


# ---------------------------------------------------------------------------
# Membership Inference Attack
# ---------------------------------------------------------------------------

class TestMembershipInference:
    @pytest.fixture
    def agent(self):
        return PPOAgent(
            observation_dim=4, action_dim=2,
            action_type="discrete", device="cpu"
        )

    def _make_trajs(self, n):
        return [
            {
                "observations": [np.random.randn(4) for _ in range(5)],
                "actions": [np.random.randint(0, 2) for _ in range(5)],
            }
            for _ in range(n)
        ]

    def test_output_structure(self, agent):
        members = self._make_trajs(3)
        non_members = self._make_trajs(3)
        scores, labels = membership_inference_attack(
            agent, members, non_members, "cpu"
        )
        assert len(scores) == 6
        assert len(labels) == 6
        assert all(isinstance(s, float) for s in scores)

    def test_labels_correct(self, agent):
        members = self._make_trajs(2)
        non_members = self._make_trajs(3)
        scores, labels = membership_inference_attack(
            agent, members, non_members, "cpu"
        )
        assert labels == [True, True, False, False, False]

    def test_empty_members(self, agent):
        scores, labels = membership_inference_attack(
            agent, [], self._make_trajs(2), "cpu"
        )
        assert len(scores) == 2
        assert all(not l for l in labels)


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------

class TestComputeAllMetrics:
    def test_all_keys_present_with_timesteps(self):
        m = UnlearningMetrics()
        result = m.compute_all_metrics(
            forget_scores=[0.8, 0.6, 0.4, 0.2],
            retain_returns=[95, 90, 92, 88],
            baseline_returns=[100, 98, 102, 99],
            timesteps=[0, 1, 2, 3],
        )
        assert "aufc" in result
        assert "forget_effectiveness" in result
        assert "retain_stability_index" in result
        assert "selectivity" in result
        assert "mean_forget_score" in result
        assert "std_forget_score" in result
        assert "mean_retain_return" in result
        assert "std_retain_return" in result

    def test_no_aufc_without_timesteps(self):
        m = UnlearningMetrics()
        result = m.compute_all_metrics(
            forget_scores=[0.8, 0.6, 0.4],
            retain_returns=[95, 90, 92],
            baseline_returns=[100, 98, 102],
        )
        assert "aufc" not in result

    def test_selectivity_is_product(self):
        m = UnlearningMetrics()
        result = m.compute_all_metrics(
            forget_scores=[0.3, 0.3, 0.3],
            retain_returns=[95, 95, 95],
            baseline_returns=[100, 100, 100],
        )
        expected = result["forget_effectiveness"] * result["retain_stability_index"]
        assert result["selectivity"] == pytest.approx(expected, abs=1e-6)

    def test_baseline_forget_scores_passed_through(self):
        m = UnlearningMetrics()
        baseline_fs = [0.9, 0.9, 0.9]
        current_fs = [0.1, 0.1, 0.1]
        result = m.compute_all_metrics(
            forget_scores=current_fs,
            retain_returns=[100, 100],
            baseline_returns=[100, 100],
            baseline_forget_scores=baseline_fs,
        )
        # With baseline scores, effectiveness should be high
        assert result["forget_effectiveness"] > 0.5

    def test_mean_std_correct(self):
        m = UnlearningMetrics()
        fs = [0.2, 0.4, 0.6]
        rr = [90, 100, 110]
        result = m.compute_all_metrics(
            forget_scores=fs,
            retain_returns=rr,
            baseline_returns=[100, 100],
        )
        assert result["mean_forget_score"] == pytest.approx(np.mean(fs))
        assert result["std_forget_score"] == pytest.approx(np.std(fs))
        assert result["mean_retain_return"] == pytest.approx(np.mean(rr))
        assert result["std_retain_return"] == pytest.approx(np.std(rr))

    def test_empty_forget_scores(self):
        m = UnlearningMetrics()
        result = m.compute_all_metrics(
            forget_scores=[],
            retain_returns=[100],
            baseline_returns=[100],
        )
        assert result["forget_effectiveness"] == 0.0
        assert result["mean_forget_score"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
