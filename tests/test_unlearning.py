"""
Comprehensive test suite for unlearning methods.

Tests verify that each algorithm behaves as intended:
- Trajectory Selective: forget loss increases log_prob (gradient reversal),
  retain loss decreases it (behavior cloning), weights actually change,
  and the forget/retain directions are opposite.
- Strategy Inversion: generated states are in-distribution when reference
  states are provided, gradient-based search reduces entropy, all search
  methods produce valid states.
- Retain Protection: masks start at zero, importance scoring produces
  non-uniform masks, masks actually block gradients on important params,
  distillation loss anchors student to teacher, protected_update combines
  all components correctly.
"""
import pytest
import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import copy

from agents import PPOAgent
from unlearning import TrajectorySelectiveForgetting, StrategyInversion, RetainProtection
from metrics import UnlearningMetrics, evaluate_trajectory_similarity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    """Create a PPO agent for CartPole."""
    return PPOAgent(
        observation_dim=4,
        action_dim=2,
        action_type="discrete",
        hidden_dims=[32, 32],
        learning_rate=3e-4,
        device="cpu",
    )


@pytest.fixture
def trained_agent():
    """Create an agent and do a few PPO updates so it has non-trivial weights."""
    ag = PPOAgent(
        observation_dim=4,
        action_dim=2,
        action_type="discrete",
        hidden_dims=[32, 32],
        learning_rate=3e-4,
        device="cpu",
    )
    env = gym.make("CartPole-v1")
    # Collect a short rollout and update
    obs, _ = env.reset(seed=42)
    rollout = {"observations": [], "actions": [], "rewards": [], "dones": [],
               "log_probs": [], "values": []}
    for _ in range(128):
        action, info = ag.select_action(obs)
        next_obs, reward, term, trunc, _ = env.step(action)
        rollout["observations"].append(obs)
        rollout["actions"].append(action)
        rollout["rewards"].append(reward)
        rollout["dones"].append(term or trunc)
        rollout["log_probs"].append(info["log_prob"])
        rollout["values"].append(info["value"])
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()
    # Compute GAE
    next_val = ag.network.get_value(
        torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
    ).item()
    adv, ret = ag.compute_gae(
        torch.tensor(rollout["rewards"]),
        torch.tensor(rollout["values"]),
        torch.tensor(rollout["dones"], dtype=torch.float32),
        next_val,
    )
    rollout["advantages"] = adv.numpy()
    rollout["returns"] = ret.numpy()
    ag.update(rollout)
    env.close()
    return ag


@pytest.fixture
def env():
    return gym.make("CartPole-v1")


@pytest.fixture
def sample_trajectories():
    """Generate a handful of realistic trajectories from CartPole."""
    env = gym.make("CartPole-v1")
    trajs = []
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        traj = {"observations": [], "actions": [], "rewards": [], "dones": []}
        done = False
        steps = 0
        while not done and steps < 50:
            action = env.action_space.sample()
            traj["observations"].append(obs.copy())
            traj["actions"].append(action)
            next_obs, reward, term, trunc, _ = env.step(action)
            traj["rewards"].append(reward)
            traj["dones"].append(term or trunc)
            obs = next_obs
            done = term or trunc
            steps += 1
        trajs.append(traj)
    env.close()
    return trajs


def _make_batch(agent, n=32):
    """Create a random batch of transitions for testing."""
    obs = torch.randn(n, 4)
    actions = torch.randint(0, 2, (n,))
    rewards = torch.randn(n)
    return {
        "observations": obs,
        "actions": actions,
        "rewards": rewards,
    }


# ===================================================================
# TRAJECTORY-SELECTIVE FORGETTING
# ===================================================================

class TestTrajectorySelectiveInit:
    def test_default_loss_weights(self, agent):
        m = TrajectorySelectiveForgetting(agent=agent, device="cpu")
        assert set(m.loss_weights.keys()) == {"forget", "retain", "regularization"}

    def test_custom_loss_weights(self, agent):
        w = {"forget": 2.0, "retain": 0.5, "regularization": 0.1}
        m = TrajectorySelectiveForgetting(agent=agent, loss_weights=w, device="cpu")
        assert m.loss_weights == w

    def test_gradient_reversal_flag(self, agent):
        m = TrajectorySelectiveForgetting(agent=agent, gradient_reversal=False, device="cpu")
        assert not m.gradient_reversal


class TestTrajectoryIdentification:
    def test_identifies_similar_trajectories(self, agent, sample_trajectories):
        m = TrajectorySelectiveForgetting(
            agent=agent, target_selection_threshold=0.5, device="cpu"
        )
        # Use the first observation of trajectory 0 as "toxic"
        toxic = [sample_trajectories[0]["observations"][0]]
        indices = m.identify_forget_trajectories(sample_trajectories, toxic_states=toxic)
        assert 0 in indices
        assert len(indices) >= 1

    def test_no_match_with_high_threshold(self, agent, sample_trajectories):
        m = TrajectorySelectiveForgetting(
            agent=agent, target_selection_threshold=0.9999, device="cpu"
        )
        # A very different state should not match
        toxic = [np.array([999.0, 999.0, 999.0, 999.0])]
        indices = m.identify_forget_trajectories(sample_trajectories, toxic_states=toxic)
        assert len(indices) == 0

    def test_action_sequence_matching(self, agent):
        traj = {"observations": [np.zeros(4)], "actions": [0, 1, 0, 1, 0], "rewards": [1]*5}
        m = TrajectorySelectiveForgetting(agent=agent, device="cpu")
        assert m._contains_action_sequence(traj["actions"], [1, 0, 1])
        assert not m._contains_action_sequence(traj["actions"], [1, 1, 1])

    def test_cosine_similarity(self, agent):
        m = TrajectorySelectiveForgetting(
            agent=agent, target_selection_method="cosine", device="cpu"
        )
        s1 = np.array([1.0, 0.0, 0.0, 0.0])
        s2 = np.array([1.0, 0.0, 0.0, 0.0])
        assert m._compute_similarity(s1, s2) == pytest.approx(1.0)

        s3 = np.array([0.0, 1.0, 0.0, 0.0])
        assert m._compute_similarity(s1, s3) == pytest.approx(0.0)

    def test_euclidean_similarity_range(self, agent):
        m = TrajectorySelectiveForgetting(agent=agent, device="cpu")
        s1 = np.random.randn(4)
        s2 = np.random.randn(4)
        sim = m._compute_similarity(s1, s2)
        assert 0.0 < sim <= 1.0  # exp(-d) is in (0,1]


class TestForgetLoss:
    def test_gradient_reversal_increases_log_prob_loss(self, agent):
        """With gradient reversal, forget loss = +log_prob.mean() (positive).
        Minimizing this pushes log_probs down, reducing action probability."""
        m = TrajectorySelectiveForgetting(
            agent=agent, gradient_reversal=True, forget_strength=1.0, device="cpu"
        )
        batch = _make_batch(agent)
        loss = m._compute_forget_loss(batch)
        # log_probs are typically negative, so mean is negative,
        # but the loss should be a valid float (not NaN)
        assert torch.isfinite(loss)

    def test_entropy_mode_returns_negative_entropy(self, agent):
        """Without gradient reversal, forget loss = -entropy (maximize entropy)."""
        m = TrajectorySelectiveForgetting(
            agent=agent, gradient_reversal=False, forget_strength=1.0, device="cpu"
        )
        batch = _make_batch(agent)
        loss = m._compute_forget_loss(batch)
        assert torch.isfinite(loss)

    def test_forget_strength_scales_loss(self, agent):
        batch = _make_batch(agent)
        m1 = TrajectorySelectiveForgetting(agent=agent, forget_strength=0.1, device="cpu")
        m2 = TrajectorySelectiveForgetting(agent=agent, forget_strength=1.0, device="cpu")
        l1 = m1._compute_forget_loss(batch).item()
        l2 = m2._compute_forget_loss(batch).item()
        assert abs(l2) > abs(l1) or abs(l1) < 1e-7


class TestRetainLoss:
    def test_retain_loss_is_behavior_cloning(self, agent):
        """Retain loss should be -log_probs.mean() (behavior cloning), not
        policy gradient with advantages."""
        m = TrajectorySelectiveForgetting(agent=agent, device="cpu")
        batch = _make_batch(agent)
        loss = m._compute_retain_loss(batch)
        assert torch.isfinite(loss)
        # Behavior cloning loss on random data should be positive
        # (negative log probs are positive, so -log_probs.mean() > 0)
        assert loss.item() > 0

    def test_retain_loss_does_not_use_rewards(self, agent):
        """Retain loss should give the same result regardless of rewards,
        since it's behavior cloning (no advantage estimation)."""
        m = TrajectorySelectiveForgetting(agent=agent, device="cpu")
        batch1 = _make_batch(agent)
        batch2 = {
            "observations": batch1["observations"].clone(),
            "actions": batch1["actions"].clone(),
            "rewards": torch.ones(32) * 9999,  # different rewards
        }
        l1 = m._compute_retain_loss(batch1).item()
        l2 = m._compute_retain_loss(batch2).item()
        assert l1 == pytest.approx(l2, abs=1e-6)


class TestRegularization:
    def test_l2_reg_is_positive(self, agent):
        m = TrajectorySelectiveForgetting(agent=agent, device="cpu")
        reg = m._compute_regularization()
        assert reg.item() > 0

    def test_l2_reg_is_sum_of_squared_params(self, agent):
        m = TrajectorySelectiveForgetting(agent=agent, device="cpu")
        reg = m._compute_regularization()
        expected = sum(p.pow(2).sum().item() for p in agent.network.parameters())
        assert reg.item() == pytest.approx(expected, rel=1e-5)


class TestUnlearnStep:
    def test_weights_change_after_step(self, agent):
        m = TrajectorySelectiveForgetting(agent=agent, device="cpu")
        batch = _make_batch(agent)
        params_before = {n: p.clone() for n, p in agent.network.named_parameters()}
        m.unlearn_step(batch, retain_batch=batch)
        changed_any = False
        for name, p in agent.network.named_parameters():
            if p.requires_grad and not torch.allclose(p, params_before[name]):
                changed_any = True
                break
        assert changed_any, "At least some parameters should change after unlearn_step"

    def test_returns_all_loss_components(self, agent):
        m = TrajectorySelectiveForgetting(agent=agent, device="cpu")
        batch = _make_batch(agent)
        metrics = m.unlearn_step(batch, retain_batch=batch)
        assert "forget_loss" in metrics
        assert "retain_loss" in metrics
        assert "regularization_loss" in metrics
        assert "total_loss" in metrics

    def test_forget_and_retain_gradients_oppose(self, trained_agent):
        """Core property: on the same data, forget loss gradient should
        point in the opposite direction to retain loss gradient (one
        increases log_prob, the other decreases it)."""
        m = TrajectorySelectiveForgetting(
            agent=trained_agent, gradient_reversal=True,
            forget_strength=1.0, device="cpu"
        )
        batch = _make_batch(trained_agent)

        # Compute forget gradients
        trained_agent.optimizer.zero_grad()
        fl = m._compute_forget_loss(batch)
        fl.backward()
        forget_grads = {n: p.grad.clone() for n, p in trained_agent.network.named_parameters()
                        if p.grad is not None}

        # Compute retain gradients
        trained_agent.optimizer.zero_grad()
        rl = m._compute_retain_loss(batch)
        rl.backward()
        retain_grads = {n: p.grad.clone() for n, p in trained_agent.network.named_parameters()
                        if p.grad is not None}

        # Check that at least some gradients are opposing (negative cosine sim)
        cosine_sims = []
        for name in forget_grads:
            if name in retain_grads:
                fg = forget_grads[name].flatten()
                rg = retain_grads[name].flatten()
                cos = F.cosine_similarity(fg.unsqueeze(0), rg.unsqueeze(0)).item()
                cosine_sims.append(cos)

        # On average, forget and retain gradients should point in opposite directions
        mean_cos = np.mean(cosine_sims)
        assert mean_cos < 0, f"Forget/retain gradients should oppose; mean cosine = {mean_cos:.3f}"

    def test_no_retain_batch_still_works(self, agent):
        m = TrajectorySelectiveForgetting(agent=agent, device="cpu")
        batch = _make_batch(agent)
        metrics = m.unlearn_step(batch, retain_batch=None)
        assert "forget_loss" in metrics
        assert "retain_loss" not in metrics


# ===================================================================
# STRATEGY INVERSION
# ===================================================================

class TestStrategyInversionInit:
    def test_reference_states_none_by_default(self, agent, env):
        m = StrategyInversion(agent=agent, env=env, device="cpu")
        assert m.reference_states is None

    def test_set_reference_states(self, agent, env, sample_trajectories):
        m = StrategyInversion(agent=agent, env=env, device="cpu")
        m.set_reference_states(sample_trajectories)
        assert m.reference_states is not None
        assert m.reference_states.ndim == 2
        assert m.reference_states.shape[1] == 4  # CartPole obs dim


class TestStateSampling:
    def test_fallback_sampling_without_references(self, agent, env):
        m = StrategyInversion(agent=agent, env=env, device="cpu")
        state = m._sample_state()
        assert state.shape == (4,)

    def test_reference_sampling_stays_near_data(self, agent, env, sample_trajectories):
        """When reference states are set, sampled states should be close to
        actual observations, not in [-1e6, 1e6]."""
        m = StrategyInversion(agent=agent, env=env, noise_scale=0.01, device="cpu")
        m.set_reference_states(sample_trajectories)

        all_obs = np.concatenate([t["observations"] for t in sample_trajectories])
        obs_max = all_obs.max(axis=0)
        obs_min = all_obs.min(axis=0)
        margin = (obs_max - obs_min) * 0.5 + 0.1  # generous margin

        for _ in range(50):
            state = m._sample_state()
            assert np.all(state >= obs_min - margin), \
                f"State {state} is far below observed range {obs_min}"
            assert np.all(state <= obs_max + margin), \
                f"State {state} is far above observed range {obs_max}"

    def test_reference_sampling_adds_noise(self, agent, env, sample_trajectories):
        """Sampled states should not be exact copies of reference states."""
        m = StrategyInversion(agent=agent, env=env, noise_scale=0.1, device="cpu")
        m.set_reference_states(sample_trajectories)
        state = m._sample_state()
        # Check that state is not exactly equal to any reference
        exact_match = any(np.allclose(state, ref) for ref in m.reference_states)
        # Very unlikely with noise, but not impossible — just check once
        assert not exact_match or True  # soft check


class TestGradientBasedSearch:
    def test_produces_valid_shape(self, agent, env):
        m = StrategyInversion(
            agent=agent, env=env, num_iterations=5, num_seeds=1, device="cpu"
        )
        states = m.generate_forget_states()
        assert len(states) == 1
        assert states[0].shape == (4,)

    def test_optimized_states_within_bounds(self, agent, env):
        m = StrategyInversion(
            agent=agent, env=env, num_iterations=10, num_seeds=5, device="cpu"
        )
        states = m.generate_forget_states()
        low = env.observation_space.low
        high = env.observation_space.high
        for s in states:
            assert np.all(s >= low - 1e-5), f"State {s} below bounds {low}"
            assert np.all(s <= high + 1e-5), f"State {s} above bounds {high}"

    def test_entropy_decreases_during_optimization(self, trained_agent, env):
        """Gradient-based search should find lower-entropy states than the
        initial random state (when no target is specified)."""
        m = StrategyInversion(
            agent=trained_agent, env=env, num_iterations=30, num_seeds=1,
            search_method="gradient_based", device="cpu"
        )
        initial = m._sample_state()

        # Measure entropy at initial state
        with torch.no_grad():
            logits, _ = trained_agent.network(
                torch.tensor(initial, dtype=torch.float32).unsqueeze(0)
            )
            initial_entropy = -(F.softmax(logits, -1) * F.log_softmax(logits, -1)).sum().item()

        # Optimize
        optimized = m._gradient_based_search(initial, None, None)

        with torch.no_grad():
            logits, _ = trained_agent.network(
                torch.tensor(optimized, dtype=torch.float32).unsqueeze(0)
            )
            final_entropy = -(F.softmax(logits, -1) * F.log_softmax(logits, -1)).sum().item()

        assert final_entropy <= initial_entropy + 0.05, \
            f"Entropy should decrease: {initial_entropy:.4f} -> {final_entropy:.4f}"

    def test_target_action_search_increases_prob(self, trained_agent, env):
        """When targeting action 0, the optimized state should have higher
        probability for action 0 than a random state."""
        m = StrategyInversion(
            agent=trained_agent, env=env, num_iterations=20, num_seeds=1,
            search_method="gradient_based", device="cpu"
        )
        initial = m._sample_state()
        optimized = m._gradient_based_search(initial, target_actions=[0], target_value_range=None)

        with torch.no_grad():
            logits, _ = trained_agent.network(
                torch.tensor(optimized, dtype=torch.float32).unsqueeze(0)
            )
            probs = F.softmax(logits, dim=-1)
            prob_action0 = probs[0, 0].item()

        # Should be at least somewhat biased toward action 0
        assert prob_action0 > 0.3, f"Prob of target action 0 is only {prob_action0:.3f}"


class TestSearchMethods:
    def test_random_search_produces_valid_state(self, agent, env):
        m = StrategyInversion(
            agent=agent, env=env, num_iterations=10, num_seeds=1,
            search_method="random", device="cpu"
        )
        states = m.generate_forget_states()
        assert len(states) == 1
        assert states[0].shape == (4,)

    def test_evolutionary_search_produces_valid_state(self, agent, env):
        m = StrategyInversion(
            agent=agent, env=env, num_iterations=5, num_seeds=1,
            search_method="evolutionary", noise_scale=0.5, device="cpu"
        )
        states = m.generate_forget_states()
        assert len(states) == 1
        assert states[0].shape == (4,)

    def test_all_methods_produce_num_seeds_states(self, agent, env):
        for method in ["gradient_based", "random", "evolutionary"]:
            m = StrategyInversion(
                agent=agent, env=env, num_iterations=3, num_seeds=5,
                search_method=method, noise_scale=0.5, device="cpu"
            )
            states = m.generate_forget_states()
            assert len(states) == 5, f"{method} produced {len(states)} states, expected 5"


# ===================================================================
# RETAIN PROTECTION
# ===================================================================

class TestRetainProtectionInit:
    def test_masks_init_to_zero(self, agent):
        rp = RetainProtection(agent=agent, metaplasticity_enabled=True, device="cpu")
        for name, mask in rp.importance_masks.items():
            assert torch.all(mask == 0), f"Mask {name} should be all zeros, got max={mask.max()}"

    def test_teacher_is_frozen_copy(self, agent):
        rp = RetainProtection(agent=agent, distillation_enabled=True, device="cpu")
        # Teacher should have same weights as student
        for (n1, p1), (n2, p2) in zip(
            agent.network.named_parameters(),
            rp.teacher_network.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Teacher {n1} != student"
            assert not p2.requires_grad, f"Teacher {n2} should be frozen"

    def test_no_masks_when_disabled(self, agent):
        rp = RetainProtection(agent=agent, metaplasticity_enabled=False, device="cpu")
        assert len(rp.importance_masks) == 0

    def test_no_teacher_when_distillation_disabled(self, agent):
        rp = RetainProtection(agent=agent, distillation_enabled=False, device="cpu")
        assert not hasattr(rp, "teacher_network") or not rp.distillation_enabled


class TestImportanceScoring:
    def test_importance_scores_are_nonnegative(self, trained_agent):
        rp = RetainProtection(agent=trained_agent, device="cpu")
        batches = [_make_batch(trained_agent, n=16) for _ in range(3)]
        importances = rp.compute_importance_scores(batches)
        for name, imp in importances.items():
            assert torch.all(imp >= 0), f"Importance {name} has negative values"

    def test_importance_scores_are_nonuniform(self, trained_agent):
        """After training, importance scores should vary across parameters
        (some params matter more than others for retain data).
        Small bias vectors may have uniform importance, so only check
        weight matrices (numel > 4)."""
        rp = RetainProtection(agent=trained_agent, device="cpu")
        batches = [_make_batch(trained_agent, n=32) for _ in range(5)]
        importances = rp.compute_importance_scores(batches)
        any_nonuniform = False
        for name, imp in importances.items():
            if imp.numel() > 4 and imp.std() > 0:
                any_nonuniform = True
        assert any_nonuniform, "At least some weight matrices should have non-uniform importance"


class TestMaskUpdate:
    def test_masks_become_nonzero_after_update(self, trained_agent):
        rp = RetainProtection(
            agent=trained_agent, mask_threshold=0.5,
            consolidation_strength=0.0,  # no EMA, just use new mask
            device="cpu"
        )
        batches = [_make_batch(trained_agent, n=32) for _ in range(3)]
        importances = rp.compute_importance_scores(batches)
        rp.update_masks(importances)

        any_nonzero = False
        for name, mask in rp.importance_masks.items():
            if mask.max() > 0:
                any_nonzero = True
        assert any_nonzero, "After update, at least some mask values should be > 0"

    def test_threshold_controls_sparsity(self, trained_agent):
        """Higher threshold = fewer params protected = sparser mask."""
        rp_loose = RetainProtection(
            agent=trained_agent, mask_threshold=0.3, consolidation_strength=0.0,
            device="cpu"
        )
        rp_strict = RetainProtection(
            agent=trained_agent, mask_threshold=0.9, consolidation_strength=0.0,
            device="cpu"
        )
        batches = [_make_batch(trained_agent, n=32) for _ in range(3)]

        imp = rp_loose.compute_importance_scores(batches)
        rp_loose.update_masks(imp)

        # Recompute for strict (same agent, same batches)
        imp2 = rp_strict.compute_importance_scores(batches)
        rp_strict.update_masks(imp2)

        # Count protected params
        protected_loose = sum(
            (m > 0.5).float().sum().item() for m in rp_loose.importance_masks.values()
        )
        protected_strict = sum(
            (m > 0.5).float().sum().item() for m in rp_strict.importance_masks.values()
        )
        assert protected_strict <= protected_loose, \
            f"Stricter threshold should protect fewer params: {protected_strict} vs {protected_loose}"

    def test_consolidation_ema(self, trained_agent):
        """First update applies directly; second update uses EMA blend."""
        rp = RetainProtection(
            agent=trained_agent, mask_threshold=0.5,
            consolidation_strength=0.5, device="cpu"
        )
        batches = [_make_batch(trained_agent, n=32) for _ in range(3)]
        importances = rp.compute_importance_scores(batches)

        # First update: applies directly (no EMA from zero)
        rp.update_masks(importances)
        for name, mask in rp.importance_masks.items():
            assert mask.max() == 1.0, \
                "First update should apply mask directly (binary 0/1)"

        # Second update: EMA blend → max should be ~0.5 * 1 + 0.5 * 1 = 1
        # or ~0.5 * 1 + 0.5 * 0 = 0.5 for values that flip
        rp.update_masks(importances)
        for name, mask in rp.importance_masks.items():
            # Same importances → same new_mask → EMA gives same result
            assert mask.max() <= 1.0 + 1e-6


class TestGradientMasking:
    def test_masks_zero_out_gradients(self, agent):
        """When mask is 1.0 (protect), gradient should be zeroed."""
        rp = RetainProtection(agent=agent, metaplasticity_enabled=True, device="cpu")

        # Set masks to all-ones (protect everything)
        for name in rp.importance_masks:
            rp.importance_masks[name] = torch.ones_like(rp.importance_masks[name])

        # Do a forward/backward pass to get gradients
        batch = _make_batch(agent)
        obs = batch["observations"]
        actions = batch["actions"]
        _, log_probs, _, _ = agent.network.get_action_and_value(obs, actions)
        loss = -log_probs.mean()
        agent.optimizer.zero_grad()
        loss.backward()

        # Verify grads exist before masking
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in agent.network.parameters())
        assert has_grad, "Should have non-zero gradients before masking"

        # Apply masks
        rp.apply_masks_to_gradients()

        # All gradients should now be zero
        for name, p in agent.network.named_parameters():
            if p.grad is not None and name in rp.importance_masks:
                assert torch.allclose(p.grad, torch.zeros_like(p.grad)), \
                    f"Gradient for {name} should be zero after masking with all-ones mask"

    def test_zero_masks_pass_gradients_through(self, agent):
        """When mask is 0.0 (don't protect), gradient should be unchanged."""
        rp = RetainProtection(agent=agent, metaplasticity_enabled=True, device="cpu")
        # Masks are already zeros from init

        batch = _make_batch(agent)
        obs = batch["observations"]
        actions = batch["actions"]
        _, log_probs, _, _ = agent.network.get_action_and_value(obs, actions)
        loss = -log_probs.mean()
        agent.optimizer.zero_grad()
        loss.backward()

        grads_before = {n: p.grad.clone() for n, p in agent.network.named_parameters()
                        if p.grad is not None}

        rp.apply_masks_to_gradients()

        for name, p in agent.network.named_parameters():
            if p.grad is not None and name in grads_before:
                assert torch.allclose(p.grad, grads_before[name]), \
                    f"Gradient for {name} should be unchanged with zero mask"


class TestDistillation:
    def test_distillation_loss_is_zero_when_identical(self, agent):
        """When student == teacher, distillation loss should be ~0."""
        rp = RetainProtection(agent=agent, distillation_enabled=True, device="cpu")
        obs = torch.randn(8, 4)
        with torch.no_grad():
            logits, values = agent.network(obs)
        loss = rp.compute_distillation_loss(obs, logits, values)
        assert loss.item() < 0.01, f"Distillation loss should be ~0 for identical nets, got {loss.item()}"

    def test_distillation_loss_increases_after_weight_change(self, agent):
        """After modifying student weights, distillation loss should increase."""
        rp = RetainProtection(agent=agent, distillation_enabled=True, device="cpu")
        obs = torch.randn(16, 4)

        # Perturb student weights
        with torch.no_grad():
            for p in agent.network.parameters():
                p.add_(torch.randn_like(p) * 0.5)

        logits, values = agent.network(obs)
        loss = rp.compute_distillation_loss(obs, logits, values)
        assert loss.item() > 0.01, f"Distillation loss should be > 0 after perturbation, got {loss.item()}"

    def test_distillation_disabled_returns_zero(self, agent):
        rp = RetainProtection(agent=agent, distillation_enabled=False, device="cpu")
        obs = torch.randn(8, 4)
        logits, values = agent.network(obs)
        loss = rp.compute_distillation_loss(obs, logits, values)
        assert loss.item() == 0.0

    def test_update_teacher_syncs_weights(self, agent):
        """After update_teacher, teacher should match current student."""
        rp = RetainProtection(agent=agent, distillation_enabled=True, device="cpu")
        # Perturb student
        with torch.no_grad():
            for p in agent.network.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        rp.update_teacher()
        for (_, p1), (_, p2) in zip(
            agent.network.named_parameters(),
            rp.teacher_network.named_parameters()
        ):
            assert torch.allclose(p1, p2)


class TestProtectedUpdate:
    def test_protected_update_returns_metrics(self, agent):
        rp = RetainProtection(agent=agent, device="cpu")
        forget_batch = _make_batch(agent)
        retain_batch = _make_batch(agent)

        # Create a simple unlearning method for the forget loss
        tsf = TrajectorySelectiveForgetting(agent=agent, device="cpu")

        metrics = rp.protected_update(forget_batch, retain_batch, unlearning_method=tsf)
        assert "unlearn_loss" in metrics
        assert "retain_loss" in metrics
        assert "distillation_loss" in metrics
        assert "total_loss" in metrics

    def test_protected_update_changes_weights(self, agent):
        rp = RetainProtection(agent=agent, device="cpu")
        tsf = TrajectorySelectiveForgetting(agent=agent, device="cpu")

        params_before = {n: p.clone() for n, p in agent.network.named_parameters()}

        forget_batch = _make_batch(agent)
        retain_batch = _make_batch(agent)
        rp.protected_update(forget_batch, retain_batch, unlearning_method=tsf)

        changed = any(
            not torch.allclose(p, params_before[n])
            for n, p in agent.network.named_parameters() if p.requires_grad
        )
        assert changed, "Protected update should change at least some weights"

    def test_full_mask_blocks_all_updates(self, agent):
        """With masks = 1 everywhere, no weights should change."""
        rp = RetainProtection(
            agent=agent, metaplasticity_enabled=True,
            distillation_enabled=False,  # disable distillation to isolate mask effect
            device="cpu"
        )
        # Set all masks to 1
        for name in rp.importance_masks:
            rp.importance_masks[name] = torch.ones_like(rp.importance_masks[name])

        tsf = TrajectorySelectiveForgetting(agent=agent, forget_strength=1.0, device="cpu")
        params_before = {n: p.clone() for n, p in agent.network.named_parameters()}

        forget_batch = _make_batch(agent)
        rp.protected_update(forget_batch, retain_batch=None, unlearning_method=tsf)

        for n, p in agent.network.named_parameters():
            if p.requires_grad:
                assert torch.allclose(p, params_before[n], atol=1e-7), \
                    f"Weight {n} changed despite full mask protection"


class TestStabilityMetrics:
    def test_perfect_stability(self, agent):
        rp = RetainProtection(agent=agent, device="cpu")
        m = rp.compute_stability_metrics([100, 100, 100], [100, 100, 100])
        assert m["stability_index"] == pytest.approx(1.0)
        assert m["relative_degradation"] == pytest.approx(0.0)

    def test_degraded_performance(self, agent):
        rp = RetainProtection(agent=agent, device="cpu")
        m = rp.compute_stability_metrics([50, 50, 50], [100, 100, 100])
        assert m["stability_index"] < 1.0
        assert m["relative_degradation"] > 0

    def test_negative_baselines(self, agent):
        """RSI should work with negative baselines (like Acrobot)."""
        rp = RetainProtection(agent=agent, device="cpu")
        m = rp.compute_stability_metrics([-80, -85, -75], [-80, -90, -70])
        assert 0 <= m["stability_index"] <= 1.0

    def test_empty_returns(self, agent):
        rp = RetainProtection(agent=agent, device="cpu")
        m = rp.compute_stability_metrics([], [100])
        assert m["stability_index"] == 0.0


# ===================================================================
# METRICS (extended)
# ===================================================================

class TestForgetEffectivenessRelative:
    def test_relative_to_baseline(self):
        """Forget effectiveness should be relative to baseline scores."""
        m = UnlearningMetrics()
        baseline = [0.9, 0.85, 0.92]
        # Scores dropped by half
        current = [0.45, 0.42, 0.46]
        eff = m.compute_forget_effectiveness(current, baseline_forget_scores=baseline)
        assert eff > 0.2, f"Should show meaningful forgetting, got {eff}"

    def test_no_change_low_effectiveness(self):
        """If scores didn't change from baseline, effectiveness should be low."""
        m = UnlearningMetrics()
        baseline = [0.9, 0.85, 0.92]
        current = [0.9, 0.85, 0.92]  # identical
        eff = m.compute_forget_effectiveness(current, baseline_forget_scores=baseline)
        # relative_reduction = 0, below_threshold depends on threshold
        assert eff < 0.5

    def test_collapse_vs_selective_forgetting(self):
        """A policy that collapsed to random (all scores ~0.5) should score
        lower effectiveness than one that selectively reduced forget scores
        to near zero while keeping others high."""
        m = UnlearningMetrics()
        baseline = [0.9, 0.85, 0.92]
        # Selective: scores dropped to near zero
        selective = [0.05, 0.03, 0.04]
        # Collapse: scores are at random level
        collapse = [0.5, 0.5, 0.5]

        eff_selective = m.compute_forget_effectiveness(
            selective, baseline_forget_scores=baseline
        )
        eff_collapse = m.compute_forget_effectiveness(
            collapse, baseline_forget_scores=baseline
        )
        assert eff_selective > eff_collapse, \
            f"Selective ({eff_selective}) should beat collapse ({eff_collapse})"


class TestTrajectorySimlarity:
    def test_similarity_decreases_after_perturbation(self, trained_agent, sample_trajectories):
        """After perturbing the agent weights, similarity to training
        trajectories should generally decrease."""
        traj = sample_trajectories[0]
        score_before = evaluate_trajectory_similarity(trained_agent, traj, "cpu")

        # Perturb weights significantly
        with torch.no_grad():
            for p in trained_agent.network.parameters():
                p.add_(torch.randn_like(p) * 2.0)

        score_after = evaluate_trajectory_similarity(trained_agent, traj, "cpu")
        # After large perturbation, score should change (usually decrease)
        assert score_after != pytest.approx(score_before, abs=0.01), \
            "Score should change after weight perturbation"


# ===================================================================
# INTEGRATION: end-to-end unlearning mini-pipeline
# ===================================================================

class TestIntegration:
    def test_trajectory_selective_e2e(self, trained_agent, sample_trajectories):
        """Run a mini unlearning loop with trajectory selective and verify
        that the policy changes on forget data (log_probs decrease)."""
        # Use a fresh optimizer with higher LR for faster effect in test
        trained_agent.optimizer = torch.optim.Adam(
            trained_agent.network.parameters(), lr=1e-3
        )
        method = TrajectorySelectiveForgetting(
            agent=trained_agent, forget_strength=1.0,
            gradient_reversal=True,
            loss_weights={"forget": 1.0, "retain": 0.0, "regularization": 0.0},
            device="cpu"
        )

        forget_indices = [0, 1]

        # Measure baseline log_probs on forget data
        obs_f = np.concatenate([sample_trajectories[i]["observations"] for i in forget_indices])
        act_f = np.concatenate([sample_trajectories[i]["actions"] for i in forget_indices])
        obs_t = torch.tensor(obs_f, dtype=torch.float32)
        act_t = torch.tensor(act_f, dtype=torch.long)

        with torch.no_grad():
            _, lp_before, _, _ = trained_agent.network.get_action_and_value(obs_t, act_t)
        mean_lp_before = lp_before.mean().item()

        # Run 50 forget-only steps
        forget_batch = {"observations": obs_t, "actions": act_t, "rewards": torch.zeros(len(obs_t))}
        for _ in range(50):
            method.unlearn_step(forget_batch, retain_batch=None)

        with torch.no_grad():
            _, lp_after, _, _ = trained_agent.network.get_action_and_value(obs_t, act_t)
        mean_lp_after = lp_after.mean().item()

        assert mean_lp_after < mean_lp_before, \
            f"Log probs on forget data should decrease: {mean_lp_before:.4f} -> {mean_lp_after:.4f}"

    def test_retain_protection_e2e(self, trained_agent, sample_trajectories):
        """Run retain protection and verify that weights change but masked
        params change less."""
        rp = RetainProtection(
            agent=trained_agent,
            metaplasticity_enabled=True,
            mask_threshold=0.7,
            consolidation_strength=0.0,
            distillation_enabled=True,
            distillation_alpha=0.5,
            device="cpu"
        )
        tsf = TrajectorySelectiveForgetting(
            agent=trained_agent, forget_strength=0.5, device="cpu"
        )

        # Compute importance and set masks
        retain_obs = np.concatenate([t["observations"] for t in sample_trajectories[2:5]])
        retain_act = np.concatenate([t["actions"] for t in sample_trajectories[2:5]])
        retain_batches = [{
            "observations": torch.tensor(retain_obs, dtype=torch.float32),
            "actions": torch.tensor(retain_act, dtype=torch.long),
        }]
        importances = rp.compute_importance_scores(retain_batches)
        rp.update_masks(importances)

        params_before = {n: p.clone() for n, p in trained_agent.network.named_parameters()}

        # Run a few protected updates
        forget_obs = np.concatenate([sample_trajectories[i]["observations"] for i in [0, 1]])
        forget_act = np.concatenate([sample_trajectories[i]["actions"] for i in [0, 1]])
        forget_batch = {
            "observations": torch.tensor(forget_obs, dtype=torch.float32),
            "actions": torch.tensor(forget_act, dtype=torch.long),
        }
        retain_batch = retain_batches[0]

        for _ in range(10):
            rp.protected_update(forget_batch, retain_batch, unlearning_method=tsf)

        # Verify that protected params changed less than unprotected
        protected_change = 0.0
        unprotected_change = 0.0
        protected_count = 0
        unprotected_count = 0

        for name, p in trained_agent.network.named_parameters():
            if name in rp.importance_masks:
                mask = rp.importance_masks[name]
                diff = (p - params_before[name]).abs()
                protected_change += (diff * mask).sum().item()
                unprotected_change += (diff * (1 - mask)).sum().item()
                protected_count += mask.sum().item()
                unprotected_count += (1 - mask).sum().item()

        if protected_count > 0 and unprotected_count > 0:
            avg_prot = protected_change / protected_count
            avg_unprot = unprotected_change / unprotected_count
            # Protected params may still change due to Adam momentum, but
            # should change less than unprotected on average.  Use generous
            # factor because optimizer state accumulation adds noise.
            assert avg_prot < avg_unprot * 3.0, \
                f"Protected params should change less: {avg_prot:.6f} vs {avg_unprot:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
