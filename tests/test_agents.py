"""
Comprehensive test suite for agents and neural network architectures.

Covers:
- MLP: construction, forward shapes, activations
- ActorCritic: discrete/continuous, forward/get_value/get_action_and_value,
  deterministic vs stochastic action selection, output shapes
- PPOAgent: init, action selection, GAE computation correctness, PPO update
  mechanics (ratio, clipping, advantage normalization), save/load round-trip,
  deterministic mode
"""
import pytest
import torch
import numpy as np
import tempfile
import os

from agents import PPOAgent, ActorCritic, MLP


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class TestMLP:
    def test_output_shape(self):
        net = MLP(input_dim=8, output_dim=3, hidden_dims=[32, 16])
        x = torch.randn(4, 8)
        out = net(x)
        assert out.shape == (4, 3)

    def test_single_hidden_layer(self):
        net = MLP(input_dim=4, output_dim=2, hidden_dims=[16])
        x = torch.randn(1, 4)
        out = net(x)
        assert out.shape == (1, 2)

    def test_activations(self):
        for act in ["tanh", "relu", "elu", "leaky_relu"]:
            net = MLP(input_dim=4, output_dim=2, hidden_dims=[16], activation=act)
            out = net(torch.randn(2, 4))
            assert out.shape == (2, 2)
            assert torch.isfinite(out).all()

    def test_unknown_activation_defaults_to_tanh(self):
        net = MLP(input_dim=4, output_dim=2, hidden_dims=[16], activation="nonexistent")
        out = net(torch.randn(2, 4))
        assert out.shape == (2, 2)

    def test_ortho_init(self):
        net = MLP(input_dim=4, output_dim=2, hidden_dims=[32], ortho_init=True)
        # Check that weights are not identity or zeros
        w = list(net.network.parameters())[0]
        assert w.abs().sum() > 0
        assert not torch.allclose(w, torch.eye(*w.shape[:2])[:w.shape[0], :w.shape[1]])

    def test_no_ortho_init(self):
        net = MLP(input_dim=4, output_dim=2, hidden_dims=[32], ortho_init=False)
        out = net(torch.randn(2, 4))
        assert out.shape == (2, 2)

    def test_gradients_flow(self):
        net = MLP(input_dim=4, output_dim=2, hidden_dims=[16, 16])
        x = torch.randn(4, 4)
        out = net(x)
        loss = out.sum()
        loss.backward()
        for p in net.parameters():
            assert p.grad is not None
            assert p.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# ActorCritic
# ---------------------------------------------------------------------------

class TestActorCriticDiscrete:
    @pytest.fixture
    def net(self):
        return ActorCritic(
            observation_dim=4, action_dim=2,
            hidden_dims=[32, 32], action_type="discrete"
        )

    def test_forward_shapes(self, net):
        obs = torch.randn(8, 4)
        logits, values = net(obs)
        assert logits.shape == (8, 2)
        assert values.shape == (8, 1)

    def test_get_value_shape(self, net):
        obs = torch.randn(8, 4)
        v = net.get_value(obs)
        assert v.shape == (8, 1)

    def test_get_action_and_value_sample(self, net):
        obs = torch.randn(8, 4)
        action, log_prob, entropy, value = net.get_action_and_value(obs)
        assert action.shape == (8,)
        assert log_prob.shape == (8,)
        assert entropy.shape == (8,)
        assert value.shape == (8, 1)
        # Actions should be valid discrete actions
        assert torch.all(action >= 0) and torch.all(action < 2)

    def test_get_action_and_value_with_action(self, net):
        obs = torch.randn(8, 4)
        given_action = torch.randint(0, 2, (8,))
        action, log_prob, entropy, value = net.get_action_and_value(obs, given_action)
        # Should return the given action back
        assert torch.equal(action, given_action)
        # log_prob should be negative (log of probability)
        assert torch.all(log_prob <= 0)

    def test_log_probs_sum_to_one(self, net):
        """Exponentiated log_probs across all actions should sum to 1."""
        obs = torch.randn(4, 4)
        logits, _ = net(obs)
        probs = torch.softmax(logits, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_entropy_is_nonnegative(self, net):
        obs = torch.randn(16, 4)
        _, _, entropy, _ = net.get_action_and_value(obs)
        assert torch.all(entropy >= 0)

    def test_entropy_range(self, net):
        """Entropy should be between 0 and log(action_dim)."""
        obs = torch.randn(32, 4)
        _, _, entropy, _ = net.get_action_and_value(obs)
        max_entropy = np.log(2)  # 2 actions
        assert torch.all(entropy <= max_entropy + 1e-5)

    def test_batch_size_one(self, net):
        obs = torch.randn(1, 4)
        action, log_prob, entropy, value = net.get_action_and_value(obs)
        assert action.shape == (1,)


class TestActorCriticContinuous:
    @pytest.fixture
    def net(self):
        return ActorCritic(
            observation_dim=3, action_dim=2,
            hidden_dims=[32, 32], action_type="continuous"
        )

    def test_forward_shapes(self, net):
        obs = torch.randn(8, 3)
        action_mean, values = net(obs)
        assert action_mean.shape == (8, 2)
        assert values.shape == (8, 1)

    def test_get_action_and_value_sample(self, net):
        obs = torch.randn(8, 3)
        action, log_prob, entropy, value = net.get_action_and_value(obs)
        assert action.shape == (8, 2)
        assert log_prob.shape == (8,)  # summed across action dims
        assert entropy.shape == (8,)  # summed across action dims

    def test_get_action_and_value_with_action(self, net):
        obs = torch.randn(8, 3)
        given_action = torch.randn(8, 2)
        action, log_prob, entropy, value = net.get_action_and_value(obs, given_action)
        assert torch.allclose(action, given_action)

    def test_logstd_is_learnable(self, net):
        assert net.actor_logstd.requires_grad


# ---------------------------------------------------------------------------
# PPOAgent
# ---------------------------------------------------------------------------

class TestPPOAgentInit:
    def test_default_init(self):
        agent = PPOAgent(observation_dim=4, action_dim=2, device="cpu")
        assert agent.network is not None
        assert agent.optimizer is not None
        assert agent.action_type == "discrete"
        assert agent.gamma == 0.99
        assert agent.clip_epsilon == 0.2

    def test_custom_hyperparams(self):
        agent = PPOAgent(
            observation_dim=8, action_dim=4,
            gamma=0.95, gae_lambda=0.9, clip_epsilon=0.1,
            learning_rate=1e-3, n_epochs=5, batch_size=32,
            device="cpu"
        )
        assert agent.gamma == 0.95
        assert agent.gae_lambda == 0.9
        assert agent.clip_epsilon == 0.1
        assert agent.n_epochs == 5
        assert agent.batch_size == 32

    def test_kwargs_accepted(self):
        """PPOAgent should accept extra kwargs without error."""
        agent = PPOAgent(
            observation_dim=4, action_dim=2,
            device="cpu", some_extra_param=42
        )
        assert agent is not None


class TestPPOActionSelection:
    @pytest.fixture
    def agent(self):
        return PPOAgent(
            observation_dim=4, action_dim=2,
            action_type="discrete", device="cpu"
        )

    def test_stochastic_action(self, agent):
        obs = np.random.randn(4).astype(np.float32)
        action, info = agent.select_action(obs, deterministic=False)
        assert action in [0, 1]
        assert "log_prob" in info
        assert "value" in info

    def test_deterministic_action(self, agent):
        """Deterministic action should always return the same action for the
        same observation."""
        obs = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        actions = set()
        for _ in range(10):
            action, _ = agent.select_action(obs, deterministic=True)
            actions.add(int(action))
        assert len(actions) == 1, "Deterministic action should be consistent"

    def test_stochastic_has_variance(self, agent):
        """Over many samples, stochastic action should produce both actions."""
        obs = np.zeros(4, dtype=np.float32)
        actions = set()
        for _ in range(100):
            action, _ = agent.select_action(obs, deterministic=False)
            actions.add(int(action))
        assert len(actions) == 2, "Stochastic should produce both actions over 100 trials"

    def test_value_is_scalar(self, agent):
        obs = np.random.randn(4).astype(np.float32)
        _, info = agent.select_action(obs)
        assert isinstance(info["value"], float)
        assert isinstance(info["log_prob"], float)


class TestPPOGAE:
    @pytest.fixture
    def agent(self):
        return PPOAgent(observation_dim=4, action_dim=2, device="cpu")

    def test_shapes(self, agent):
        n = 10
        rewards = torch.ones(n)
        values = torch.ones(n) * 0.5
        dones = torch.zeros(n)
        dones[-1] = 1.0
        advantages, returns = agent.compute_gae(rewards, values, dones, 0.0)
        assert advantages.shape == (n,)
        assert returns.shape == (n,)

    def test_returns_equal_advantages_plus_values(self, agent):
        """By definition, returns = advantages + values."""
        rewards = torch.tensor([1.0, 1.0, 1.0, 0.0])
        values = torch.tensor([0.5, 0.6, 0.7, 0.1])
        dones = torch.tensor([0.0, 0.0, 0.0, 1.0])
        advantages, returns = agent.compute_gae(rewards, values, dones, 0.0)
        assert torch.allclose(returns, advantages + values, atol=1e-5)

    def test_terminal_state_no_bootstrap(self, agent):
        """At terminal state, next value should not bootstrap."""
        rewards = torch.tensor([1.0])
        values = torch.tensor([0.5])
        dones = torch.tensor([1.0])
        advantages, returns = agent.compute_gae(rewards, values, dones, 100.0)
        # With done=1, next_value (100.0) should be ignored
        # delta = reward + gamma * next_value * (1-done) - value = 1.0 + 0 - 0.5 = 0.5
        expected_advantage = 0.5
        assert advantages[0].item() == pytest.approx(expected_advantage, abs=1e-5)

    def test_non_terminal_bootstraps(self, agent):
        """Without terminal, next_value should be used for bootstrapping."""
        rewards = torch.tensor([1.0])
        values = torch.tensor([0.5])
        dones = torch.tensor([0.0])
        next_value = 1.0
        advantages, returns = agent.compute_gae(rewards, values, dones, next_value)
        # delta = 1.0 + 0.99 * 1.0 * 1.0 - 0.5 = 1.49
        expected_advantage = 1.49
        assert advantages[0].item() == pytest.approx(expected_advantage, abs=1e-4)

    def test_all_done_no_propagation(self, agent):
        """If every step is terminal, advantages should be simple TD errors."""
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.5, 1.0, 1.5])
        dones = torch.tensor([1.0, 1.0, 1.0])
        advantages, _ = agent.compute_gae(rewards, values, dones, 0.0)
        # Each advantage = reward - value (no bootstrapping)
        expected = torch.tensor([0.5, 1.0, 1.5])
        assert torch.allclose(advantages, expected, atol=1e-5)

    def test_discount_propagation(self, agent):
        """Advantages at earlier timesteps should reflect future rewards."""
        rewards = torch.tensor([0.0, 0.0, 10.0])
        values = torch.zeros(3)
        dones = torch.tensor([0.0, 0.0, 0.0])  # no terminal, so rewards propagate
        advantages, _ = agent.compute_gae(rewards, values, dones, 0.0)
        # Last step gets the reward, earlier steps get discounted versions
        assert advantages[2].item() > 0
        assert advantages[1].item() > 0  # propagated reward
        assert advantages[0].item() > 0
        assert advantages[0].item() < advantages[1].item()  # more discounted


class TestPPOUpdate:
    @pytest.fixture
    def agent(self):
        return PPOAgent(
            observation_dim=4, action_dim=2,
            n_epochs=2, batch_size=8, device="cpu"
        )

    def _make_rollout(self, n=32):
        return {
            "observations": np.random.randn(n, 4).astype(np.float32),
            "actions": np.random.randint(0, 2, size=(n,)).astype(np.float32),
            "log_probs": np.random.randn(n).astype(np.float32),
            "advantages": np.random.randn(n).astype(np.float32),
            "returns": np.random.randn(n).astype(np.float32),
        }

    def test_returns_metrics(self, agent):
        rollout = self._make_rollout()
        metrics = agent.update(rollout)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics

    def test_weights_change(self, agent):
        params_before = {n: p.clone() for n, p in agent.network.named_parameters()}
        rollout = self._make_rollout()
        agent.update(rollout)
        changed = any(
            not torch.allclose(p, params_before[n])
            for n, p in agent.network.named_parameters()
        )
        assert changed

    def test_value_loss_positive(self, agent):
        rollout = self._make_rollout()
        metrics = agent.update(rollout)
        assert metrics["value_loss"] >= 0

    def test_entropy_positive(self, agent):
        rollout = self._make_rollout()
        metrics = agent.update(rollout)
        assert metrics["entropy"] > 0

    def test_multiple_updates_converge(self, agent):
        """Policy loss should decrease (or stay stable) over many updates
        with the same data (overfitting check)."""
        rollout = self._make_rollout(64)
        losses = []
        for _ in range(5):
            m = agent.update(rollout)
            losses.append(m["policy_loss"])
        # Not strictly decreasing, but should not explode
        assert losses[-1] < losses[0] * 10, "Policy loss should not explode"


class TestPPOSaveLoad:
    def test_save_load_roundtrip(self):
        agent = PPOAgent(observation_dim=4, action_dim=2, device="cpu")
        obs = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)

        # Get action before save
        action_before, info_before = agent.select_action(obs, deterministic=True)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent.save(path)

            # Create new agent and load
            agent2 = PPOAgent(observation_dim=4, action_dim=2, device="cpu")
            agent2.load(path)

            action_after, info_after = agent2.select_action(obs, deterministic=True)
            assert action_before == action_after
            assert info_before["value"] == pytest.approx(info_after["value"], abs=1e-5)
        finally:
            os.unlink(path)

    def test_save_contains_network_and_optimizer(self):
        agent = PPOAgent(observation_dim=4, action_dim=2, device="cpu")
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent.save(path)
            checkpoint = torch.load(path, map_location="cpu")
            assert "network" in checkpoint
            assert "optimizer" in checkpoint
        finally:
            os.unlink(path)

    def test_state_dict_roundtrip(self):
        agent = PPOAgent(observation_dim=4, action_dim=2, device="cpu")
        sd = agent.state_dict()
        assert "network" in sd
        assert "optimizer" in sd

        agent2 = PPOAgent(observation_dim=4, action_dim=2, device="cpu")
        agent2.load_state_dict(sd)

        # Verify weights match
        for (_, p1), (_, p2) in zip(
            agent.network.named_parameters(),
            agent2.network.named_parameters()
        ):
            assert torch.allclose(p1, p2)


class TestMetaplasticityMask:
    """Test the standalone MetaplasticityMask module."""

    def test_init_masks_to_zeros(self):
        """Masks init to zero (protect nothing) until importance is computed."""
        from agents import MetaplasticityMask
        net = ActorCritic(observation_dim=4, action_dim=2, hidden_dims=[16])
        mm = MetaplasticityMask(net)
        for name, mask in mm.masks.items():
            assert torch.all(mask == 0.0)

    def test_update_masks_creates_binary(self):
        from agents import MetaplasticityMask
        net = ActorCritic(observation_dim=4, action_dim=2, hidden_dims=[16])
        mm = MetaplasticityMask(net)
        importances = {n: torch.randn_like(p).abs()
                       for n, p in net.named_parameters() if p.requires_grad}
        mm.update_masks(importances, threshold=0.5)
        for name, mask in mm.masks.items():
            unique = mask.unique()
            assert all(v in [0.0, 1.0] for v in unique.tolist())

    def test_apply_masks_modifies_gradients(self):
        """mask=1 means 'important, protect' → grad *= (1 - 1) = 0."""
        from agents import MetaplasticityMask
        net = ActorCritic(observation_dim=4, action_dim=2, hidden_dims=[16])
        mm = MetaplasticityMask(net)
        # Set masks to one for all params (protect everything → zero grads)
        for name in mm.masks:
            mm.masks[name] = torch.ones_like(mm.masks[name])

        obs = torch.randn(4, 4)
        logits, val = net(obs)
        loss = logits.sum() + val.sum()
        loss.backward()

        # Before applying: grads exist
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in net.parameters())
        assert has_grad

        mm.apply_masks()

        # After applying with mask=1 (protect): grads should be zero
        for name, p in net.named_parameters():
            if p.grad is not None and name in mm.masks:
                assert torch.allclose(p.grad, torch.zeros_like(p.grad))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
