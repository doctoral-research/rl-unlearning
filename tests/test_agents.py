"""
Test suite for agents
"""
import pytest
import torch
import numpy as np
from agents import PPOAgent, ActorCritic


def test_actor_critic_discrete():
    """Test ActorCritic with discrete actions."""
    network = ActorCritic(
        observation_dim=4,
        action_dim=2,
        hidden_dims=[32, 32],
        action_type="discrete",
    )
    
    # Test forward pass
    obs = torch.randn(8, 4)
    logits, values = network(obs)
    
    assert logits.shape == (8, 2)
    assert values.shape == (8, 1)


def test_actor_critic_continuous():
    """Test ActorCritic with continuous actions."""
    network = ActorCritic(
        observation_dim=3,
        action_dim=1,
        hidden_dims=[32, 32],
        action_type="continuous",
    )
    
    # Test forward pass
    obs = torch.randn(8, 3)
    action_mean, values = network(obs)
    
    assert action_mean.shape == (8, 1)
    assert values.shape == (8, 1)


def test_ppo_agent_initialization():
    """Test PPO agent initialization."""
    agent = PPOAgent(
        observation_dim=4,
        action_dim=2,
        action_type="discrete",
        device="cpu",
    )
    
    assert agent.network is not None
    assert agent.optimizer is not None


def test_ppo_select_action():
    """Test action selection."""
    agent = PPOAgent(
        observation_dim=4,
        action_dim=2,
        action_type="discrete",
        device="cpu",
    )
    
    obs = np.random.randn(4)
    action, info = agent.select_action(obs)
    
    assert action in [0, 1]
    assert "log_prob" in info
    assert "value" in info


def test_ppo_compute_gae():
    """Test GAE computation."""
    agent = PPOAgent(
        observation_dim=4,
        action_dim=2,
        device="cpu",
    )
    
    rewards = torch.tensor([1.0, 1.0, 1.0, 0.0])
    values = torch.tensor([0.5, 0.6, 0.7, 0.1])
    dones = torch.tensor([0.0, 0.0, 0.0, 1.0])
    next_value = 0.0
    
    advantages, returns = agent.compute_gae(rewards, values, dones, next_value)
    
    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape


def test_ppo_update():
    """Test policy update."""
    agent = PPOAgent(
        observation_dim=4,
        action_dim=2,
        device="cpu",
        n_epochs=1,
        batch_size=4,
    )
    
    # Create dummy rollout
    rollout = {
        "observations": np.random.randn(16, 4),
        "actions": np.random.randint(0, 2, size=(16,)),
        "log_probs": np.random.randn(16),
        "advantages": np.random.randn(16),
        "returns": np.random.randn(16),
    }
    
    metrics = agent.update(rollout)
    
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics


if __name__ == "__main__":
    pytest.main([__file__])
