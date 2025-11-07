"""
Test suite for unlearning methods
"""
import pytest
import torch
import numpy as np
import gymnasium as gym
from agents import PPOAgent
from unlearning import TrajectorySelectiveForgetting, StrategyInversion, RetainProtection


@pytest.fixture
def simple_agent():
    """Create simple agent for testing."""
    return PPOAgent(
        observation_dim=4,
        action_dim=2,
        device="cpu",
    )


@pytest.fixture
def simple_env():
    """Create simple environment for testing."""
    return gym.make("CartPole-v1")


def test_trajectory_selective_initialization(simple_agent):
    """Test trajectory selective forgetting initialization."""
    method = TrajectorySelectiveForgetting(
        agent=simple_agent,
        device="cpu",
    )
    
    assert method.agent is not None
    assert method.forget_strength > 0


def test_trajectory_similarity_computation(simple_agent):
    """Test similarity computation."""
    method = TrajectorySelectiveForgetting(
        agent=simple_agent,
        device="cpu",
    )
    
    state1 = np.array([1.0, 2.0, 3.0, 4.0])
    state2 = np.array([1.1, 2.1, 3.0, 4.0])
    
    similarity = method._compute_similarity(state1, state2)
    assert 0 <= similarity <= 1


def test_strategy_inversion_initialization(simple_agent, simple_env):
    """Test strategy inversion initialization."""
    method = StrategyInversion(
        agent=simple_agent,
        env=simple_env,
        num_seeds=10,
        device="cpu",
    )
    
    assert method.agent is not None
    assert method.num_seeds == 10


def test_strategy_inversion_state_sampling(simple_agent, simple_env):
    """Test state sampling."""
    method = StrategyInversion(
        agent=simple_agent,
        env=simple_env,
        device="cpu",
    )
    
    state = method._sample_state()
    assert state.shape == simple_env.observation_space.shape


def test_retain_protection_initialization(simple_agent):
    """Test retain protection initialization."""
    method = RetainProtection(
        agent=simple_agent,
        metaplasticity_enabled=True,
        distillation_enabled=True,
        device="cpu",
    )
    
    assert method.metaplasticity_enabled
    assert method.distillation_enabled
    assert method.teacher_network is not None


def test_retain_protection_masks(simple_agent):
    """Test metaplasticity mask creation."""
    method = RetainProtection(
        agent=simple_agent,
        metaplasticity_enabled=True,
        device="cpu",
    )
    
    assert len(method.importance_masks) > 0


if __name__ == "__main__":
    pytest.main([__file__])
