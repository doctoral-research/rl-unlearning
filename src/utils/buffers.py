"""
Replay buffer implementations
"""
import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
from collections import deque


class ReplayBuffer:
    """Simple replay buffer for storing and sampling transitions."""
    
    def __init__(self, capacity: int, observation_shape: Tuple, action_shape: Tuple):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        
        self.observations = np.zeros((capacity,) + observation_shape, dtype=np.float32)
        self.actions = np.zeros((capacity,) + action_shape, dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_observations = np.zeros((capacity,) + observation_shape, dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add transition to buffer."""
        self.observations[self.position] = obs
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_obs
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample random batch from buffer."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            "observations": torch.as_tensor(self.observations[indices], dtype=torch.float32),
            "actions": torch.as_tensor(self.actions[indices], dtype=torch.float32),
            "rewards": torch.as_tensor(self.rewards[indices], dtype=torch.float32),
            "next_observations": torch.as_tensor(self.next_observations[indices], dtype=torch.float32),
            "dones": torch.as_tensor(self.dones[indices], dtype=torch.float32),
        }
    
    def __len__(self) -> int:
        return self.size


class TrajectoryBuffer:
    """Buffer for storing complete trajectories."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.trajectories = deque(maxlen=capacity)
    
    def add_trajectory(self, trajectory: Dict[str, List]) -> None:
        """Add complete trajectory to buffer."""
        self.trajectories.append(trajectory)
    
    def sample_trajectories(self, n: int) -> List[Dict[str, List]]:
        """Sample n random trajectories."""
        if n > len(self.trajectories):
            return list(self.trajectories)
        indices = np.random.choice(len(self.trajectories), size=n, replace=False)
        return [self.trajectories[i] for i in indices]
    
    def get_all(self) -> List[Dict[str, List]]:
        """Get all trajectories."""
        return list(self.trajectories)
    
    def clear(self) -> None:
        """Clear buffer."""
        self.trajectories.clear()
    
    def __len__(self) -> int:
        return len(self.trajectories)


class DualBuffer:
    """Dual buffer system for forget and retain data."""
    
    def __init__(
        self,
        forget_capacity: int,
        retain_capacity: int,
        observation_shape: Tuple,
        action_shape: Tuple,
    ):
        self.forget_buffer = ReplayBuffer(forget_capacity, observation_shape, action_shape)
        self.retain_buffer = ReplayBuffer(retain_capacity, observation_shape, action_shape)
    
    def add_forget(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add transition to forget buffer."""
        self.forget_buffer.add(obs, action, reward, next_obs, done)
    
    def add_retain(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add transition to retain buffer."""
        self.retain_buffer.add(obs, action, reward, next_obs, done)
    
    def sample_mixed(self, batch_size: int, forget_ratio: float = 0.5) -> Dict[str, torch.Tensor]:
        """Sample mixed batch from both buffers."""
        forget_size = int(batch_size * forget_ratio)
        retain_size = batch_size - forget_size
        
        forget_batch = self.forget_buffer.sample(forget_size) if len(self.forget_buffer) > 0 else None
        retain_batch = self.retain_buffer.sample(retain_size) if len(self.retain_buffer) > 0 else None
        
        if forget_batch is None:
            return retain_batch
        if retain_batch is None:
            return forget_batch
        
        # Combine batches
        combined = {}
        for key in forget_batch.keys():
            combined[key] = torch.cat([forget_batch[key], retain_batch[key]], dim=0)
            combined[f"{key}_mask"] = torch.cat([
                torch.ones(forget_size),
                torch.zeros(retain_size)
            ])
        
        return combined
    
    def __len__(self) -> Tuple[int, int]:
        return len(self.forget_buffer), len(self.retain_buffer)
