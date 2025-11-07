"""
PPO (Proximal Policy Optimization) Agent
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional
from .networks import ActorCritic


class PPOAgent:
    """Proximal Policy Optimization agent."""
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        action_type: str = "discrete",
        hidden_dims: list = [64, 64],
        activation: str = "tanh",
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        clip_value: bool = True,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = "cpu",
        **kwargs
    ):
        self.device = torch.device(device)
        self.action_type = action_type
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.clip_value = clip_value
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Network
        self.network = ActorCritic(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            ortho_init=True,
            action_type=action_type,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-5)
        
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict]:
        """Select action given observation."""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(obs_tensor)
        
        if deterministic and self.action_type == "discrete":
            features = self.network.shared(obs_tensor)
            logits = self.network.actor(features)
            action = torch.argmax(logits, dim=-1)
        
        action_np = action.cpu().numpy()[0]
        
        info = {
            "log_prob": log_prob.item(),
            "value": value.item(),
        }
        
        return action_np, info
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, rollout_buffer: Dict) -> Dict[str, float]:
        """Update policy using PPO."""
        observations = torch.FloatTensor(rollout_buffer["observations"]).to(self.device)
        actions = torch.FloatTensor(rollout_buffer["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout_buffer["log_probs"]).to(self.device)
        advantages = torch.FloatTensor(rollout_buffer["advantages"]).to(self.device)
        returns = torch.FloatTensor(rollout_buffer["returns"]).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for epoch in range(self.n_epochs):
            # Shuffle data
            indices = torch.randperm(len(observations))
            
            for start in range(0, len(observations), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get current policy outputs
                _, log_probs, entropy, values = self.network.get_action_and_value(
                    batch_obs, batch_actions
                )
                
                # Policy loss
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                values = values.flatten()
                if self.clip_value:
                    values_clipped = batch_returns + torch.clamp(
                        values - batch_returns,
                        -self.clip_epsilon,
                        self.clip_epsilon
                    )
                    value_loss = torch.max(
                        (values - batch_returns) ** 2,
                        (values_clipped - batch_returns) ** 2
                    ).mean()
                else:
                    value_loss = ((values - batch_returns) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                n_updates += 1
        
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }
    
    def save(self, path: str) -> None:
        """Save agent."""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str) -> None:
        """Load agent."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
    
    def state_dict(self):
        """Get state dict."""
        return {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.network.load_state_dict(state_dict["network"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
