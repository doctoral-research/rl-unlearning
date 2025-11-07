"""
Neural network architectures for RL agents
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import numpy as np


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize layer with orthogonal initialization."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MLP(nn.Module):
    """Multi-layer perceptron."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: str = "tanh",
        ortho_init: bool = True,
    ):
        super().__init__()
        
        self.activation = self._get_activation(activation)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layer = nn.Linear(prev_dim, hidden_dim)
            if ortho_init:
                layer = layer_init(layer)
            layers.append(layer)
            layers.append(self.activation)
            prev_dim = hidden_dim
        
        output_layer = nn.Linear(prev_dim, output_dim)
        if ortho_init:
            output_layer = layer_init(output_layer, std=0.01)
        layers.append(output_layer)
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
        }
        return activations.get(name, nn.Tanh())


class ActorCritic(nn.Module):
    """Actor-Critic network for on-policy RL."""
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [64, 64],
        activation: str = "tanh",
        ortho_init: bool = True,
        action_type: str = "discrete",
    ):
        super().__init__()
        
        self.action_type = action_type
        self.action_dim = action_dim
        
        # Shared feature extractor (optional)
        self.shared = MLP(
            input_dim=observation_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            activation=activation,
            ortho_init=ortho_init,
        )
        
        # Actor (policy) head
        if action_type == "discrete":
            self.actor = nn.Linear(hidden_dims[-1], action_dim)
            if ortho_init:
                self.actor = layer_init(self.actor, std=0.01)
        else:  # continuous
            self.actor_mean = nn.Linear(hidden_dims[-1], action_dim)
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
            if ortho_init:
                self.actor_mean = layer_init(self.actor_mean, std=0.01)
        
        # Critic (value) head
        self.critic = nn.Linear(hidden_dims[-1], 1)
        if ortho_init:
            self.critic = layer_init(self.critic, std=1.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits/mean and value."""
        features = self.shared(x)
        
        if self.action_type == "discrete":
            action_logits = self.actor(features)
            value = self.critic(features)
            return action_logits, value
        else:
            action_mean = self.actor_mean(features)
            value = self.critic(features)
            return action_mean, value
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        features = self.shared(x)
        return self.critic(features)
    
    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, entropy, and value."""
        features = self.shared(x)
        value = self.critic(features)
        
        if self.action_type == "discrete":
            logits = self.actor(features)
            probs = torch.distributions.Categorical(logits=logits)
            
            if action is None:
                action = probs.sample()
            
            return action, probs.log_prob(action), probs.entropy(), value
        else:
            action_mean = self.actor_mean(features)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = torch.distributions.Normal(action_mean, action_std)
            
            if action is None:
                action = probs.sample()
            
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value


class MetaplasticityMask(nn.Module):
    """Metaplasticity mask for protecting important weights."""
    
    def __init__(self, model: nn.Module, mask_type: str = "importance"):
        super().__init__()
        self.model = model
        self.mask_type = mask_type
        self.masks = {}
        
        # Initialize masks
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.masks[name] = torch.ones_like(param.data)
    
    def compute_importance(self, data_loader, criterion):
        """Compute parameter importance scores."""
        importances = {name: torch.zeros_like(param) 
                      for name, param in self.model.named_parameters() 
                      if param.requires_grad}
        
        self.model.eval()
        for batch in data_loader:
            self.model.zero_grad()
            loss = criterion(batch)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    importances[name] += param.grad.abs()
        
        # Normalize
        for name in importances:
            importances[name] /= len(data_loader)
        
        return importances
    
    def update_masks(self, importances: dict, threshold: float = 0.5):
        """Update masks based on importance scores."""
        for name, importance in importances.items():
            # Threshold-based masking
            threshold_value = torch.quantile(importance.flatten(), threshold)
            self.masks[name] = (importance >= threshold_value).float()
    
    def apply_masks(self):
        """Apply masks to gradients."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.masks:
                if param.grad is not None:
                    param.grad *= self.masks[name]
