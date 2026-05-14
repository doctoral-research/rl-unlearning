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


class ConvEncoder(nn.Module):
    """Atari-style CNN encoder for 64x64x3 pixel observations.

    Used by Procgen / Atari-like envs. The output_dim matches the MLP
    hidden_dim so the rest of ActorCritic doesn't need to know it's a
    pixel env vs a flat-vector env.
    """

    def __init__(self, output_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, kernel_size=8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)), nn.ReLU(),
            nn.Flatten(),
        )
        # For 64x64 input the conv output is 64*4*4 = 1024
        self.fc = nn.Sequential(
            layer_init(nn.Linear(1024, output_dim)), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, H, W, C) and permute to (B, C, H, W)
        if x.dim() == 4 and x.shape[-1] in (1, 3, 4):
            x = x.permute(0, 3, 1, 2).contiguous()
        return self.fc(self.conv(x))


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
    """Actor-Critic network for on-policy RL.

    Continuous-action policy supports two modes:
    - Standard Gaussian (default): single learnable log_std vector, IID
      noise sampled per step. Fails on envs like Pendulum where temporally-
      correlated exploration is required.
    - gSDE (Raffin et al. 2021, "Generalized State Dependent Exploration"):
      noise = latent_sde @ noise_matrix, where noise_matrix ~ N(0, std^2)
      is resampled every `sde_sample_freq` env steps (NOT every step), and
      latent_sde = features from the shared MLP. Enables coherent
      exploration trajectories.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [64, 64],
        activation: str = "tanh",
        ortho_init: bool = True,
        action_type: str = "discrete",
        use_sde: bool = False,
        sde_log_std_init: float = -2.0,
        tanh_squash: bool = False,
        action_scale: float = 1.0,
        use_conv: bool = False,
    ):
        super().__init__()

        self.action_type = action_type
        self.action_dim = action_dim
        self.use_sde = bool(use_sde) and action_type == "continuous"
        self.tanh_squash = bool(tanh_squash) and action_type == "continuous"
        self.action_scale = float(action_scale)
        self.use_conv = bool(use_conv)

        # Shared feature extractor: ConvEncoder for pixel envs (Procgen,
        # Atari), MLP for flat-vector envs (everything else).
        if self.use_conv:
            self.shared = ConvEncoder(output_dim=hidden_dims[-1])
        else:
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
            if ortho_init:
                self.actor_mean = layer_init(self.actor_mean, std=0.01)
            if self.use_sde:
                self._latent_sde_dim = hidden_dims[-1]
                # log_std lives over (latent_sde_dim, action_dim) — full
                # state-dependent diagonal covariance per Raffin 2021.
                self.log_std = nn.Parameter(
                    torch.ones(self._latent_sde_dim, action_dim) * sde_log_std_init,
                )
                # Exploration matrix: sampled noise weights, refreshed
                # periodically via sample_sde_noise(). Registered as buffer
                # so it moves with .to(device).
                self.register_buffer(
                    "exploration_matrix",
                    torch.zeros(self._latent_sde_dim, action_dim),
                )
                self.sample_sde_noise(batch_size=1)
            else:
                self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        # Critic (value) head
        self.critic = nn.Linear(hidden_dims[-1], 1)
        if ortho_init:
            self.critic = layer_init(self.critic, std=1.0)

    def _sde_std(self):
        """Bounded std for gSDE. SB3's `expln` transform: std grows linearly
        for log_std > 0 instead of exponentially. Keeps the noise scale
        from blowing up during training (the failure mode we hit on
        HalfCheetah where action magnitudes diverged)."""
        below = torch.exp(self.log_std)
        above = self.log_std + 1.0  # linear for log_std > 0
        return torch.where(self.log_std <= 0.0, below, above)

    def sample_sde_noise(self, batch_size: int = 1) -> None:
        """Resample the gSDE exploration matrix. Call every K env steps."""
        if not self.use_sde:
            return
        with torch.no_grad():
            std = self._sde_std()
            eps = torch.randn_like(std)
            self.exploration_matrix.copy_(eps * std)

    def _sde_action_and_logprob(self, features, action=None):
        """gSDE: action = mean + features @ exploration_matrix, with a
        Gaussian log-prob whose per-dim std is sqrt(features**2 @ std**2).

        With tanh_squash, the raw Gaussian sample is passed through tanh
        before being returned, with the log-prob corrected via the
        change-of-variables Jacobian. Standard SAC-style trick that keeps
        actions bounded without the gradient-killing effect of hard clip.
        """
        mean = self.actor_mean(features)
        std_per_dim = self._sde_std()  # (latent_dim, action_dim), bounded
        # State-dependent variance: (B, A)
        variance = (features ** 2) @ (std_per_dim ** 2)
        std = torch.sqrt(variance + 1e-6)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            # Sample the pre-squash u from the gSDE distribution
            noise = features @ self.exploration_matrix
            u = mean + noise
        else:
            # Recover pre-squash u from the squashed-and-scaled action
            if self.tanh_squash:
                a_unscaled = (action / self.action_scale).clamp(-0.999, 0.999)
                u = torch.atanh(a_unscaled)
            else:
                u = action
        log_prob = dist.log_prob(u).sum(-1)
        if self.tanh_squash:
            squashed = torch.tanh(u)
            # log-det Jacobian of tanh + action scaling
            #   a = scale * tanh(u) => da/du = scale * (1 - tanh^2(u))
            log_prob = log_prob - (
                torch.log(1.0 - squashed ** 2 + 1e-6).sum(-1)
                + self.action_dim * float(np.log(self.action_scale))
            )
            out_action = (squashed * self.action_scale) if action is None else action
        else:
            out_action = u
        return out_action, log_prob, dist.entropy().sum(-1)
    
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
        elif self.use_sde:
            action, log_prob, entropy = self._sde_action_and_logprob(features, action)
            return action, log_prob, entropy, value
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
        
        # Initialize masks to zero (protect nothing by default).
        # Masks are set via update_masks() from actual importance scores.
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.masks[name] = torch.zeros_like(param.data)
    
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
        """Apply masks to gradients.

        Masks follow the 'importance' convention: mask=1 means the
        parameter is important and should be *protected* (gradient
        zeroed), consistent with RetainProtection.apply_masks_to_gradients.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.masks:
                if param.grad is not None:
                    param.grad *= (1 - self.masks[name])
