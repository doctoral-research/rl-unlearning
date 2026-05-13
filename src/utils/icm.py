"""Intrinsic Curiosity Module (Pathak et al. 2017) for PPO.

Standard recipe used by community PPO+Taxi solutions: vanilla PPO with
count-based or no exploration can't solve sparse-reward discrete envs
like Taxi-v3 (the +20 dropoff is never sampled), but PPO + ICM does.

The module owns three small MLPs:
- feature encoder φ: obs -> feature_dim
- inverse model g: (φ(s), φ(s')) -> action logits
- forward model f: (φ(s), one_hot(a)) -> φ̂(s')

Intrinsic reward at step t:   r_i = (eta/2) * ||φ̂(s_{t+1}) - φ(s_{t+1})||^2

The encoder is trained ONLY via the inverse loss (so it learns features
relevant to controllable state changes, ignoring uncontrollable noise),
while the forward model is trained on top of the (detached) features.
This is the Pathak design that decouples curiosity from random-noise
distractors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ICM(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        feature_dim: int = 32,
        hidden: int = 64,
        eta: float = 0.1,
        beta: float = 0.2,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.feature_dim = int(feature_dim)
        self.eta = float(eta)
        self.beta = float(beta)
        self.device = device

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, feature_dim),
        )
        self.inverse_head = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        self.forward_head = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, feature_dim),
        )
        self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def _one_hot(self, actions: torch.Tensor) -> torch.Tensor:
        return F.one_hot(actions.long(), num_classes=self.action_dim).float()

    @torch.no_grad()
    def intrinsic_reward(self, obs, action, next_obs) -> float:
        """Per-step intrinsic reward used during rollout collection."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        act_t = torch.as_tensor([action], dtype=torch.long, device=self.device)
        phi = self.encoder(obs_t)
        phi_next = self.encoder(next_t)
        phi_pred = self.forward_head(torch.cat([phi, self._one_hot(act_t)], dim=-1))
        return float(0.5 * self.eta * ((phi_pred - phi_next) ** 2).sum(-1).item())

    def update(self, observations, actions, next_observations):
        """Batch update on a rollout. Returns loss dict."""
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(next_observations, dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(actions, dtype=torch.long, device=self.device)

        phi = self.encoder(obs)
        phi_next = self.encoder(next_obs)
        # Inverse: predict action from (phi, phi_next). Trains encoder.
        action_logits = self.inverse_head(torch.cat([phi, phi_next], dim=-1))
        inv_loss = F.cross_entropy(action_logits, acts)
        # Forward: predict phi_next from (phi.detach(), one_hot(a)). Encoder
        # is detached here so forward-model error can't shape the encoder.
        a_oh = self._one_hot(acts)
        phi_pred = self.forward_head(torch.cat([phi.detach(), a_oh], dim=-1))
        fwd_loss = 0.5 * ((phi_pred - phi_next.detach()) ** 2).sum(-1).mean()

        loss = (1.0 - self.beta) * inv_loss + self.beta * fwd_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "icm_loss": float(loss.item()),
            "icm_inv_loss": float(inv_loss.item()),
            "icm_fwd_loss": float(fwd_loss.item()),
        }
