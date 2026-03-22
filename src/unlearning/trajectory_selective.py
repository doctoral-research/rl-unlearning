"""
Trajectory-Selective Forgetting Method
Pillar 1: Targeted negative learning signals with loss decomposition
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


class TrajectorySelectiveForgetting:
    """Trajectory-selective forgetting with loss decomposition."""
    
    def __init__(
        self,
        agent,
        forget_strength: float = 0.5,
        loss_decomposition: bool = True,
        target_selection_method: str = "similarity",
        target_selection_threshold: float = 0.8,
        gradient_reversal: bool = True,
        loss_weights: Dict[str, float] = None,
        device: str = "cpu",
    ):
        self.agent = agent
        self.forget_strength = forget_strength
        self.loss_decomposition = loss_decomposition
        self.target_selection_method = target_selection_method
        self.target_selection_threshold = target_selection_threshold
        self.gradient_reversal = gradient_reversal
        self.device = torch.device(device)
        
        self.loss_weights = loss_weights or {
            "forget": 1.0,
            "retain": 1.0,
            "regularization": 0.001,
        }
        
        self.forget_trajectories = []
        self.retain_trajectories = []
    
    def identify_forget_trajectories(
        self,
        trajectories: List[Dict],
        toxic_states: List[np.ndarray] = None,
        toxic_actions: List = None,
    ) -> List[int]:
        """Identify trajectories to forget based on criteria."""
        forget_indices = []
        
        for i, traj in enumerate(trajectories):
            if self._should_forget(traj, toxic_states, toxic_actions):
                forget_indices.append(i)
        
        return forget_indices
    
    def _should_forget(
        self,
        trajectory: Dict,
        toxic_states: List[np.ndarray] = None,
        toxic_actions: List = None,
    ) -> bool:
        """Determine if trajectory should be forgotten."""
        if toxic_states is not None:
            # Check state-based forgetting
            observations = trajectory["observations"]
            for toxic_state in toxic_states:
                for obs in observations:
                    similarity = self._compute_similarity(obs, toxic_state)
                    if similarity > self.target_selection_threshold:
                        return True
        
        if toxic_actions is not None:
            # Check action sequence forgetting
            actions = trajectory["actions"]
            if self._contains_action_sequence(actions, toxic_actions):
                return True
        
        return False
    
    def _compute_similarity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """Compute similarity between states."""
        if self.target_selection_method == "similarity":
            # Euclidean distance-based similarity
            distance = np.linalg.norm(state1 - state2)
            return np.exp(-distance)
        elif self.target_selection_method == "cosine":
            # Cosine similarity (guard against zero-norm vectors)
            norm_product = np.linalg.norm(state1) * np.linalg.norm(state2)
            if norm_product < 1e-8:
                return 0.0
            return np.dot(state1, state2) / norm_product
        else:
            return 0.0
    
    def _contains_action_sequence(self, actions: List, target_sequence: List) -> bool:
        """Check if action sequence contains target sequence."""
        if len(actions) < len(target_sequence):
            return False
        
        for i in range(len(actions) - len(target_sequence) + 1):
            if actions[i:i+len(target_sequence)] == target_sequence:
                return True
        return False
    
    def unlearn_step(
        self,
        forget_batch: Dict[str, torch.Tensor],
        retain_batch: Dict[str, torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Perform one unlearning step with loss decomposition."""
        metrics = {}
        
        # Forget loss (negative learning)
        forget_loss = self._compute_forget_loss(forget_batch)
        metrics["forget_loss"] = forget_loss.item()
        
        # Retain loss (preserve knowledge)
        retain_loss = 0.0
        if retain_batch is not None:
            retain_loss = self._compute_retain_loss(retain_batch)
            metrics["retain_loss"] = retain_loss.item()
        
        # Regularization
        reg_loss = self._compute_regularization()
        metrics["regularization_loss"] = reg_loss.item()
        
        # Combined loss
        total_loss = (
            self.loss_weights["forget"] * forget_loss +
            self.loss_weights["retain"] * retain_loss +
            self.loss_weights["regularization"] * reg_loss
        )
        
        # Backprop and update
        self.agent.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.agent.network.parameters(), self.agent.max_grad_norm)
        self.agent.optimizer.step()
        
        metrics["total_loss"] = total_loss.item()
        return metrics
    
    def _compute_forget_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for forget data (negative learning signal).

        Two complementary objectives:
        1. Gradient ascent on log_prob: increase the loss the policy would
           normally minimize, pushing it away from these actions.
        2. Entropy maximisation: push the policy toward uniform randomness
           on forget states so it has no preference.
        """
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)

        # Get policy outputs
        action_output, value = self.agent.network(observations)

        if self.agent.action_type == "discrete":
            dist = torch.distributions.Categorical(logits=action_output)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            if self.gradient_reversal:
                # Ascend on log-prob (make these actions less likely)
                # AND push toward uniform (maximize entropy)
                policy_loss = log_probs.mean() - entropy.mean()
            else:
                policy_loss = -entropy.mean()
        else:
            action_mean = action_output
            action_std = torch.exp(self.agent.network.actor_logstd.expand_as(action_mean))
            dist = torch.distributions.Normal(action_mean, action_std)
            log_probs = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1)

            if self.gradient_reversal:
                policy_loss = log_probs.mean() - entropy.mean()
            else:
                policy_loss = -entropy.mean()

        return self.forget_strength * policy_loss
    
    def _compute_retain_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for retain data (behavior cloning to preserve knowledge).

        Uses behavior cloning (maximize log-probability of recorded actions)
        rather than policy gradient with advantages.  The retain objective is
        simply to keep the policy close to its original behavior on retain
        states — no return estimation is needed for that.
        """
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)

        _, log_probs, entropy, _ = self.agent.network.get_action_and_value(
            observations, actions
        )

        # Behavior cloning: maximize probability of recorded retain actions
        policy_loss = -log_probs.mean()

        # Entropy bonus: subtracting entropy from the loss encourages the
        # policy to maintain stochasticity on retain states rather than
        # collapsing to a deterministic distribution.
        return policy_loss - 0.01 * entropy.mean()
    
    def _compute_regularization(self) -> torch.Tensor:
        """Compute L2 regularization term (weight decay)."""
        l2_reg = torch.tensor(0.0, device=self.device)
        for param in self.agent.network.parameters():
            l2_reg += param.pow(2).sum()
        return l2_reg
