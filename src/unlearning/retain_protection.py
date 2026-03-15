"""
Retain Protection with Metaplasticity
Pillar 3: Metaplasticity masks and selective distillation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Any
import copy


class RetainProtection:
    """Retain protection using metaplasticity and distillation."""
    
    def __init__(
        self,
        agent,
        metaplasticity_enabled: bool = True,
        mask_type: str = "importance",
        mask_threshold: float = 0.5,
        consolidation_strength: float = 0.9,
        distillation_enabled: bool = True,
        distillation_temperature: float = 2.0,
        distillation_alpha: float = 0.5,
        device: str = "cpu",
    ):
        self.agent = agent
        self.device = torch.device(device)
        
        # Metaplasticity settings
        self.metaplasticity_enabled = metaplasticity_enabled
        self.mask_type = mask_type
        self.mask_threshold = mask_threshold
        self.consolidation_strength = consolidation_strength
        
        # Distillation settings
        self.distillation_enabled = distillation_enabled
        self.temperature = distillation_temperature
        self.alpha = distillation_alpha
        
        # Initialize teacher network for distillation
        if self.distillation_enabled:
            self.teacher_network = copy.deepcopy(agent.network)
            self.teacher_network.eval()
            for param in self.teacher_network.parameters():
                param.requires_grad = False
        
        # Initialize importance masks
        self.importance_masks = {}
        self.gradient_accumulators = {}
        
        if self.metaplasticity_enabled:
            self._initialize_masks()
    
    def _initialize_masks(self):
        """Initialize importance masks for all parameters."""
        for name, param in self.agent.network.named_parameters():
            if param.requires_grad:
                self.importance_masks[name] = torch.ones_like(param.data)
                self.gradient_accumulators[name] = torch.zeros_like(param.data)
    
    def compute_importance_scores(self, data_loader) -> Dict[str, torch.Tensor]:
        """Compute parameter importance scores from retain data."""
        importances = {}
        
        for name, param in self.agent.network.named_parameters():
            if param.requires_grad:
                importances[name] = torch.zeros_like(param.data)
        
        self.agent.network.eval()
        n_batches = 0
        
        for batch in data_loader:
            self.agent.network.zero_grad()
            
            # Forward pass
            observations = batch["observations"].to(self.device)
            actions = batch["actions"].to(self.device)
            
            _, log_probs, _, values = self.agent.network.get_action_and_value(
                observations, actions
            )
            
            # Compute loss
            loss = -log_probs.mean() + 0.5 * values.pow(2).mean()
            loss.backward()
            
            # Accumulate gradient magnitudes
            for name, param in self.agent.network.named_parameters():
                if param.requires_grad and param.grad is not None:
                    importances[name] += param.grad.abs()
            
            n_batches += 1
        
        # Normalize by number of batches
        for name in importances:
            importances[name] /= max(n_batches, 1)
        
        self.agent.network.train()
        return importances
    
    def update_masks(self, importances: Dict[str, torch.Tensor]):
        """Update metaplasticity masks based on importance scores."""
        for name, importance in importances.items():
            if name in self.importance_masks:
                # Threshold-based masking
                threshold_value = torch.quantile(importance.flatten(), self.mask_threshold)
                new_mask = (importance >= threshold_value).float()
                
                # Consolidate with existing mask
                self.importance_masks[name] = (
                    self.consolidation_strength * self.importance_masks[name] +
                    (1 - self.consolidation_strength) * new_mask
                )
    
    def apply_masks_to_gradients(self):
        """Apply importance masks to gradients during backprop."""
        if not self.metaplasticity_enabled:
            return
        
        for name, param in self.agent.network.named_parameters():
            if param.requires_grad and name in self.importance_masks:
                if param.grad is not None:
                    # Reduce gradients for important parameters (protect them)
                    param.grad *= (1 - self.importance_masks[name])
    
    def compute_distillation_loss(
        self,
        observations: torch.Tensor,
        student_action_output: torch.Tensor,
        student_value: torch.Tensor,
    ) -> torch.Tensor:
        """Compute knowledge distillation loss."""
        if not self.distillation_enabled:
            return torch.tensor(0.0, device=self.device)
        
        with torch.no_grad():
            teacher_action_output, teacher_value = self.teacher_network(observations)
        
        # Distillation for policy
        if self.agent.action_type == "discrete":
            student_log_probs = F.log_softmax(student_action_output / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_action_output / self.temperature, dim=-1)
            policy_distill_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="batchmean"
            ) * (self.temperature ** 2)
        else:
            policy_distill_loss = F.mse_loss(student_action_output, teacher_action_output)
        
        # Distillation for value
        value_distill_loss = F.mse_loss(student_value, teacher_value)
        
        return self.alpha * (policy_distill_loss + value_distill_loss)
    
    def update_teacher(self):
        """Update teacher network with current student weights."""
        if self.distillation_enabled:
            self.teacher_network.load_state_dict(self.agent.network.state_dict())
            self.teacher_network.eval()
    
    def protected_update(
        self,
        forget_batch: Dict[str, torch.Tensor],
        retain_batch: Optional[Dict[str, torch.Tensor]] = None,
        unlearning_method = None,
    ) -> Dict[str, float]:
        """Perform protected update with metaplasticity and distillation."""
        metrics = {}
        
        # Standard unlearning loss
        if unlearning_method is not None:
            unlearn_loss = unlearning_method._compute_forget_loss(forget_batch)
            metrics["unlearn_loss"] = unlearn_loss.item()
        else:
            unlearn_loss = torch.tensor(0.0, device=self.device)
        
        # Retain loss with distillation
        retain_loss = torch.tensor(0.0, device=self.device)
        distill_loss = torch.tensor(0.0, device=self.device)
        
        if retain_batch is not None:
            observations = retain_batch["observations"].to(self.device)
            actions = retain_batch["actions"].to(self.device)
            
            # Get student outputs
            action_output, value = self.agent.network(observations)
            
            # Distillation loss
            distill_loss = self.compute_distillation_loss(observations, action_output, value)
            metrics["distillation_loss"] = distill_loss.item()
            
            # Standard retain loss
            _, log_probs, entropy, values = self.agent.network.get_action_and_value(
                observations, actions
            )
            
            if "returns" in retain_batch:
                returns = retain_batch["returns"].to(self.device)
                advantages = returns - values.flatten().detach()
                policy_loss = -(log_probs * advantages).mean()
                value_loss = ((values.flatten() - returns) ** 2).mean()
                retain_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
                metrics["retain_loss"] = retain_loss.item()
        
        # Total loss
        total_loss = unlearn_loss + retain_loss + distill_loss
        
        # Backpropagation
        self.agent.optimizer.zero_grad()
        total_loss.backward()
        
        # Apply metaplasticity masks
        self.apply_masks_to_gradients()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(self.agent.network.parameters(), self.agent.max_grad_norm)
        
        # Update
        self.agent.optimizer.step()
        
        metrics["total_loss"] = total_loss.item()
        return metrics
    
    def compute_stability_metrics(self, eval_returns: list, baseline_returns: list) -> Dict[str, float]:
        """Compute retain stability metrics."""
        if len(eval_returns) == 0 or len(baseline_returns) == 0:
            return {"stability_index": 0.0, "relative_degradation": 0.0}

        current_mean = np.mean(eval_returns)
        baseline_mean = np.mean(baseline_returns)
        baseline_std = np.std(baseline_returns)

        scale = max(abs(baseline_mean), baseline_std, 1.0)
        degradation = abs(current_mean - baseline_mean) / scale
        stability_index = max(0.0, 1.0 - degradation)

        return {
            "stability_index": stability_index,
            "relative_degradation": degradation,
            "current_performance": current_mean,
            "baseline_performance": baseline_mean,
        }
