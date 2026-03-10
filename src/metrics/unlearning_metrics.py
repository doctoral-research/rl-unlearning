"""
Evaluation metrics for unlearning
"""
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import auc
import torch


class UnlearningMetrics:
    """Compute unlearning evaluation metrics."""
    
    def __init__(self):
        self.forget_scores_history = []
        self.retain_scores_history = []
        self.baseline_retain_performance = None
    
    def set_baseline(self, baseline_returns: List[float]):
        """Set baseline performance before unlearning."""
        self.baseline_retain_performance = np.mean(baseline_returns)
    
    def compute_aufc(
        self,
        forget_scores: List[float],
        timesteps: List[int] = None,
    ) -> float:
        """
        Compute Area Under Forgetting Curve (AUFC).
        Measures cumulative forgetting effectiveness over time.
        Higher is better (more effective forgetting).
        """
        if len(forget_scores) < 2:
            return 0.0
        
        if timesteps is None:
            timesteps = list(range(len(forget_scores)))
        
        # Normalize timesteps to [0, 1]
        timesteps_norm = np.array(timesteps)
        timesteps_norm = (timesteps_norm - timesteps_norm.min()) / (timesteps_norm.max() - timesteps_norm.min() + 1e-8)
        
        # Compute AUC
        return auc(timesteps_norm, forget_scores)
    
    def compute_retain_stability_index(
        self,
        current_returns: List[float],
        baseline_returns: List[float] = None,
    ) -> float:
        """
        Compute Retain Stability Index (RSI).
        Measures how well non-target behaviors are preserved.
        Range: [0, 1], where 1 means perfect preservation.
        """
        if baseline_returns is None:
            baseline_returns = self.baseline_retain_performance
            if baseline_returns is None:
                return 0.0
            baseline_mean = baseline_returns
        else:
            baseline_mean = np.mean(baseline_returns)
        
        current_mean = np.mean(current_returns)
        
        # Compute relative performance
        if abs(baseline_mean) < 1e-8:
            return 1.0 if abs(current_mean) < 1e-8 else 0.0
        
        relative_performance = current_mean / baseline_mean
        
        # RSI is high when performance is maintained
        rsi = min(1.0, max(0.0, relative_performance))
        
        return rsi
    
    def compute_selectivity(
        self,
        forget_effectiveness: float,
        retain_stability: float,
    ) -> float:
        """
        Compute Selectivity metric.
        Measures precision of targeted forgetting vs collateral damage.
        High selectivity = effective forgetting with minimal retain impact.
        Range: [0, 1]
        """
        # Selectivity is high when forget is effective AND retain is stable
        selectivity = forget_effectiveness * retain_stability
        return selectivity
    
    def compute_forget_effectiveness(
        self,
        forget_trajectories_scores: List[float],
        threshold: float = 0.3,
    ) -> float:
        """
        Compute forgetting effectiveness.
        Measures how much the agent has forgotten target behaviors.
        Range: [0, 1], where 1 means complete forgetting.
        """
        if len(forget_trajectories_scores) == 0:
            return 0.0
        
        # Effectiveness is high when scores are below threshold
        below_threshold = np.mean([score < threshold for score in forget_trajectories_scores])
        
        # Also consider magnitude of reduction
        score_reduction = 1.0 - np.mean(forget_trajectories_scores)
        
        # Combine both measures
        effectiveness = 0.5 * below_threshold + 0.5 * max(0, score_reduction)
        
        return effectiveness
    
    def compute_privacy_auc(
        self,
        membership_inference_scores: List[float],
        true_members: List[bool],
    ) -> float:
        """
        Compute Privacy AUC.
        Measures privacy guarantees via membership inference attack resistance.
        Lower is better (harder to infer membership).
        """
        if len(membership_inference_scores) != len(true_members):
            return 0.0
        
        from sklearn.metrics import roc_auc_score
        
        try:
            privacy_auc = 1.0 - roc_auc_score(true_members, membership_inference_scores)
        except (ValueError, TypeError):
            privacy_auc = 0.5  # Random guessing
        
        return privacy_auc
    
    def compute_all_metrics(
        self,
        forget_scores: List[float],
        retain_returns: List[float],
        baseline_returns: List[float] = None,
        timesteps: List[int] = None,
    ) -> Dict[str, float]:
        """Compute all unlearning metrics."""
        metrics = {}
        
        # AUFC
        if len(forget_scores) > 1:
            metrics["aufc"] = self.compute_aufc(forget_scores, timesteps)
        
        # Forget effectiveness
        metrics["forget_effectiveness"] = self.compute_forget_effectiveness(forget_scores)
        
        # Retain stability
        metrics["retain_stability_index"] = self.compute_retain_stability_index(
            retain_returns, baseline_returns
        )
        
        # Selectivity
        metrics["selectivity"] = self.compute_selectivity(
            metrics["forget_effectiveness"],
            metrics["retain_stability_index"]
        )
        
        # Additional stats
        metrics["mean_forget_score"] = np.mean(forget_scores) if forget_scores else 0.0
        metrics["std_forget_score"] = np.std(forget_scores) if forget_scores else 0.0
        metrics["mean_retain_return"] = np.mean(retain_returns) if retain_returns else 0.0
        metrics["std_retain_return"] = np.std(retain_returns) if retain_returns else 0.0
        
        return metrics


def evaluate_trajectory_similarity(
    agent,
    trajectory: Dict,
    device: str = "cpu",
) -> float:
    """
    Evaluate how likely the agent is to reproduce a trajectory.
    Returns score in [0, 1] where high score = likely to reproduce.
    """
    observations = torch.FloatTensor(trajectory["observations"]).to(device)
    actions = torch.FloatTensor(trajectory["actions"]).to(device)
    
    with torch.no_grad():
        _, log_probs, _, _ = agent.network.get_action_and_value(observations, actions)
    
    # Average probability of taking the recorded actions
    avg_log_prob = log_probs.mean().item()
    probability = np.exp(avg_log_prob)
    
    return probability


def membership_inference_attack(
    agent,
    member_trajectories: List[Dict],
    non_member_trajectories: List[Dict],
    device: str = "cpu",
) -> Tuple[List[float], List[bool]]:
    """
    Perform membership inference attack.
    Returns (scores, true_labels) for privacy evaluation.
    """
    scores = []
    labels = []
    
    # Evaluate member trajectories
    for traj in member_trajectories:
        score = evaluate_trajectory_similarity(agent, traj, device)
        scores.append(score)
        labels.append(True)
    
    # Evaluate non-member trajectories
    for traj in non_member_trajectories:
        score = evaluate_trajectory_similarity(agent, traj, device)
        scores.append(score)
        labels.append(False)
    
    return scores, labels
