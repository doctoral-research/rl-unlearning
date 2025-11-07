"""
Utility functions and helpers
"""
import random
import numpy as np
import torch
from typing import Optional, Dict, Any
from pathlib import Path
import logging


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """Get torch device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def create_directories(config: Dict[str, Any]) -> None:
    """Create necessary directories for experiment."""
    directories = [
        config.get("output_dir", "experiments/outputs"),
        config.get("log_dir", "experiments/logs"),
        config["training"].get("checkpoint_dir", "experiments/checkpoints"),
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """Setup logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_checkpoint(
    agent,
    optimizer: torch.optim.Optimizer,
    timestep: int,
    metrics: Dict[str, float],
    path: str,
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        "timestep": timestep,
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str, agent, optimizer: Optional[torch.optim.Optimizer] = None):
    """Load training checkpoint."""
    checkpoint = torch.load(path)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["timestep"], checkpoint["metrics"]


class RunningMeanStd:
    """Running mean and standard deviation tracker."""
    
    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray) -> None:
        """Update statistics with new batch of data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        """Update from computed moments."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
