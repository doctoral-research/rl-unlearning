"""
Test suite for metrics
"""
import pytest
import numpy as np
from metrics import UnlearningMetrics


def test_aufc_computation():
    """Test AUFC computation."""
    metrics = UnlearningMetrics()
    
    forget_scores = [0.9, 0.7, 0.5, 0.3, 0.1]
    aufc = metrics.compute_aufc(forget_scores)
    
    assert 0 <= aufc <= 1
    assert aufc > 0  # Should be positive for decreasing scores


def test_retain_stability_index():
    """Test retain stability index."""
    metrics = UnlearningMetrics()
    
    baseline = [100, 105, 95, 98, 102]
    current = [90, 95, 85, 88, 92]
    
    rsi = metrics.compute_retain_stability_index(current, baseline)
    
    assert 0 <= rsi <= 1


def test_selectivity():
    """Test selectivity metric."""
    metrics = UnlearningMetrics()
    
    forget_eff = 0.8
    retain_stab = 0.9
    
    selectivity = metrics.compute_selectivity(forget_eff, retain_stab)
    
    assert 0 <= selectivity <= 1
    assert selectivity == pytest.approx(0.72)


def test_forget_effectiveness():
    """Test forget effectiveness."""
    metrics = UnlearningMetrics()
    
    # High scores = not forgotten
    high_scores = [0.9, 0.8, 0.85, 0.95]
    eff_high = metrics.compute_forget_effectiveness(high_scores, threshold=0.5)
    
    # Low scores = forgotten
    low_scores = [0.1, 0.2, 0.15, 0.05]
    eff_low = metrics.compute_forget_effectiveness(low_scores, threshold=0.5)
    
    assert eff_low > eff_high


def test_compute_all_metrics():
    """Test computing all metrics."""
    metrics = UnlearningMetrics()
    
    forget_scores = [0.8, 0.6, 0.4, 0.2]
    retain_returns = [95, 90, 92, 88]
    baseline_returns = [100, 98, 102, 99]
    
    all_metrics = metrics.compute_all_metrics(
        forget_scores=forget_scores,
        retain_returns=retain_returns,
        baseline_returns=baseline_returns,
    )
    
    assert "aufc" in all_metrics
    assert "forget_effectiveness" in all_metrics
    assert "retain_stability_index" in all_metrics
    assert "selectivity" in all_metrics


if __name__ == "__main__":
    pytest.main([__file__])
