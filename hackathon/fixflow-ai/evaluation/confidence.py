"""
Confidence interval module.
"""
from typing import Tuple, List, Callable, Any
import numpy as np

def bootstrap_confidence_interval(
    y_true: List[Any], 
    y_pred: List[Any], 
    metric_fn: Callable, 
    n_bootstrap: int = 1000, 
    alpha: float = 0.05, 
    seed: int = 42
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0
    metrics = []
    
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_t = [y_true[i] for i in idx]
        y_p = [y_pred[i] for i in idx]
        metrics.append(metric_fn(y_t, y_p))
        
    metrics.sort()
    lower = float(np.percentile(metrics, 100 * (alpha / 2)))
    upper = float(np.percentile(metrics, 100 * (1 - alpha / 2)))
    mean = float(np.mean(metrics))
    return lower, mean, upper

def hoeffding_lower_bound(accuracy: float, n_samples: int, delta: float = 0.05) -> float:
    """Compute Hoeffding lower bound for accuracy."""
    bound = np.sqrt(np.log(1 / delta) / (2 * n_samples))
    return float(accuracy - bound)
