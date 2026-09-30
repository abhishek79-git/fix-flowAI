"""
Reproducibility evaluation module.
"""
import random
import numpy as np
from typing import Dict, Any, Callable

def set_all_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def check_determinism(train_fn: Callable, args: tuple, n_runs: int = 3) -> Dict[str, Any]:
    """Check if training is deterministic."""
    results = []
    for _ in range(n_runs):
        set_all_seeds(42)
        results.append(train_fn(*args))
        
    variance = float(np.var(results)) if len(results) > 0 else 0.0
    is_deterministic = variance < 1e-6
    
    return {
        "results": results,
        "is_deterministic": is_deterministic,
        "variance": variance
    }
