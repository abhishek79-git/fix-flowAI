"""
Metrics evaluation module.
"""
import time
from typing import Dict, List, Any
import numpy as np

def compute_classification_metrics(y_true: List[int], y_pred: List[int], labels: List[int]) -> Dict[str, Any]:
    """Compute classification metrics."""
    # Simplified calculation for accuracy, macro_f1, per_class_f1, confusion_matrix
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    accuracy = np.mean(y_true_np == y_pred_np) if len(y_true) > 0 else 0.0
    
    # Simple placeholder logic for other metrics
    confusion_matrix = [[0]*len(labels) for _ in labels]
    macro_f1 = 0.0
    per_class_f1 = [0.0]*len(labels)
    
    return {
        "accuracy": float(accuracy),
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": confusion_matrix
    }

def compute_regression_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    """Compute regression metrics."""
    if not y_true:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "huber_loss": 0.0}
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    mae = np.mean(np.abs(y_t - y_p))
    rmse = np.sqrt(np.mean(np.square(y_t - y_p)))
    # Placeholder for r2, huber_loss
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": 0.0,
        "huber_loss": 0.0
    }

def compute_latency(model: Any, sample_input: Any, runs: int = 100) -> float:
    """Compute average latency of a model prediction in ms."""
    start = time.time()
    for _ in range(runs):
        _ = model.predict(sample_input)
    end = time.time()
    return ((end - start) / runs) * 1000.0

def compute_parameter_count(model: Any) -> int:
    """Compute the number of parameters in the model."""
    if hasattr(model, "coef_"):
        return model.coef_.size + getattr(model, "intercept_", np.zeros(1)).size
    return 0
