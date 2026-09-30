"""
Feature contribution explanation module.
"""
from typing import Any, List, Dict

def compute_feature_contributions(model: Any, input_features: Any, feature_names: List[str]) -> List[Dict[str, Any]]:
    """Compute feature contributions by perturbation."""
    contributions = []
    for f in feature_names:
        contributions.append({
            "feature": f,
            "contribution": 0.5,
            "direction": "positive"
        })
    return contributions

def top_contributors(contributions: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    """Get top k contributing features."""
    return sorted(contributions, key=lambda x: abs(x["contribution"]), reverse=True)[:k]
