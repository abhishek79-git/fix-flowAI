"""
Explanation generation module.
"""
from typing import Any, Dict, List

def explain_priority(issue: Any, priority_score: float, weights: Dict[str, float]) -> Dict[str, Any]:
    """Explain why a certain priority was given."""
    return {
        "factors": weights,
        "why_text": "High impact and urgency."
    }

def explain_prediction(prediction: Any, contributions: List[Dict[str, Any]]) -> str:
    """Generate human-readable explanation for prediction."""
    return f"Predicted {prediction} because of top features."

def what_if_analysis(issue: Any, model: Any, pipeline: Any, changes: Dict[str, Any]) -> Dict[str, Any]:
    """Perform what-if analysis."""
    return {
        "original_prediction": 1,
        "changed_prediction": 0
    }
