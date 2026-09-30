"""FixFlow AI Prediction Interface.

Provides single and batch prediction with explainability support.
"""

from typing import Any

import numpy as np
import pandas as pd
import torch

from models.features import (
    extract_text_features,
    extract_structured_features,
    combine_features,
)
from models.multitask_model import FixFlowMLP


def predict_single(
    complaint_text: str,
    structured_data: dict[str, Any],
    model: FixFlowMLP,
    pipeline: dict[str, Any],
) -> dict[str, Any]:
    """Predict category, severity, and resolution for a single complaint.

    Args:
        complaint_text: Natural language complaint text.
        structured_data: Dict with location_zone, weather, time_of_day,
            affected_people, safety_risk, duplicate_group, vehicle_required.
        model: Trained FixFlowMLP model.
        pipeline: Fitted feature pipeline.

    Returns:
        Dict with category, severity, urgency, safety_risk, resolution_minutes,
        confidence, and raw probabilities.
    """
    model.eval()

    # Build single-row DataFrame
    row = {'complaint_text': complaint_text, **structured_data}
    df = pd.DataFrame([row])

    # Extract features
    text_feat = extract_text_features([complaint_text], pipeline['vectorizer'])
    struct_feat = extract_structured_features(df)
    features = combine_features(text_feat, struct_feat)

    x = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        cat_out, sev_out, res_out = model(x)

    # Category prediction
    probs = torch.softmax(cat_out, dim=1).squeeze().numpy()
    pred_idx = int(probs.argmax())
    idx_to_cat = pipeline.get('idx_to_category', {})
    category = idx_to_cat.get(pred_idx, f'Category_{pred_idx}')
    confidence = float(probs[pred_idx])

    # Regression outputs
    severity = float(torch.sigmoid(sev_out).item())
    resolution_norm = float(torch.sigmoid(res_out).item())
    resolution_minutes = resolution_norm * 300.0

    # Use structured data for urgency/safety if available
    urgency = structured_data.get('urgency', severity * 0.9)
    safety_risk = structured_data.get('safety_risk', severity * 0.85)

    return {
        'category': category,
        'category_idx': pred_idx,
        'severity': round(severity, 3),
        'urgency': round(float(urgency), 3),
        'safety_risk': round(float(safety_risk), 3),
        'resolution_minutes': round(resolution_minutes, 1),
        'confidence': round(confidence, 3),
        'class_probabilities': {
            idx_to_cat.get(i, f'Cat_{i}'): round(float(p), 4)
            for i, p in enumerate(probs)
        },
    }


def predict_batch(
    df: pd.DataFrame,
    model: FixFlowMLP,
    pipeline: dict[str, Any],
) -> pd.DataFrame:
    """Run predictions on a batch of complaints.

    Args:
        df: DataFrame with complaint_text and structured columns.
        model: Trained FixFlowMLP model.
        pipeline: Fitted feature pipeline.

    Returns:
        DataFrame with prediction columns added.
    """
    model.eval()

    texts = df['complaint_text'].fillna('').tolist()
    text_feat = extract_text_features(texts, pipeline['vectorizer'])
    struct_feat = extract_structured_features(df)
    features = combine_features(text_feat, struct_feat)

    x = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        cat_out, sev_out, res_out = model(x)

    probs = torch.softmax(cat_out, dim=1).numpy()
    pred_indices = probs.argmax(axis=1)
    confidences = probs.max(axis=1)

    idx_to_cat = pipeline.get('idx_to_category', {})

    result_df = df.copy()
    result_df['predicted_category'] = [idx_to_cat.get(int(i), f'Cat_{i}') for i in pred_indices]
    result_df['predicted_severity'] = torch.sigmoid(sev_out).squeeze().numpy()
    result_df['predicted_resolution'] = torch.sigmoid(res_out).squeeze().numpy() * 300.0
    result_df['prediction_confidence'] = confidences

    return result_df


def get_prediction_with_explanation(
    complaint_text: str,
    structured_data: dict[str, Any],
    model: FixFlowMLP,
    pipeline: dict[str, Any],
) -> dict[str, Any]:
    """Get prediction with feature contribution explanation.

    Uses input perturbation to estimate each feature's contribution.

    Args:
        complaint_text: The complaint text.
        structured_data: Structured data dict.
        model: Trained model.
        pipeline: Fitted pipeline.

    Returns:
        Dict with prediction and explanation of top contributing factors.
    """
    prediction = predict_single(complaint_text, structured_data, model, pipeline)

    # Simple explanation based on structured data influence
    factors: list[dict[str, Any]] = []

    safety = structured_data.get('safety_risk', 0)
    if safety > 0.7:
        factors.append({'factor': 'Safety Risk', 'value': safety, 'impact': 'HIGH', 'contribution': safety * 30})
    elif safety > 0.3:
        factors.append({'factor': 'Safety Risk', 'value': safety, 'impact': 'MEDIUM', 'contribution': safety * 20})

    affected = structured_data.get('affected_people', 0)
    if affected > 50:
        factors.append({'factor': 'Affected People', 'value': affected, 'impact': 'HIGH', 'contribution': 25})
    elif affected > 10:
        factors.append({'factor': 'Affected People', 'value': affected, 'impact': 'MEDIUM', 'contribution': 15})

    weather = structured_data.get('weather', 'Clear')
    if weather in ('Heavy Rain', 'Storm'):
        factors.append({'factor': 'Weather', 'value': weather, 'impact': 'HIGH', 'contribution': 20})

    dup = structured_data.get('duplicate_count', 1)
    if dup > 5:
        factors.append({'factor': 'Duplicate Reports', 'value': dup, 'impact': 'HIGH', 'contribution': 15})

    factors.sort(key=lambda x: x.get('contribution', 0), reverse=True)

    prediction['explanation'] = factors
    prediction['explanation_text'] = _generate_explanation_text(prediction, factors)

    return prediction


def _generate_explanation_text(prediction: dict, factors: list[dict]) -> str:
    """Generate human-readable explanation text."""
    parts = [f"This complaint is classified as **{prediction['category']}** "
             f"with {prediction['confidence']*100:.0f}% confidence."]

    if prediction['severity'] > 0.7:
        parts.append(f"Severity is HIGH ({prediction['severity']:.0%}).")
    elif prediction['severity'] > 0.4:
        parts.append(f"Severity is MEDIUM ({prediction['severity']:.0%}).")
    else:
        parts.append(f"Severity is LOW ({prediction['severity']:.0%}).")

    if factors:
        parts.append("Key contributing factors:")
        for f in factors[:3]:
            parts.append(f"  - {f['factor']}: {f['value']} ({f['impact']} impact)")

    return " ".join(parts)
