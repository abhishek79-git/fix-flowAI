"""FixFlow AI Calibration Evaluation.

Measures Expected Calibration Error (ECE) and produces reliability diagram data.
A well-calibrated model's predicted confidence should match actual accuracy.
"""

import numpy as np
from typing import Any


def expected_calibration_error(
    confidences: np.ndarray | list[float],
    accuracies: np.ndarray | list[float],
    num_bins: int = 10,
) -> float:
    """Compute ECE from pre-binned confidences and accuracies.

    Args:
        confidences: Per-bin average predicted confidence.
        accuracies: Per-bin actual accuracy.
        num_bins: Number of bins (for weighting).

    Returns:
        Expected Calibration Error.
    """
    conf = np.asarray(confidences)
    acc = np.asarray(accuracies)
    return float(np.mean(np.abs(conf - acc)))


def compute_ece_from_predictions(
    y_true: np.ndarray | list[int],
    y_proba: np.ndarray | list[float],
    num_bins: int = 10,
) -> float:
    """Compute ECE directly from true labels and predicted probabilities.

    For each bin, computes the absolute difference between average confidence
    and actual accuracy, weighted by the fraction of samples in each bin.

    Args:
        y_true: True binary labels (0 or 1).
        y_proba: Predicted probabilities for the positive class.
        num_bins: Number of confidence bins.

    Returns:
        ECE value (lower is better).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    n = len(y_true)

    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        mask = (y_proba > bin_edges[i]) & (y_proba <= bin_edges[i + 1])
        bin_count = mask.sum()

        if bin_count == 0:
            continue

        bin_accuracy = y_true[mask].mean()
        bin_confidence = y_proba[mask].mean()
        ece += (bin_count / n) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def reliability_diagram_data(
    y_true: np.ndarray | list[int],
    y_proba: np.ndarray | list[float],
    num_bins: int = 10,
) -> dict[str, list[float]]:
    """Compute binned data for a reliability (calibration) diagram.

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities.
        num_bins: Number of bins.

    Returns:
        Dict with bin_confidences, bin_accuracies, and bin_counts.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_confidences: list[float] = []
    bin_accuracies: list[float] = []
    bin_counts: list[float] = []

    for i in range(num_bins):
        mask = (y_proba > bin_edges[i]) & (y_proba <= bin_edges[i + 1])
        count = int(mask.sum())
        bin_counts.append(float(count))

        if count == 0:
            bin_confidences.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_accuracies.append(0.0)
        else:
            bin_confidences.append(float(y_proba[mask].mean()))
            bin_accuracies.append(float(y_true[mask].mean()))

    return {
        'bin_confidences': bin_confidences,
        'bin_accuracies': bin_accuracies,
        'bin_counts': bin_counts,
    }
