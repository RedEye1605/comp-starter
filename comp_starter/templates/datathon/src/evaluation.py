"""Evaluation metrics and validation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    classification_report,
)


def evaluate(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    task_type: str = "classification",
    average: str = "weighted",
) -> dict:
    """Evaluate predictions with standard metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        task_type: 'classification' or 'regression'.
        average: Averaging mode for multi-class metrics.

    Returns:
        Dictionary of metric name -> value.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if task_type == "classification":
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
            "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        }
    else:
        return {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }


def print_evaluation(y_true, y_pred, task_type: str = "classification") -> None:
    """Print a formatted evaluation report."""
    metrics = evaluate(y_true, y_pred, task_type)

    print("=" * 40)
    print("EVALUATION REPORT")
    print("=" * 40)

    for name, value in metrics.items():
        print(f"  {name:>12s}: {value:.4f}")

    if task_type == "classification":
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))

    print("=" * 40)
