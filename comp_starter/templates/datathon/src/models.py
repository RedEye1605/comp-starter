"""Model training and prediction utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

MODELS_DIR = Path(__file__).parent.parent / "models"


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model=None,
    cv: int = 5,
    scoring: str = "accuracy",
    random_state: int = 42,
) -> tuple:
    """Train a model with cross-validation.

    Args:
        X: Feature matrix.
        y: Target vector.
        model: Sklearn-compatible model. Defaults to RandomForest.
        cv: Number of CV folds.
        scoring: Scoring metric.
        random_state: Random seed.

    Returns:
        Tuple of (trained_model, cv_scores).
    """
    if model is None:
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)

    # Auto-detect classification vs regression for stratification
    try:
        n_classes = len(np.unique(y))
        if n_classes <= 20 and y.dtype in [np.int64, np.int32, np.float64]:
            cv_strategy = StratifiedKFold(cv, shuffle=True, random_state=random_state)
        else:
            cv_strategy = KFold(cv, shuffle=True, random_state=random_state)
    except Exception:
        cv_strategy = KFold(cv, shuffle=True, random_state=random_state)

    scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring)
    model.fit(X, y)

    return model, scores


def save_model(model, name: str, models_dir: Path | None = None) -> Path:
    """Save a trained model to disk using joblib."""
    import joblib

    dest = models_dir or MODELS_DIR
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name: str, models_dir: Path | None = None):
    """Load a saved model."""
    import joblib

    src = models_dir or MODELS_DIR
    path = src / f"{name}.joblib"
    return joblib.load(path)
