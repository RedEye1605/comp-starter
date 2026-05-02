"""Feature engineering pipeline."""

from __future__ import annotations

import pandas as pd


def build_features(
    df: pd.DataFrame,
    mode: str = "train",
    encoders: dict | None = None,
) -> pd.DataFrame:
    """Build features from raw data.

    Args:
        df: Input DataFrame.
        mode: 'train' or 'test' — controls fitting vs transforming.
        encoders: Pre-fitted encoders from training (pass for test mode).

    Returns:
        Feature-engineered DataFrame.
    """
    feat = df.copy()
    _encoders = encoders or {}

    # TODO: Add your feature engineering here

    # Example: Encode categorical columns
    # cat_cols = feat.select_dtypes(include=['object']).columns
    # for col in cat_cols:
    #     if mode == 'train':
    #         le = LabelEncoder()
    #         feat[col] = le.fit_transform(feat[col].astype(str))
    #         _encoders[col] = le
    #     else:
    #         le = _encoders.get(col)
    #         if le:
    #             feat[col] = feat[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    return feat, _encoders


def create_interaction_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Create pairwise interaction features for given columns."""
    result = df.copy()
    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1 :]:
            result[f"{col_a}_x_{col_b}"] = df[col_a] * df[col_b]
    return result


def create_aggregate_features(
    df: pd.DataFrame,
    group_col: str,
    value_cols: list[str],
    aggs: list[str] | None = None,
) -> pd.DataFrame:
    """Create group-by aggregate features."""
    aggs = aggs or ["mean", "std", "min", "max"]
    grouped = df.groupby(group_col)[value_cols].agg(aggs)
    grouped.columns = [f"{group_col}_{col}_{agg}" for col, agg in grouped.columns]
    return grouped.reset_index()
