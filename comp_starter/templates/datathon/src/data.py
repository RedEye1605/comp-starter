"""Data loading and preprocessing utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


def load_data(raw_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test datasets.

    Args:
        raw_dir: Override default data/raw directory.

    Returns:
        Tuple of (train_df, test_df).
    """
    raw = raw_dir or (DATA_DIR / "raw")

    train = pd.read_csv(raw / "train.csv")
    test = pd.read_csv(raw / "test.csv")
    return train, test


def save_processed(df: pd.DataFrame, name: str, processed_dir: Path | None = None) -> Path:
    """Save a processed DataFrame to data/processed/.

    Args:
        df: DataFrame to save.
        name: Filename (without directory).
        processed_dir: Override default data/processed directory.

    Returns:
        Path to saved file.
    """
    dest = processed_dir or (DATA_DIR / "processed")
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / name
    df.to_csv(path, index=False)
    return path


def quick_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a quick summary of a DataFrame.

    Returns a DataFrame with columns: dtype, missing, missing_pct, nunique.
    """
    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "missing": df.isnull().sum(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2),
        "nunique": df.nunique(),
    })
    return summary.sort_values("missing_pct", ascending=False)
