"""Submission file generation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"


def create_submission(
    predictions: np.ndarray | pd.Series,
    test_df: pd.DataFrame,
    id_col: str = "id",
    target_col: str = "target",
    path: str | Path | None = None,
) -> Path:
    """Create a submission CSV file.

    Args:
        predictions: Model predictions.
        test_df: Test DataFrame (must contain id_col).
        id_col: Name of the ID column.
        target_col: Name of the target/prediction column.
        path: Output path. Defaults to submissions/submission_TIMESTAMP.csv.

    Returns:
        Path to the saved submission file.
    """

    submission = pd.DataFrame({
        id_col: test_df[id_col],
        target_col: predictions,
    })

    if path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = SUBMISSIONS_DIR / f"submission_{timestamp}.csv"
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    submission.to_csv(path, index=False)
    return path


def validate_submission(submission_path: Path, sample_path: Path) -> list[str]:
    """Validate a submission against the sample submission.

    Returns a list of issues (empty = valid).
    """
    issues: list[str] = []

    sub = pd.read_csv(submission_path)
    sample = pd.read_csv(sample_path)

    if list(sub.columns) != list(sample.columns):
        issues.append(f"Column mismatch: got {list(sub.columns)}, expected {list(sample.columns)}")

    if len(sub) != len(sample):
        issues.append(f"Row count mismatch: got {len(sub)}, expected {len(sample)}")

    if not sub.iloc[:, 0].equals(sample.iloc[:, 0]):
        issues.append("ID column values don't match sample submission")

    return issues
