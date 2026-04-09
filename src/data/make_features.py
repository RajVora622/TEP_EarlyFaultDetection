from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.data.load_data import feature_columns, save_dataframe
from src.utils.config import CONFIG


def add_lag_features(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    group_column: str = "run_id",
    lags: Sequence[int] = (1, 3, 5),
) -> pd.DataFrame:
    """Create lagged features within each run."""
    result = df.copy()

    for col in feature_columns:
        for lag in lags:
            result[f"{col}_lag_{lag}"] = result.groupby(group_column)[col].shift(lag)

    return result


def add_rolling_features(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    group_column: str = "run_id",
    windows: Sequence[int] = (5, 10),
) -> pd.DataFrame:
    """Add rolling mean and standard deviation features per run."""
    result = df.copy()

    for col in feature_columns:
        grouped = result.groupby(group_column)[col]
        for window in windows:
            result[f"{col}_roll_mean_{window}"] = grouped.transform(
                lambda s: s.rolling(window=window, min_periods=1).mean()
            )
            result[f"{col}_roll_std_{window}"] = grouped.transform(
                lambda s: s.rolling(window=window, min_periods=1).std().fillna(0.0)
            )

    return result


def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply simple cleanup after feature generation."""
    return df.dropna().reset_index(drop=True)


def build_tep_features(
    input_path: str | Path,
    output_path: str | Path,
    group_column: str = CONFIG.run_column,
) -> pd.DataFrame:
    """Create a feature-enriched version of a processed TEP CSV."""
    df = pd.read_csv(input_path)
    cols = feature_columns(df.columns)
    enriched = add_lag_features(df, cols, group_column=group_column)
    enriched = add_rolling_features(enriched, cols, group_column=group_column)
    enriched = finalize_features(enriched)
    save_dataframe(enriched, output_path)
    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate lag and rolling features for a processed TEP CSV.")
    parser.add_argument("--input", default=str(CONFIG.processed_data_dir / "tep_train.csv"), help="Input CSV path.")
    parser.add_argument(
        "--output",
        default=str(CONFIG.processed_data_dir / "tep_train_features.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--group-column", default=CONFIG.run_column, help="Run/group identifier column.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_tep_features(args.input, args.output, group_column=args.group_column)


if __name__ == "__main__":
    main()
