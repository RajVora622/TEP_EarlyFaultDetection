from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.data.load_data import save_dataframe
from src.utils.config import CONFIG


@dataclass
class RunSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_by_run(
    df: pd.DataFrame,
    run_column: str = "run_id",
    stratify_column: str = "faultNumber",
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> RunSplit:
    """Split a dataframe while keeping all rows from a run together."""
    if run_column not in df.columns:
        raise KeyError(f"'{run_column}' column not found.")
    if stratify_column not in df.columns:
        raise KeyError(f"'{stratify_column}' column not found.")

    run_targets = df.groupby(run_column)[stratify_column].first().reset_index()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(
        splitter.split(run_targets, y=run_targets[stratify_column], groups=run_targets[run_column])
    )
    train_val_runs = set(run_targets.iloc[train_val_idx][run_column])
    test_runs = set(run_targets.iloc[test_idx][run_column])

    train_val = df[df[run_column].isin(train_val_runs)].reset_index(drop=True)
    test = df[df[run_column].isin(test_runs)].reset_index(drop=True)

    train_val_targets = train_val.groupby(run_column)[stratify_column].first().reset_index()
    second_split = GroupShuffleSplit(n_splits=1, test_size=validation_size, random_state=random_state)
    train_idx, val_idx = next(
        second_split.split(
            train_val_targets,
            y=train_val_targets[stratify_column],
            groups=train_val_targets[run_column],
        )
    )
    train_runs = set(train_val_targets.iloc[train_idx][run_column])
    val_runs = set(train_val_targets.iloc[val_idx][run_column])

    train = train_val[train_val[run_column].isin(train_runs)].reset_index(drop=True)
    validation = train_val[train_val[run_column].isin(val_runs)].reset_index(drop=True)
    return RunSplit(train=train, validation=validation, test=test)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create run-aware train/validation/test splits from a TEP CSV.")
    parser.add_argument("--data", default=str(CONFIG.processed_data_dir / "tep_train.csv"), help="Input processed CSV.")
    parser.add_argument(
        "--output-dir",
        default=str(CONFIG.processed_data_dir),
        help="Directory where split CSVs will be written.",
    )
    parser.add_argument("--run-column", default=CONFIG.run_column, help="Run identifier column.")
    parser.add_argument("--target-column", default=CONFIG.multiclass_target_column, help="Target column for balancing.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of runs held out for test.")
    parser.add_argument("--validation-size", type=float, default=0.2, help="Fraction of remaining runs for validation.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data)
    split = split_by_run(
        df,
        run_column=args.run_column,
        stratify_column=args.target_column,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )
    output_dir = Path(args.output_dir)
    save_dataframe(split.train, output_dir / "tep_train_split.csv")
    save_dataframe(split.validation, output_dir / "tep_validation_split.csv")
    save_dataframe(split.test, output_dir / "tep_internal_test_split.csv")


if __name__ == "__main__":
    main()
