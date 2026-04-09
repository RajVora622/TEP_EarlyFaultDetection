from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.config import CONFIG


SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".parquet"}
TEP_RAW_FILES = {
    "train": {
        "fault_free": "TEP_FaultFree_Training.csv",
        "faulty": "TEP_Faulty_Training.csv",
    },
    "test": {
        "fault_free": "TEP_FaultFree_Testing.csv",
        "faulty": "TEP_Faulty_Testing.csv",
    },
}
IDENTIFIER_COLUMNS = ["faultNumber", "simulationRun", "sample", "run_id", "dataset_split", "source_file", "is_faulty"]


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a single tabular file from disk."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep=None, engine="python")
    return pd.read_parquet(path)


def load_raw_directory(directory: str | Path) -> pd.DataFrame:
    """Load and concatenate every supported table in a directory."""
    directory = Path(directory)
    files = sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.name != ".gitkeep" and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        raise FileNotFoundError(f"No supported files found in {directory}")

    frames = []
    for file_path in files:
        frame = load_table(file_path)
        frame["source_file"] = file_path.name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def save_dataframe(df: pd.DataFrame, output_path: str | Path) -> None:
    """Persist a dataframe to CSV or Parquet based on file extension."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)


def feature_columns(columns: Iterable[str]) -> list[str]:
    """Return the TEP process measurement and manipulated-variable columns."""
    return [col for col in columns if col.startswith("xmeas_") or col.startswith("xmv_")]


def load_tep_raw_file(path: str | Path, split: str) -> pd.DataFrame:
    """Load a raw TEP CSV and standardize metadata columns."""
    df = pd.read_csv(path)
    df["source_file"] = Path(path).name
    df[CONFIG.split_column] = split
    df[CONFIG.binary_target_column] = (df[CONFIG.multiclass_target_column] != 0).astype(int)
    df[CONFIG.run_column] = (
        df[CONFIG.split_column].astype(str)
        + "_fault_"
        + df[CONFIG.multiclass_target_column].astype(str)
        + "_run_"
        + df["simulationRun"].astype(str)
    )
    return df


def load_tep_dataset(raw_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Load the four canonical TEP CSV files into train and test dataframes."""
    raw_dir = Path(raw_dir or CONFIG.raw_data_dir)
    datasets: dict[str, pd.DataFrame] = {}

    for split, file_map in TEP_RAW_FILES.items():
        frames = []
        for file_name in file_map.values():
            path = raw_dir / file_name
            if not path.exists():
                raise FileNotFoundError(f"Missing expected TEP raw file: {path}")
            frames.append(load_tep_raw_file(path, split=split))
        datasets[split] = pd.concat(frames, ignore_index=True)

    return datasets


def save_tep_processed_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> None:
    """Save processed train and test TEP datasets."""
    output_dir = Path(output_dir or CONFIG.processed_data_dir)
    save_dataframe(train_df, output_dir / "tep_train.csv")
    save_dataframe(test_df, output_dir / "tep_test.csv")


def write_feature_metadata(output_dir: str | Path | None = None, columns: Iterable[str] | None = None) -> None:
    """Persist the list of model feature columns for downstream scripts."""
    output_dir = Path(output_dir or CONFIG.metadata_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = list(columns) if columns is not None else []
    metadata = pd.DataFrame({"feature_column": cols})
    metadata.to_csv(output_dir / "feature_columns.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and standardize the raw TEP CSV files.")
    parser.add_argument("--raw-dir", default=str(CONFIG.raw_data_dir), help="Directory containing raw TEP CSV files.")
    parser.add_argument(
        "--output-dir",
        default=str(CONFIG.processed_data_dir),
        help="Directory where processed train/test CSVs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = load_tep_dataset(args.raw_dir)
    save_tep_processed_datasets(datasets["train"], datasets["test"], args.output_dir)
    write_feature_metadata(columns=feature_columns(datasets["train"].columns))


if __name__ == "__main__":
    main()
