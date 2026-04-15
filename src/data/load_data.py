from __future__ import annotations

import argparse
from dataclasses import dataclass, field
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
TEP_FEATURE_NAME_MAP = {
    "xmeas_1": "a_feed_stream",
    "xmeas_2": "d_feed_stream",
    "xmeas_3": "e_feed_stream",
    "xmeas_4": "total_fresh_feed_to_strip",
    "xmeas_5": "recycle_flow_to_reactor",
    "xmeas_6": "reactor_feed_rate",
    "xmeas_7": "reactor_pressure",
    "xmeas_8": "reactor_level",
    "xmeas_9": "reactor_temperature",
    "xmeas_10": "purge_rate",
    "xmeas_11": "separator_temperature",
    "xmeas_12": "separator_level",
    "xmeas_13": "separator_pressure",
    "xmeas_14": "separator_underflow",
    "xmeas_15": "stripper_level",
    "xmeas_16": "stripper_pressure",
    "xmeas_17": "stripper_underflow",
    "xmeas_18": "stripper_temperature",
    "xmeas_19": "stripper_steam_flow",
    "xmeas_20": "compressor_work",
    "xmeas_21": "reactor_cooling_water_outlet_temp",
    "xmeas_22": "condenser_cooling_water_outlet_temp",
    "xmeas_23": "reactor_feed_a_fraction",
    "xmeas_24": "reactor_feed_b_fraction",
    "xmeas_25": "reactor_feed_c_fraction",
    "xmeas_26": "reactor_feed_d_fraction",
    "xmeas_27": "reactor_feed_e_fraction",
    "xmeas_28": "reactor_feed_f_fraction",
    "xmeas_29": "purge_a_fraction",
    "xmeas_30": "purge_b_fraction",
    "xmeas_31": "purge_c_fraction",
    "xmeas_32": "purge_d_fraction",
    "xmeas_33": "purge_e_fraction",
    "xmeas_34": "purge_f_fraction",
    "xmeas_35": "purge_g_fraction",
    "xmeas_36": "purge_h_fraction",
    "xmeas_37": "product_d_fraction",
    "xmeas_38": "product_e_fraction",
    "xmeas_39": "product_f_fraction",
    "xmeas_40": "product_g_fraction",
    "xmeas_41": "product_h_fraction",
    "xmv_1": "d_feed_flow_valve",
    "xmv_2": "e_feed_flow_valve",
    "xmv_3": "a_feed_flow_valve",
    "xmv_4": "total_feed_flow_to_strip_valve",
    "xmv_5": "compressor_recycle_valve",
    "xmv_6": "purge_valve",
    "xmv_7": "separator_pot_liquid_flow_valve",
    "xmv_8": "stripper_liquid_product_flow_valve",
    "xmv_9": "stripper_steam_valve",
    "xmv_10": "reactor_cooling_water_flow_valve",
    "xmv_11": "condenser_cooling_water_flow_valve",
}
TEP_METADATA_COLUMN_MAP = {
    "faultnumber": "faultNumber",
    "simulationrun": "simulationRun",
    "sample": "sample",
}
EXPECTED_SAMPLES_PER_RUN = {"train": 500, "test": 960}
TEP_RAW_FEATURE_COLUMNS = list(TEP_FEATURE_NAME_MAP)
TEP_READABLE_FEATURE_COLUMNS = list(TEP_FEATURE_NAME_MAP.values())
IDENTIFIER_COLUMNS = ["faultNumber", "simulationRun", "sample", "run_id", "dataset_split", "source_file", "is_faulty"]


@dataclass
class RunValidationState:
    expected_samples: int
    row_counts: dict[tuple[int, int], int] = field(default_factory=dict)
    sample_masks: dict[tuple[int, int], int] = field(default_factory=dict)


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
    available = set(columns)
    readable = [col for col in TEP_READABLE_FEATURE_COLUMNS if col in available]
    if readable:
        return readable
    return [col for col in TEP_RAW_FEATURE_COLUMNS if col in available]


def _normalized_column_name(column: str) -> str:
    return column.strip().lower()


def standardize_tep_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize metadata names and map raw TEP sensor names to readable names."""
    rename_map: dict[str, str] = {}

    for column in df.columns:
        normalized = _normalized_column_name(column)
        if normalized in TEP_METADATA_COLUMN_MAP:
            rename_map[column] = TEP_METADATA_COLUMN_MAP[normalized]
        elif normalized in TEP_FEATURE_NAME_MAP:
            rename_map[column] = TEP_FEATURE_NAME_MAP[normalized]

    return df.rename(columns=rename_map)


def validate_tep_frame(df: pd.DataFrame, split: str, source_name: str) -> None:
    """Check that a standardized TEP frame has the expected columns and run structure."""
    required_columns = {"faultNumber", "simulationRun", "sample", *TEP_READABLE_FEATURE_COLUMNS}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"{source_name} is missing required TEP columns: {missing_columns}")

    if split not in EXPECTED_SAMPLES_PER_RUN:
        raise ValueError(f"Unsupported split '{split}' for TEP validation.")

    expected_samples = EXPECTED_SAMPLES_PER_RUN[split]
    run_sizes = (
        df.groupby(["faultNumber", "simulationRun"])["sample"]
        .agg(row_count="size", unique_samples="nunique")
        .reset_index()
    )
    invalid_runs = run_sizes[
        (run_sizes["row_count"] != expected_samples) | (run_sizes["unique_samples"] != expected_samples)
    ]
    if not invalid_runs.empty:
        preview = invalid_runs.head(5).to_dict(orient="records")
        raise ValueError(
            f"{source_name} has runs that do not match the expected {expected_samples} samples. "
            f"Examples: {preview}"
        )


def _initialize_validation_state(split: str) -> RunValidationState:
    if split not in EXPECTED_SAMPLES_PER_RUN:
        raise ValueError(f"Unsupported split '{split}' for TEP validation.")
    return RunValidationState(expected_samples=EXPECTED_SAMPLES_PER_RUN[split])


def update_run_validation(state: RunValidationState, df: pd.DataFrame, source_name: str) -> None:
    """Accumulate per-run sample coverage for chunked validation."""
    validate_tep_frame(df.head(0), split="train" if state.expected_samples == 500 else "test", source_name=source_name)

    grouped = df.groupby(["faultNumber", "simulationRun"], sort=False)["sample"]
    for (fault_number, simulation_run), samples in grouped:
        key = (int(fault_number), int(simulation_run))
        sample_values = pd.to_numeric(samples, errors="raise").astype(int)

        if ((sample_values < 1) | (sample_values > state.expected_samples)).any():
            raise ValueError(
                f"{source_name} contains sample values outside the expected 1..{state.expected_samples} range "
                f"for run {key}."
            )

        state.row_counts[key] = state.row_counts.get(key, 0) + len(sample_values)

        sample_mask = state.sample_masks.get(key, 0)
        for sample_value in pd.unique(sample_values):
            sample_mask |= 1 << (int(sample_value) - 1)
        state.sample_masks[key] = sample_mask


def finalize_run_validation(state: RunValidationState, source_name: str) -> None:
    """Ensure every run accumulated from chunked reads has the expected sample coverage."""
    expected_mask = (1 << state.expected_samples) - 1
    invalid_runs = []

    for key, row_count in state.row_counts.items():
        if row_count != state.expected_samples or state.sample_masks.get(key, 0) != expected_mask:
            invalid_runs.append(
                {
                    "faultNumber": key[0],
                    "simulationRun": key[1],
                    "row_count": row_count,
                    "unique_samples": int(state.sample_masks.get(key, 0).bit_count()),
                }
            )
        if len(invalid_runs) == 5:
            break

    if invalid_runs:
        raise ValueError(
            f"{source_name} has runs that do not match the expected {state.expected_samples} samples. "
            f"Examples: {invalid_runs}"
        )


def load_tep_raw_file(path: str | Path, split: str) -> pd.DataFrame:
    """Load a raw TEP CSV and standardize metadata columns."""
    df = standardize_tep_columns(pd.read_csv(path))
    validate_tep_frame(df, split=split, source_name=Path(path).name)
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


def build_tep_split_file(
    raw_paths: Iterable[str | Path],
    split: str,
    output_path: str | Path,
    chunk_size: int = 100_000,
) -> None:
    """Stream one processed split to disk while validating run structure."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = True

    for raw_path in raw_paths:
        raw_path = Path(raw_path)
        validation_state = _initialize_validation_state(split)

        for chunk in pd.read_csv(raw_path, chunksize=chunk_size):
            chunk = standardize_tep_columns(chunk)
            update_run_validation(validation_state, chunk, source_name=raw_path.name)
            chunk["source_file"] = raw_path.name
            chunk[CONFIG.split_column] = split
            chunk[CONFIG.binary_target_column] = (chunk[CONFIG.multiclass_target_column] != 0).astype(int)
            chunk[CONFIG.run_column] = (
                chunk[CONFIG.split_column].astype(str)
                + "_fault_"
                + chunk[CONFIG.multiclass_target_column].astype(str)
                + "_run_"
                + chunk["simulationRun"].astype(str)
            )
            chunk.to_csv(output_path, mode="w" if write_header else "a", index=False, header=write_header)
            write_header = False

        finalize_run_validation(validation_state, source_name=raw_path.name)


def build_and_save_tep_processed_datasets(
    raw_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    chunk_size: int = 100_000,
) -> None:
    """Build the canonical processed train/test datasets directly from raw CSVs."""
    raw_dir = Path(raw_dir or CONFIG.raw_data_dir)
    output_dir = Path(output_dir or CONFIG.processed_data_dir)

    for split, file_map in TEP_RAW_FILES.items():
        raw_paths = []
        for file_name in file_map.values():
            path = raw_dir / file_name
            if not path.exists():
                raise FileNotFoundError(f"Missing expected TEP raw file: {path}")
            raw_paths.append(path)

        build_tep_split_file(
            raw_paths=raw_paths,
            split=split,
            output_path=output_dir / f"tep_{split}.csv",
            chunk_size=chunk_size,
        )


def write_feature_metadata(output_dir: str | Path | None = None, columns: Iterable[str] | None = None) -> None:
    """Persist the list of model feature columns and the raw-to-readable mapping."""
    output_dir = Path(output_dir or CONFIG.metadata_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = list(columns) if columns is not None else []
    metadata = pd.DataFrame({"feature_column": cols})
    metadata.to_csv(output_dir / "feature_columns.csv", index=False)

    mapping_rows = []
    included = set(cols)
    for raw_column, feature_column in TEP_FEATURE_NAME_MAP.items():
        prefix, feature_number = raw_column.split("_")
        mapping_rows.append(
            {
                "raw_feature_column": raw_column,
                "feature_column": feature_column,
                "feature_family": "measured" if prefix == "xmeas" else "manipulated",
                "feature_number": int(feature_number),
                "included_in_processed_dataset": feature_column in included,
            }
        )

    pd.DataFrame(mapping_rows).to_csv(output_dir / "feature_mapping.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and standardize the raw TEP CSV files.")
    parser.add_argument("--raw-dir", default=str(CONFIG.raw_data_dir), help="Directory containing raw TEP CSV files.")
    parser.add_argument(
        "--output-dir",
        default=str(CONFIG.processed_data_dir),
        help="Directory where processed train/test CSVs will be written.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of rows per chunk when streaming the large raw CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_and_save_tep_processed_datasets(args.raw_dir, args.output_dir, chunk_size=args.chunk_size)
    write_feature_metadata(columns=TEP_READABLE_FEATURE_COLUMNS)


if __name__ == "__main__":
    main()
