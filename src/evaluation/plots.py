from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.data.load_data import feature_columns, load_table
from src.utils.config import CONFIG


sns.set_theme(style="whitegrid")


def _ensure_parent_dir(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_figure(fig, output_path: str | Path) -> Path:
    path = _ensure_parent_dir(output_path)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _numeric_feature_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    frame = df.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    return frame.dropna(axis=1, how="all")


def _select_top_variance_features(df: pd.DataFrame, columns: Sequence[str], top_n: int) -> list[str]:
    numeric = _numeric_feature_frame(df, columns)
    if numeric.empty:
        return []
    variances = numeric.var().sort_values(ascending=False)
    return variances.head(top_n).index.tolist()


def dataset_overview(
    df: pd.DataFrame,
    target_column: str = CONFIG.multiclass_target_column,
    binary_target_column: str = CONFIG.binary_target_column,
    run_column: str = CONFIG.run_column,
    split_column: str = CONFIG.split_column,
) -> pd.DataFrame:
    """Build a compact dataset summary at the overall and split levels."""
    scopes: list[tuple[str, pd.DataFrame]] = [("overall", df)]

    if split_column in df.columns:
        for split_name, split_df in df.groupby(split_column, sort=True):
            scopes.append((f"{split_column}={split_name}", split_df))

    records = []
    for scope_name, scope_df in scopes:
        run_count = scope_df[run_column].nunique() if run_column in scope_df.columns else pd.NA
        run_targets = (
            scope_df.groupby(run_column)[target_column].first()
            if run_column in scope_df.columns and target_column in scope_df.columns
            else pd.Series(dtype="float64")
        )
        fault_run_count = int((run_targets != 0).sum()) if not run_targets.empty else pd.NA
        samples_per_run = scope_df.groupby(run_column).size() if run_column in scope_df.columns else pd.Series(dtype="float64")

        records.append(
            {
                "scope": scope_name,
                "rows": len(scope_df),
                "unique_runs": run_count,
                "unique_fault_labels": scope_df[target_column].nunique() if target_column in scope_df.columns else pd.NA,
                "faulty_rows": int((scope_df[binary_target_column] == 1).sum())
                if binary_target_column in scope_df.columns
                else pd.NA,
                "faulty_row_fraction": float((scope_df[binary_target_column] == 1).mean())
                if binary_target_column in scope_df.columns
                else pd.NA,
                "faulty_runs": fault_run_count,
                "avg_samples_per_run": float(samples_per_run.mean()) if not samples_per_run.empty else pd.NA,
                "median_samples_per_run": float(samples_per_run.median()) if not samples_per_run.empty else pd.NA,
            }
        )

    return pd.DataFrame.from_records(records)


def write_feature_summary(df: pd.DataFrame, columns: Sequence[str], output_path: str | Path) -> pd.DataFrame:
    """Write summary statistics for model features."""
    numeric = _numeric_feature_frame(df, columns)
    if numeric.empty:
        raise ValueError("No numeric feature columns were found for summary generation.")

    summary = numeric.agg(["mean", "std", "min", "median", "max"]).T.reset_index()
    summary = summary.rename(columns={"index": "feature"})
    summary["missing_fraction"] = numeric.isna().mean().values
    summary["variance"] = numeric.var().values
    summary = summary.sort_values("variance", ascending=False).reset_index(drop=True)

    output_path = _ensure_parent_dir(output_path)
    summary.to_csv(output_path, index=False)
    return summary


def plot_confusion_matrix(y_true, y_pred, output_path: str | Path) -> None:
    """Save a confusion matrix figure to disk."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, colorbar=False)
    _save_figure(fig, output_path)


def plot_feature_distribution(df: pd.DataFrame, column: str, label_column: str, output_path: str | Path) -> None:
    """Plot a feature distribution split by class label."""
    if column not in df.columns:
        raise KeyError(f"'{column}' column not found.")
    if label_column not in df.columns:
        raise KeyError(f"'{label_column}' column not found.")

    plot_df = df[[column, label_column]].copy()
    plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(subset=[column])

    if plot_df.empty:
        raise ValueError(f"No valid values available to plot for '{column}'.")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        data=plot_df,
        x=column,
        hue=label_column,
        bins=40,
        stat="density",
        common_norm=False,
        element="step",
        fill=False,
        ax=ax,
    )
    ax.set_title(f"{column} distribution by {label_column}")
    _save_figure(fig, output_path)


def plot_fault_distribution(
    df: pd.DataFrame,
    target_column: str,
    run_column: str,
    output_path: str | Path,
) -> None:
    """Plot fault frequencies at both row and run level."""
    if target_column not in df.columns:
        raise KeyError(f"'{target_column}' column not found.")
    if run_column not in df.columns:
        raise KeyError(f"'{run_column}' column not found.")

    row_counts = df[target_column].value_counts().sort_index()
    run_counts = df.groupby(run_column)[target_column].first().value_counts().sort_index()
    plot_df = pd.DataFrame(
        {
            "faultNumber": list(row_counts.index) + list(run_counts.index),
            "count": list(row_counts.values) + list(run_counts.values),
            "level": ["rows"] * len(row_counts) + ["runs"] * len(run_counts),
        }
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=plot_df, x="faultNumber", y="count", hue="level", ax=ax)
    ax.set_title("Fault distribution across rows and runs")
    ax.set_xlabel(target_column)
    ax.set_ylabel("Count")
    _save_figure(fig, output_path)


def plot_missingness(df: pd.DataFrame, columns: Sequence[str], output_path: str | Path, top_n: int = 20) -> None:
    """Plot the most-missing feature columns."""
    if not columns:
        raise ValueError("At least one column is required to plot missingness.")

    missing_fraction = df.loc[:, list(columns)].isna().mean().sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=missing_fraction.values, y=missing_fraction.index, ax=ax, orient="h")
    ax.set_title(f"Top {len(missing_fraction)} columns by missing-value fraction")
    ax.set_xlabel("Missing fraction")
    ax.set_ylabel("Column")
    _save_figure(fig, output_path)


def plot_correlation_heatmap(df: pd.DataFrame, columns: Sequence[str], output_path: str | Path) -> None:
    """Plot a feature-correlation heatmap."""
    numeric = _numeric_feature_frame(df, columns)
    if numeric.shape[1] < 2:
        raise ValueError("At least two numeric feature columns are required for a correlation heatmap.")

    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0.0, ax=ax)
    ax.set_title("Feature correlation heatmap")
    _save_figure(fig, output_path)


def select_representative_runs(
    df: pd.DataFrame,
    run_column: str = CONFIG.run_column,
    target_column: str = CONFIG.multiclass_target_column,
    max_normal_runs: int = 2,
    max_fault_runs: int = 3,
) -> list[str]:
    """Pick a few normal and faulty runs to visualize over time."""
    if run_column not in df.columns or target_column not in df.columns:
        return []

    run_targets = df.groupby(run_column)[target_column].first().reset_index()
    normal_runs = run_targets.loc[run_targets[target_column] == 0, run_column].head(max_normal_runs).tolist()
    faulty_runs = (
        run_targets.loc[run_targets[target_column] != 0]
        .drop_duplicates(subset=[target_column])
        .head(max_fault_runs)[run_column]
        .tolist()
    )
    return normal_runs + faulty_runs


def plot_feature_trajectories(
    df: pd.DataFrame,
    column: str,
    run_ids: Iterable[str],
    output_path: str | Path,
    run_column: str = CONFIG.run_column,
    time_column: str = CONFIG.time_column,
    target_column: str = CONFIG.multiclass_target_column,
) -> None:
    """Plot a selected feature over time for a small set of runs."""
    run_ids = list(run_ids)
    if not run_ids:
        raise ValueError("At least one run id is required to plot feature trajectories.")
    if column not in df.columns:
        raise KeyError(f"'{column}' column not found.")
    if run_column not in df.columns:
        raise KeyError(f"'{run_column}' column not found.")
    if time_column not in df.columns:
        raise KeyError(f"'{time_column}' column not found.")

    plot_df = df.loc[df[run_column].isin(run_ids), [run_column, time_column, column] + ([target_column] if target_column in df.columns else [])].copy()
    plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(subset=[column]).sort_values([run_column, time_column])

    if plot_df.empty:
        raise ValueError(f"No trajectory values available for '{column}'.")

    if target_column in plot_df.columns:
        run_faults = plot_df.groupby(run_column)[target_column].first()
        plot_df["run_label"] = plot_df[run_column].map(lambda run_id: f"{run_id} | fault={run_faults.loc[run_id]}")
    else:
        plot_df["run_label"] = plot_df[run_column].astype(str)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=plot_df, x=time_column, y=column, hue="run_label", ax=ax)
    ax.set_title(f"{column} trajectories across representative runs")
    ax.set_xlabel(time_column)
    ax.set_ylabel(column)
    _save_figure(fig, output_path)


def generate_dataset_exploration(
    input_path: str | Path,
    output_dir: str | Path,
    selected_features: Sequence[str] | None = None,
    top_n_features: int = 4,
    correlation_top_n: int = 12,
    run_column: str = CONFIG.run_column,
    time_column: str = CONFIG.time_column,
    target_column: str = CONFIG.multiclass_target_column,
    binary_target_column: str = CONFIG.binary_target_column,
    split_column: str = CONFIG.split_column,
) -> list[Path]:
    """Generate a small collection of reusable dataset-exploration artifacts."""
    input_path = Path(input_path)
    if not input_path.exists():
        default_input = CONFIG.processed_data_dir / "tep_train.csv"
        detail = f"Input dataset not found: {input_path}."
        if input_path == default_input:
            detail += (
                " Build the processed dataset first with "
                "`python -m src.data.load_data` after placing the raw TEP CSV files in `data/raw/`, "
                "or pass `--input` with the path to an existing processed CSV or Parquet file."
            )
        else:
            detail += " Pass `--input` with the path to an existing processed CSV or Parquet file."
        raise FileNotFoundError(detail)

    df = load_table(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_features = feature_columns(df.columns)
    if not all_features:
        raise ValueError("No TEP feature columns were found in the input file.")

    selected = [feature for feature in (selected_features or []) if feature in df.columns]
    if not selected:
        selected = _select_top_variance_features(df, all_features, top_n_features)

    correlation_features = _select_top_variance_features(df, all_features, correlation_top_n)
    artifact_paths: list[Path] = []

    overview = dataset_overview(
        df,
        target_column=target_column,
        binary_target_column=binary_target_column,
        run_column=run_column,
        split_column=split_column,
    )
    overview_path = output_dir / "dataset_overview.csv"
    overview.to_csv(_ensure_parent_dir(overview_path), index=False)
    artifact_paths.append(overview_path)

    feature_summary_path = output_dir / "feature_summary.csv"
    write_feature_summary(df, all_features, feature_summary_path)
    artifact_paths.append(feature_summary_path)

    if run_column in df.columns and target_column in df.columns:
        fault_distribution_path = output_dir / "fault_distribution.png"
        plot_fault_distribution(df, target_column=target_column, run_column=run_column, output_path=fault_distribution_path)
        artifact_paths.append(fault_distribution_path)

    missingness_path = output_dir / "missingness.png"
    plot_missingness(df, all_features, missingness_path)
    artifact_paths.append(missingness_path)

    if len(correlation_features) >= 2:
        correlation_path = output_dir / "correlation_heatmap.png"
        plot_correlation_heatmap(df, correlation_features, correlation_path)
        artifact_paths.append(correlation_path)

    if binary_target_column in df.columns:
        for feature in selected:
            distribution_path = output_dir / f"{feature}_distribution.png"
            plot_feature_distribution(df, column=feature, label_column=binary_target_column, output_path=distribution_path)
            artifact_paths.append(distribution_path)

    if run_column in df.columns and time_column in df.columns:
        representative_runs = select_representative_runs(df, run_column=run_column, target_column=target_column)
        if representative_runs:
            for feature in selected:
                trajectory_path = output_dir / f"{feature}_trajectories.png"
                plot_feature_trajectories(
                    df,
                    column=feature,
                    run_ids=representative_runs,
                    output_path=trajectory_path,
                    run_column=run_column,
                    time_column=time_column,
                    target_column=target_column,
                )
                artifact_paths.append(trajectory_path)

    return artifact_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate exploratory plots and summary tables for a TEP dataset.")
    parser.add_argument(
        "--input",
        default=str(CONFIG.processed_data_dir / "tep_train.csv"),
        help="Input CSV/Parquet file. Defaults to the processed training dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(CONFIG.results_dir / "eda"),
        help="Directory where EDA artifacts will be written.",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Optional list of specific feature columns to visualize.",
    )
    parser.add_argument(
        "--top-n-features",
        type=int,
        default=4,
        help="Number of high-variance features to plot when --features is not provided.",
    )
    parser.add_argument(
        "--correlation-top-n",
        type=int,
        default=12,
        help="Number of high-variance features to include in the correlation heatmap.",
    )
    parser.add_argument("--run-column", default=CONFIG.run_column, help="Run identifier column.")
    parser.add_argument("--time-column", default=CONFIG.time_column, help="Time/sample column.")
    parser.add_argument("--target-column", default=CONFIG.multiclass_target_column, help="Fault-label column.")
    parser.add_argument("--binary-target-column", default=CONFIG.binary_target_column, help="Binary label column.")
    parser.add_argument("--split-column", default=CONFIG.split_column, help="Dataset split column.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset_exploration(
        input_path=args.input,
        output_dir=args.output_dir,
        selected_features=args.features,
        top_n_features=args.top_n_features,
        correlation_top_n=args.correlation_top_n,
        run_column=args.run_column,
        time_column=args.time_column,
        target_column=args.target_column,
        binary_target_column=args.binary_target_column,
        split_column=args.split_column,
    )


if __name__ == "__main__":
    main()
