from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

print("[train_svm] Starting script and importing dependencies...", file=sys.stderr, flush=True)
from src.utils.config import CONFIG


def log_progress(message: str) -> None:
    """Print a simple progress message that shows up immediately in the terminal."""
    print(f"[train_svm] {message}", file=sys.stderr, flush=True)


def timed_import_start(label: str) -> float:
    """Log the start of an expensive import and return its start time."""
    log_progress(f"Importing {label}...")
    return time.perf_counter()


def timed_import_end(label: str, start_time: float) -> None:
    """Log how long an expensive import took."""
    elapsed = time.perf_counter() - start_time
    log_progress(f"Imported {label} in {elapsed:.2f}s")


def read_csv_header(path: str | Path) -> list[str]:
    """Read only the first CSV row so we can inspect columns without loading the full file."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def ordered_unique_columns(columns: list[str]) -> list[str]:
    """Preserve column order while dropping duplicates."""
    return list(dict.fromkeys(columns))


def _should_log_chunk(chunk_counter: int) -> bool:
    """Log frequently at the start, then less often once the pattern is clear."""
    return chunk_counter <= 3 or chunk_counter % 10 == 0


def load_svm_dataframe(
    path: str | Path,
    required_columns: list[str],
    chunksize: int,
):
    """Load a large CSV in chunks so the user can see progress while it streams."""
    import pandas as pd

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix != ".csv":
        log_progress(f"Loading non-CSV input directly from {path}")
        return pd.read_parquet(path, columns=required_columns)

    log_progress(f"Streaming rows from {path}")
    frames: list[pd.DataFrame] = []
    kept_rows = 0
    chunk_counter = 0
    for chunk in pd.read_csv(path, usecols=required_columns, chunksize=chunksize):
        chunk_counter += 1
        frames.append(chunk)
        kept_rows += len(chunk)

        if _should_log_chunk(chunk_counter):
            log_progress(f"Loaded chunk {chunk_counter} from {path.name}; accumulated {kept_rows:,} rows")

    if not frames:
        raise ValueError(f"No rows were loaded from {path}.")

    return pd.concat(frames, ignore_index=True)


def sample_runs(
    df,
    run_column: str,
    max_runs: int | None,
    random_state: int,
):
    """Optionally shrink the dataset by sampling whole runs instead of rows."""
    if max_runs is None:
        return df

    unique_runs = df[run_column].drop_duplicates()
    if len(unique_runs) <= max_runs:
        return df

    chosen_runs = unique_runs.sample(n=max_runs, random_state=random_state)
    return df.loc[df[run_column].isin(chosen_runs)].reset_index(drop=True)


def split_train_validation_by_run(
    df,
    run_column: str,
    stratify_column: str,
    validation_size: float,
    random_state: int,
):
    """Create a run-aware train/validation split from the training dataset.

    The key idea is that all samples from one TEP simulation run stay together.
    This avoids training on one part of a run and validating on a nearby part of
    the same run, which would make the validation numbers too optimistic.
    """
    from sklearn.model_selection import GroupShuffleSplit

    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be between 0 and 1.")

    run_targets = df.groupby(run_column)[stratify_column].first().reset_index()
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_size, random_state=random_state)
    train_idx, validation_idx = next(
        splitter.split(run_targets, y=run_targets[stratify_column], groups=run_targets[run_column])
    )
    train_runs = set(run_targets.iloc[train_idx][run_column])
    validation_runs = set(run_targets.iloc[validation_idx][run_column])
    train_split = df.loc[df[run_column].isin(train_runs)].reset_index(drop=True)
    validation_split = df.loc[df[run_column].isin(validation_runs)].reset_index(drop=True)
    return train_split, validation_split


def train_svm(
    data_path: str,
    target_column: str,
    output_dir: str,
    eval_path: str | None = None,
    run_column: str = CONFIG.run_column,
    validation_size: float = 0.2,
    max_train_runs: int | None = 400,
    max_validation_runs: int | None = 100,
    max_eval_runs: int | None = 150,
    threshold_steps: int = 25,
    max_false_alarm_rate: float = 0.05,
    random_state: int = 42,
    chunksize: int = 100_000,
) -> None:
    start_time = timed_import_start("pandas")
    import pandas as pd
    timed_import_end("pandas", start_time)

    start_time = timed_import_start("scikit-learn")
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    timed_import_end("scikit-learn", start_time)

    start_time = timed_import_start("local data/evaluation helpers")
    from src.data.load_data import feature_columns
    from src.evaluation.alarm_metrics import (
        binary_classification_metrics,
        binary_threshold_sweep,
        choose_operating_point,
        default_threshold_grid,
    )
    from src.evaluation.metrics import classification_report_dict
    timed_import_end("local data/evaluation helpers", start_time)

    log_progress(f"Inspecting input columns from {data_path}")
    input_columns = read_csv_header(data_path) if Path(data_path).suffix.lower() == ".csv" else pd.read_parquet(data_path).columns.tolist()
    feature_cols = feature_columns(input_columns)
    required_columns = ordered_unique_columns([run_column, target_column, *feature_cols])

    log_progress(f"Loading training data from {data_path}")
    train_df = load_svm_dataframe(data_path, required_columns=required_columns, chunksize=chunksize)
    if eval_path:
        log_progress(f"Loading evaluation data from {eval_path}")
        eval_df = load_svm_dataframe(eval_path, required_columns=required_columns, chunksize=chunksize)
    else:
        eval_df = train_df.copy()

    train_df[run_column] = train_df[run_column].astype(str)
    eval_df[run_column] = eval_df[run_column].astype(str)

    log_progress("Creating a run-aware train/validation split from the training data")
    train_fit_df, validation_df = split_train_validation_by_run(
        train_df,
        run_column=run_column,
        stratify_column=target_column,
        validation_size=validation_size,
        random_state=random_state,
    )
    train_fit_df = sample_runs(train_fit_df, run_column=run_column, max_runs=max_train_runs, random_state=random_state)
    validation_df = sample_runs(
        validation_df,
        run_column=run_column,
        max_runs=max_validation_runs,
        random_state=random_state,
    )
    eval_df = sample_runs(eval_df, run_column=run_column, max_runs=max_eval_runs, random_state=random_state)
    log_progress(
        f"Using {train_fit_df[run_column].nunique():,} train runs, "
        f"{validation_df[run_column].nunique():,} validation runs, and "
        f"{eval_df[run_column].nunique():,} evaluation runs"
    )

    X_train = train_fit_df[feature_cols]
    y_train = train_fit_df[target_column]
    X_validation = validation_df[feature_cols]
    y_validation = validation_df[target_column]
    X_eval = eval_df[feature_columns(eval_df.columns)]
    y_eval = eval_df[target_column]
    unique_classes = sorted(pd.Series(y_train).dropna().unique().tolist())
    log_progress(
        f"Preparing a {'binary' if len(unique_classes) == 2 else 'multiclass'} SVM with "
        f"{len(feature_cols)} features and classes {unique_classes}"
    )

    model = Pipeline(
        [
            # SVMs are scale-sensitive, so we normalize before fitting.
            ("scaler", StandardScaler()),
            # We use LinearSVC because a full kernel SVM is usually too expensive
            # for a multi-million-row process dataset like TEP.
            # LinearSVC also supports multiclass classification internally with
            # a one-vs-rest style scheme, which makes it a practical starter
            # baseline for both binary and multiclass fault problems.
            ("classifier", LinearSVC(class_weight="balanced", dual="auto", max_iter=5000, random_state=random_state)),
        ]
    )
    log_progress("Fitting LinearSVC model")
    model.fit(X_train, y_train)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_progress("Running evaluation on the held-out data")
    predictions = model.predict(X_eval)
    summary: dict[str, object] = {
        "model": "LinearSVC",
        "target_column": target_column,
        "task_type": "binary" if len(unique_classes) == 2 else "multiclass",
        "class_labels": unique_classes,
        "feature_count": len(feature_cols),
        "train_rows_used": int(len(train_fit_df)),
        "validation_rows_used": int(len(validation_df)),
        "eval_rows_used": int(len(eval_df)),
        "train_runs_used": int(train_fit_df[run_column].nunique()),
        "validation_runs_used": int(validation_df[run_column].nunique()),
        "eval_runs_used": int(eval_df[run_column].nunique()),
        "metrics": classification_report_dict(y_eval, predictions),
    }

    if len(unique_classes) == 2:
        # For binary fault detection we can use the signed decision score as an
        # alarm score and sweep thresholds to study false alarms vs misses.
        # We tune that threshold on a held-out validation set made of whole runs,
        # then report the chosen threshold on the external evaluation set.
        log_progress(f"Computing binary threshold sweep on validation runs with {threshold_steps} candidate thresholds")
        validation_scores = model.decision_function(X_validation)
        threshold_grid = default_threshold_grid(validation_scores, num_thresholds=threshold_steps)
        validation_threshold_metrics = binary_threshold_sweep(y_validation, validation_scores, threshold_grid)
        threshold_metrics_path = output_path / "svm_validation_threshold_metrics.csv"
        validation_threshold_metrics.to_csv(threshold_metrics_path, index=False)

        recommended = choose_operating_point(validation_threshold_metrics, max_false_alarm_rate=max_false_alarm_rate)
        eval_scores = model.decision_function(X_eval)
        eval_threshold_predictions = (eval_scores >= float(recommended["threshold"])).astype(int)
        eval_threshold_metrics = binary_classification_metrics(y_eval, eval_threshold_predictions)
        summary["default_threshold"] = 0.0
        summary["default_threshold_eval_metrics"] = classification_report_dict(y_eval, predictions)
        summary["validation_threshold_selection"] = {
            "threshold": float(recommended["threshold"]),
            "false_alarm_rate": float(recommended["false_alarm_rate"]),
            "recall": float(recommended["recall"]),
            "f1": float(recommended["f1"]),
        }
        summary["recommended_threshold"] = {
            "threshold": float(recommended["threshold"]),
            "validation_false_alarm_rate": float(recommended["false_alarm_rate"]),
            "validation_recall": float(recommended["recall"]),
            "validation_f1": float(recommended["f1"]),
            "eval_false_alarm_rate": float(eval_threshold_metrics["fp"] / max(1, eval_threshold_metrics["fp"] + eval_threshold_metrics["tn"])),
            "eval_recall": float(eval_threshold_metrics["recall"]),
            "eval_precision": float(eval_threshold_metrics["precision"]),
            "eval_f1": float(eval_threshold_metrics["f1"]),
        }
        summary["threshold_metrics_path"] = str(threshold_metrics_path)
        log_progress(
            "Recommended binary operating point from validation runs: "
            f"threshold={recommended['threshold']:.4f}, "
            f"validation_false_alarm_rate={recommended['false_alarm_rate']:.4f}, "
            f"eval_recall={eval_threshold_metrics['recall']:.4f}"
        )
    else:
        # In multiclass mode there is no single alarm threshold with the same
        # interpretation, so we save direct class predictions and class metrics.
        log_progress("Saving multiclass predictions and per-class metrics")
        prediction_df = pd.DataFrame(
            {
                "actual": y_eval.reset_index(drop=True),
                "predicted": pd.Series(predictions).reset_index(drop=True),
            }
        )
        prediction_path = output_path / "svm_multiclass_predictions.csv"
        prediction_df.to_csv(prediction_path, index=False)
        summary["prediction_path"] = str(prediction_path)

    with (output_path / "svm_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    log_progress(f"Finished. Outputs written to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a simple linear SVM baseline for binary or multiclass TEP fault classification."
    )
    parser.add_argument("--data", default=str(CONFIG.processed_data_dir / "tep_train.csv"), help="Training CSV path.")
    parser.add_argument(
        "--eval-data",
        default=str(CONFIG.processed_data_dir / "tep_test.csv"),
        help="Evaluation CSV path. Defaults to the canonical TEP test set.",
    )
    parser.add_argument(
        "--target",
        default=CONFIG.binary_target_column,
        help="Target column name, for example is_faulty for binary or faultNumber for multiclass.",
    )
    parser.add_argument("--run-column", default=CONFIG.run_column, help="Run identifier column for run-aware splitting.")
    parser.add_argument("--output-dir", default="results", help="Directory to save outputs.")
    parser.add_argument("--validation-size", type=float, default=0.2, help="Fraction of training runs used for validation.")
    parser.add_argument("--max-train-runs", type=int, default=400, help="Optional cap on training runs.")
    parser.add_argument("--max-validation-runs", type=int, default=100, help="Optional cap on validation runs.")
    parser.add_argument("--max-eval-runs", type=int, default=150, help="Optional cap on evaluation runs.")
    parser.add_argument(
        "--threshold-steps",
        type=int,
        default=25,
        help="Number of thresholds for binary tradeoff analysis. Ignored for multiclass targets.",
    )
    parser.add_argument(
        "--max-false-alarm-rate",
        type=float,
        default=0.05,
        help="False-alarm budget used when choosing a recommended threshold for binary targets.",
    )
    parser.add_argument("--chunksize", type=int, default=100000, help="CSV chunk size used while streaming large inputs.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_svm(
        args.data,
        args.target,
        args.output_dir,
        eval_path=args.eval_data,
        run_column=args.run_column,
        validation_size=args.validation_size,
        max_train_runs=args.max_train_runs,
        max_validation_runs=args.max_validation_runs,
        max_eval_runs=args.max_eval_runs,
        threshold_steps=args.threshold_steps,
        max_false_alarm_rate=args.max_false_alarm_rate,
        random_state=args.random_state,
        chunksize=args.chunksize,
    )
