from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

print("[train_lstm] Starting script and importing dependencies...", file=sys.stderr, flush=True)
from src.utils.config import CONFIG


def log_progress(message: str) -> None:
    """Print a simple progress message that shows up immediately in the terminal."""
    print(f"[train_lstm] {message}", file=sys.stderr, flush=True)


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


def sample_runs(df, run_column: str, max_runs: int | None, random_state: int):
    """Optionally shrink the dataset by sampling entire runs instead of rows."""
    if max_runs is None:
        return df

    unique_runs = df[run_column].drop_duplicates()
    if len(unique_runs) <= max_runs:
        return df

    chosen_runs = unique_runs.sample(n=max_runs, random_state=random_state)
    return df.loc[df[run_column].isin(chosen_runs)].reset_index(drop=True)


def add_alarm_target(
    df,
    target_column: str,
    fault_column: str,
    time_column: str,
    onset_sample: int,
):
    """Create a target that only turns on after the assumed fault-onset time.

    This is the key early-detection idea: we do not want the model to get credit
    for alarming before the process is actually supposed to be faulty.
    """
    result = df.copy()
    result[target_column] = ((result[fault_column] != 0) & (result[time_column] >= onset_sample)).astype(int)
    return result


def build_windows(
    df,
    feature_cols: list[str],
    target_column: str,
    run_column: str,
    time_column: str,
    window_size: int,
    stride: int,
):
    """Convert a run-wise table into fixed-length windows for an LSTM."""
    import numpy as np
    import pandas as pd

    windows: list[np.ndarray] = []
    labels: list[int] = []
    metadata_rows: list[dict[str, str | int | float]] = []

    for run_id, run_df in df.groupby(run_column, sort=False):
        ordered = run_df.sort_values(time_column).reset_index(drop=True)
        feature_values = ordered[feature_cols].to_numpy(dtype=np.float32)
        targets = ordered[target_column].to_numpy(dtype=np.int64)

        for end_idx in range(window_size - 1, len(ordered), stride):
            start_idx = end_idx - window_size + 1
            windows.append(feature_values[start_idx : end_idx + 1])
            labels.append(int(targets[end_idx]))
            metadata_rows.append(
                {
                    run_column: run_id,
                    time_column: int(ordered.loc[end_idx, time_column]),
                    target_column: int(targets[end_idx]),
                }
            )

    if not windows:
        raise ValueError("No sequence windows were created. Try a smaller window size or more runs.")

    return np.stack(windows), np.asarray(labels, dtype=np.int64), pd.DataFrame(metadata_rows)


def sample_windows(
    X,
    y,
    metadata,
    max_windows: int | None,
    random_state: int,
):
    """Keep a manageable number of windows while preserving class balance."""
    if max_windows is None or len(y) <= max_windows:
        return X, y, metadata

    import numpy as np

    rng = np.random.default_rng(random_state)
    indices = np.arange(len(y))

    if len(np.unique(y)) <= 1:
        chosen = rng.choice(indices, size=max_windows, replace=False)
    else:
        class_zero = indices[y == 0]
        class_one = indices[y == 1]
        class_one_target = max(1, int(round(max_windows * (len(class_one) / len(indices)))))
        class_one_target = min(len(class_one), class_one_target)
        class_zero_target = min(len(class_zero), max_windows - class_one_target)

        if class_zero_target + class_one_target < max_windows:
            remaining = max_windows - (class_zero_target + class_one_target)
            extra_zero = min(len(class_zero) - class_zero_target, remaining)
            class_zero_target += extra_zero
            remaining -= extra_zero
            extra_one = min(len(class_one) - class_one_target, remaining)
            class_one_target += extra_one

        chosen_zero = rng.choice(class_zero, size=class_zero_target, replace=False) if class_zero_target > 0 else np.array([], dtype=int)
        chosen_one = rng.choice(class_one, size=class_one_target, replace=False) if class_one_target > 0 else np.array([], dtype=int)
        chosen = np.concatenate([chosen_zero, chosen_one])

    chosen = np.sort(chosen)
    return X[chosen], y[chosen], metadata.iloc[chosen].reset_index(drop=True)


def _sample_run_ids_from_csv(
    path: str | Path,
    run_column: str,
    max_runs: int | None,
    random_state: int,
    chunksize: int,
) -> set[str] | None:
    """Collect run ids from a large CSV, then sample a subset if requested."""
    if max_runs is None:
        return None

    import numpy as np
    import pandas as pd

    run_ids: list[str] = []
    seen_runs: set[str] = set()
    chunk_counter = 0
    for chunk in pd.read_csv(path, usecols=[run_column], chunksize=chunksize):
        chunk_counter += 1
        for run_id in chunk[run_column].drop_duplicates():
            run_id_str = str(run_id)
            if run_id_str not in seen_runs:
                seen_runs.add(run_id_str)
                run_ids.append(run_id_str)

        if _should_log_chunk(chunk_counter):
            log_progress(f"Scanned {chunk_counter} chunks while collecting candidate runs from {path}")

    if len(run_ids) <= max_runs:
        return set(run_ids)

    rng = np.random.default_rng(random_state)
    chosen = rng.choice(np.asarray(run_ids, dtype=object), size=max_runs, replace=False)
    return {str(run_id) for run_id in chosen.tolist()}


def load_lstm_dataframe(
    path: str | Path,
    required_columns: list[str],
    run_column: str,
    max_runs: int | None,
    random_state: int,
    chunksize: int,
):
    """Load only the columns and runs needed for LSTM training.

    The main reason for this helper is practical: the processed TEP CSV files are
    several gigabytes, so reading everything eagerly can timeout or exhaust memory.
    We therefore stream CSV files in chunks and keep only the relevant columns.
    """
    import pandas as pd

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        selected_runs = _sample_run_ids_from_csv(
            path,
            run_column=run_column,
            max_runs=max_runs,
            random_state=random_state,
            chunksize=chunksize,
        )
        if selected_runs is None:
            log_progress(f"Streaming all runs from {path}")
        else:
            log_progress(f"Streaming {len(selected_runs):,} sampled runs from {path}")

        frames: list[pd.DataFrame] = []
        kept_rows = 0
        chunk_counter = 0
        for chunk in pd.read_csv(path, usecols=required_columns, chunksize=chunksize):
            chunk_counter += 1
            chunk[run_column] = chunk[run_column].astype(str)
            if selected_runs is not None:
                chunk = chunk.loc[chunk[run_column].isin(selected_runs)]

            if not chunk.empty:
                frames.append(chunk)
                kept_rows += len(chunk)

            if _should_log_chunk(chunk_counter):
                log_progress(f"Loaded {chunk_counter} chunks and kept {kept_rows:,} rows from {path.name}")

        if not frames:
            raise ValueError(f"No rows were loaded from {path}. Check the chosen run limits and input columns.")

        return pd.concat(frames, ignore_index=True)

    log_progress(f"Loading non-CSV input directly from {path}")
    df = pd.read_parquet(path, columns=required_columns)
    df[run_column] = df[run_column].astype(str)
    return sample_runs(df, run_column=run_column, max_runs=max_runs, random_state=random_state)


def train_lstm_detector(
    data_path: str,
    output_dir: str,
    eval_path: str | None = None,
    run_column: str = CONFIG.run_column,
    time_column: str = CONFIG.time_column,
    fault_column: str = CONFIG.multiclass_target_column,
    train_onset_sample: int = 20,
    eval_onset_sample: int = 160,
    window_size: int = 30,
    stride: int = 5,
    max_train_runs: int | None = 300,
    max_eval_runs: int | None = 150,
    max_train_windows: int | None = 50_000,
    max_eval_windows: int | None = 25_000,
    batch_size: int = 128,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    threshold_steps: int = 25,
    max_false_alarm_rate: float = 0.05,
    random_state: int = 42,
    chunksize: int = 100_000,
) -> None:
    start_time = timed_import_start("numpy and pandas")
    import numpy as np
    import pandas as pd
    timed_import_end("numpy and pandas", start_time)

    start_time = timed_import_start("PyTorch")
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    timed_import_end("PyTorch", start_time)

    start_time = timed_import_start("local data/evaluation helpers")
    from src.data.load_data import feature_columns
    from src.evaluation.alarm_metrics import choose_operating_point, default_threshold_grid, early_detection_threshold_sweep
    timed_import_end("local data/evaluation helpers", start_time)

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    log_progress(f"Inspecting input columns from {data_path}")
    input_columns = read_csv_header(data_path) if Path(data_path).suffix.lower() == ".csv" else pd.read_parquet(data_path).columns.tolist()
    feature_cols = feature_columns(input_columns)
    required_columns = ordered_unique_columns([run_column, time_column, fault_column, *feature_cols])

    log_progress(f"Loading training data from {data_path}")
    train_df = load_lstm_dataframe(
        data_path,
        required_columns=required_columns,
        run_column=run_column,
        max_runs=max_train_runs,
        random_state=random_state,
        chunksize=chunksize,
    )
    if eval_path:
        log_progress(f"Loading evaluation data from {eval_path}")
        eval_df = load_lstm_dataframe(
            eval_path,
            required_columns=required_columns,
            run_column=run_column,
            max_runs=max_eval_runs,
            random_state=random_state,
            chunksize=chunksize,
        )
    else:
        eval_df = train_df.copy()

    log_progress(
        f"Using {train_df[run_column].nunique():,} training runs and "
        f"{eval_df[run_column].nunique():,} evaluation runs after run sampling"
    )

    log_progress(f"Scaling {len(feature_cols)} feature columns")
    train_means = train_df[feature_cols].mean()
    train_stds = train_df[feature_cols].std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    # We compute scaling statistics on the training set only so the evaluation
    # set does not leak information into the fitted preprocessing step.
    train_df.loc[:, feature_cols] = (train_df[feature_cols] - train_means) / train_stds
    eval_df.loc[:, feature_cols] = (eval_df[feature_cols] - train_means) / train_stds

    train_df = add_alarm_target(
        train_df,
        target_column="alarm_target",
        fault_column=fault_column,
        time_column=time_column,
        onset_sample=train_onset_sample,
    )
    eval_df = add_alarm_target(
        eval_df,
        target_column="alarm_target",
        fault_column=fault_column,
        time_column=time_column,
        onset_sample=eval_onset_sample,
    )
    log_progress(
        f"Created early-detection targets with training onset sample {train_onset_sample} "
        f"and evaluation onset sample {eval_onset_sample}"
    )

    log_progress(f"Building sequence windows with window_size={window_size} and stride={stride}")
    X_train, y_train, train_metadata = build_windows(
        train_df,
        feature_cols=feature_cols,
        target_column="alarm_target",
        run_column=run_column,
        time_column=time_column,
        window_size=window_size,
        stride=stride,
    )
    X_eval, y_eval, eval_metadata = build_windows(
        eval_df,
        feature_cols=feature_cols,
        target_column="alarm_target",
        run_column=run_column,
        time_column=time_column,
        window_size=window_size,
        stride=stride,
    )

    X_train, y_train, train_metadata = sample_windows(
        X_train, y_train, train_metadata, max_windows=max_train_windows, random_state=random_state
    )
    X_eval, y_eval, eval_metadata = sample_windows(
        X_eval, y_eval, eval_metadata, max_windows=max_eval_windows, random_state=random_state
    )
    log_progress(
        f"Prepared {len(train_metadata):,} training windows and {len(eval_metadata):,} evaluation windows"
    )

    if y_train.sum() == 0:
        raise ValueError(
            "The training windows contain no positive alarm labels. "
            "Try lowering --train-onset-sample or checking how the faulty runs are encoded."
        )

    log_progress("Converting training windows into PyTorch tensors")
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    log_progress(f"Built DataLoader with {len(train_loader)} batches")

    class SimpleLSTMDetector(nn.Module):
        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.classifier = nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _, (hidden_state, _) = self.lstm(x)
            return self.classifier(hidden_state[-1]).squeeze(-1)

    model = SimpleLSTMDetector(input_size=len(feature_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log_progress("Initialized LSTM model and Adam optimizer")

    positive_count = max(1, int(y_train.sum()))
    negative_count = max(1, int(len(y_train) - positive_count))
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    log_progress(
        f"Starting LSTM training for {epochs} epoch(s) with batch_size={batch_size} "
        f"and positive-class weight {float(pos_weight.item()):.3f}"
    )

    model.train()
    batch_report_interval = max(1, len(train_loader) // 5)
    for epoch in range(epochs):
        log_progress(f"Starting epoch {epoch + 1}/{epochs}")
        running_loss = 0.0
        for batch_index, (batch_x, batch_y) in enumerate(train_loader, start=1):
            if batch_index == 1:
                log_progress(
                    f"Epoch {epoch + 1}/{epochs} | first batch shape="
                    f"{tuple(batch_x.shape)}"
                )
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

            if batch_index % batch_report_interval == 0 or batch_index == len(train_loader):
                avg_loss_so_far = running_loss / batch_index
                log_progress(
                    f"Epoch {epoch + 1}/{epochs} | batch {batch_index}/{len(train_loader)} "
                    f"| avg_loss={avg_loss_so_far:.4f}"
                )

        epoch_loss = running_loss / max(1, len(train_loader))
        log_progress(f"Completed epoch {epoch + 1}/{epochs} with mean training loss {epoch_loss:.4f}")

    model.eval()
    log_progress("Running evaluation and converting logits to alarm scores")
    with torch.no_grad():
        eval_logits = model(torch.tensor(X_eval, dtype=torch.float32))
        eval_scores = torch.sigmoid(eval_logits).cpu().numpy()

    eval_results = eval_metadata.copy()
    eval_results["score"] = eval_scores
    eval_results["alarm_target"] = y_eval
    threshold_grid = default_threshold_grid(eval_scores, num_thresholds=threshold_steps)
    log_progress(f"Evaluating {len(threshold_grid)} threshold(s) for alarm tradeoff analysis")
    threshold_metrics = early_detection_threshold_sweep(
        eval_results,
        score_column="score",
        true_column="alarm_target",
        time_column=time_column,
        run_column=run_column,
        thresholds=threshold_grid,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    threshold_metrics_path = output_path / "lstm_threshold_metrics.csv"
    prediction_path = output_path / "lstm_window_predictions.csv"
    threshold_metrics.to_csv(threshold_metrics_path, index=False)
    eval_results.to_csv(prediction_path, index=False)

    recommended = choose_operating_point(threshold_metrics, max_false_alarm_rate=max_false_alarm_rate)
    log_progress(
        "Recommended operating point: "
        f"threshold={recommended['threshold']:.4f}, "
        f"false_alarm_rate={recommended['false_alarm_rate']:.4f}, "
        f"event_recall={recommended['event_recall']:.4f}"
    )
    summary = {
        "model": "SimpleLSTMDetector",
        "feature_count": len(feature_cols),
        "window_size": window_size,
        "stride": stride,
        "train_runs_used": int(train_metadata[run_column].nunique()),
        "eval_runs_used": int(eval_metadata[run_column].nunique()),
        "train_windows_used": int(len(train_metadata)),
        "eval_windows_used": int(len(eval_metadata)),
        "train_onset_sample": train_onset_sample,
        "eval_onset_sample": eval_onset_sample,
        "recommended_threshold": {
            "threshold": float(recommended["threshold"]),
            "false_alarm_rate": float(recommended["false_alarm_rate"]),
            "event_recall": float(recommended["event_recall"]),
            "row_f1": float(recommended["row_f1"]),
            "median_detection_delay": None
            if pd.isna(recommended["median_detection_delay"])
            else float(recommended["median_detection_delay"]),
        },
        "threshold_metrics_path": str(threshold_metrics_path),
        "prediction_path": str(prediction_path),
    }

    with (output_path / "lstm_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    log_progress(f"Finished. Outputs written to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple LSTM baseline for early TEP fault detection.")
    parser.add_argument("--data", default=str(CONFIG.processed_data_dir / "tep_train.csv"), help="Training CSV path.")
    parser.add_argument(
        "--eval-data",
        default=str(CONFIG.processed_data_dir / "tep_test.csv"),
        help="Evaluation CSV path. Defaults to the canonical TEP test set.",
    )
    parser.add_argument("--output-dir", default="results", help="Directory to save outputs.")
    parser.add_argument("--run-column", default=CONFIG.run_column, help="Run identifier column.")
    parser.add_argument("--time-column", default=CONFIG.time_column, help="Time/sample column.")
    parser.add_argument("--fault-column", default=CONFIG.multiclass_target_column, help="Fault label column.")
    parser.add_argument(
        "--train-onset-sample",
        type=int,
        default=20,
        help="Sample where the fault is assumed to begin in training runs.",
    )
    parser.add_argument(
        "--eval-onset-sample",
        type=int,
        default=160,
        help="Sample where the fault is assumed to begin in evaluation runs.",
    )
    parser.add_argument("--window-size", type=int, default=30, help="Number of timesteps in each LSTM window.")
    parser.add_argument("--stride", type=int, default=5, help="Step size between consecutive windows.")
    parser.add_argument("--max-train-runs", type=int, default=300, help="Optional cap on training runs.")
    parser.add_argument("--max-eval-runs", type=int, default=150, help="Optional cap on evaluation runs.")
    parser.add_argument("--max-train-windows", type=int, default=50_000, help="Optional cap on training windows.")
    parser.add_argument("--max-eval-windows", type=int, default=25_000, help="Optional cap on evaluation windows.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--threshold-steps", type=int, default=25, help="Number of thresholds for alarm analysis.")
    parser.add_argument("--chunksize", type=int, default=100000, help="CSV chunk size used while streaming large inputs.")
    parser.add_argument(
        "--max-false-alarm-rate",
        type=float,
        default=0.05,
        help="False-alarm budget used when choosing a recommended threshold.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_lstm_detector(
        args.data,
        args.output_dir,
        eval_path=args.eval_data,
        run_column=args.run_column,
        time_column=args.time_column,
        fault_column=args.fault_column,
        train_onset_sample=args.train_onset_sample,
        eval_onset_sample=args.eval_onset_sample,
        window_size=args.window_size,
        stride=args.stride,
        max_train_runs=args.max_train_runs,
        max_eval_runs=args.max_eval_runs,
        max_train_windows=args.max_train_windows,
        max_eval_windows=args.max_eval_windows,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        threshold_steps=args.threshold_steps,
        max_false_alarm_rate=args.max_false_alarm_rate,
        random_state=args.random_state,
        chunksize=args.chunksize,
    )
