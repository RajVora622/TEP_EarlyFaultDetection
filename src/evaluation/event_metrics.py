from __future__ import annotations

import pandas as pd


def detection_delay(
    df: pd.DataFrame,
    true_column: str,
    pred_column: str,
    time_column: str,
    run_column: str = "run_id",
) -> pd.DataFrame:
    """Measure the first detection delay for each run."""
    rows = []
    for run_id, run_df in df.groupby(run_column):
        positives = run_df.loc[run_df[true_column] == 1, time_column]
        predicted = run_df.loc[run_df[pred_column] == 1, time_column]

        if positives.empty:
            continue

        true_start = positives.min()
        pred_start = predicted.min() if not predicted.empty else None
        delay = None if pred_start is None else pred_start - true_start
        rows.append({"run_id": run_id, "true_start": true_start, "pred_start": pred_start, "delay": delay})

    return pd.DataFrame(rows)


def event_recall(
    df: pd.DataFrame,
    true_column: str,
    pred_column: str,
    run_column: str = "run_id",
) -> float:
    """Compute the fraction of runs where an event was detected at least once."""
    event_runs = 0
    detected_runs = 0

    for _, run_df in df.groupby(run_column):
        has_event = (run_df[true_column] == 1).any()
        if not has_event:
            continue

        event_runs += 1
        if ((run_df[true_column] == 1) & (run_df[pred_column] == 1)).any():
            detected_runs += 1

    if event_runs == 0:
        return 0.0
    return detected_runs / event_runs
