from __future__ import annotations

from collections.abc import Sequence


def _binary_confusion_counts(y_true, y_pred) -> tuple[int, int, int, int]:
    """Compute TN, FP, FN, TP for binary labels without sklearn."""
    tn = fp = fn = tp = 0
    for true_value, pred_value in zip(y_true, y_pred):
        if int(true_value) == 1 and int(pred_value) == 1:
            tp += 1
        elif int(true_value) == 0 and int(pred_value) == 1:
            fp += 1
        elif int(true_value) == 1 and int(pred_value) == 0:
            fn += 1
        else:
            tn += 1
    return tn, fp, fn, tp


def _safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def binary_classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Return basic binary metrics without depending on sklearn."""
    tn, fp, fn, tp = _binary_confusion_counts(y_true, y_pred)
    total = tn + fp + fn + tp
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    accuracy = _safe_divide(tp + tn, total)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def default_threshold_grid(scores: Sequence[float], num_thresholds: int = 25):
    """Create a compact threshold grid from score quantiles.

    We use quantiles instead of a fixed linear range because model scores can be
    very unevenly distributed. Quantiles give us operating points that are more
    informative for the precision/recall tradeoff.
    """
    import numpy as np

    score_array = np.asarray(scores, dtype=float)
    if score_array.size == 0:
        raise ValueError("At least one score is required to build a threshold grid.")

    quantiles = np.linspace(0.0, 1.0, num=max(2, num_thresholds))
    return np.unique(np.quantile(score_array, quantiles))


def binary_threshold_sweep(
    y_true: Sequence[int],
    scores: Sequence[float],
    thresholds: Sequence[float],
):
    """Evaluate common binary-classification metrics across thresholds."""
    import numpy as np
    import pandas as pd

    y_true_array = np.asarray(y_true, dtype=int)
    score_array = np.asarray(scores, dtype=float)
    rows: list[dict[str, float]] = []

    for threshold in thresholds:
        predictions = (score_array >= threshold).astype(int)
        metric_row = binary_classification_metrics(y_true_array, predictions)
        tn = int(metric_row["tn"])
        fp = int(metric_row["fp"])
        fn = int(metric_row["fn"])
        tp = int(metric_row["tp"])
        negative_count = max(1, tn + fp)
        positive_count = max(1, tp + fn)

        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(metric_row["accuracy"]),
                "precision": float(metric_row["precision"]),
                "recall": float(metric_row["recall"]),
                "f1": float(metric_row["f1"]),
                # In the binary detector case, false positive rate and false alarm
                # rate mean the same thing, so we store both names to keep the
                # downstream analysis code easy to read.
                "false_alarm_rate": float(fp / negative_count),
                "false_positive_rate": float(fp / negative_count),
                "false_negative_rate": float(fn / positive_count),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
        )

    return pd.DataFrame(rows)


def early_detection_threshold_sweep(
    df,
    score_column: str,
    true_column: str,
    time_column: str,
    run_column: str,
    thresholds: Sequence[float],
):
    """Measure alarm-management tradeoffs for an early-detection model.

    The input dataframe should already contain one row per evaluation point,
    usually one window ending at a given sample. The `true_column` should turn
    on only after the assumed fault onset so that event recall and delay are
    measured against the actual alarming region.
    """
    if score_column not in df.columns:
        raise KeyError(f"'{score_column}' column not found.")
    if true_column not in df.columns:
        raise KeyError(f"'{true_column}' column not found.")
    if time_column not in df.columns:
        raise KeyError(f"'{time_column}' column not found.")
    if run_column not in df.columns:
        raise KeyError(f"'{run_column}' column not found.")

    import pandas as pd

    from src.evaluation.event_metrics import detection_delay, event_recall

    rows: list[dict[str, float | int | None]] = []

    for threshold in thresholds:
        frame = df.copy()
        frame["prediction"] = (frame[score_column] >= threshold).astype(int)
        binary_metrics = binary_classification_metrics(frame[true_column], frame["prediction"])

        delay_df = detection_delay(
            frame,
            true_column=true_column,
            pred_column="prediction",
            time_column=time_column,
            run_column=run_column,
        )
        valid_delays = delay_df["delay"].dropna() if not delay_df.empty else pd.Series(dtype="float64")

        rows.append(
            {
                "threshold": float(threshold),
                "row_precision": float(binary_metrics["precision"]),
                "row_recall": float(binary_metrics["recall"]),
                "row_f1": float(binary_metrics["f1"]),
                "false_alarm_rate": float(frame.loc[frame[true_column] == 0, "prediction"].mean()),
                "event_recall": float(
                    event_recall(frame, true_column=true_column, pred_column="prediction", run_column=run_column)
                ),
                "mean_detection_delay": float(valid_delays.mean()) if not valid_delays.empty else None,
                "median_detection_delay": float(valid_delays.median()) if not valid_delays.empty else None,
                "detected_runs": int(delay_df["pred_start"].notna().sum()) if not delay_df.empty else 0,
                "faulty_runs": int(len(delay_df)),
            }
        )

    return pd.DataFrame(rows)


def choose_operating_point(metrics_df, max_false_alarm_rate: float):
    """Pick a threshold that respects a false-alarm budget when possible."""
    if metrics_df.empty:
        raise ValueError("At least one metric row is required.")

    false_alarm_column = None
    for candidate in ["false_alarm_rate", "false_positive_rate"]:
        if candidate in metrics_df.columns:
            false_alarm_column = candidate
            break

    if false_alarm_column is None:
        raise KeyError(
            "Expected a false-alarm column in the metric table. "
            "Supported names are 'false_alarm_rate' and 'false_positive_rate'."
        )

    eligible = metrics_df.loc[metrics_df[false_alarm_column] <= max_false_alarm_rate]
    if eligible.empty:
        eligible = metrics_df

    # Student-style heuristic:
    # 1. stay under the nuisance-alarm budget if possible,
    # 2. maximize recall/event recall,
    # 3. then break ties by F1.
    ranking_columns = [col for col in ["event_recall", "recall", "row_f1", "f1"] if col in eligible.columns]
    ascending = [False] * len(ranking_columns)
    return eligible.sort_values(ranking_columns, ascending=ascending).iloc[0]
