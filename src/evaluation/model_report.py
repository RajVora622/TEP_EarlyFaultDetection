from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import CONFIG


def ensure_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_plot(fig, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_svm_threshold_tradeoff(df: pd.DataFrame, chosen_threshold: float, output_path: str | Path) -> None:
    """Plot the main alarm tradeoff for the binary SVM."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["false_alarm_rate"], df["recall"], marker="o", label="Recall")
    ax.plot(df["false_alarm_rate"], df["f1"], marker="s", label="F1")

    chosen_row = df.loc[(df["threshold"] - chosen_threshold).abs().idxmin()]
    ax.scatter(
        [chosen_row["false_alarm_rate"]],
        [chosen_row["recall"]],
        color="crimson",
        s=70,
        label="Chosen threshold",
        zorder=3,
    )
    ax.set_xlabel("False Alarm Rate")
    ax.set_ylabel("Metric Value")
    ax.set_title("SVM Validation Tradeoff")
    ax.legend()
    ax.grid(alpha=0.3)
    save_plot(fig, output_path)


def plot_lstm_alarm_tradeoff(df: pd.DataFrame, chosen_threshold: float, output_path: str | Path) -> None:
    """Plot the event-level alarm tradeoff for the LSTM."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["false_alarm_rate"], df["event_recall"], marker="o", label="Event recall")
    ax.plot(df["false_alarm_rate"], df["row_f1"], marker="s", label="Row F1")

    chosen_row = df.loc[(df["threshold"] - chosen_threshold).abs().idxmin()]
    ax.scatter(
        [chosen_row["false_alarm_rate"]],
        [chosen_row["event_recall"]],
        color="crimson",
        s=70,
        label="Chosen threshold",
        zorder=3,
    )
    ax.set_xlabel("False Alarm Rate")
    ax.set_ylabel("Metric Value")
    ax.set_title("LSTM Alarm Tradeoff")
    ax.legend()
    ax.grid(alpha=0.3)
    save_plot(fig, output_path)


def plot_lstm_delay_curve(df: pd.DataFrame, chosen_threshold: float, output_path: str | Path) -> None:
    """Plot how detection delay changes as the threshold becomes stricter."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["threshold"], df["median_detection_delay"], marker="o", label="Median detection delay")

    chosen_row = df.loc[(df["threshold"] - chosen_threshold).abs().idxmin()]
    ax.scatter(
        [chosen_row["threshold"]],
        [chosen_row["median_detection_delay"]],
        color="crimson",
        s=70,
        label="Chosen threshold",
        zorder=3,
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Delay (samples)")
    ax.set_title("LSTM Detection Delay by Threshold")
    ax.legend()
    ax.grid(alpha=0.3)
    save_plot(fig, output_path)


def multiclass_report_frame(multiclass_metrics: dict) -> pd.DataFrame:
    """Convert the sklearn-style multiclass report into a tidy dataframe."""
    rows = []
    for label, values in multiclass_metrics["metrics"]["classification_report"].items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        rows.append(
            {
                "class_label": int(label),
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1": float(values["f1-score"]),
                "support": int(values["support"]),
            }
        )
    return pd.DataFrame(rows).sort_values("class_label").reset_index(drop=True)


def plot_multiclass_class_recall(df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot recall by fault class for the multiclass SVM."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(df["class_label"].astype(str), df["recall"])
    ax.set_xlabel("Fault Class")
    ax.set_ylabel("Recall")
    ax.set_title("Multiclass SVM Recall by Fault Class")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    save_plot(fig, output_path)


def _format_float(value, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def build_report_text(
    svm_metrics: dict,
    svm_threshold_df: pd.DataFrame,
    lstm_metrics: dict,
    lstm_threshold_df: pd.DataFrame,
    multiclass_metrics: dict | None,
    multiclass_df: pd.DataFrame | None,
    output_dir: Path,
) -> str:
    """Write the interpretation as a compact report.

    The goal here is not to compare models directly. Instead, we explain what
    each model's output means and how to reason about threshold choice.
    """
    svm_choice = svm_metrics["recommended_threshold"]
    svm_default = svm_metrics["default_threshold_eval_metrics"]["classification_report"]
    lstm_choice = lstm_metrics["recommended_threshold"]
    multiclass_top_note = ""
    multiclass_bottom_note = ""

    if multiclass_metrics is not None and multiclass_df is not None and not multiclass_df.empty:
        top_classes = multiclass_df.sort_values("recall", ascending=False).head(5)
        bottom_classes = multiclass_df.sort_values("recall", ascending=True).head(5)
        multiclass_top_note = ", ".join(
            f"class {int(row.class_label)} ({_format_float(row.recall)})" for row in top_classes.itertuples()
        )
        multiclass_bottom_note = ", ".join(
            f"class {int(row.class_label)} ({_format_float(row.recall)})" for row in bottom_classes.itertuples()
        )

    svm_low_alarm = svm_threshold_df.loc[svm_threshold_df["false_alarm_rate"] <= 0.05].head(1)
    svm_low_alarm_note = ""
    if not svm_low_alarm.empty:
        row = svm_low_alarm.iloc[0]
        svm_low_alarm_note = (
            f"On the run-aware validation split, a threshold near {_format_float(row['threshold'])} "
            f"keeps the false alarm rate around {_format_float(row['false_alarm_rate'])} while recall is "
            f"{_format_float(row['recall'])}."
        )

    lstm_low_alarm = lstm_threshold_df.loc[lstm_threshold_df["false_alarm_rate"] <= 0.05].head(1)
    lstm_low_alarm_note = ""
    if not lstm_low_alarm.empty:
        row = lstm_low_alarm.iloc[0]
        lstm_low_alarm_note = (
            f"Near a false alarm rate of {_format_float(row['false_alarm_rate'])}, the LSTM still has "
            f"event recall {_format_float(row['event_recall'])} with median delay "
            f"{_format_float(row['median_detection_delay'], 1)} samples."
        )

    lines = [
        "# Model Results Report",
        "",
        "## How To Read These Results",
        "",
        "The main theme is alarm management: every threshold choice trades off nuisance alarms against missed or delayed detections.",
        "For the SVM, the important values are false alarm rate, recall, and precision on held-out runs.",
        "For the LSTM, the important values are false alarm rate, event recall, and detection delay, because early warning is a sequence problem rather than a single-row problem.",
        "",
        "## SVM Interpretation",
        "",
        f"- The SVM was trained on {svm_metrics['train_runs_used']} runs, tuned on {svm_metrics['validation_runs_used']} validation runs, and evaluated on {svm_metrics['eval_runs_used']} external runs.",
        f"- At the chosen threshold {_format_float(svm_choice['threshold'])}, the external evaluation false alarm rate is {_format_float(svm_choice['eval_false_alarm_rate'])}, recall is {_format_float(svm_choice['eval_recall'])}, precision is {_format_float(svm_choice['eval_precision'])}, and F1 is {_format_float(svm_choice['eval_f1'])}.",
        f"- The default threshold produces positive-class recall { _format_float(svm_default['1']['recall']) } and positive-class precision { _format_float(svm_default['1']['precision']) } on the external evaluation split.",
        f"- The normal class recall is { _format_float(svm_default['0']['recall']) }, which means the model is fairly good at recognizing normal points when they appear, but the dataset is still heavily dominated by faulty rows.",
        f"- {svm_low_alarm_note}" if svm_low_alarm_note else "- The validation threshold curve shows how recall falls as the alarm policy becomes stricter.",
        "- Interpretation: the SVM is a conservative detector. It can keep nuisance alarms low, but lowering false alarms also reduces the fraction of faulty rows that get detected.",
        "",
        "## LSTM Interpretation",
        "",
        f"- The LSTM was trained on {lstm_metrics['train_runs_used']} runs and evaluated on {lstm_metrics['eval_runs_used']} runs using windows of length {lstm_metrics['window_size']} and stride {lstm_metrics['stride']}.",
        f"- At the chosen threshold {_format_float(lstm_choice['threshold'])}, the false alarm rate is {_format_float(lstm_choice['false_alarm_rate'])}, event recall is {_format_float(lstm_choice['event_recall'])}, row F1 is {_format_float(lstm_choice['row_f1'])}, and median detection delay is {_format_float(lstm_choice['median_detection_delay'], 1)} samples.",
        f"- {lstm_low_alarm_note}" if lstm_low_alarm_note else "- The threshold curve shows how stricter thresholds reduce nuisance alarms but increase detection delay.",
        "- Negative delay values at very low thresholds mean the model is alarming before the assumed fault-onset time. That can be useful as a warning sign, but it also creates nuisance alarms if the threshold is too loose.",
        "- Interpretation: the LSTM output should be read as an operating curve. Lower thresholds detect faults earlier and more often, while higher thresholds are quieter but slower.",
        "",
        "## Multiclass SVM Interpretation",
        "",
    ]

    if multiclass_metrics is not None and multiclass_df is not None and not multiclass_df.empty:
        lines.extend(
            [
                f"- The multiclass SVM was trained on {multiclass_metrics['train_runs_used']} runs and evaluated on {multiclass_metrics['eval_runs_used']} runs across {len(multiclass_metrics['class_labels'])} classes.",
                f"- Overall multiclass accuracy is {_format_float(multiclass_metrics['metrics']['accuracy'])}, weighted F1 is {_format_float(multiclass_metrics['metrics']['f1_weighted'])}, and macro F1 is {_format_float(multiclass_metrics['metrics']['classification_report']['macro avg']['f1-score'])}.",
                f"- The best-recalled classes are {multiclass_top_note}.",
                f"- The hardest classes are {multiclass_bottom_note}.",
                "- Interpretation: the multiclass SVM can identify some fault types reasonably well, but performance is very uneven across classes. This suggests that some faults have distinct static signatures while others overlap heavily in the 52-variable snapshot space.",
                "- In practice, this means the multiclass output should be treated as a diagnostic aid rather than a uniformly reliable fault identifier across all 21 classes.",
            ]
        )
    else:
        lines.extend(
            [
                "- No multiclass SVM metrics file was provided, so this section could not be generated.",
            ]
        )

    lines.extend(
        [
            "",
        "## Practical Takeaways",
        "",
        "- Do not rely on accuracy alone. The threshold-specific alarm metrics are more informative for this project.",
        "- A reasonable deployment-style choice is to start from a target false alarm budget, then choose the threshold that gives the best recall or event recall under that budget.",
        "- For the SVM, that budget mostly controls the tradeoff between missed rows and nuisance alarms.",
        "- For the LSTM, that budget also controls how early the alarm tends to fire.",
        "- For multiclass classification, the per-class recall plot matters more than the overall accuracy because some fault labels are much easier than others.",
        "",
        "## Generated Artifacts",
        "",
        f"- SVM plot: [{(output_dir / 'svm_tradeoff.png').name}]({output_dir / 'svm_tradeoff.png'})",
        f"- LSTM tradeoff plot: [{(output_dir / 'lstm_tradeoff.png').name}]({output_dir / 'lstm_tradeoff.png'})",
        f"- LSTM delay plot: [{(output_dir / 'lstm_delay_curve.png').name}]({output_dir / 'lstm_delay_curve.png'})",
        f"- Multiclass recall plot: [{(output_dir / 'svm_multiclass_recall.png').name}]({output_dir / 'svm_multiclass_recall.png'})",
        ]
    )
    return "\n".join(lines)


def generate_model_report(
    svm_metrics_path: str | Path,
    svm_threshold_path: str | Path,
    lstm_metrics_path: str | Path,
    lstm_threshold_path: str | Path,
    multiclass_metrics_path: str | Path | None,
    output_dir: str | Path,
) -> Path:
    output_dir = ensure_dir(output_dir)
    svm_metrics = read_json(svm_metrics_path)
    svm_threshold_df = pd.read_csv(svm_threshold_path)
    lstm_metrics = read_json(lstm_metrics_path)
    lstm_threshold_df = pd.read_csv(lstm_threshold_path)
    multiclass_metrics = None
    multiclass_df = None
    if multiclass_metrics_path and Path(multiclass_metrics_path).exists():
        multiclass_metrics = read_json(multiclass_metrics_path)
        multiclass_df = multiclass_report_frame(multiclass_metrics)

    plot_svm_threshold_tradeoff(
        svm_threshold_df,
        chosen_threshold=float(svm_metrics["recommended_threshold"]["threshold"]),
        output_path=output_dir / "svm_tradeoff.png",
    )
    plot_lstm_alarm_tradeoff(
        lstm_threshold_df,
        chosen_threshold=float(lstm_metrics["recommended_threshold"]["threshold"]),
        output_path=output_dir / "lstm_tradeoff.png",
    )
    plot_lstm_delay_curve(
        lstm_threshold_df,
        chosen_threshold=float(lstm_metrics["recommended_threshold"]["threshold"]),
        output_path=output_dir / "lstm_delay_curve.png",
    )
    if multiclass_df is not None:
        plot_multiclass_class_recall(multiclass_df, output_path=output_dir / "svm_multiclass_recall.png")

    report_text = build_report_text(
        svm_metrics=svm_metrics,
        svm_threshold_df=svm_threshold_df,
        lstm_metrics=lstm_metrics,
        lstm_threshold_df=lstm_threshold_df,
        multiclass_metrics=multiclass_metrics,
        multiclass_df=multiclass_df,
        output_dir=output_dir,
    )
    report_path = output_dir / "model_results_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a compact markdown report and plots from model outputs.")
    parser.add_argument("--svm-metrics", default="results/svm/svm_metrics.json", help="Path to SVM metrics JSON.")
    parser.add_argument(
        "--svm-thresholds",
        default="results/svm/svm_validation_threshold_metrics.csv",
        help="Path to SVM validation threshold metrics CSV.",
    )
    parser.add_argument("--lstm-metrics", default="results/lstm/lstm_metrics.json", help="Path to LSTM metrics JSON.")
    parser.add_argument(
        "--lstm-thresholds",
        default="results/lstm/lstm_threshold_metrics.csv",
        help="Path to LSTM threshold metrics CSV.",
    )
    parser.add_argument(
        "--multiclass-metrics",
        default="results/svm_multiclass/svm_metrics.json",
        help="Optional path to multiclass SVM metrics JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(CONFIG.results_dir / "report"),
        help="Directory where the markdown report and plots will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_model_report(
        svm_metrics_path=args.svm_metrics,
        svm_threshold_path=args.svm_thresholds,
        lstm_metrics_path=args.lstm_metrics,
        lstm_threshold_path=args.lstm_thresholds,
        multiclass_metrics_path=args.multiclass_metrics,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
