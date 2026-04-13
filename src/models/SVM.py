from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data.load_data import feature_columns
from src.evaluation.metrics import classification_report_dict
from src.utils.config import CONFIG


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
KERNEL = "rbf"
C = 10.0
GAMMA = "scale"


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _segment_stats(series: np.ndarray) -> list[float]:
    """Compute mean, std, min, max for a 1D array."""
    return [series.mean(), series.std(), series.min(), series.max()]


def extract_run_features(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-run features with temporal segmentation.

    For each of the 52 sensors, computes:
      - Global stats: mean, std, min, max, slope (5)
      - Early segment stats (first quarter): mean, std, min, max (4)
      - Late segment stats (last quarter): mean, std, min, max (4)
      - Late - early mean difference (1)
      - Late / early std ratio (1)
    Total: 15 features per sensor × 52 sensors = 780 features per run.

    Returns (X, y) where X is [num_runs, 780] and y is [num_runs].
    """
    runs = []
    labels = []
    for _, group in df.groupby("run_id"):
        group = group.sort_values("sample")
        values = group[feature_cols].to_numpy(dtype=np.float32)  # [T, 52]
        T = len(values)
        quarter = T // 4
        timesteps = np.arange(T, dtype=np.float32)

        run_features = []
        for col_idx in range(values.shape[1]):
            series = values[:, col_idx]
            early = series[:quarter]
            late = series[-quarter:]

            # Global stats
            slope = np.polyfit(timesteps, series, 1)[0] if T > 1 else 0.0
            run_features.extend([
                series.mean(), series.std(), series.min(), series.max(), slope,
            ])

            # Segment stats
            run_features.extend(_segment_stats(early))
            run_features.extend(_segment_stats(late))

            # Temporal change indicators
            run_features.append(late.mean() - early.mean())
            early_std = early.std()
            run_features.append(late.std() / early_std if early_std > 1e-8 else 1.0)

        runs.append(run_features)
        labels.append(group["faultNumber"].iloc[0])

    return np.array(runs, dtype=np.float32), np.array(labels)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_svm(
    data_path: str,
    target_column: str,
    output_dir: str,
    eval_path: str | None = None,
) -> None:
    print("Loading training data...")
    df = pd.read_csv(data_path)
    feat_cols = feature_columns(df.columns)
    print(f"Training rows: {len(df)}  Runs: {df['run_id'].nunique()}")

    print("Extracting run-level features...")
    X_train, y_train = extract_run_features(df, feat_cols)
    print(f"Training feature matrix: {X_train.shape}  ({X_train.shape[1]} features per run)")

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    print(f"\n{'='*60}")
    print(f"Hyperparameters:")
    print(f"  Kernel:   {KERNEL}")
    print(f"  C:        {C}")
    print(f"  Gamma:    {GAMMA}")
    print(f"  Classes:  {len(np.unique(y_train))}")
    print(f"{'='*60}")

    print("\nTraining SVM...")
    model = SVC(kernel=KERNEL, C=C, gamma=GAMMA, class_weight="balanced", decision_function_shape="ovr")
    model.fit(X_train, y_train)
    print("Training complete.")

    train_preds = model.predict(X_train)
    train_acc = (train_preds == y_train).mean()
    print(f"Train accuracy: {train_acc:.4f}")

    # Evaluate
    if eval_path:
        print("\nLoading test data...")
        eval_df = pd.read_csv(eval_path)
        print(f"Test rows: {len(eval_df)}  Runs: {eval_df['run_id'].nunique()}")
        print("Extracting test features...")
        X_test, y_test = extract_run_features(eval_df, feat_cols)
        X_test = scaler.transform(X_test)
    else:
        X_test, y_test = X_train, y_train

    predictions = model.predict(X_test)
    metrics = classification_report_dict(y_test, predictions)
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}  Test F1: {metrics['f1_weighted']:.4f}")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with (output_path / f"svm_metrics_{timestamp}.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Metrics saved to {output_path / f'svm_metrics_{timestamp}.json'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an SVM for TEP fault classification.")
    parser.add_argument("--data", default=str(CONFIG.processed_data_dir / "tep_train.csv"), help="Training CSV path.")
    parser.add_argument(
        "--eval-data",
        default=str(CONFIG.processed_data_dir / "tep_test.csv"),
        help="Evaluation CSV path.",
    )
    parser.add_argument("--target", default=CONFIG.multiclass_target_column, help="Target column name.")
    parser.add_argument("--output-dir", default="results", help="Directory to save outputs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_svm(args.data, args.target, args.output_dir, eval_path=args.eval_data)
