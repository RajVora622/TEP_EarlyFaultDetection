from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.data.load_data import feature_columns
from src.evaluation.metrics import classification_report_dict
from src.utils.config import CONFIG


def train_random_forest(data_path: str, target_column: str, output_dir: str, eval_path: str | None = None) -> None:
    df = pd.read_csv(data_path)
    X = df[feature_columns(df.columns)]
    y = df[target_column]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(X, y)

    eval_df = pd.read_csv(eval_path) if eval_path else df
    eval_X = eval_df[feature_columns(eval_df.columns)]
    eval_y = eval_df[target_column]
    predictions = model.predict(eval_X)
    metrics = classification_report_dict(eval_y, predictions)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "rf_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a random forest baseline.")
    parser.add_argument("--data", default=str(CONFIG.processed_data_dir / "tep_train.csv"), help="Training CSV path.")
    parser.add_argument(
        "--eval-data",
        default=str(CONFIG.processed_data_dir / "tep_test.csv"),
        help="Evaluation CSV path. Defaults to the canonical TEP test set.",
    )
    parser.add_argument("--target", default=CONFIG.multiclass_target_column, help="Target column name.")
    parser.add_argument("--output-dir", default="results", help="Directory to save outputs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_random_forest(args.data, args.target, args.output_dir, eval_path=args.eval_data)
