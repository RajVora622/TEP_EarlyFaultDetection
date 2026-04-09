from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.load_data import feature_columns
from src.evaluation.metrics import classification_report_dict
from src.utils.config import CONFIG


class SimpleCNN(nn.Module):
    def __init__(self, input_length: int, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(32 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        self.input_length = input_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_cnn(
    data_path: str,
    target_column: str,
    output_dir: str,
    eval_path: str | None = None,
    epochs: int = 10,
) -> None:
    df = pd.read_csv(data_path)
    feature_cols = feature_columns(df.columns)
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_column]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    features = torch.tensor(X).unsqueeze(1)
    labels = torch.tensor(y_encoded, dtype=torch.long)

    loader = DataLoader(TensorDataset(features, labels), batch_size=64, shuffle=True)
    model = SimpleCNN(input_length=X.shape[1], num_classes=len(encoder.classes_))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    eval_df = pd.read_csv(eval_path) if eval_path else df
    eval_features = torch.tensor(eval_df[feature_columns(eval_df.columns)].to_numpy(dtype=np.float32)).unsqueeze(1)
    eval_labels = eval_df[target_column]

    model.eval()
    with torch.no_grad():
        logits = model(eval_features)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    predicted_labels = encoder.inverse_transform(predictions)
    metrics = classification_report_dict(eval_labels, predicted_labels)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "cnn_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple 1D CNN baseline.")
    parser.add_argument("--data", default=str(CONFIG.processed_data_dir / "tep_train.csv"), help="Training CSV path.")
    parser.add_argument(
        "--eval-data",
        default=str(CONFIG.processed_data_dir / "tep_test.csv"),
        help="Evaluation CSV path. Defaults to the canonical TEP test set.",
    )
    parser.add_argument("--target", default=CONFIG.multiclass_target_column, help="Target column name.")
    parser.add_argument("--output-dir", default="results", help="Directory to save outputs.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_cnn(args.data, args.target, args.output_dir, eval_path=args.eval_data, epochs=args.epochs)
