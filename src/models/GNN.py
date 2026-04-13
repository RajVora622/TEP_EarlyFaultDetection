from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

from src.data.load_data import feature_columns
from src.evaluation.metrics import classification_report_dict
from src.utils.config import CONFIG


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 50
HIDDEN_DIM = 64
PATIENCE = 5
DROPOUT = 0.3
WEIGHT_DECAY = 1e-4
TEMPORAL_KERNEL = 7
TEMPORAL_POOL = 4
NUM_NODES = 52


# ---------------------------------------------------------------------------
# TEP plant topology (Downs & Vogel 1993)
# ---------------------------------------------------------------------------
# Node indices follow feature_columns() order:
#   0-40 = xmeas_1 .. xmeas_41
#   41-51 = xmv_1 .. xmv_11
#
# Condenser and separator are merged into one unit — the Fortran source code
# (teprob.f) models them with shared state variables and a single cooling
# water system.

UNIT_MEMBERS = {
    "reactor":             [6, 7, 8, 20, 50],
    "condenser_separator": [10, 11, 12, 13, 21, 47, 51],
    "compressor":          [4, 9, 19, 28, 29, 30, 31, 32, 33, 34, 35, 45, 46],
    "stripper":            [3, 14, 15, 16, 17, 18, 36, 37, 38, 39, 40, 44, 48, 49],
    "mixer":               [0, 1, 2, 5, 22, 23, 24, 25, 26, 27, 41, 42, 43],
}

# Stream-based inter-unit edges following TEP material flow.
# Verified against Downs & Vogel (1993) and teprob.f source code.
# Each tuple is (source_node_idx, dest_node_idx) following physical flow.
# Both directions are added in build_edge_index() to make edges undirected.
#
# Node index reference:
#   xmeas_N → index N-1    (e.g. xmeas_7 = idx 6)
#   xmv_N   → index 40+N   (e.g. xmv_10 = idx 50)

STREAM_EDGES = [
    # --- Stream 6: Mixer → Reactor ---
    *[(src, dst) for src in [5, 22, 23, 24, 25, 26, 27]
                 for dst in [6, 7, 8]],

    # --- Stream 7: Reactor → Condenser/Separator ---
    *[(src, dst) for src in [6, 7, 8]
                 for dst in [10, 11, 12, 21]],

    # --- Stream 12: Cond/Sep → Compressor ---
    (12, 19), (12, 4), (12, 9),
    (10, 19),
    *[(10, d) for d in range(28, 36)],
    (11, 4), (11, 9),

    # --- Stream 8: Compressor → Mixer ---
    (4, 5),
    (45, 5),
    *[(28 + i, 22 + i) for i in range(6)],
    (34, 5), (35, 5),

    # --- Stream 10: Cond/Sep → Stripper ---
    *[(src, dst) for src in [13, 47]
                 for dst in [14, 15, 17]],
    (11, 14),
    *[(10, d) for d in [36, 37, 38, 39, 40]],

    # --- Stream 5: Stripper → Cond/Sep (overhead vapor return) ---
    *[(src, dst) for src in [15, 17]
                 for dst in [10, 11, 12]],
    (18, 12), (49, 12),

    # --- Pressure coupling ---
    (12, 5),
]


def build_edge_index() -> torch.Tensor:
    """Build the static 52-node edge index from the TEP plant topology."""
    sources: list[int] = []
    targets: list[int] = []

    for members in UNIT_MEMBERS.values():
        for u, v in combinations(members, 2):
            sources += [u, v]
            targets += [v, u]

    for src, dst in STREAM_EDGES:
        sources += [src, dst]
        targets += [dst, src]

    return torch.tensor([sources, targets], dtype=torch.long)


EDGE_INDEX = build_edge_index()


# ---------------------------------------------------------------------------
# Data conversion
# ---------------------------------------------------------------------------

def dataframe_to_graphs(
    df: pd.DataFrame,
    feature_cols: list[str],
    edge_index: torch.Tensor,
    target_column: str = "faultNumber",
) -> list[Data]:
    """Convert a TEP DataFrame into per-run PyG Data objects.

    Each run_id becomes one graph. Node features are the full time series:
    shape [52, T] where T = number of timesteps in the run.
    """
    graphs = []
    for _, group in df.groupby("run_id"):
        group = group.sort_values("sample")
        x = torch.tensor(
            group[feature_cols].to_numpy(dtype=np.float32).T,
            dtype=torch.float,
        )  # [52, T]
        y = torch.tensor(group[target_column].iloc[0], dtype=torch.long)
        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    return graphs


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TEPGraphNet(nn.Module):
    """GNN for per-run TEP fault classification.

    1. Temporal Encoder: two 1D conv layers compress each node's
       variable-length time series into a fixed embedding.
    2. Graph Convolution: three GCNConv layers propagate information
       between sensors following the plant topology.
    3. Concatenation Readout: all 52 node embeddings are concatenated
       (preserving sensor identity) and classified.
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_classes: int = 21,
        num_nodes: int = NUM_NODES,
        dropout: float = DROPOUT,
        kernel_size: int = TEMPORAL_KERNEL,
        temporal_pool: int = TEMPORAL_POOL,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2

        # Temporal encoder: Conv1d → pool to temporal_pool bins → learned compression
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(temporal_pool),  # [nodes, hidden_dim, temporal_pool]
        )
        # Learned compression: flatten pooled output → hidden_dim
        self.temporal_compress = nn.Sequential(
            nn.Linear(hidden_dim * temporal_pool, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # GCN layers with batch norm
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.gcn_dropout = nn.Dropout(dropout)

        self.num_nodes = num_nodes
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_nodes * hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # x: [total_nodes, T]  (total_nodes = num_graphs * 52)

        # Temporal encoding: conv → pool to 16 bins → learned compression to hidden_dim
        x = x.unsqueeze(1)                    # [total_nodes, 1, T]
        x = self.temporal_conv(x)              # [total_nodes, hidden_dim, 16]
        x = x.flatten(1)                       # [total_nodes, hidden_dim * 16]
        x = self.temporal_compress(x)          # [total_nodes, hidden_dim]

        # Graph convolution with residual connections
        residual = x
        x = self.gcn_dropout(torch.relu(self.bn1(self.conv1(x, edge_index))))
        x = x + residual

        residual = x
        x = self.gcn_dropout(torch.relu(self.bn2(self.conv2(x, edge_index))))
        x = x + residual

        residual = x
        x = self.gcn_dropout(torch.relu(self.bn3(self.conv3(x, edge_index))))
        x = x + residual

        # Concatenation readout (preserves sensor identity)
        num_graphs = batch.max().item() + 1
        x = x.view(num_graphs, self.num_nodes * x.size(-1))  # [num_graphs, 52*hidden_dim]

        return self.classifier(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _evaluate(model, loader, encoder, criterion):
    """Run evaluation and return loss, accuracy, and predicted labels."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    with torch.no_grad():
        for data in loader:
            logits = model(data.x, data.edge_index, data.batch)
            loss = criterion(logits, data.y)
            total_loss += loss.item() * data.num_graphs
            preds = torch.argmax(logits, dim=1)
            correct += (preds == data.y).sum().item()
            total += data.num_graphs
            all_preds.append(preds.cpu().numpy())
    predictions = np.concatenate(all_preds)
    predicted_labels = encoder.inverse_transform(predictions)
    return total_loss / total, correct / total, predicted_labels


def _plot_curves(history: dict, output_path: Path, timestamp: str = "") -> None:
    """Save training/validation loss and accuracy plots."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    if history.get("stopped_epoch"):
        ax1.axvline(x=history["stopped_epoch"], color="gray", linestyle="--", label="Early Stop")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Accuracy", markersize=4)
    ax2.plot(epochs, history["val_acc"], "r-o", label="Val Accuracy", markersize=4)
    if history.get("stopped_epoch"):
        ax2.axvline(x=history["stopped_epoch"], color="gray", linestyle="--", label="Early Stop")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = f"_{timestamp}" if timestamp else ""
    curves_file = output_path / f"gnn_training_curves{suffix}.png"
    plt.savefig(curves_file, dpi=150)
    plt.close()
    print(f"Training curves saved to {curves_file}")


def train_gnn(
    data_path: str,
    target_column: str,
    output_dir: str,
    eval_path: str | None = None,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
) -> None:
    print("Loading data...")
    df = pd.read_csv(data_path)
    feat_cols = feature_columns(df.columns)
    print(f"Training rows: {len(df)}  Runs: {df['run_id'].nunique()}")

    # Normalize on the flat DataFrame before grouping into runs
    scaler = StandardScaler()
    df[feat_cols] = scaler.fit_transform(df[feat_cols].to_numpy(dtype=np.float32))

    # Build per-run graphs and split into train/val (same sequence length)
    print("Building graphs...")
    edge_index = build_edge_index()
    all_graphs = dataframe_to_graphs(df, feat_cols, edge_index)
    print(f"Total graphs: {len(all_graphs)}  Node features: [{NUM_NODES}, {all_graphs[0].x.shape[1]}]")

    # LabelEncoder for inverse_transform at eval time
    encoder = LabelEncoder()
    all_labels = [g.y.item() for g in all_graphs]
    encoder.fit(all_labels)
    print(f"Classes: {len(encoder.classes_)} (faults {min(encoder.classes_)}-{max(encoder.classes_)})")

    # 90/10 train/val split (stratified by fault, same sequence length)
    rng = np.random.RandomState(42)
    val_indices = set()
    for cls in encoder.classes_:
        cls_idx = [i for i, l in enumerate(all_labels) if l == cls]
        n_val = max(1, len(cls_idx) // 10)
        val_indices.update(rng.choice(cls_idx, size=n_val, replace=False))
    train_graphs = [g for i, g in enumerate(all_graphs) if i not in val_indices]
    val_graphs = [g for i, g in enumerate(all_graphs) if i in val_indices]
    print(f"Train: {len(train_graphs)}  Val: {len(val_graphs)}")

    # Load test data separately for final evaluation
    if eval_path:
        print("Preparing test data...")
        eval_df = pd.read_csv(eval_path)
        eval_df[feat_cols] = scaler.transform(eval_df[feat_cols].to_numpy(dtype=np.float32))
        eval_graphs = dataframe_to_graphs(eval_df, feat_cols, edge_index)
    else:
        eval_graphs = all_graphs

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
    num_batches = len(train_loader)
    print(f"Validation graphs: {len(val_graphs)}")
    print(f"Batches/epoch: {num_batches}")

    model = TEPGraphNet(
        hidden_dim=HIDDEN_DIM,
        num_classes=len(encoder.classes_),
        num_nodes=NUM_NODES,
        dropout=DROPOUT,
        kernel_size=TEMPORAL_KERNEL,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # Tracking
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "stopped_epoch": None}
    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    print(f"\n{'='*60}")
    print(f"Hyperparameters:")
    print(f"  Learning rate:     {LEARNING_RATE}")
    print(f"  Batch size:        {BATCH_SIZE}")
    print(f"  Epochs:            {epochs}")
    print(f"  Hidden dim:        {HIDDEN_DIM}")
    print(f"  Dropout:           {DROPOUT}")
    print(f"  Temporal kernel:   {TEMPORAL_KERNEL}")
    print(f"  Patience:          {patience}")
    print(f"  Weight decay:      {WEIGHT_DECAY}")
    print(f"  Optimizer:         Adam")
    print(f"  Loss:              CrossEntropyLoss")
    print(f"{'='*60}")
    train_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index, data.batch)
            loss = criterion(logits, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            correct += (torch.argmax(logits, dim=1) == data.y).sum().item()
            total += data.num_graphs

            if (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - epoch_start
                eta = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)
                print(f"  batch {batch_idx + 1}/{num_batches}  "
                      f"loss={total_loss / total:.4f}  "
                      f"acc={correct / total:.4f}  "
                      f"ETA={eta:.0f}s")

        train_loss = total_loss / len(train_graphs)
        train_acc = correct / total

        # --- Validate ---
        val_loss, val_acc, _ = _evaluate(model, val_loader, encoder, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - train_start
        remaining = epoch_time * (epochs - epoch - 1)
        print(f"Epoch {epoch + 1}/{epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"time={epoch_time:.0f}s  ETA={remaining:.0f}s")

        # --- Early stopping + checkpoint ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output_path / "gnn_checkpoint_best.pt")
            print(f"  Checkpoint saved (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                history["stopped_epoch"] = epoch + 1
                break

    print(f"\nTraining complete in {time.time() - train_start:.0f}s")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model (val_loss={best_val_loss:.4f})")

    # --- Final evaluation on full test set ---
    print("\nFinal evaluation on full test set...")
    eval_loader = DataLoader(eval_graphs, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Eval graphs: {len(eval_graphs)}")

    _, test_acc, predicted_labels = _evaluate(model, eval_loader, encoder, criterion)
    true_labels = [g.y.item() for g in eval_graphs]
    true_labels = encoder.inverse_transform(true_labels)
    metrics = classification_report_dict(true_labels, predicted_labels)
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}  Test F1: {metrics['f1_weighted']:.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with (output_path / f"gnn_metrics_{timestamp}.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Metrics saved to {output_path / f'gnn_metrics_{timestamp}.json'}")

    model_file = output_path / f"gnn_model_{timestamp}.pt"
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file}")

    _plot_curves(history, output_path, timestamp)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GNN for TEP fault classification.")
    parser.add_argument("--data", default=str(CONFIG.processed_data_dir / "tep_train.csv"), help="Training CSV path.")
    parser.add_argument(
        "--eval-data",
        default=str(CONFIG.processed_data_dir / "tep_test.csv"),
        help="Evaluation CSV path.",
    )
    parser.add_argument("--target", default=CONFIG.multiclass_target_column, help="Target column name.")
    parser.add_argument("--output-dir", default="results", help="Directory to save outputs.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs.")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Early stopping patience.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_gnn(
        args.data, args.target, args.output_dir,
        eval_path=args.eval_data, epochs=args.epochs,
        patience=args.patience,
    )
