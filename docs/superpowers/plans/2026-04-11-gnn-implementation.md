# GNN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a Graph Neural Network for multiclass fault classification on the TEP dataset, where the graph encodes the physical plant topology (5 process units as nodes, material flows as directed edges).

**Architecture:** Each timestep is classified independently. The 52 sensor variables are grouped by process unit into 5 node feature vectors. The adjacency matrix encodes the physical flow connections (Reactor→Condenser→Separator→Compressor→Mixer→Reactor, with Separator↔Stripper branch). Two GCN message-passing layers produce node embeddings, which are mean-pooled into a graph-level vector and classified via a linear head into 21 classes (faultNumber 0-20).

**Tech Stack:** PyTorch, PyTorch Geometric (torch_geometric), pandas, numpy, scikit-learn (LabelEncoder, metrics)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/models/GNN.py` | TEP graph definition (adjacency, node feature mapping), GNN model class, training function, CLI entry point |
| `tests/test_gnn.py` | Unit tests for graph construction, model forward pass, and training output |

The existing file `src/models/GNN.py` will house everything — this follows the pattern of the other model files (e.g., `train_cnn.py` puts model class + training function + CLI in one file).

---

### Task 1: Define the TEP Graph Structure

**Files:**
- Modify: `src/models/GNN.py`
- Create: `tests/test_gnn.py`

This task defines the physical TEP plant topology as a PyG-compatible graph: which sensors belong to which process unit (node features), and which units are connected by material flows (edge index).

- [ ] **Step 1: Write the failing test for graph construction**

Create `tests/test_gnn.py`:

```python
from __future__ import annotations

import torch
from src.models.GNN import build_tep_graph, UNIT_FEATURES


def test_unit_features_cover_all_52_variables():
    """Every xmeas/xmv column must be assigned to exactly one process unit."""
    all_vars = []
    for vars_list in UNIT_FEATURES.values():
        all_vars.extend(vars_list)
    assert len(all_vars) == 52, f"Expected 52 variables, got {len(all_vars)}"
    assert len(set(all_vars)) == 52, "Duplicate variable assignments found"


def test_build_tep_graph_shape():
    """build_tep_graph should return a PyG Data object with correct shapes."""
    import pandas as pd
    import numpy as np

    # Create a fake row with 52 feature columns
    feature_cols = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
    row = pd.Series(np.random.randn(52), index=feature_cols)

    data = build_tep_graph(row)

    assert data.x.shape[0] == 5, "Should have 5 nodes (process units)"
    assert data.edge_index.shape[0] == 2, "edge_index should have 2 rows (source, target)"
    assert data.edge_index.shape[1] > 0, "Should have at least one edge"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gnn.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_tep_graph' from 'src.models.GNN'`

- [ ] **Step 3: Implement the graph structure in GNN.py**

Write to `src/models/GNN.py`:

```python
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data

from src.data.load_data import feature_columns
from src.evaluation.metrics import classification_report_dict
from src.utils.config import CONFIG


# --- TEP Plant Topology ---
# Maps each process unit to its sensor columns (xmeas/xmv).
# Based on Downs & Vogel (1993) process flow diagram.
UNIT_NAMES = ["reactor", "condenser", "separator", "compressor", "stripper"]

UNIT_FEATURES: OrderedDict[str, list[str]] = OrderedDict({
    "reactor": [
        "xmeas_7", "xmeas_8", "xmeas_9", "xmeas_21",  # pressure, level, temp, CW outlet
        "xmeas_23", "xmeas_24", "xmeas_25", "xmeas_26", "xmeas_27", "xmeas_28",  # feed compositions A-F
        "xmv_10",  # reactor CW flow
    ],
    "condenser": [
        "xmeas_22",  # separator/condenser CW outlet temp
        "xmv_11",    # condenser CW flow
    ],
    "separator": [
        "xmeas_11", "xmeas_12", "xmeas_13", "xmeas_14",  # temp, level, pressure, underflow
        "xmv_7",  # separator liquid flow valve
    ],
    "compressor": [
        "xmeas_5", "xmeas_10", "xmeas_20",  # recycle flow, purge rate, compressor work
        "xmeas_29", "xmeas_30", "xmeas_31", "xmeas_32",  # purge compositions A-D
        "xmeas_33", "xmeas_34", "xmeas_35", "xmeas_36",  # purge compositions E-H
        "xmv_5", "xmv_6",  # recycle valve, purge valve
    ],
    "stripper": [
        "xmeas_1", "xmeas_2", "xmeas_3", "xmeas_4", "xmeas_6",  # feed flows + reactor feed rate
        "xmeas_15", "xmeas_16", "xmeas_17", "xmeas_18", "xmeas_19",  # level, pressure, underflow, temp, steam
        "xmeas_37", "xmeas_38", "xmeas_39", "xmeas_40", "xmeas_41",  # product compositions D-H
        "xmv_1", "xmv_2", "xmv_3", "xmv_4",  # feed valves D, E, A, A+C
        "xmv_8", "xmv_9",  # stripper liquid valve, steam valve
    ],
})

# Directed edges following physical material flow.
# Indices: 0=reactor, 1=condenser, 2=separator, 3=compressor, 4=stripper
#   reactor -> condenser    (stream 7)
#   condenser -> separator  (stream 13)
#   separator -> compressor (stream 12)
#   separator -> stripper   (stream 10)
#   compressor -> reactor   (stream 8, recycle through mixer)
#   stripper -> separator   (stream 5, vapor return)
EDGE_INDEX = torch.tensor([
    [0, 1, 2, 2, 3, 4],  # source nodes
    [1, 2, 3, 4, 0, 2],  # target nodes
], dtype=torch.long)


def build_tep_graph(row: pd.Series, label: int | None = None) -> Data:
    """Convert a single timestep (row of sensor readings) into a PyG Data object.

    Each of the 5 process units becomes a node. Node features are the sensor
    readings assigned to that unit, zero-padded to the size of the largest group.
    """
    max_features = max(len(v) for v in UNIT_FEATURES.values())
    node_features = []
    for unit_name in UNIT_NAMES:
        cols = UNIT_FEATURES[unit_name]
        values = [row[col] for col in cols]
        # Zero-pad to uniform width
        padded = values + [0.0] * (max_features - len(values))
        node_features.append(padded)

    x = torch.tensor(node_features, dtype=torch.float)
    data = Data(x=x, edge_index=EDGE_INDEX)
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)
    return data
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gnn.py -v`
Expected: PASS — both `test_unit_features_cover_all_52_variables` and `test_build_tep_graph_shape` pass.

- [ ] **Step 5: Commit**

```bash
git add src/models/GNN.py tests/test_gnn.py
git commit -m "feat(gnn): define TEP graph topology and node feature mapping"
```

---

### Task 2: Implement the GNN Model Class

**Files:**
- Modify: `src/models/GNN.py`
- Modify: `tests/test_gnn.py`

This task implements the GNN model using GCNConv layers with mean pooling for graph-level classification.

- [ ] **Step 1: Write the failing test for model forward pass**

Append to `tests/test_gnn.py`:

```python
def test_gnn_model_forward_pass():
    """The model should accept a batch of graphs and return logits for 21 classes."""
    from torch_geometric.data import Batch
    from src.models.GNN import TEPGraphNet, UNIT_FEATURES

    max_features = max(len(v) for v in UNIT_FEATURES.values())
    num_classes = 21

    model = TEPGraphNet(in_channels=max_features, num_classes=num_classes)

    # Create a batch of 4 fake graphs
    graphs = []
    for _ in range(4):
        x = torch.randn(5, max_features)
        data = Data(x=x, edge_index=EDGE_INDEX)
        graphs.append(data)

    batch = Batch.from_data_list(graphs)

    model.eval()
    with torch.no_grad():
        logits = model(batch)

    assert logits.shape == (4, num_classes), f"Expected (4, {num_classes}), got {logits.shape}"


def test_gnn_model_output_changes_with_different_input():
    """Sanity check: different inputs should produce different outputs."""
    from torch_geometric.data import Batch
    from src.models.GNN import TEPGraphNet, UNIT_FEATURES

    max_features = max(len(v) for v in UNIT_FEATURES.values())
    model = TEPGraphNet(in_channels=max_features, num_classes=21)
    model.eval()

    g1 = Data(x=torch.zeros(5, max_features), edge_index=EDGE_INDEX)
    g2 = Data(x=torch.ones(5, max_features), edge_index=EDGE_INDEX)

    batch1 = Batch.from_data_list([g1])
    batch2 = Batch.from_data_list([g2])

    with torch.no_grad():
        out1 = model(batch1)
        out2 = model(batch2)

    assert not torch.allclose(out1, out2), "Different inputs should produce different outputs"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gnn.py::test_gnn_model_forward_pass -v`
Expected: FAIL — `ImportError: cannot import name 'TEPGraphNet' from 'src.models.GNN'`

- [ ] **Step 3: Implement the GNN model class**

Add to `src/models/GNN.py` after the `build_tep_graph` function:

```python
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool


class TEPGraphNet(nn.Module):
    """Graph Neural Network for TEP fault classification.

    Architecture:
        GCNConv(in → hidden) → ReLU → GCNConv(hidden → hidden) → ReLU
        → global_mean_pool → Linear(hidden → num_classes)
    """

    def __init__(self, in_channels: int, num_classes: int, hidden: int = 64) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.classifier = nn.Linear(hidden, num_classes)
        self.relu = nn.ReLU()

    def forward(self, batch: Data) -> torch.Tensor:
        x, edge_index = batch.x, batch.edge_index
        batch_index = batch.batch if hasattr(batch, "batch") and batch.batch is not None else torch.zeros(x.size(0), dtype=torch.long)

        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch_index)  # (num_graphs, hidden)
        return self.classifier(x)
```

Note: the imports `from torch import nn` and `from torch_geometric.nn import GCNConv, global_mean_pool` should go at the top of the file with the other imports.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gnn.py -v`
Expected: PASS — all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/models/GNN.py tests/test_gnn.py
git commit -m "feat(gnn): implement TEPGraphNet model with GCNConv layers"
```

---

### Task 3: Implement Dataset Conversion (DataFrame → Graph List)

**Files:**
- Modify: `src/models/GNN.py`
- Modify: `tests/test_gnn.py`

This task adds a function to convert an entire DataFrame of TEP readings into a list of PyG Data objects, one per row/timestep.

- [ ] **Step 1: Write the failing test for dataset conversion**

Append to `tests/test_gnn.py`:

```python
def test_dataframe_to_graph_list():
    """Converting a DataFrame should produce one graph per row with correct labels."""
    from src.models.GNN import dataframe_to_graphs, UNIT_FEATURES

    feature_cols = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
    n_rows = 10
    df = pd.DataFrame(np.random.randn(n_rows, 52), columns=feature_cols)
    df["faultNumber"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    graphs = dataframe_to_graphs(df, target_column="faultNumber")

    assert len(graphs) == n_rows
    assert graphs[0].y.item() == 0
    assert graphs[5].y.item() == 5
    max_features = max(len(v) for v in UNIT_FEATURES.values())
    assert graphs[0].x.shape == (5, max_features)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gnn.py::test_dataframe_to_graph_list -v`
Expected: FAIL — `ImportError: cannot import name 'dataframe_to_graphs'`

- [ ] **Step 3: Implement dataframe_to_graphs**

Add to `src/models/GNN.py`:

```python
def dataframe_to_graphs(df: pd.DataFrame, target_column: str) -> list[Data]:
    """Convert a TEP DataFrame into a list of PyG Data objects."""
    encoder = LabelEncoder()
    labels = encoder.fit_transform(df[target_column])

    graphs = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        graph = build_tep_graph(row, label=int(labels[idx]))
        graphs.append(graph)

    return graphs, encoder
```

Update the test to unpack the tuple:

```python
def test_dataframe_to_graph_list():
    """Converting a DataFrame should produce one graph per row with correct labels."""
    from src.models.GNN import dataframe_to_graphs, UNIT_FEATURES

    feature_cols = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
    n_rows = 10
    df = pd.DataFrame(np.random.randn(n_rows, 52), columns=feature_cols)
    df["faultNumber"] = list(range(10))

    graphs, encoder = dataframe_to_graphs(df, target_column="faultNumber")

    assert len(graphs) == n_rows
    assert graphs[0].y.item() == encoder.transform([0])[0]
    assert graphs[5].y.item() == encoder.transform([5])[0]
    max_features = max(len(v) for v in UNIT_FEATURES.values())
    assert graphs[0].x.shape == (5, max_features)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gnn.py -v`
Expected: PASS — all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/models/GNN.py tests/test_gnn.py
git commit -m "feat(gnn): add dataframe-to-graph conversion for TEP data"
```

---

### Task 4: Implement Training Function and CLI

**Files:**
- Modify: `src/models/GNN.py`
- Modify: `tests/test_gnn.py`

This task adds the `train_gnn()` function and CLI argument parser, following the same pattern as `train_cnn.py`.

- [ ] **Step 1: Write the failing test for training output**

Append to `tests/test_gnn.py`:

```python
import json
import tempfile


def test_train_gnn_produces_metrics(tmp_path):
    """Training on small synthetic data should produce a metrics JSON file."""
    from src.models.GNN import train_gnn

    # Create small synthetic CSVs
    feature_cols = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
    n_rows = 50

    train_df = pd.DataFrame(np.random.randn(n_rows, 52), columns=feature_cols)
    train_df["faultNumber"] = np.random.randint(0, 3, n_rows)

    eval_df = pd.DataFrame(np.random.randn(20, 52), columns=feature_cols)
    eval_df["faultNumber"] = np.random.randint(0, 3, 20)

    train_path = tmp_path / "train.csv"
    eval_path = tmp_path / "eval.csv"
    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    output_dir = tmp_path / "results"
    train_gnn(
        data_path=str(train_path),
        target_column="faultNumber",
        output_dir=str(output_dir),
        eval_path=str(eval_path),
        epochs=2,
    )

    metrics_file = output_dir / "gnn_metrics.json"
    assert metrics_file.exists(), "Should produce gnn_metrics.json"

    with open(metrics_file) as f:
        metrics = json.load(f)

    assert "accuracy" in metrics
    assert "f1_weighted" in metrics
    assert "confusion_matrix" in metrics
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gnn.py::test_train_gnn_produces_metrics -v`
Expected: FAIL — `ImportError: cannot import name 'train_gnn'`

- [ ] **Step 3: Implement train_gnn and CLI**

Add to `src/models/GNN.py`:

```python
from torch_geometric.loader import DataLoader


def train_gnn(
    data_path: str,
    target_column: str,
    output_dir: str,
    eval_path: str | None = None,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden: int = 64,
) -> None:
    """Train the TEP GNN and save evaluation metrics."""
    # Load and convert training data
    train_df = pd.read_csv(data_path)
    train_graphs, encoder = dataframe_to_graphs(train_df, target_column)
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)

    # Determine dimensions
    in_channels = train_graphs[0].x.shape[1]
    num_classes = len(encoder.classes_)

    # Build model
    model = TEPGraphNet(in_channels=in_channels, num_classes=num_classes, hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        print(f"Epoch {epoch + 1}/{epochs}  loss={total_loss / len(train_graphs):.4f}")

    # Evaluate
    eval_df = pd.read_csv(eval_path) if eval_path else train_df
    eval_graphs, _ = dataframe_to_graphs(eval_df, target_column)
    eval_loader = DataLoader(eval_graphs, batch_size=batch_size, shuffle=False)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in eval_loader:
            logits = model(batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

    predicted_labels = encoder.inverse_transform(all_preds)
    eval_labels = eval_df[target_column]
    metrics = classification_report_dict(eval_labels, predicted_labels)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "gnn_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GNN on the TEP dataset.")
    parser.add_argument("--data", default=str(CONFIG.processed_data_dir / "tep_train.csv"), help="Training CSV path.")
    parser.add_argument(
        "--eval-data",
        default=str(CONFIG.processed_data_dir / "tep_test.csv"),
        help="Evaluation CSV path.",
    )
    parser.add_argument("--target", default=CONFIG.multiclass_target_column, help="Target column name.")
    parser.add_argument("--output-dir", default="results", help="Directory to save outputs.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden dimension size.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_gnn(
        args.data,
        args.target,
        args.output_dir,
        eval_path=args.eval_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gnn.py -v`
Expected: PASS — all 6 tests pass.

- [ ] **Step 5: Run the model on synthetic data end-to-end**

Run: `python -m pytest tests/test_gnn.py::test_train_gnn_produces_metrics -v -s`
Expected: See epoch loss output and PASS.

- [ ] **Step 6: Commit**

```bash
git add src/models/GNN.py tests/test_gnn.py
git commit -m "feat(gnn): add training function and CLI entry point"
```

---

### Task 5: Update Requirements and Run on Real Data

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add torch-geometric to requirements.txt**

Add the following lines to `requirements.txt`:

```
torch-geometric>=2.5
torch-scatter>=2.1
torch-sparse>=0.6
torch-cluster>=1.6
```

- [ ] **Step 2: Run the GNN on the actual TEP dataset**

Run:
```bash
python -m src.models.GNN --data data/processed/tep_train.csv --eval-data data/processed/tep_test.csv --target faultNumber --epochs 30
```
Expected: Epoch loss printed for 30 epochs, `results/gnn_metrics.json` created with accuracy, F1, confusion matrix.

- [ ] **Step 3: Verify metrics file was created and inspect results**

Run:
```bash
python -c "import json; m = json.load(open('results/gnn_metrics.json')); print(f'Accuracy: {m[\"accuracy\"]:.4f}, F1: {m[\"f1_weighted\"]:.4f}')"
```

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add torch-geometric dependencies to requirements.txt"
```
