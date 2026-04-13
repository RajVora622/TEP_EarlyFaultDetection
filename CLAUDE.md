# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tennessee Eastman Process (TEP) early fault detection system. ML pipeline for detecting and classifying faults in chemical plant simulation data using multiple models (Logistic Regression, Random Forest, XGBoost, 1D CNN).

## Commands

All scripts run as Python modules from the project root. No test suite exists; evaluation is via metrics JSONs and EDA plots.

```bash
# Install dependencies
pip install -r requirements.txt

# Data pipeline (run in order)
python -m src.data.load_data                          # raw CSVs → processed train/test
python -m src.data.make_features --input data/processed/tep_train.csv --output data/processed/tep_train_features.csv
python -m src.data.split_runs --data data/processed/tep_train.csv --output-dir data/processed

# EDA
python -m src.evaluation.plots --input data/processed/tep_train.csv --output-dir results/eda

# Train models (each outputs results/{model}_metrics.json)
python -m src.models.train_logreg --data data/processed/tep_train.csv --eval-data data/processed/tep_test.csv --target faultNumber
python -m src.models.train_rf     --data data/processed/tep_train.csv --eval-data data/processed/tep_test.csv
python -m src.models.train_xgb    --data data/processed/tep_train.csv --eval-data data/processed/tep_test.csv
python -m src.models.train_cnn    --data data/processed/tep_train.csv --eval-data data/processed/tep_test.csv --epochs 10
```

## Architecture

**Data flow:** `data/raw/` → `src/data/load_data.py` → `data/processed/` → feature engineering / splitting → model training → `results/`

- **`src/utils/config.py`** — Central `ProjectConfig` dataclass defining all paths and column names. Imported as `CONFIG` throughout.
- **`src/data/`** — Data loading (`load_data.py`), lag/rolling-window feature engineering (`make_features.py`), run-aware GroupShuffleSplit (`split_runs.py`).
- **`src/models/`** — One script per model. All accept `--data`, `--eval-data`, `--target`, `--output-dir` args. Features are auto-detected by `xmeas_*`/`xmv_*` column prefixes (52 columns).
- **`src/evaluation/`** — Standard classification metrics (`metrics.py`), time-series event metrics like detection delay and event recall (`event_metrics.py`), and EDA plotting (`plots.py`).

**Key design patterns:**
- `run_id` groups all samples from a single simulation run; `sample` is the time step within a run. Splits use GroupShuffleSplit on `run_id` to prevent data leakage.
- Two target modes: `faultNumber` (multiclass, 0-20) and `is_faulty` (binary).
- Raw TEP files are gitignored. Expected names: `TEP_FaultFree_Training.csv`, `TEP_Faulty_Training.csv`, `TEP_FaultFree_Testing.csv`, `TEP_Faulty_Testing.csv`.
