# TEP Fault Warning

Starter project structure for working with the Tennessee Eastman Process (TEP) dataset.

## Structure

```text
tep-fault-warning/
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
├── notebooks/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   └── utils/
├── results/
├── requirements.txt
└── README.md
```

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Put the original TEP files into `data/raw/`.
4. Build processed train and test files:

```bash
.venv/bin/python -m src.data.load_data
```

5. Optionally generate lag and rolling-window features:

```bash
.venv/bin/python -m src.data.make_features --input data/processed/tep_train.csv --output data/processed/tep_train_features.csv
```

6. Optionally generate exploratory summaries and plots:

```bash
.venv/bin/python -m src.evaluation.plots --input data/processed/tep_train.csv --output-dir results/eda
```

7. Run model scripts as modules from the project root, for example:

```bash
.venv/bin/python -m src.models.train_logreg --data data/processed/tep_train.csv --eval-data data/processed/tep_test.csv --target faultNumber
```

Binary SVM baseline:

```bash
.venv/bin/python -m src.models.train_svm --data data/processed/tep_train.csv --eval-data data/processed/tep_test.csv --target is_faulty --output-dir results/svm
```

Multiclass SVM baseline:

```bash
.venv/bin/python -m src.models.train_svm --data data/processed/tep_train.csv --eval-data data/processed/tep_test.csv --target faultNumber --output-dir results/svm_multiclass
```

Early-detection LSTM baseline:

```bash
.venv/bin/python -m src.models.train_lstm --data data/processed/tep_train.csv --eval-data data/processed/tep_test.csv --output-dir results/lstm
```

## Notes

- The raw TEP CSV files are expected to be named `TEP_FaultFree_Training.csv`, `TEP_Faulty_Training.csv`, `TEP_FaultFree_Testing.csv`, and `TEP_Faulty_Testing.csv`.
- The processing script creates `run_id`, `dataset_split`, and `is_faulty` columns and renames the 52 TEP process variables from raw `xmeas_*` / `xmv_*` names to readable physical labels.
- The loader validates that every `(faultNumber, simulationRun)` run has 500 samples in training files and 960 samples in testing files before saving the processed datasets.
- `data/metadata/feature_mapping.csv` records the raw-to-readable feature-name mapping, and `data/metadata/feature_columns.csv` lists the processed feature columns used by the models.
- Use `faultNumber` for multiclass fault classification or `is_faulty` for binary detection.
- `src.models.train_svm` supports both binary and multiclass targets. It now uses a run-aware train/validation split inside the training data so threshold tuning is done on held-out runs instead of shuffled rows.
- In binary mode, `src.models.train_svm` saves `svm_validation_threshold_metrics.csv`, which you can use to inspect the false-positive / false-negative tradeoff on held-out validation runs before applying the chosen threshold to the external evaluation set.
- `src.models.train_lstm` builds simple sequence windows, evaluates alarm thresholds, and reports event recall plus detection delay to support early-detection analysis.
- The LSTM script defaults to a common TEP early-detection assumption: faults begin around sample 20 in training runs and sample 160 in test runs. Adjust `--train-onset-sample` and `--eval-onset-sample` if your data variant uses different fault-onset timing.
- The EDA script writes summary CSVs plus distribution, missingness, correlation, and per-run trajectory plots under `results/eda/`.
