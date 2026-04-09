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
python -m src.data.load_data
```

5. Optionally generate lag and rolling-window features:

```bash
python -m src.data.make_features --input data/processed/tep_train.csv --output data/processed/tep_train_features.csv
```

6. Optionally generate exploratory summaries and plots:

```bash
python -m src.evaluation.plots --input data/processed/tep_train.csv --output-dir results/eda
```

7. Run model scripts as modules from the project root, for example:

```bash
python -m src.models.train_logreg --data data/processed/tep_train.csv --eval-data data/processed/tep_test.csv --target faultNumber
```

## Notes

- The raw TEP CSV files are expected to be named `TEP_FaultFree_Training.csv`, `TEP_Faulty_Training.csv`, `TEP_FaultFree_Testing.csv`, and `TEP_Faulty_Testing.csv`.
- The processing script creates `run_id`, `dataset_split`, and `is_faulty` columns in addition to the original TEP columns.
- Model scripts train on the 52 process columns (`xmeas_*` and `xmv_*`) and ignore metadata columns automatically.
- Use `faultNumber` for multiclass fault classification or `is_faulty` for binary detection.
- The EDA script writes summary CSVs plus distribution, missingness, correlation, and per-run trajectory plots under `results/eda/`.
