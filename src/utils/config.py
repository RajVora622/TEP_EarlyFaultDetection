from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProjectConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    raw_data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    metadata_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    run_column: str = "run_id"
    time_column: str = "sample"
    multiclass_target_column: str = "faultNumber"
    binary_target_column: str = "is_faulty"
    split_column: str = "dataset_split"
    training_label: str = "train"
    testing_label: str = "test"

    def __post_init__(self) -> None:
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.processed_data_dir = self.project_root / "data" / "processed"
        self.metadata_dir = self.project_root / "data" / "metadata"
        self.results_dir = self.project_root / "results"


CONFIG = ProjectConfig()
