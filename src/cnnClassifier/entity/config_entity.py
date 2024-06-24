from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    
@dataclass(frozen=True)
class DatapreparationConfig:
    root_dir: Path
    source_path : Path
    train_path : Path
    val_path : Path
    split_ratio: float
    
@dataclass(frozen=True)
class ModelprepConfig:
    root_dir: Path
    model_path: Path
    params_image_size: list
    params_classes: int
    
@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    untrained_model_path: Path
    trained_model_path: Path
    training_directory: Path
    validation_directory: Path

@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    model_history_path : Path
    
