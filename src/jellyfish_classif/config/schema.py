from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class DatasetConfig:
    img_size: int = 224
    batch_size: int = 16
    seed: int = 42


@dataclass
class ModelConfig:
    num_classes: int = 5
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    dropout_rate: float = 0.3
    fine_tune_layers: int = 30
    optimizer: str = "adam"
    learning_rate: float = 0.001


@dataclass
class TrainingConfig:
    epochs: int = 100
    early_stopping_patience: int = 8
    checkpoint_monitor: str = "val_accuracy"
    checkpoint_mode: str = "max"
    save_best_only: bool = True
    verbose: int = 1
    output_dir: str = "outputs"


@dataclass
class DownloadConfig:
    max_images_per_species: int = 1000
    per_page: int = 50
    api_sleep_time: int = 2
    image_size: str = "medium"  # square, small, medium, large, original
    species: List[Dict[str, str | int]] | None = None
