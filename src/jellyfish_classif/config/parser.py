from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from jellyfish_classif.config.schema import (
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
    DownloadConfig,
)


class Config:
    """Singleton configuration loader for the Jellyfish Classifier project.

    This class ensures that configuration data is loaded only once from the YAML file,
    and subsequent instantiations return the same instance.

    Attributes:
        path (Path): Path to the YAML configuration file.
        dataset (DatasetConfig): Dataset configuration section.
        model (ModelConfig): Model configuration section.
        training (TrainingConfig): Training configuration section.
        download (DownloadConfig): Download configuration section.
    """

    _instance: Optional["Config"] = None

    def __new__(cls, path: str | Path = "config.yaml") -> "Config":
        """Ensure only one instance of Config exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, path: str | Path = "config.yaml") -> None:
        """Initialize the Config singleton (executed only once)."""
        # Prevent reinitialization of the singleton
        if getattr(self, "_initialized", False):
            return

        self.path: Path = Path(path)
        self._data: Dict[str, Any] = self._load_yaml()

        self.dataset: DatasetConfig = DatasetConfig(**self._data.get("dataset", {}))
        self.model: ModelConfig = ModelConfig(**self._data.get("model", {}))
        self.training: TrainingConfig = TrainingConfig(**self._data.get("training", {}))
        self.download: DownloadConfig = DownloadConfig(**self._data.get("download", {}))

        self._initialized: bool = True

    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML configuration file content."""
        if not self.path.exists():
            raise FileNotFoundError(f"Config file not found at {self.path}")
        with open(self.path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def __repr__(self) -> str:
        """Return a string representation of the configuration object."""
        return f"<Config dataset={self.dataset} model={self.model}>"
