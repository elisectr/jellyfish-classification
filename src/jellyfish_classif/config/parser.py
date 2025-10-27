import yaml
from pathlib import Path
from config.schema import DatasetConfig, ModelConfig, TrainingConfig, DownloadConfig


class Config:
    def __init__(self, path: str | Path = "config/config.yaml"):
        self.path = Path(path)
        self._data = self._load_yaml()

        self.dataset = DatasetConfig(**self._data.get("dataset", {}))
        self.model = ModelConfig(**self._data.get("model", {}))
        self.training = TrainingConfig(**self._data.get("training", {}))
        self.download = DownloadConfig(**self._data.get("download", {}))

    def _load_yaml(self) -> dict:
        if not self.path.exists():
            raise FileNotFoundError(f"Config file not found at {self.path}")
        with open(self.path, "r") as f:
            return yaml.safe_load(f) or {}

    def __repr__(self):
        return f"<Config dataset={self.dataset} model={self.model}>"
