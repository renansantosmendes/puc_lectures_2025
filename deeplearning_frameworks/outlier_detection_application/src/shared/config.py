"""
Configuration module for model hyperparameters and paths.
"""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Model hyperparameters configuration."""
    ticker: str = "AAPL"
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-01"
    sequence_length: int = 30
    batch_size: int = 64
    hidden_dim: int = 64
    latent_dim: int = 16
    number_of_layers: int = 2
    dropout_rate: float = 0.2
    learning_rate: float = 1e-3
    number_of_epochs: int = 100
    early_stopping_patience: int = 12
    train_validation_split: float = 0.8
    outlier_percentile: int = 95
    input_dimension: int = 3  # Number of features: Return, LogVolume, HighLowSpread


@dataclass
class PathConfig:
    """Paths configuration for artifacts."""
    base_dir: Path = Path(__file__).parent.parent.parent
    artifacts_dir: Path = base_dir / "artifacts"
    model_path: Path = artifacts_dir / "forecast_lstm_ae.pth"
    scaler_path: Path = artifacts_dir / "scaler.gz"
    config_path: Path = artifacts_dir / "model_config.json"
    
    def __post_init__(self):
        """Ensure artifacts directory exists."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


# Global instances
model_config = ModelConfig()
path_config = PathConfig()
