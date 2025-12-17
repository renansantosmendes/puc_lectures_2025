"""
Training service for the LSTM Autoencoder outlier detection model.

This module handles the complete training pipeline including:
- Data preparation
- Model training with early stopping
- Saving model artifacts for the prediction service
"""
import os
import sys
import json
import random
import joblib
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.model import ForecastLSTMAutoencoder
from shared.config import ModelConfig, PathConfig
from shared.data_processing import prepare_data_for_training


class TimeSeriesForecastDataset(Dataset):
    """
    PyTorch Dataset for time series forecasting.

    Attributes:
        input_sequences: Tensor of input sequences with shape (n_samples, seq_length, n_features).
        target_values: Tensor of target values with shape (n_samples, n_features).
    """

    def __init__(self, input_sequences: np.ndarray, target_values: np.ndarray) -> None:
        """
        Initialize the dataset.

        Args:
            input_sequences: Array of input sequences.
            target_values: Array of target values.
        """
        self.input_sequences: torch.Tensor = torch.FloatTensor(input_sequences)
        self.target_values: torch.Tensor = torch.FloatTensor(target_values)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.input_sequences)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Tuple containing input sequence and target value tensors.
        """
        return self.input_sequences[index], self.target_values[index]


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    number_of_epochs: int,
    patience: int,
    training_device: torch.device,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    loss_criterion: nn.Module
) -> Tuple[List[float], List[float], int]:
    """
    Train the model with validation and early stopping.

    Args:
        model: The neural network model to train.
        train_loader: DataLoader for training data.
        validation_loader: DataLoader for validation data.
        number_of_epochs: Maximum number of training epochs.
        patience: Number of epochs to wait before early stopping.
        training_device: Device to train on (CPU or CUDA).
        optimizer: Optimizer for training.
        scheduler: Learning rate scheduler.
        loss_criterion: Loss function.

    Returns:
        Tuple containing training losses, validation losses, and best epoch number.
    """
    best_validation_loss = float("inf")
    best_epoch = 0
    best_model_state = None
    training_losses = []
    validation_losses = []
    patience_counter = 0

    print("Starting training...")

    for epoch in range(1, number_of_epochs + 1):
        model.train()
        epoch_training_loss = 0.0

        for input_batch, target_batch in train_loader:
            input_batch = input_batch.to(training_device)
            target_batch = target_batch.to(training_device)

            optimizer.zero_grad()
            predictions = model(input_batch)
            loss = loss_criterion(predictions, target_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_training_loss += loss.item()

        average_training_loss = epoch_training_loss / len(train_loader)
        training_losses.append(average_training_loss)

        model.eval()
        epoch_validation_loss = 0.0

        with torch.no_grad():
            for validation_input, validation_target in validation_loader:
                validation_input = validation_input.to(training_device)
                validation_target = validation_target.to(training_device)
                validation_predictions = model(validation_input)
                validation_loss = loss_criterion(validation_predictions, validation_target)
                epoch_validation_loss += validation_loss.item()

        average_validation_loss = epoch_validation_loss / len(validation_loader)
        validation_losses.append(average_validation_loss)

        current_learning_rate = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {average_training_loss:.6f} | "
            f"Val Loss: {average_validation_loss:.6f} | "
            f"LR: {current_learning_rate:.6e}"
        )

        if average_validation_loss < best_validation_loss - 1e-8:
            best_validation_loss = average_validation_loss
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best Val Loss: {best_validation_loss:.6f} (epoch {best_epoch})"
            )
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Model restored to best epoch {best_epoch} with Val Loss {best_validation_loss:.6f}")

    return training_losses, validation_losses, best_epoch


def save_model_artifacts(
    model: nn.Module,
    scaler,
    config: ModelConfig,
    paths: PathConfig
) -> None:
    """
    Save model weights, scaler, and configuration to disk.

    Args:
        model: Trained model.
        scaler: Fitted scaler.
        config: Model configuration.
        paths: Path configuration.
    """
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    torch.save(model.state_dict(), paths.model_path)
    print(f"Model saved to {paths.model_path}")
    
    # Save scaler
    joblib.dump(scaler, paths.scaler_path)
    print(f"Scaler saved to {paths.scaler_path}")
    
    # Save configuration as JSON
    config_dict = {
        "ticker": config.ticker,
        "sequence_length": config.sequence_length,
        "hidden_dim": config.hidden_dim,
        "latent_dim": config.latent_dim,
        "number_of_layers": config.number_of_layers,
        "dropout_rate": config.dropout_rate,
        "input_dimension": config.input_dimension
    }
    with open(paths.config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Configuration saved to {paths.config_path}")


def run_training(config: ModelConfig = None, paths: PathConfig = None) -> None:
    """
    Main training pipeline.

    Args:
        config: Model configuration. Uses default if None.
        paths: Path configuration. Uses default if None.
    """
    if config is None:
        config = ModelConfig()
    if paths is None:
        paths = PathConfig()
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    input_sequences, target_values, scaler, dataframe = prepare_data_for_training(
        ticker=config.ticker,
        start_date=config.start_date,
        end_date=config.end_date,
        sequence_length=config.sequence_length
    )
    
    # Create dataset and data loaders
    dataset = TimeSeriesForecastDataset(input_sequences, target_values)
    train_size = int(config.train_validation_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    print(f"Training: {len(train_dataset)} sequences | Validation: {len(val_dataset)} sequences")
    
    # Initialize model
    model = ForecastLSTMAutoencoder(
        sequence_length=config.sequence_length,
        input_dimension=config.input_dimension,
        hidden_dimension=config.hidden_dim,
        latent_dimension=config.latent_dim,
        number_of_layers=config.number_of_layers,
        dropout_rate=config.dropout_rate
    ).to(device)
    
    # Loss, optimizer, and scheduler
    loss_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.number_of_epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate * 10,
        total_steps=total_steps
    )
    
    # Train model
    training_losses, validation_losses, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=val_loader,
        number_of_epochs=config.number_of_epochs,
        patience=config.early_stopping_patience,
        training_device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_criterion=loss_criterion
    )
    
    # Save artifacts
    save_model_artifacts(model, scaler, config, paths)
    
    print("\nTraining completed successfully!")
    print(f"Best validation loss: {min(validation_losses):.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    run_training()
