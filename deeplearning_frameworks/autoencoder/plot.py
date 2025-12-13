import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Sequence
from numpy.typing import ArrayLike


class PlotOutlierAnalyzer:
    """
    Visualization class for time series outlier detection model results.

    This class generates various plots to analyze the performance of a
    prediction-based outlier detection model, including:
        - Price chart with detected outliers highlighted
        - Prediction error timeline with threshold
        - Error distribution histogram
        - Model training curves

    Attributes:
        dates: Array of dates corresponding to the price data.
        prices: Array of price values.
        errors: Array of prediction errors (MSE).
        outliers: Boolean mask indicating outlier positions.
        threshold: Error threshold used for outlier detection.
        train_losses: Training loss values per epoch.
        val_losses: Validation loss values per epoch.
        best_epoch: Epoch number with best validation performance.
        seq_length: Sequence length used by the model.
    """

    def __init__(
        self,
        dates: ArrayLike,
        prices: ArrayLike,
        errors: ArrayLike,
        outliers_mask: ArrayLike,
        threshold: float,
        train_losses: Sequence[float],
        val_losses: Sequence[float],
        best_epoch: int,
        seq_length: int
    ) -> None:
        """
        Initialize the PlotOutlierAnalyzer with model results.

        Args:
            dates: Sequence of dates corresponding to price observations.
            prices: Sequence of price values.
            errors: Sequence of prediction errors (MSE) for each timestep.
            outliers_mask: Boolean mask where True indicates an outlier.
            threshold: Error threshold value for outlier classification.
            train_losses: List of training loss values per epoch.
            val_losses: List of validation loss values per epoch.
            best_epoch: The epoch number with the best validation loss.
            seq_length: The sequence length used by the prediction model.
        """
        self.dates = np.array(dates)
        self.prices = np.array(prices).reshape(-1)
        self.errors = np.array(errors)
        self.outliers = np.array(outliers_mask)
        self.threshold = threshold
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.best_epoch = best_epoch
        self.seq_length = seq_length

        self.error_dates = self.dates[self.seq_length : self.seq_length + len(self.errors)]
        self.prices_next = self.prices[self.seq_length : self.seq_length + len(self.errors)]
        self.outlier_dates = self.error_dates[self.outliers]
        self.outlier_prices = self.prices_next[self.outliers]

    def plot_price_with_outliers(self, ticker: str = "Unknown") -> None:
        """
        Plot the price series with detected outliers highlighted.

        Creates a line plot of the full price series with red scatter points
        marking the positions where outliers were detected.

        Args:
            ticker: Stock ticker symbol to display in the plot title.
        """
        plt.figure(figsize=(14, 5))
        plt.plot(self.dates, self.prices, label="Price", linewidth=1)
        plt.scatter(
            self.outlier_dates,
            self.outlier_prices,
            color="red",
            s=40,
            label="Outliers",
            zorder=3
        )
        plt.title(f"Stock Price for {ticker} with Detected Outliers")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_error_timeline(self) -> None:
        """
        Plot the prediction error over time with the outlier threshold.

        Creates a line plot showing the MSE error at each timestep,
        with a horizontal line indicating the threshold and shaded
        regions highlighting where errors exceed the threshold.
        """
        plt.figure(figsize=(14, 5))
        plt.plot(self.error_dates, self.errors, label="Error (MSE)", linewidth=1)
        plt.axhline(
            self.threshold,
            color="red",
            linestyle="--",
            label=f"Threshold ({self.threshold:.6e})"
        )
        plt.fill_between(
            self.error_dates,
            self.threshold,
            self.errors,
            where=(self.errors > self.threshold),
            color="red",
            alpha=0.3,
            label="Outliers"
        )
        plt.title("Model Prediction Error (MSE)")
        plt.xlabel("Date")
        plt.ylabel("Error (MSE)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_error_histogram(self) -> None:
        """
        Plot the distribution of prediction errors as a histogram.

        Creates a histogram showing the frequency distribution of MSE
        errors, with a vertical line indicating the outlier threshold.
        """
        plt.figure(figsize=(14, 5))
        plt.hist(self.errors, bins=50, alpha=0.7, edgecolor="black")
        plt.axvline(
            self.threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Threshold"
        )
        plt.title("Prediction Error Distribution")
        plt.xlabel("Error (MSE)")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_training_curve(self) -> None:
        """
        Plot the model training and validation loss curves.

        Creates a line plot showing both training and validation loss
        over epochs, with a vertical line marking the best epoch
        (lowest validation loss).
        """
        plt.figure(figsize=(14, 5))
        plt.plot(self.train_losses, label="Train Loss", linewidth=2)
        plt.plot(self.val_losses, label="Val Loss", linewidth=2)
        plt.axvline(
            self.best_epoch - 1,
            color="green",
            linestyle="--",
            label=f"Best Epoch ({self.best_epoch})"
        )
        plt.title("Model Training Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_all(self, ticker: str = "UNKNOWN") -> None:
        """
        Generate all available visualization plots.

        Convenience method that calls all individual plotting methods
        in sequence to provide a complete analysis overview.

        Args:
            ticker: Stock ticker symbol to display in relevant plot titles.
        """
        self.plot_price_with_outliers(ticker)
        self.plot_error_timeline()
        self.plot_error_histogram()
        self.plot_training_curve()