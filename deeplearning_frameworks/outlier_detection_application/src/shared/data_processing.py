"""
Data processing utilities for feature engineering and sequence creation.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple, List
from sklearn.preprocessing import MinMaxScaler


def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Download stock data from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol.
        start_date: Start date for data retrieval in YYYY-MM-DD format.
        end_date: End date for data retrieval in YYYY-MM-DD format.

    Returns:
        DataFrame containing the stock data.
    """
    print("Downloading data...")
    dataframe = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False
    )
    return dataframe


def engineer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features from raw stock data.

    Args:
        dataframe: Raw stock data DataFrame.

    Returns:
        DataFrame with additional engineered features.
    """
    dataframe["LogVolume"] = np.log1p(dataframe["Volume"].fillna(0))
    dataframe["Return"] = dataframe["Close"].pct_change().fillna(0)
    dataframe["HighLowSpread"] = (
        (dataframe["High"] - dataframe["Low"]) / dataframe["Close"]
    ).fillna(0)
    return dataframe


def create_sequences_and_targets(
    data: np.ndarray,
    sequence_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input sequences and corresponding target values.

    Args:
        data: Normalized feature array.
        sequence_length: Number of timesteps in each input sequence.

    Returns:
        Tuple containing input sequences (X) and target values (y).
    """
    input_sequences: List[np.ndarray] = []
    target_values: List[np.ndarray] = []

    for index in range(len(data) - sequence_length):
        input_sequences.append(data[index:index + sequence_length])
        target_values.append(data[index + sequence_length])

    return np.array(input_sequences), np.array(target_values)


def prepare_data_for_training(
    ticker: str,
    start_date: str,
    end_date: str,
    sequence_length: int,
    feature_columns: List[str] = None
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, pd.DataFrame]:
    """
    Full data preparation pipeline for training.

    Args:
        ticker: Stock ticker symbol.
        start_date: Start date for data retrieval.
        end_date: End date for data retrieval.
        sequence_length: Number of timesteps in each input sequence.
        feature_columns: List of feature column names to use.

    Returns:
        Tuple of (input_sequences, target_values, fitted_scaler, raw_dataframe).
    """
    if feature_columns is None:
        feature_columns = ["Return", "LogVolume", "HighLowSpread"]

    # Download and engineer features
    dataframe = download_stock_data(ticker, start_date, end_date)
    dataframe = engineer_features(dataframe)
    
    # Extract features and scale
    features = dataframe[feature_columns].values
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Create sequences
    input_sequences, target_values = create_sequences_and_targets(
        scaled_features, sequence_length
    )
    
    print(f"Total sequences created: {len(input_sequences)} | "
          f"Input shape: {input_sequences[0].shape} | "
          f"Target shape: {target_values.shape}")
    
    return input_sequences, target_values, scaler, dataframe
