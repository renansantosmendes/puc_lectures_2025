"""
Pydantic schemas for API request and response validation.
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import date


class SequenceData(BaseModel):
    """Input sequence data for prediction."""
    values: List[List[float]] = Field(
        ...,
        description="2D array of input values with shape (sequence_length, n_features). "
                    "Each inner list should have 3 values: [Return, LogVolume, HighLowSpread]",
        example=[
            [0.01, 17.5, 0.02],
            [0.02, 17.8, 0.03],
            [-0.01, 17.2, 0.015]
        ]
    )


class PredictionRequest(BaseModel):
    """Request body for single prediction."""
    sequence: SequenceData = Field(..., description="Input sequence for prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sequence": {
                    "values": [
                        [0.01, 17.5, 0.02] for _ in range(30)
                    ]
                }
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request body for batch prediction."""
    sequences: List[SequenceData] = Field(
        ..., 
        description="List of input sequences for batch prediction"
    )


class PredictionResponse(BaseModel):
    """Response for single prediction."""
    prediction: List[float] = Field(
        ..., 
        description="Predicted values for the next timestep [Return, LogVolume, HighLowSpread]"
    )
    reconstruction_error: float = Field(
        ..., 
        description="Mean squared error of the prediction (can be used for anomaly detection)"
    )
    is_anomaly: bool = Field(
        ..., 
        description="Whether this prediction is flagged as an anomaly based on the threshold"
    )
    anomaly_threshold: float = Field(
        ..., 
        description="Current threshold used for anomaly detection"
    )


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""
    predictions: List[PredictionResponse] = Field(
        ..., 
        description="List of predictions for each input sequence"
    )
    total_anomalies: int = Field(
        ..., 
        description="Total number of anomalies detected in the batch"
    )


class OutlierDetectionRequest(BaseModel):
    """Request for detecting outliers in stock data."""
    ticker: str = Field(
        default="AAPL", 
        description="Stock ticker symbol"
    )
    start_date: date = Field(
        default=date(2020, 1, 1), 
        description="Start date for data retrieval"
    )
    end_date: date = Field(
        default=date(2025, 12, 1), 
        description="End date for data retrieval"
    )
    sigma_multiplier: float = Field(
        default=3.0, 
        description="Number of standard deviations for anomaly threshold"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "start_date": "2020-01-01",
                "end_date": "2025-12-01",
                "sigma_multiplier": 3.0
            }
        }


class OutlierInfo(BaseModel):
    """Information about a detected outlier."""
    index: int = Field(..., description="Index of the outlier in the dataset")
    date: str = Field(..., description="Date of the outlier occurrence")
    close_price: float = Field(..., description="Closing price on the outlier date")
    mse: float = Field(..., description="Mean squared reconstruction error")


class OutlierDetectionResponse(BaseModel):
    """Response for outlier detection analysis."""
    ticker: str = Field(..., description="Analyzed stock ticker")
    total_sequences: int = Field(..., description="Total number of sequences analyzed")
    outliers_detected: int = Field(..., description="Number of outliers detected")
    outlier_percentage: float = Field(..., description="Percentage of outliers")
    threshold: float = Field(..., description="Calculated anomaly threshold")
    error_statistics: dict = Field(..., description="Statistical summary of reconstruction errors")
    outliers: List[OutlierInfo] = Field(..., description="List of detected outliers")


class HealthCheckResponse(BaseModel):
    """Response for health check endpoint."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    scaler_loaded: bool = Field(..., description="Whether the scaler is loaded")
    device: str = Field(..., description="Compute device being used")


class ModelInfoResponse(BaseModel):
    """Response for model information endpoint."""
    model_type: str = Field(..., description="Type of the model")
    sequence_length: int = Field(..., description="Expected input sequence length")
    input_dimension: int = Field(..., description="Number of input features")
    hidden_dimension: int = Field(..., description="LSTM hidden state size")
    latent_dimension: int = Field(..., description="Latent representation size")
    device: str = Field(..., description="Compute device")
    feature_names: List[str] = Field(..., description="Names of input features")
