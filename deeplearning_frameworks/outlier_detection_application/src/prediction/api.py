"""
Prediction API service using FastAPI.

This module provides a RESTful API for:
- Making predictions using the trained LSTM Autoencoder
- Detecting outliers in stock price data
- Health checks and model information
"""
import os
import sys
import json
from typing import List, Optional
from contextlib import asynccontextmanager

import numpy as np
import torch
import joblib
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.model import ForecastLSTMAutoencoder
from shared.config import PathConfig
from shared.data_processing import (
    download_stock_data, 
    engineer_features, 
    create_sequences_and_targets
)
from prediction.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    OutlierDetectionRequest,
    OutlierDetectionResponse,
    OutlierInfo,
    HealthCheckResponse,
    ModelInfoResponse
)


# Global state for model and scaler
class ModelManager:
    """Manages the loaded model and scaler state."""
    
    def __init__(self):
        self.model: Optional[ForecastLSTMAutoencoder] = None
        self.scaler = None
        self.device: torch.device = torch.device("cpu")
        self.config: dict = {}
        self.anomaly_threshold: float = 0.05  # Default threshold
        
    def load_artifacts(self, paths: PathConfig) -> bool:
        """
        Load model and scaler from disk.
        
        Args:
            paths: Path configuration.
            
        Returns:
            True if loading was successful, False otherwise.
        """
        try:
            # Load configuration
            if paths.config_path.exists():
                with open(paths.config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                # Use defaults
                self.config = {
                    "sequence_length": 30,
                    "input_dimension": 3,
                    "hidden_dim": 64,
                    "latent_dim": 16,
                    "number_of_layers": 2,
                    "dropout_rate": 0.2
                }
            
            # Setup device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Initialize model
            self.model = ForecastLSTMAutoencoder(
                sequence_length=self.config.get("sequence_length", 30),
                input_dimension=self.config.get("input_dimension", 3),
                hidden_dimension=self.config.get("hidden_dim", 64),
                latent_dimension=self.config.get("latent_dim", 16),
                number_of_layers=self.config.get("number_of_layers", 2),
                dropout_rate=self.config.get("dropout_rate", 0.2)
            )
            
            # Load weights if available
            if paths.model_path.exists():
                self.model.load_state_dict(
                    torch.load(paths.model_path, map_location=self.device)
                )
                self.model.to(self.device)
                self.model.eval()
                print(f"Model loaded from {paths.model_path}")
            else:
                print(f"Warning: Model file not found at {paths.model_path}")
                return False
            
            # Load scaler
            if paths.scaler_path.exists():
                self.scaler = joblib.load(paths.scaler_path)
                print(f"Scaler loaded from {paths.scaler_path}")
            else:
                print(f"Warning: Scaler file not found at {paths.scaler_path}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            return False


# Global model manager
model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI application."""
    # Startup: Load model artifacts
    paths = PathConfig()
    success = model_manager.load_artifacts(paths)
    if success:
        print("Model artifacts loaded successfully!")
    else:
        print("Warning: Failed to load model artifacts. Some endpoints may not work.")
    
    yield
    
    # Shutdown: Cleanup
    print("Shutting down prediction service...")


# FastAPI application
app = FastAPI(
    title="Outlier Detection API",
    description="""
## LSTM Autoencoder for Stock Price Anomaly Detection

This API provides endpoints for detecting anomalies in stock price time series data
using a trained LSTM Autoencoder with Multi-Head Attention.

### Features:
- **Single Prediction**: Make a prediction for a single input sequence
- **Batch Prediction**: Make predictions for multiple sequences at once
- **Outlier Detection**: Analyze stock data and detect anomalies
- **Health Check**: Verify service status and model availability

### Model Architecture:
The model uses an LSTM encoder with multi-head attention mechanism to learn
normal patterns in stock price movements. Anomalies are detected based on
reconstruction error exceeding a threshold calculated using log-normal distribution.

### Input Features:
1. **Return**: Percentage change in closing price
2. **LogVolume**: Log-transformed trading volume
3. **HighLowSpread**: Normalized high-low price spread
    """,
    version="1.0.0",
    contact={
        "name": "Renan Santos Mendes",
        "email": "renansantosmendes@gmail.com"
    },
    license_info={
        "name": "MIT License"
    },
    lifespan=lifespan
)


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["System"],
    summary="Health Check",
    description="Check the health status of the prediction service."
)
async def health_check() -> HealthCheckResponse:
    """
    Perform a health check on the prediction service.
    
    Returns information about:
    - Service status
    - Model availability
    - Scaler availability
    - Compute device being used
    """
    return HealthCheckResponse(
        status="healthy" if model_manager.model is not None else "degraded",
        model_loaded=model_manager.model is not None,
        scaler_loaded=model_manager.scaler is not None,
        device=str(model_manager.device)
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Model Information",
    description="Get information about the loaded model architecture and configuration."
)
async def get_model_info() -> ModelInfoResponse:
    """
    Get detailed information about the loaded model.
    
    Returns model architecture details including:
    - Model type
    - Sequence length
    - Input/hidden/latent dimensions
    - Feature names
    """
    if model_manager.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please ensure model artifacts are available."
        )
    
    return ModelInfoResponse(
        model_type="ForecastLSTMAutoencoder",
        sequence_length=model_manager.config.get("sequence_length", 30),
        input_dimension=model_manager.config.get("input_dimension", 3),
        hidden_dimension=model_manager.config.get("hidden_dim", 64),
        latent_dimension=model_manager.config.get("latent_dim", 16),
        device=str(model_manager.device),
        feature_names=["Return", "LogVolume", "HighLowSpread"]
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Single Prediction",
    description="Make a prediction for a single input sequence and detect if it's an anomaly."
)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Make a prediction for a single input sequence.
    
    The input sequence should have shape (sequence_length, 3) where 3 represents
    the features: Return, LogVolume, HighLowSpread.
    
    Returns the predicted next timestep values and anomaly detection results.
    """
    if model_manager.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Convert input to tensor
        sequence = np.array(request.sequence.values)
        expected_length = model_manager.config.get("sequence_length", 30)
        
        if sequence.shape[0] != expected_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Sequence length must be {expected_length}, got {sequence.shape[0]}"
            )
        
        if sequence.shape[1] != 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Number of features must be 3, got {sequence.shape[1]}"
            )
        
        # Scale input if scaler is available
        if model_manager.scaler is not None:
            sequence = model_manager.scaler.transform(sequence)
        
        # Make prediction
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(model_manager.device)
        
        with torch.no_grad():
            prediction = model_manager.model(input_tensor)
        
        prediction_np = prediction.cpu().numpy()[0]
        
        # Calculate reconstruction error (using the last value as target)
        target = sequence[-1]
        mse = float(np.mean((target - prediction_np) ** 2))
        
        # Determine if anomaly
        is_anomaly = mse > model_manager.anomaly_threshold
        
        # Inverse transform if scaler available
        if model_manager.scaler is not None:
            prediction_np = model_manager.scaler.inverse_transform(
                prediction_np.reshape(1, -1)
            )[0]
        
        return PredictionResponse(
            prediction=prediction_np.tolist(),
            reconstruction_error=mse,
            is_anomaly=is_anomaly,
            anomaly_threshold=model_manager.anomaly_threshold
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Batch Prediction",
    description="Make predictions for multiple input sequences at once."
)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Make predictions for multiple input sequences in a single request.
    
    This is more efficient than making multiple single prediction requests.
    """
    if model_manager.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    predictions = []
    for sequence_data in request.sequences:
        single_request = PredictionRequest(sequence=sequence_data)
        pred = await predict(single_request)
        predictions.append(pred)
    
    total_anomalies = sum(1 for p in predictions if p.is_anomaly)
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_anomalies=total_anomalies
    )


@app.post(
    "/detect-outliers",
    response_model=OutlierDetectionResponse,
    tags=["Outlier Detection"],
    summary="Detect Outliers in Stock Data",
    description="Download stock data, analyze it, and detect anomalies using the trained model."
)
async def detect_outliers(request: OutlierDetectionRequest) -> OutlierDetectionResponse:
    """
    Detect outliers in stock price data.
    
    This endpoint:
    1. Downloads historical stock data for the specified ticker
    2. Engineers features (Return, LogVolume, HighLowSpread)
    3. Creates sequences and runs them through the model
    4. Calculates reconstruction errors and identifies outliers
    
    The anomaly threshold is calculated using lognormal distribution statistics.
    """
    if model_manager.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    if model_manager.scaler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scaler not loaded"
        )
    
    try:
        # Download and process data
        dataframe = download_stock_data(
            request.ticker,
            str(request.start_date),
            str(request.end_date)
        )
        dataframe = engineer_features(dataframe)
        
        # Extract features
        feature_columns = ["Return", "LogVolume", "HighLowSpread"]
        features = dataframe[feature_columns].values
        scaled_features = model_manager.scaler.transform(features)
        
        # Create sequences
        sequence_length = model_manager.config.get("sequence_length", 30)
        input_sequences, target_values = create_sequences_and_targets(
            scaled_features, sequence_length
        )
        
        # Compute reconstruction errors
        input_tensor = torch.FloatTensor(input_sequences).to(model_manager.device)
        
        with torch.no_grad():
            predictions_list = []
            batch_size = 1024
            for i in range(0, len(input_tensor), batch_size):
                batch = input_tensor[i:i + batch_size]
                predictions_list.append(model_manager.model(batch).cpu().numpy())
            predictions = np.vstack(predictions_list)
        
        reconstruction_errors = np.mean((target_values - predictions) ** 2, axis=1)
        
        # Calculate threshold using lognormal distribution
        log_errors = np.log1p(reconstruction_errors)
        mean_log = np.mean(log_errors)
        std_log = np.std(log_errors)
        threshold = float(np.expm1(mean_log + request.sigma_multiplier * std_log))
        
        # Detect outliers
        outlier_mask = reconstruction_errors > threshold
        outlier_indices = np.where(outlier_mask)[0]
        
        # Get dates and prices
        dates = dataframe.index.values
        prices = features
        error_dates = dates[sequence_length:sequence_length + len(reconstruction_errors)]
        next_step_prices = prices[sequence_length:sequence_length + len(reconstruction_errors), 0]
        
        # Create outlier info list
        outliers = []
        for idx in outlier_indices[:20]:  # Limit to 20 outliers
            outliers.append(OutlierInfo(
                index=int(idx),
                date=str(error_dates[idx])[:10],
                close_price=float(next_step_prices[idx]),
                mse=float(reconstruction_errors[idx])
            ))
        
        # Update the model manager's threshold
        model_manager.anomaly_threshold = threshold
        
        return OutlierDetectionResponse(
            ticker=request.ticker,
            total_sequences=len(reconstruction_errors),
            outliers_detected=int(len(outlier_indices)),
            outlier_percentage=float(len(outlier_indices) / len(reconstruction_errors) * 100),
            threshold=threshold,
            error_statistics={
                "mean": float(reconstruction_errors.mean()),
                "median": float(np.median(reconstruction_errors)),
                "max": float(reconstruction_errors.max()),
                "min": float(reconstruction_errors.min()),
                "std": float(reconstruction_errors.std())
            },
            outliers=outliers
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Outlier detection error: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
