"""
FastAPI application for Fetal Health Classification.

This application provides REST API endpoints for making predictions using
trained machine learning models for fetal health classification.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from models import ModelManager
from schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)


# Global model manager instance
model_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Loads models on startup and cleans up on shutdown.
    """
    global model_manager
    
    # Startup: Load models
    print("Loading machine learning models...")
    model_manager = ModelManager()
    model_manager.load_models()
    print("Models loaded successfully!")
    
    yield
    
    # Shutdown: Cleanup
    print("Shutting down application...")


# Initialize FastAPI application
app = FastAPI(
    title="Fetal Health Classification API",
    description="""
    REST API for predicting fetal health status using machine learning models.
    
    This API provides endpoints to:
    - Make single predictions
    - Make batch predictions
    - Get information about available models
    - Check API health status
    
    The models classify fetal health into three categories:
    - **Normal**: Healthy fetal condition
    - **Suspect**: Requires attention
    - **Pathological**: Requires immediate medical intervention
    """,
    version="1.0.0",
    lifespan=lifespan,
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    response_model=HealthResponse,
    tags=["Health"],
    summary="API Health Check",
    description="Check if the API is running and models are loaded."
)
async def root():
    """
    Root endpoint for API health check.
    
    Returns:
        HealthResponse: API status and available models
    """
    return HealthResponse(
        status="healthy",
        message="Fetal Health Classification API is running",
        models_loaded=model_manager.get_loaded_models() if model_manager else []
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Detailed Health Check",
    description="Get detailed health status of the API and loaded models."
)
async def health_check():
    """
    Detailed health check endpoint.
    
    Returns:
        HealthResponse: Detailed API status
    """
    if not model_manager or not model_manager.models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded"
        )
    
    return HealthResponse(
        status="healthy",
        message="All systems operational",
        models_loaded=model_manager.get_loaded_models()
    )


@app.get(
    "/models",
    response_model=list[ModelInfoResponse],
    tags=["Models"],
    summary="List Available Models",
    description="Get information about all available models."
)
async def list_models():
    """
    List all available models with their information.
    
    Returns:
        list[ModelInfoResponse]: List of available models
    """
    if not model_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not initialized"
        )
    
    return model_manager.get_models_info()


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Make Single Prediction",
    description="""
    Make a prediction for a single fetal health record.
    
    Provide the fetal health features and optionally specify which model to use.
    If no model is specified, the default model (Gradient Boosting) will be used.
    """
)
async def predict(request: PredictionRequest):
    """
    Make a single prediction.
    
    Args:
        request: Prediction request with features and optional model name
        
    Returns:
        PredictionResponse: Prediction result with health status
        
    Raises:
        HTTPException: If model not found or prediction fails
    """
    if not model_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not initialized"
        )
    
    try:
        result = model_manager.predict(
            features=request.features,
            model_name=request.model_name
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Predictions"],
    summary="Make Batch Predictions",
    description="""
    Make predictions for multiple fetal health records at once.
    
    Provide a list of feature sets and optionally specify which model to use.
    This endpoint is more efficient for processing multiple records.
    """
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions.
    
    Args:
        request: Batch prediction request with multiple feature sets
        
    Returns:
        BatchPredictionResponse: List of prediction results
        
    Raises:
        HTTPException: If model not found or prediction fails
    """
    if not model_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not initialized"
        )
    
    try:
        results = model_manager.predict_batch(
            features_list=request.features_list,
            model_name=request.model_name
        )
        return BatchPredictionResponse(predictions=results)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
