"""
Pydantic schemas for request and response models.

Defines the data structures for API requests and responses.
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class FetalHealthFeatures(BaseModel):
    """
    Fetal health features for prediction.
    
    All features should be numerical values representing various
    fetal health indicators from the reduced dataset.
    """
    severe_decelerations: float = Field(
        ...,
        description="Number of severe decelerations per second",
        example=0.0
    )
    accelerations: float = Field(
        ...,
        description="Number of accelerations per second",
        example=0.0
    )
    fetal_movement: float = Field(
        ...,
        description="Number of fetal movements per second",
        example=0.0
    )
    uterine_contractions: float = Field(
        ...,
        description="Number of uterine contractions per second",
        example=0.0
    )

    def to_list(self) -> list[float]:
        """Convert features to a list in the correct order."""
        return [
            self.severe_decelerations,
            self.accelerations,
            self.fetal_movement,
            self.uterine_contractions,
        ]


class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    features: FetalHealthFeatures = Field(
        ...,
        description="Fetal health features for prediction"
    )
    model_name: Optional[str] = Field(
        default="gradient_boosting",
        description="Name of the model to use for prediction",
        example="gradient_boosting"
    )

    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v):
        """Validate model name."""
        if v is None:
            return "gradient_boosting"
        valid_models = ["decision_tree", "gradient_boosting"]
        if v not in valid_models:
            raise ValueError(f"Model must be one of {valid_models}")
        return v


class PredictionResponse(BaseModel):
    """Response model for single prediction."""
    prediction_code: float = Field(
        ...,
        description="Numerical prediction code (1.0, 2.0, or 3.0)",
        example=1.0
    )
    health_status: str = Field(
        ...,
        description="Human-readable health status",
        example="Normal"
    )
    model_used: str = Field(
        ...,
        description="Name of the model used for prediction",
        example="gradient_boosting"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Prediction confidence (if available)",
        example=0.95
    )


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    features_list: list[FetalHealthFeatures] = Field(
        ...,
        description="List of fetal health features for batch prediction",
        min_length=1
    )
    model_name: Optional[str] = Field(
        default="gradient_boosting",
        description="Name of the model to use for predictions",
        example="gradient_boosting"
    )

    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v):
        """Validate model name."""
        if v is None:
            return "gradient_boosting"
        valid_models = ["decision_tree", "gradient_boosting"]
        if v not in valid_models:
            raise ValueError(f"Model must be one of {valid_models}")
        return v


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: list[PredictionResponse] = Field(
        ...,
        description="List of prediction results"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoints."""
    status: str = Field(
        ...,
        description="API health status",
        example="healthy"
    )
    message: str = Field(
        ...,
        description="Status message",
        example="All systems operational"
    )
    models_loaded: list[str] = Field(
        ...,
        description="List of loaded model names",
        example=["decision_tree", "gradient_boosting"]
    )


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    name: str = Field(
        ...,
        description="Model name",
        example="gradient_boosting"
    )
    type: str = Field(
        ...,
        description="Model type/algorithm",
        example="GradientBoostingClassifier"
    )
    loaded: bool = Field(
        ...,
        description="Whether the model is loaded",
        example=True
    )
    file_path: str = Field(
        ...,
        description="Path to the model file",
        example="ia_solutions/models/gradient_boosting_model.pkl"
    )
