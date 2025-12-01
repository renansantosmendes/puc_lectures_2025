"""
Unit tests for the API schemas module.

Tests Pydantic models for validation and serialization.
"""

import pytest
from pydantic import ValidationError

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

from schemas import (
    FetalHealthFeatures,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)


class TestFetalHealthFeatures:
    """Tests for FetalHealthFeatures schema."""
    
    def test_valid_features(self):
        """Test creating features with valid data."""
        features = FetalHealthFeatures(
            severe_decelerations=0.0,
            accelerations=0.0,
            fetal_movement=0.0,
            uterine_contractions=0.0
        )
        assert features.severe_decelerations == 0.0
        assert features.accelerations == 0.0
        assert features.fetal_movement == 0.0
        assert features.uterine_contractions == 0.0
    
    def test_to_list_conversion(self):
        """Test converting features to list."""
        features = FetalHealthFeatures(
            severe_decelerations=0.001,
            accelerations=0.002,
            fetal_movement=0.003,
            uterine_contractions=0.004
        )
        feature_list = features.to_list()
        assert feature_list == [0.001, 0.002, 0.003, 0.004]
        assert len(feature_list) == 4
    
    def test_missing_field(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError):
            FetalHealthFeatures(
                severe_decelerations=0.0,
                accelerations=0.0,
                fetal_movement=0.0
                # Missing uterine_contractions
            )
    
    def test_invalid_type(self):
        """Test that invalid types raise validation error."""
        with pytest.raises(ValidationError):
            FetalHealthFeatures(
                severe_decelerations="invalid",  # Should be float
                accelerations=0.0,
                fetal_movement=0.0,
                uterine_contractions=0.0
            )


class TestPredictionRequest:
    """Tests for PredictionRequest schema."""
    
    def test_valid_request(self):
        """Test creating a valid prediction request."""
        features = FetalHealthFeatures(
            severe_decelerations=0.0,
            accelerations=0.0,
            fetal_movement=0.0,
            uterine_contractions=0.0
        )
        request = PredictionRequest(
            features=features,
            model_name="gradient_boosting"
        )
        assert request.features == features
        assert request.model_name == "gradient_boosting"
    
    def test_default_model_name(self):
        """Test that model_name defaults to gradient_boosting."""
        features = FetalHealthFeatures(
            severe_decelerations=0.0,
            accelerations=0.0,
            fetal_movement=0.0,
            uterine_contractions=0.0
        )
        request = PredictionRequest(features=features)
        assert request.model_name == "gradient_boosting"
    
    def test_invalid_model_name(self):
        """Test that invalid model names raise validation error."""
        features = FetalHealthFeatures(
            severe_decelerations=0.0,
            accelerations=0.0,
            fetal_movement=0.0,
            uterine_contractions=0.0
        )
        with pytest.raises(ValidationError):
            PredictionRequest(
                features=features,
                model_name="invalid_model"
            )
    
    def test_valid_model_names(self):
        """Test that valid model names are accepted."""
        features = FetalHealthFeatures(
            severe_decelerations=0.0,
            accelerations=0.0,
            fetal_movement=0.0,
            uterine_contractions=0.0
        )
        for model_name in ["decision_tree", "gradient_boosting"]:
            request = PredictionRequest(
                features=features,
                model_name=model_name
            )
            assert request.model_name == model_name


class TestPredictionResponse:
    """Tests for PredictionResponse schema."""
    
    def test_valid_response(self):
        """Test creating a valid prediction response."""
        response = PredictionResponse(
            prediction_code=1.0,
            health_status="Normal",
            model_used="gradient_boosting",
            confidence=0.95
        )
        assert response.prediction_code == 1.0
        assert response.health_status == "Normal"
        assert response.model_used == "gradient_boosting"
        assert response.confidence == 0.95
    
    def test_optional_confidence(self):
        """Test that confidence is optional."""
        response = PredictionResponse(
            prediction_code=2.0,
            health_status="Suspect",
            model_used="decision_tree"
        )
        assert response.confidence is None


class TestBatchPredictionRequest:
    """Tests for BatchPredictionRequest schema."""
    
    def test_valid_batch_request(self):
        """Test creating a valid batch prediction request."""
        features_list = [
            FetalHealthFeatures(
                severe_decelerations=0.0,
                accelerations=0.0,
                fetal_movement=0.0,
                uterine_contractions=0.0
            ),
            FetalHealthFeatures(
                severe_decelerations=0.001,
                accelerations=0.002,
                fetal_movement=0.003,
                uterine_contractions=0.004
            )
        ]
        request = BatchPredictionRequest(
            features_list=features_list,
            model_name="gradient_boosting"
        )
        assert len(request.features_list) == 2
        assert request.model_name == "gradient_boosting"
    
    def test_empty_features_list(self):
        """Test that empty features list raises validation error."""
        with pytest.raises(ValidationError):
            BatchPredictionRequest(
                features_list=[],
                model_name="gradient_boosting"
            )


class TestBatchPredictionResponse:
    """Tests for BatchPredictionResponse schema."""
    
    def test_valid_batch_response(self):
        """Test creating a valid batch prediction response."""
        predictions = [
            PredictionResponse(
                prediction_code=1.0,
                health_status="Normal",
                model_used="gradient_boosting",
                confidence=0.95
            ),
            PredictionResponse(
                prediction_code=2.0,
                health_status="Suspect",
                model_used="gradient_boosting",
                confidence=0.87
            )
        ]
        response = BatchPredictionResponse(predictions=predictions)
        assert len(response.predictions) == 2


class TestHealthResponse:
    """Tests for HealthResponse schema."""
    
    def test_valid_health_response(self):
        """Test creating a valid health response."""
        response = HealthResponse(
            status="healthy",
            message="All systems operational",
            models_loaded=["decision_tree", "gradient_boosting"]
        )
        assert response.status == "healthy"
        assert response.message == "All systems operational"
        assert len(response.models_loaded) == 2


class TestModelInfoResponse:
    """Tests for ModelInfoResponse schema."""
    
    def test_valid_model_info(self):
        """Test creating a valid model info response."""
        response = ModelInfoResponse(
            name="gradient_boosting",
            type="GradientBoostingClassifier",
            loaded=True,
            file_path="ia_solutions/models/gradient_boosting_model.pkl"
        )
        assert response.name == "gradient_boosting"
        assert response.type == "GradientBoostingClassifier"
        assert response.loaded is True
        assert "gradient_boosting_model.pkl" in response.file_path
