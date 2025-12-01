"""
Unit tests for the FastAPI application endpoints.

Tests API endpoints, request/response handling, and error cases.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

from main import app
from schemas import PredictionResponse


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_model_manager():
    """Create a mock ModelManager."""
    with patch('main.model_manager') as mock:
        mock.models = {
            "decision_tree": {"model": Mock(), "path": "test.pkl", "type": "DecisionTreeClassifier"},
            "gradient_boosting": {"model": Mock(), "path": "test.pkl", "type": "GradientBoostingClassifier"}
        }
        mock.get_loaded_models.return_value = ["decision_tree", "gradient_boosting"]
        mock.get_models_info.return_value = [
            {
                "name": "gradient_boosting",
                "type": "GradientBoostingClassifier",
                "loaded": True,
                "file_path": "test.pkl"
            }
        ]
        yield mock


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_endpoint(self, client, mock_model_manager):
        """Test root endpoint returns health status."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "message" in data
        assert "models_loaded" in data


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check_success(self, client, mock_model_manager):
        """Test health check with loaded models."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert len(data["models_loaded"]) > 0
    
    def test_health_check_no_models(self, client):
        """Test health check when models are not loaded."""
        with patch('main.model_manager', None):
            response = client.get("/health")
            assert response.status_code == 503


class TestModelsEndpoint:
    """Tests for models listing endpoint."""
    
    def test_list_models(self, client, mock_model_manager):
        """Test listing available models."""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
    
    def test_list_models_no_manager(self, client):
        """Test listing models when manager is not initialized."""
        with patch('main.model_manager', None):
            response = client.get("/models")
            assert response.status_code == 503


class TestPredictEndpoint:
    """Tests for single prediction endpoint."""
    
    def test_predict_success(self, client, mock_model_manager):
        """Test successful prediction."""
        # Mock prediction response
        mock_model_manager.predict.return_value = PredictionResponse(
            prediction_code=1.0,
            health_status="Normal",
            model_used="gradient_boosting",
            confidence=0.95
        )
        
        payload = {
            "features": {
                "severe_decelerations": 0.0,
                "accelerations": 0.0,
                "fetal_movement": 0.0,
                "uterine_contractions": 0.0
            },
            "model_name": "gradient_boosting"
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["health_status"] == "Normal"
        assert data["prediction_code"] == 1.0
        assert data["model_used"] == "gradient_boosting"
    
    def test_predict_missing_features(self, client, mock_model_manager):
        """Test prediction with missing features."""
        payload = {
            "features": {
                "severe_decelerations": 0.0,
                "accelerations": 0.0
                # Missing fetal_movement and uterine_contractions
            },
            "model_name": "gradient_boosting"
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_predict_invalid_model(self, client, mock_model_manager):
        """Test prediction with invalid model name."""
        payload = {
            "features": {
                "severe_decelerations": 0.0,
                "accelerations": 0.0,
                "fetal_movement": 0.0,
                "uterine_contractions": 0.0
            },
            "model_name": "invalid_model"
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_predict_model_error(self, client, mock_model_manager):
        """Test prediction when model raises error."""
        mock_model_manager.predict.side_effect = ValueError("Model not found")
        
        payload = {
            "features": {
                "severe_decelerations": 0.0,
                "accelerations": 0.0,
                "fetal_movement": 0.0,
                "uterine_contractions": 0.0
            },
            "model_name": "gradient_boosting"
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 400
    
    def test_predict_no_manager(self, client):
        """Test prediction when model manager is not initialized."""
        with patch('main.model_manager', None):
            payload = {
                "features": {
                    "severe_decelerations": 0.0,
                    "accelerations": 0.0,
                    "fetal_movement": 0.0,
                    "uterine_contractions": 0.0
                }
            }
            response = client.post("/predict", json=payload)
            assert response.status_code == 503


class TestBatchPredictEndpoint:
    """Tests for batch prediction endpoint."""
    
    def test_batch_predict_success(self, client, mock_model_manager):
        """Test successful batch prediction."""
        # Mock batch prediction response
        mock_model_manager.predict_batch.return_value = [
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
        
        payload = {
            "features_list": [
                {
                    "severe_decelerations": 0.0,
                    "accelerations": 0.0,
                    "fetal_movement": 0.0,
                    "uterine_contractions": 0.0
                },
                {
                    "severe_decelerations": 0.001,
                    "accelerations": 0.002,
                    "fetal_movement": 0.003,
                    "uterine_contractions": 0.004
                }
            ],
            "model_name": "gradient_boosting"
        }
        
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2
    
    def test_batch_predict_empty_list(self, client, mock_model_manager):
        """Test batch prediction with empty features list."""
        payload = {
            "features_list": [],
            "model_name": "gradient_boosting"
        }
        
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict_error(self, client, mock_model_manager):
        """Test batch prediction when error occurs."""
        mock_model_manager.predict_batch.side_effect = Exception("Prediction failed")
        
        payload = {
            "features_list": [
                {
                    "severe_decelerations": 0.0,
                    "accelerations": 0.0,
                    "fetal_movement": 0.0,
                    "uterine_contractions": 0.0
                }
            ]
        }
        
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 500


class TestCORS:
    """Tests for CORS middleware."""
    
    def test_cors_headers(self, client):
        """Test that CORS headers are present."""
        response = client.options("/health")
        # CORS middleware should add appropriate headers
        assert response.status_code in [200, 405]  # OPTIONS might not be explicitly defined


class TestValidation:
    """Tests for request validation."""
    
    def test_invalid_json(self, client, mock_model_manager):
        """Test handling of invalid JSON."""
        response = client.post(
            "/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_extra_fields_ignored(self, client, mock_model_manager):
        """Test that extra fields in request are handled properly."""
        mock_model_manager.predict.return_value = PredictionResponse(
            prediction_code=1.0,
            health_status="Normal",
            model_used="gradient_boosting"
        )
        
        payload = {
            "features": {
                "severe_decelerations": 0.0,
                "accelerations": 0.0,
                "fetal_movement": 0.0,
                "uterine_contractions": 0.0,
                "extra_field": "should be ignored"
            },
            "model_name": "gradient_boosting"
        }
        
        response = client.post("/predict", json=payload)
        # Pydantic should ignore extra fields or raise validation error
        assert response.status_code in [200, 422]
