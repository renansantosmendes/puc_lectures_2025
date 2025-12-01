"""
Unit tests for the ModelManager class.

Tests model loading, preprocessing, and prediction functionality.
"""

import pytest
import os
import pickle
import numpy as np
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

from models import ModelManager
from schemas import FetalHealthFeatures, PredictionResponse


class TestModelManager:
    """Tests for ModelManager class."""
    
    @pytest.fixture
    def model_manager(self):
        """Create a ModelManager instance for testing."""
        return ModelManager()
    
    def test_initialization(self, model_manager):
        """Test ModelManager initialization."""
        assert model_manager.models == {}
        assert model_manager.scaler is not None
        assert model_manager._scaler_fitted is False
    
    def test_health_mapping(self, model_manager):
        """Test health status mapping."""
        assert model_manager.HEALTH_MAPPING[1.0] == "Normal"
        assert model_manager.HEALTH_MAPPING[2.0] == "Suspect"
        assert model_manager.HEALTH_MAPPING[3.0] == "Pathological"
    
    def test_get_loaded_models_empty(self, model_manager):
        """Test getting loaded models when none are loaded."""
        assert model_manager.get_loaded_models() == []
    
    def test_get_loaded_models_with_models(self, model_manager):
        """Test getting loaded models when models are loaded."""
        # Mock loaded models
        model_manager.models = {
            "decision_tree": {"model": Mock(), "path": "path1", "type": "DecisionTreeClassifier"},
            "gradient_boosting": {"model": Mock(), "path": "path2", "type": "GradientBoostingClassifier"}
        }
        loaded = model_manager.get_loaded_models()
        assert len(loaded) == 2
        assert "decision_tree" in loaded
        assert "gradient_boosting" in loaded
    
    def test_get_models_info(self, model_manager):
        """Test getting models information."""
        # Mock loaded models
        model_manager.models = {
            "gradient_boosting": {
                "model": Mock(),
                "path": "test/path.pkl",
                "type": "GradientBoostingClassifier"
            }
        }
        info = model_manager.get_models_info()
        assert len(info) == 1
        assert info[0]["name"] == "gradient_boosting"
        assert info[0]["type"] == "GradientBoostingClassifier"
        assert info[0]["loaded"] is True
    
    def test_interpret_prediction(self, model_manager):
        """Test prediction interpretation."""
        assert model_manager._interpret_prediction(1.0) == "Normal"
        assert model_manager._interpret_prediction(2.0) == "Suspect"
        assert model_manager._interpret_prediction(3.0) == "Pathological"
        assert model_manager._interpret_prediction(99.0) == "Unknown"
    
    def test_preprocess_features(self, model_manager):
        """Test feature preprocessing."""
        features = [0.0, 0.0, 0.0, 0.0]
        preprocessed = model_manager._preprocess_features(features)
        
        assert isinstance(preprocessed, np.ndarray)
        assert preprocessed.shape == (1, 4)
        assert model_manager._scaler_fitted is True
    
    def test_preprocess_features_shape(self, model_manager):
        """Test that preprocessing maintains correct shape."""
        features = [0.001, 0.002, 0.003, 0.004]
        preprocessed = model_manager._preprocess_features(features)
        
        assert preprocessed.shape[0] == 1  # Single sample
        assert preprocessed.shape[1] == 4  # Four features
    
    def test_predict_model_not_found(self, model_manager):
        """Test prediction with non-existent model."""
        features = FetalHealthFeatures(
            severe_decelerations=0.0,
            accelerations=0.0,
            fetal_movement=0.0,
            uterine_contractions=0.0
        )
        
        with pytest.raises(ValueError, match="Model .* not found"):
            model_manager.predict(features, model_name="non_existent")
    
    def test_predict_success(self, model_manager):
        """Test successful prediction."""
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1.0])
        mock_model.predict_proba.return_value = np.array([[0.1, 0.85, 0.05]])
        
        model_manager.models = {
            "test_model": {
                "model": mock_model,
                "path": "test.pkl",
                "type": "TestClassifier"
            }
        }
        
        features = FetalHealthFeatures(
            severe_decelerations=0.0,
            accelerations=0.0,
            fetal_movement=0.0,
            uterine_contractions=0.0
        )
        
        result = model_manager.predict(features, model_name="test_model")
        
        assert isinstance(result, PredictionResponse)
        assert result.prediction_code == 1.0
        assert result.health_status == "Normal"
        assert result.model_used == "test_model"
        assert result.confidence == 0.85
    
    def test_predict_without_confidence(self, model_manager):
        """Test prediction with model that doesn't support predict_proba."""
        # Mock model without predict_proba
        mock_model = Mock()
        mock_model.predict.return_value = np.array([2.0])
        delattr(mock_model, 'predict_proba')  # Remove predict_proba
        
        model_manager.models = {
            "test_model": {
                "model": mock_model,
                "path": "test.pkl",
                "type": "TestClassifier"
            }
        }
        
        features = FetalHealthFeatures(
            severe_decelerations=0.001,
            accelerations=0.002,
            fetal_movement=0.003,
            uterine_contractions=0.004
        )
        
        result = model_manager.predict(features, model_name="test_model")
        
        assert result.prediction_code == 2.0
        assert result.health_status == "Suspect"
        assert result.confidence is None
    
    def test_predict_batch(self, model_manager):
        """Test batch predictions."""
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1.0])
        mock_model.predict_proba.return_value = np.array([[0.9, 0.05, 0.05]])
        
        model_manager.models = {
            "test_model": {
                "model": mock_model,
                "path": "test.pkl",
                "type": "TestClassifier"
            }
        }
        
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
        
        results = model_manager.predict_batch(features_list, model_name="test_model")
        
        assert len(results) == 2
        assert all(isinstance(r, PredictionResponse) for r in results)
    
    @patch('os.path.exists')
    @patch('builtins.open', create=True)
    @patch('pickle.load')
    def test_load_models_success(self, mock_pickle_load, mock_open, mock_exists, model_manager):
        """Test successful model loading."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock pickle load
        mock_model = Mock()
        mock_model.__class__.__name__ = "GradientBoostingClassifier"
        mock_pickle_load.return_value = mock_model
        
        # Load models
        model_manager.load_models()
        
        # Verify models were loaded
        assert len(model_manager.models) > 0
    
    @patch('os.path.exists')
    def test_load_models_file_not_found(self, mock_exists, model_manager, capsys):
        """Test model loading when files don't exist."""
        # Mock file doesn't exist
        mock_exists.return_value = False
        
        # Load models (should print warning but not raise)
        model_manager.load_models()
        
        # Check warning was printed
        captured = capsys.readouterr()
        assert "Warning" in captured.out or len(model_manager.models) == 0


class TestModelManagerIntegration:
    """Integration tests for ModelManager with real models if available."""
    
    def test_model_files_exist(self):
        """Test that model files exist in the expected location."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, "models")
        
        # Check if models directory exists
        if os.path.exists(models_dir):
            # Check for model files
            dt_path = os.path.join(models_dir, "decision_tree_model.pkl")
            gb_path = os.path.join(models_dir, "gradient_boosting_model.pkl")
            
            # At least one model should exist for integration tests
            assert os.path.exists(dt_path) or os.path.exists(gb_path), \
                "No model files found for integration testing"
