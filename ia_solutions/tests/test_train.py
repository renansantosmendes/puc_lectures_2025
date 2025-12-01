"""
Unit tests for the training pipeline.

Tests data loading, preprocessing, model training, and saving functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock, mock_open
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))

from train import (
    load_fetal_health_data,
    prepare_features_and_target,
    scale_features,
    split_train_test_data,
    train_decision_tree,
    train_gradient_boosting,
    evaluate_model,
    save_model,
)


class TestLoadFetalHealthData:
    """Tests for data loading function."""
    
    @patch('train.pd.read_csv')
    def test_load_data_success(self, mock_read_csv):
        """Test successful data loading."""
        # Mock DataFrame
        mock_df = pd.DataFrame({
            'severe_decelerations': [0.0, 0.001],
            'accelerations': [0.0, 0.002],
            'fetal_movement': [0.0, 0.003],
            'uterine_contractions': [0.0, 0.004],
            'fetal_health': [1.0, 2.0]
        })
        mock_read_csv.return_value = mock_df
        
        result = load_fetal_health_data("http://test.url", max_retries=1)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        mock_read_csv.assert_called_once()
    
    @patch('train.pd.read_csv')
    @patch('train.time.sleep')
    def test_load_data_retry(self, mock_sleep, mock_read_csv):
        """Test data loading with retry on failure."""
        from urllib.error import URLError
        
        # First call fails, second succeeds
        mock_df = pd.DataFrame({'col': [1, 2]})
        mock_read_csv.side_effect = [URLError("Connection failed"), mock_df]
        
        result = load_fetal_health_data("http://test.url", max_retries=2)
        
        assert isinstance(result, pd.DataFrame)
        assert mock_read_csv.call_count == 2
        mock_sleep.assert_called_once()
    
    @patch('train.pd.read_csv')
    def test_load_data_max_retries_exceeded(self, mock_read_csv):
        """Test data loading fails after max retries."""
        from urllib.error import URLError
        
        mock_read_csv.side_effect = URLError("Connection failed")
        
        with pytest.raises(Exception, match="Could not load data"):
            load_fetal_health_data("http://test.url", max_retries=2)


class TestPrepareFeatureAndTarget:
    """Tests for feature and target preparation."""
    
    def test_prepare_features_and_target(self):
        """Test separating features and target."""
        data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'fetal_health': [1.0, 2.0, 3.0]
        })
        
        features, target = prepare_features_and_target(data)
        
        assert isinstance(features, pd.DataFrame)
        assert isinstance(target, pd.Series)
        assert len(features.columns) == 2
        assert 'fetal_health' not in features.columns
        assert target.name == 'fetal_health'
        assert len(target) == 3


class TestScaleFeatures:
    """Tests for feature scaling."""
    
    def test_scale_features(self):
        """Test feature scaling."""
        features = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [10.0, 20.0, 30.0]
        })
        
        scaled = scale_features(features)
        
        assert isinstance(scaled, pd.DataFrame)
        assert scaled.shape == features.shape
        assert list(scaled.columns) == list(features.columns)
        # Scaled features should have mean close to 0 and std close to 1
        assert abs(scaled['feature1'].mean()) < 1e-10
        assert abs(scaled['feature1'].std() - 1.0) < 1e-10
    
    def test_scale_features_preserves_columns(self):
        """Test that scaling preserves column names."""
        features = pd.DataFrame({
            'col_a': [1, 2, 3],
            'col_b': [4, 5, 6]
        })
        
        scaled = scale_features(features)
        
        assert list(scaled.columns) == ['col_a', 'col_b']


class TestSplitTrainTestData:
    """Tests for train/test split."""
    
    def test_split_train_test_data(self):
        """Test splitting data into train and test sets."""
        features = pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100, 200)
        })
        target = pd.Series(range(100))
        
        X_train, X_test, y_train, y_test = split_train_test_data(
            features, target, test_size=0.3, random_state=42
        )
        
        assert len(X_train) == 70
        assert len(X_test) == 30
        assert len(y_train) == 70
        assert len(y_test) == 30
    
    def test_split_reproducibility(self):
        """Test that split is reproducible with same random state."""
        features = pd.DataFrame({'feature': range(100)})
        target = pd.Series(range(100))
        
        split1 = split_train_test_data(features, target, random_state=42)
        split2 = split_train_test_data(features, target, random_state=42)
        
        # Check that splits are identical
        pd.testing.assert_frame_equal(split1[0], split2[0])
        pd.testing.assert_frame_equal(split1[1], split2[1])


class TestTrainDecisionTree:
    """Tests for decision tree training."""
    
    def test_train_decision_tree(self):
        """Test training decision tree model."""
        X_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1]
        })
        y_train = pd.Series([1, 1, 2, 2, 3])
        
        model = train_decision_tree(X_train, y_train, max_depth=3, random_state=42)
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'fit')
    
    def test_decision_tree_parameters(self):
        """Test that decision tree uses correct parameters."""
        X_train = pd.DataFrame({'feature': [1, 2, 3]})
        y_train = pd.Series([1, 2, 3])
        
        model = train_decision_tree(X_train, y_train, max_depth=5, random_state=123)
        
        assert model.max_depth == 5
        assert model.random_state == 123


class TestTrainGradientBoosting:
    """Tests for gradient boosting training."""
    
    def test_train_gradient_boosting(self):
        """Test training gradient boosting model."""
        X_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1]
        })
        y_train = pd.Series([1, 1, 2, 2, 3])
        
        model = train_gradient_boosting(
            X_train, y_train,
            max_depth=3,
            n_estimators=10,
            learning_rate=0.1,
            random_state=42
        )
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'fit')
    
    def test_gradient_boosting_parameters(self):
        """Test that gradient boosting uses correct parameters."""
        X_train = pd.DataFrame({'feature': [1, 2, 3]})
        y_train = pd.Series([1, 2, 3])
        
        model = train_gradient_boosting(
            X_train, y_train,
            max_depth=5,
            n_estimators=50,
            learning_rate=0.05,
            random_state=123
        )
        
        assert model.max_depth == 5
        assert model.n_estimators == 50
        assert model.learning_rate == 0.05
        assert model.random_state == 123


class TestEvaluateModel:
    """Tests for model evaluation."""
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        # Create a simple mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1, 1, 2, 2, 3])
        
        X_test = pd.DataFrame({'feature': [1, 2, 3, 4, 5]})
        y_test = pd.Series([1, 1, 2, 2, 3])
        
        accuracy = evaluate_model(mock_model, X_test, y_test)
        
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
        assert accuracy == 1.0  # Perfect predictions
    
    def test_evaluate_model_imperfect(self):
        """Test evaluation with imperfect predictions."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1, 2, 2, 2, 3])
        
        X_test = pd.DataFrame({'feature': [1, 2, 3, 4, 5]})
        y_test = pd.Series([1, 1, 2, 2, 3])
        
        accuracy = evaluate_model(mock_model, X_test, y_test)
        
        assert accuracy == 0.8  # 4 out of 5 correct


class TestSaveModel:
    """Tests for model saving."""
    
    @patch('train.os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('train.pickle.dump')
    def test_save_model(self, mock_pickle_dump, mock_file, mock_makedirs):
        """Test saving model to disk."""
        mock_model = Mock()
        
        result = save_model(mock_model, "test_model", "test_dir")
        
        assert "test_model.pkl" in result
        mock_makedirs.assert_called_once_with("test_dir", exist_ok=True)
        mock_file.assert_called_once()
        mock_pickle_dump.assert_called_once()
    
    @patch('train.os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('train.pickle.dump')
    def test_save_model_default_dir(self, mock_pickle_dump, mock_file, mock_makedirs):
        """Test saving model with default directory."""
        mock_model = Mock()
        
        result = save_model(mock_model, "test_model")
        
        # Should use default MODELS_DIR
        assert "test_model.pkl" in result
        mock_makedirs.assert_called_once()


class TestIntegration:
    """Integration tests for the training pipeline."""
    
    @patch('train.pd.read_csv')
    def test_full_pipeline(self, mock_read_csv):
        """Test the full training pipeline."""
        # Mock data
        mock_df = pd.DataFrame({
            'severe_decelerations': [0.0, 0.001, 0.002, 0.0, 0.001] * 20,
            'accelerations': [0.0, 0.002, 0.003, 0.001, 0.0] * 20,
            'fetal_movement': [0.0, 0.003, 0.001, 0.002, 0.0] * 20,
            'uterine_contractions': [0.0, 0.004, 0.002, 0.003, 0.001] * 20,
            'fetal_health': [1.0, 2.0, 3.0, 1.0, 2.0] * 20
        })
        mock_read_csv.return_value = mock_df
        
        # Load data
        data = load_fetal_health_data("http://test.url", max_retries=1)
        
        # Prepare features and target
        features, target = prepare_features_and_target(data)
        
        # Scale features
        scaled_features = scale_features(features)
        
        # Split data
        X_train, X_test, y_train, y_test = split_train_test_data(
            scaled_features, target, test_size=0.3, random_state=42
        )
        
        # Train models
        dt_model = train_decision_tree(X_train, y_train)
        gb_model = train_gradient_boosting(X_train, y_train)
        
        # Evaluate models
        dt_accuracy = evaluate_model(dt_model, X_test, y_test)
        gb_accuracy = evaluate_model(gb_model, X_test, y_test)
        
        # Verify results
        assert 0.0 <= dt_accuracy <= 1.0
        assert 0.0 <= gb_accuracy <= 1.0
