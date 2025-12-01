"""
Model manager for loading and managing ML models.

Handles model loading, preprocessing, and predictions.
"""

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from schemas import FetalHealthFeatures, PredictionResponse


class ModelManager:
    """
    Manager class for handling machine learning models.
    
    Loads models on initialization and provides methods for making predictions.
    """
    
    # Model configuration - use absolute path
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    MODEL_FILES = {
        "decision_tree": "decision_tree_model.pkl",
        "gradient_boosting": "gradient_boosting_model.pkl",
    }
    
    # Health status mapping
    HEALTH_MAPPING = {
        1.0: "Normal",
        2.0: "Suspect",
        3.0: "Pathological"
    }
    
    def __init__(self):
        """Initialize the model manager."""
        self.models = {}
        self.scaler = StandardScaler()
        self._scaler_fitted = False
    
    def load_models(self) -> None:
        """
        Load all available models from disk.
        
        Raises:
            FileNotFoundError: If model files are not found
            Exception: If model loading fails
        """
        for model_name, filename in self.MODEL_FILES.items():
            model_path = os.path.join(self.MODELS_DIR, filename)
            
            if not os.path.exists(model_path):
                print(f"Warning: Model file not found: {model_path}")
                continue
            
            try:
                with open(model_path, 'rb') as file:
                    model = pickle.load(file)
                self.models[model_name] = {
                    "model": model,
                    "path": model_path,
                    "type": type(model).__name__
                }
                print(f"Loaded model: {model_name} ({type(model).__name__})")
            except Exception as e:
                print(f"Error loading model {model_name}: {str(e)}")
                raise
    
    def get_loaded_models(self) -> list[str]:
        """
        Get list of loaded model names.
        
        Returns:
            list[str]: List of loaded model names
        """
        return list(self.models.keys())
    
    def get_models_info(self) -> list[dict]:
        """
        Get information about all models.
        
        Returns:
            list[dict]: List of model information dictionaries
        """
        models_info = []
        for name, info in self.models.items():
            models_info.append({
                "name": name,
                "type": info["type"],
                "loaded": True,
                "file_path": info["path"]
            })
        return models_info
    
    def _preprocess_features(self, features: list[float]) -> np.ndarray:
        """
        Preprocess features for prediction.
        
        Args:
            features: List of feature values
            
        Returns:
            np.ndarray: Preprocessed features
        """
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # For prediction, we fit the scaler on the input data
        # In production, you should save and load the scaler used during training
        if not self._scaler_fitted:
            # This is a simplified approach - ideally load the training scaler
            self.scaler.fit(features_array)
            self._scaler_fitted = True
        
        # Scale features
        scaled_features = self.scaler.transform(features_array)
        
        return scaled_features
    
    def _interpret_prediction(self, prediction: float) -> str:
        """
        Interpret numerical prediction into health status.
        
        Args:
            prediction: Numerical prediction value
            
        Returns:
            str: Health status label
        """
        return self.HEALTH_MAPPING.get(prediction, "Unknown")
    
    def predict(
        self,
        features: FetalHealthFeatures,
        model_name: Optional[str] = "gradient_boosting"
    ) -> PredictionResponse:
        """
        Make a single prediction.
        
        Args:
            features: Fetal health features
            model_name: Name of the model to use
            
        Returns:
            PredictionResponse: Prediction result
            
        Raises:
            ValueError: If model not found or invalid features
        """
        if model_name not in self.models:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available models: {list(self.models.keys())}"
            )
        
        # Convert features to list
        features_list = features.to_list()
        
        # Preprocess features
        preprocessed_features = self._preprocess_features(features_list)
        
        # Get model
        model = self.models[model_name]["model"]
        
        # Make prediction
        prediction = model.predict(preprocessed_features)[0]
        
        # Interpret prediction
        health_status = self._interpret_prediction(prediction)
        
        # Get confidence if available (for models that support predict_proba)
        confidence = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(preprocessed_features)[0]
                confidence = float(max(probabilities))
            except Exception:
                pass
        
        return PredictionResponse(
            prediction_code=float(prediction),
            health_status=health_status,
            model_used=model_name,
            confidence=confidence
        )
    
    def predict_batch(
        self,
        features_list: list[FetalHealthFeatures],
        model_name: Optional[str] = "gradient_boosting"
    ) -> list[PredictionResponse]:
        """
        Make batch predictions.
        
        Args:
            features_list: List of fetal health features
            model_name: Name of the model to use
            
        Returns:
            list[PredictionResponse]: List of prediction results
            
        Raises:
            ValueError: If model not found or invalid features
        """
        results = []
        for features in features_list:
            result = self.predict(features, model_name)
            results.append(result)
        
        return results
