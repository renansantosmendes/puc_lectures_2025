"""
Module for training fetal health classification models.

This script performs data loading, preprocessing, and training of machine
learning models for fetal health classification using Decision Tree and
Gradient Boosting.
"""

import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# Constants
DATA_URL = 'https://raw.githubusercontent.com/renansantosmendes/lectures-cdas-2023/master/fetal_health_reduced.csv'
TEST_SIZE = 0.3
RANDOM_STATE = 42
MAX_DEPTH = 10
N_ESTIMATORS = 100
LEARNING_RATE = 0.01
MODELS_DIR = 'ia_solutions/models'


def load_fetal_health_data(url: str, max_retries: int = 3) -> pd.DataFrame:
    """
    Load fetal health data from a URL with retry mechanism.
    
    Args:
        url: URL of the CSV file containing the data
        max_retries: Maximum number of retry attempts
        
    Returns:
        DataFrame with fetal health data
        
    Raises:
        Exception: If data cannot be loaded after all retries
    """
    import time
    from urllib.error import URLError
    
    for attempt in range(max_retries):
        try:
            print(f"Attempting to load data (attempt {attempt + 1}/{max_retries})...")
            data = pd.read_csv(url)
            print("Data loaded successfully!")
            return data
        except (URLError, ConnectionResetError, TimeoutError, Exception) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"Connection failed: {type(e).__name__}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"\n⚠️  Failed to load data from URL after {max_retries} attempts.")
                print("This is likely due to network restrictions or firewall blocking the connection.")
                print("\nSuggested solutions:")
                print("1. Try running from a different network")
                print("2. Download the file manually and update the DATA_URL to the local path")
                print(f"3. Download from: {url}")
                raise Exception(f"Could not load data after {max_retries} attempts. Last error: {e}")


def prepare_features_and_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target from the data.
    
    Args:
        data: DataFrame with complete data
        
    Returns:
        Tuple containing (features, target_labels)
    """
    features = data.iloc[:, :-1]
    target_labels = data["fetal_health"]
    return features, target_labels


def scale_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize features using StandardScaler.
    
    Args:
        features: DataFrame with unnormalized features
        
    Returns:
        DataFrame with normalized features
    """
    scaler = StandardScaler()
    scaled_features_array = scaler.fit_transform(features)
    scaled_features = pd.DataFrame(
        scaled_features_array, 
        columns=list(features.columns)
    )
    return scaled_features


def split_train_test_data(
    features: pd.DataFrame, 
    target: pd.Series, 
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Args:
        features: DataFrame with features
        target: Series with target labels
        test_size: Proportion of data for testing
        random_state: Seed for reproducibility
        
    Returns:
        Tuple containing (features_train, features_test, target_train, target_test)
    """
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state
    )


def train_decision_tree(
    features_train: pd.DataFrame,
    target_train: pd.Series,
    max_depth: int = MAX_DEPTH,
    random_state: int = RANDOM_STATE
) -> DecisionTreeClassifier:
    """
    Train a Decision Tree Classifier model.
    
    Args:
        features_train: Training features
        target_train: Training labels
        max_depth: Maximum depth of the tree
        random_state: Seed for reproducibility
        
    Returns:
        Trained Decision Tree model
    """
    decision_tree_classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state
    )
    decision_tree_classifier.fit(features_train, target_train)
    return decision_tree_classifier


def train_gradient_boosting(
    features_train: pd.DataFrame,
    target_train: pd.Series,
    max_depth: int = MAX_DEPTH,
    n_estimators: int = N_ESTIMATORS,
    learning_rate: float = LEARNING_RATE,
    random_state: int = RANDOM_STATE
) -> GradientBoostingClassifier:
    """
    Train a Gradient Boosting Classifier model.
    
    Args:
        features_train: Training features
        target_train: Training labels
        max_depth: Maximum depth of the trees
        n_estimators: Number of estimators
        learning_rate: Learning rate
        random_state: Seed for reproducibility
        
    Returns:
        Trained Gradient Boosting model
    """
    gradient_boosting_classifier = GradientBoostingClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )
    gradient_boosting_classifier.fit(features_train, target_train)
    return gradient_boosting_classifier


def save_model(model, model_name: str, output_dir: str = MODELS_DIR) -> str:
    """
    Save a trained model to disk using pickle.
    
    Args:
        model: Trained model to save
        model_name: Name for the model file (without extension)
        output_dir: Directory to save the model
        
    Returns:
        Path to the saved model file
    """
    # Create models directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create full path for the model file
    model_path = os.path.join(output_dir, f"{model_name}.pkl")
    
    # Save the model
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"✅ Model saved to: {model_path}")
    return model_path


def evaluate_model(
    model,
    features_test: pd.DataFrame,
    target_test: pd.Series
) -> float:
    """
    Evaluate model accuracy.
    
    Args:
        model: Trained model
        features_test: Test features
        target_test: Test labels
        
    Returns:
        Model accuracy
    """
    predictions = model.predict(features_test)
    accuracy = accuracy_score(y_true=target_test, y_pred=predictions)
    return accuracy


def main():
    """Main function to execute the training pipeline."""
    # Data loading
    print("Loading data...")
    fetal_health_data = load_fetal_health_data(DATA_URL)
    
    # Data preparation
    print("Preparing features and target...")
    features, target_labels = prepare_features_and_target(fetal_health_data)
    
    # Normalization
    print("Normalizing features...")
    scaled_features = scale_features(features)
    
    # Train/test split
    print("Splitting data into train and test sets...")
    features_train, features_test, target_train, target_test = split_train_test_data(
        scaled_features,
        target_labels
    )
    
    # Training and evaluation - Decision Tree
    print("\n=== Decision Tree Classifier ===")
    decision_tree_model = train_decision_tree(features_train, target_train)
    decision_tree_accuracy = evaluate_model(
        decision_tree_model,
        features_test,
        target_test
    )
    print(f"Decision Tree Accuracy: {decision_tree_accuracy:.4f}")
    
    # Save Decision Tree model
    save_model(decision_tree_model, "decision_tree_model")
    
    # Training and evaluation - Gradient Boosting
    print("\n=== Gradient Boosting Classifier ===")
    gradient_boosting_model = train_gradient_boosting(features_train, target_train)
    gradient_boosting_accuracy = evaluate_model(
        gradient_boosting_model,
        features_test,
        target_test
    )
    print(f"Gradient Boosting Accuracy: {gradient_boosting_accuracy:.4f}")
    
    # Save Gradient Boosting model
    save_model(gradient_boosting_model, "gradient_boosting_model")
    
    print("\n🎉 Training completed! Models saved successfully.")


if __name__ == "__main__":
    main()
