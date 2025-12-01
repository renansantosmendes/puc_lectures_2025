"""
Example script to test the Fetal Health Classification API.

This script demonstrates how to interact with the API endpoints.
"""

import requests
import json


# API Configuration
BASE_URL = "http://localhost:8000"


def print_section(title: str):
    """Print a formatted section title."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")


def test_health_check():
    """Test the health check endpoint."""
    print_section("Testing Health Check Endpoint")
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_list_models():
    """Test the list models endpoint."""
    print_section("Testing List Models Endpoint")
    
    response = requests.get(f"{BASE_URL}/models")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_single_prediction():
    """Test single prediction endpoint."""
    print_section("Testing Single Prediction Endpoint")
    
    # Example features (Normal case) - 4 features from reduced dataset
    payload = {
        "features": {
            "severe_decelerations": 0.0,
            "accelerations": 0.0,
            "fetal_movement": 0.0,
            "uterine_contractions": 0.0
        },
        "model_name": "gradient_boosting"
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_batch_prediction():
    """Test batch prediction endpoint."""
    print_section("Testing Batch Prediction Endpoint")
    
    # Example features for batch prediction
    payload = {
        "features_list": [
            {
                "severe_decelerations": 0.0,
                "accelerations": 0.0,
                "fetal_movement": 0.0,
                "uterine_contractions": 0.0
            },
            {
                "severe_decelerations": 0.0,
                "accelerations": 0.006,
                "fetal_movement": 0.0,
                "uterine_contractions": 0.006
            },
            {
                "severe_decelerations": 0.001,
                "accelerations": 0.003,
                "fetal_movement": 0.001,
                "uterine_contractions": 0.004
            }
        ],
        "model_name": "gradient_boosting"
    }
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_different_models():
    """Test predictions with different models."""
    print_section("Testing Different Models")
    
    features = {
        "severe_decelerations": 0.0,
        "accelerations": 0.0,
        "fetal_movement": 0.0,
        "uterine_contractions": 0.0
    }
    
    for model_name in ["decision_tree", "gradient_boosting"]:
        print(f"\nTesting with {model_name}:")
        payload = {"features": features, "model_name": model_name}
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        print(f"  Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"  Prediction: {response.json()['health_status']}")
            if response.json().get('confidence'):
                print(f"  Confidence: {response.json()['confidence']:.2%}")
        else:
            print(f"  Error: {response.json()}")


def test_various_cases():
    """Test various prediction cases."""
    print_section("Testing Various Cases")
    
    test_cases = [
        {
            "name": "Normal Case 1",
            "features": {
                "severe_decelerations": 0.0,
                "accelerations": 0.0,
                "fetal_movement": 0.0,
                "uterine_contractions": 0.0
            }
        },
        {
            "name": "Normal Case 2",
            "features": {
                "severe_decelerations": 0.0,
                "accelerations": 0.006,
                "fetal_movement": 0.0,
                "uterine_contractions": 0.006
            }
        },
        {
            "name": "Suspect Case",
            "features": {
                "severe_decelerations": 0.001,
                "accelerations": 0.003,
                "fetal_movement": 0.001,
                "uterine_contractions": 0.004
            }
        }
    ]
    
    for case in test_cases:
        print(f"\n{case['name']}:")
        payload = {"features": case['features'], "model_name": "gradient_boosting"}
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"  Status: {result['health_status']}")
            print(f"  Code: {result['prediction_code']}")
            if result.get('confidence'):
                print(f"  Confidence: {result['confidence']:.2%}")
        else:
            print(f"  Error: {response.json()}")


def main():
    """Run all tests."""
    print("\n" + "#"*80)
    print("# Fetal Health Classification API - Test Suite")
    print("#"*80)
    
    try:
        # Test all endpoints
        test_health_check()
        test_list_models()
        test_single_prediction()
        test_batch_prediction()
        test_different_models()
        test_various_cases()
        
        print_section("All Tests Completed Successfully!")
        
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Could not connect to the API.")
        print("Make sure the API is running at:", BASE_URL)
        print("\nStart the API with:")
        print("  python -m uvicorn main:app --reload")
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
