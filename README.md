# PUC Lectures 2025

Repository for PUC lectures containing machine learning projects, deep learning frameworks, and generative AI applications.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [IA Solutions](#ia-solutions)
  - [Machine Learning Models](#machine-learning-models)
  - [FastAPI Application](#fastapi-application)
  - [Training Pipeline](#training-pipeline)
  - [Prediction Scripts](#prediction-scripts)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This repository contains educational materials and practical implementations for PUC's 2025 lecture series, focusing on:

- **Machine Learning**: Fetal health classification using ensemble methods
- **Deep Learning**: Framework implementations and examples
- **Generative AI**: Advanced analytics and financial market applications
- **REST APIs**: Production-ready FastAPI applications for ML model serving

## 📁 Project Structure

```
puc_lectures_2025/
├── ia_solutions/                    # Main ML solutions directory
│   ├── api/                        # FastAPI REST API application
│   │   ├── main.py                # API endpoints and application
│   │   ├── models.py              # Model manager and ML logic
│   │   ├── schemas.py             # Pydantic validation schemas
│   │   ├── test_api.py            # API testing script
│   │   ├── example_data.json      # Sample data for testing
│   │   ├── requirements.txt       # API dependencies
│   │   └── README.md              # API documentation
│   ├── models/                     # Trained ML models
│   │   ├── decision_tree_model.pkl
│   │   └── gradient_boosting_model.pkl
│   ├── train/                      # Model training scripts
│   │   └── train.py               # Training pipeline
│   └── predict/                    # Prediction scripts
│       └── predict.py             # Standalone prediction script
├── deeplearning_frameworks/        # Deep learning implementations
├── genai_and_advanced_analytics/   # Generative AI projects
├── genai_for_financial_market/     # Financial AI applications
├── drafts/                         # Work in progress
└── README.md                       # This file
```

## 🏥 IA Solutions

The `ia_solutions` directory contains a complete machine learning solution for fetal health classification.

### Machine Learning Models

Two ensemble models are trained and available:

1. **Decision Tree Classifier**
   - Max depth: 10
   - Random state: 42
   - Accuracy: ~80.72%

2. **Gradient Boosting Classifier** (Default)
   - Max depth: 10
   - N estimators: 100
   - Learning rate: 0.01
   - Accuracy: ~80.56%

#### Health Classifications

The models classify fetal health into three categories:

- **Normal (1.0)**: Healthy fetal condition
- **Suspect (2.0)**: Requires medical attention
- **Pathological (3.0)**: Requires immediate medical intervention

#### Features

The models use 4 features from the reduced dataset:

1. `severe_decelerations` - Number of severe decelerations per second
2. `accelerations` - Number of accelerations per second
3. `fetal_movement` - Number of fetal movements per second
4. `uterine_contractions` - Number of uterine contractions per second

### FastAPI Application

A production-ready REST API for serving the trained models.

#### Features

- ✅ Automatic model loading on startup
- ✅ Interactive API documentation (Swagger UI)
- ✅ Single and batch predictions
- ✅ Model selection support
- ✅ Health check endpoints
- ✅ CORS enabled
- ✅ Comprehensive error handling
- ✅ Type validation with Pydantic

#### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root health check |
| GET | `/health` | Detailed health status |
| GET | `/models` | List available models |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |

#### Quick Start - API

```bash
# Navigate to API directory
cd ia_solutions/api

# Install dependencies
pip install -r requirements.txt

# Start the server
python -m uvicorn main:app --reload

# Access documentation
# http://localhost:8000/docs
```

#### Example API Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "severe_decelerations": 0.0,
      "accelerations": 0.0,
      "fetal_movement": 0.0,
      "uterine_contractions": 0.0
    },
    "model_name": "gradient_boosting"
  }'
```

#### Example API Response

```json
{
  "prediction_code": 1.0,
  "health_status": "Normal",
  "model_used": "gradient_boosting",
  "confidence": 0.95
}
```

### Training Pipeline

The training pipeline (`ia_solutions/train/train.py`) implements:

- ✅ Data loading with retry mechanism
- ✅ Feature preprocessing and normalization
- ✅ Train/test split
- ✅ Model training (Decision Tree & Gradient Boosting)
- ✅ Model evaluation
- ✅ Automatic model persistence

#### Training Features

- **Clean Code**: Modular functions with single responsibilities
- **PEP 8 Compliant**: Follows Python style guidelines
- **Type Hints**: Full type annotations
- **Comprehensive Documentation**: Detailed docstrings
- **Error Handling**: Robust network error handling with exponential backoff

#### Run Training

```bash
cd ia_solutions/train
python train.py
```

The script will:
1. Load data from remote source
2. Preprocess features
3. Train both models
4. Evaluate accuracy
5. Save models to `ia_solutions/models/`

### Prediction Scripts

Standalone prediction script for testing trained models.

```bash
cd ia_solutions/predict
python predict.py
```

Features:
- Load saved models
- Preprocess sample data
- Make predictions
- Display formatted results

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/renansantosmendes/puc_lectures_2025.git
cd puc_lectures_2025
```

2. **Create virtual environment** (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

3. **Install dependencies**

For the API:
```bash
cd ia_solutions/api
pip install -r requirements.txt
```

For training:
```bash
pip install pandas scikit-learn numpy
```

## 📖 Quick Start

### 1. Train Models

```bash
cd ia_solutions/train
python train.py
```

### 2. Test Predictions

```bash
cd ia_solutions/predict
python predict.py
```

### 3. Start API Server

```bash
cd ia_solutions/api
python -m uvicorn main:app --reload
```

### 4. Test API

```bash
# In a new terminal
cd ia_solutions/api
python test_api.py
```

Or access the interactive documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 💡 Usage Examples

### Python Client Example

```python
import requests

# API base URL
BASE_URL = "http://localhost:8000"

# Make a prediction
response = requests.post(
    f"{BASE_URL}/predict",
    json={
        "features": {
            "severe_decelerations": 0.0,
            "accelerations": 0.0,
            "fetal_movement": 0.0,
            "uterine_contractions": 0.0
        },
        "model_name": "gradient_boosting"
    }
)

result = response.json()
print(f"Health Status: {result['health_status']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Batch Predictions

```python
response = requests.post(
    f"{BASE_URL}/predict/batch",
    json={
        "features_list": [
            {
                "severe_decelerations": 0.0,
                "accelerations": 0.0,
                "fetal_movement": 0.0,
                "uterine_contractions": 0.0
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
)

results = response.json()
for i, pred in enumerate(results['predictions']):
    print(f"Sample {i+1}: {pred['health_status']}")
```

## 🧪 Testing

### API Tests

Run the comprehensive test suite:

```bash
cd ia_solutions/api
python test_api.py
```

The test suite covers:
- Health check endpoints
- Model listing
- Single predictions
- Batch predictions
- Different model comparisons
- Various test cases

### Manual Testing

Use the Swagger UI for interactive testing:
1. Start the API server
2. Navigate to http://localhost:8000/docs
3. Try out different endpoints with the interactive interface

## 📊 Model Performance

| Model | Accuracy | Type |
|-------|----------|------|
| Decision Tree | 80.72% | DecisionTreeClassifier |
| Gradient Boosting | 80.56% | GradientBoostingClassifier |

## 🛠️ Development

### Code Quality

The project follows:
- **PEP 8**: Python style guide
- **Clean Code**: Meaningful names, single responsibility
- **Type Hints**: Full type annotations
- **Documentation**: Comprehensive docstrings
- **Error Handling**: Robust exception handling

### Project Standards

- All code in English (comments, docstrings, variables)
- Modular architecture
- Separation of concerns
- Configuration via constants
- Comprehensive logging

## 📝 API Documentation

Detailed API documentation is available in `ia_solutions/api/README.md`.

Key features:
- Complete endpoint reference
- Request/response schemas
- Example payloads
- Error codes
- Usage examples

## 🤝 Contributing

This is an educational repository for PUC lectures. For contributions:

1. Fork the repository
2. Create a feature branch
3. Follow the existing code style
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is part of PUC Lectures 2025 educational materials.

## 📧 Contact

For questions or support regarding this repository, please contact the course instructors.

## 🔗 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

## 🎓 Course Information

**Institution**: PUC (Pontifícia Universidade Católica)  
**Year**: 2025  
**Topics**: Machine Learning, Deep Learning, Generative AI

---

**Last Updated**: November 2025  
**Repository**: [puc_lectures_2025](https://github.com/renansantosmendes/puc_lectures_2025)