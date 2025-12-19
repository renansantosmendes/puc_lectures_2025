from setuptools import setup, find_packages

setup(
    name="ai-runtime-core",
    version="0.1.0",
    description="Core runtime utilities for AI and deep learning frameworks",
    author="Seu Nome ou Organização",
    python_requires=">=3.9",
    packages=find_packages(
        exclude=("tests", "notebooks", "examples")
    ),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "scikit-learn",
        "matplotlib",
        "yfinance",
        "joblib"
    ],
    include_package_data=True,
    zip_safe=False,
)
