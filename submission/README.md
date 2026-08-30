# Cats vs Dogs MLOps

End-to-end MLOps pipeline for binary image classification
using the Cats vs Dogs dataset.

## Assignment

AIMLCZG523 - MLOps Assignment 2

## Objective

Design and implement an end-to-end MLOps pipeline for
Cats vs Dogs binary image classification covering:

- Data and code versioning
- Model development
- Experiment tracking
- Model packaging
- Containerization
- Automated testing
- CI/CD
- Deployment
- Monitoring and logging
- Post-deployment model performance tracking

## Dataset

Cats and Dogs binary classification dataset from Kaggle.

Images will be preprocessed to 224x224 RGB and divided into
training, validation, and test sets.

## MLOps Stack

- Python
- PyTorch
- Git
- DVC
- MLflow
- FastAPI
- Docker
- GitHub Actions
- Docker Compose
- Prometheus

## Project Structure

```text
cats-dogs-mlops/
├── data/
│   ├── raw/
│   └── processed/
├── src/
├── tests/
├── app/
├── .github/
│   └── workflows/
├── k8s/
├── mlruns/
├── Dockerfile
├── requirements.txt
├── dvc.yaml
├── .gitignore
└── README.md