# Cats vs Dogs MLOps

End-to-end MLOps pipeline for binary image classification using a Convolutional Neural Network (CNN) built with PyTorch.

This project was developed as part of the **AIMLCZG523 - MLOps Assignment 2** and demonstrates the machine learning lifecycle from data preparation and model training through API serving, containerization, CI/CD, and Kubernetes deployment.

---

## Assignment

**Course:** AIMLCZG523 - MLOps Assignment 2

**Project:** Cats vs Dogs Image Classification MLOps Pipeline

---

## Objective

The objective of this project is to design and implement an end-to-end MLOps pipeline for binary image classification covering:

- Data and code versioning
- Data preprocessing
- CNN model development
- Model training and evaluation
- Experiment tracking using MLflow
- Model artifact management
- FastAPI model serving
- Automated testing using Pytest
- Docker containerization
- GitHub Actions CI/CD
- Docker image publishing to GitHub Container Registry
- Kubernetes deployment
- Kubernetes health and readiness checks
- Post-deployment API validation

---

## 1. Problem Statement

The goal is to build an image classification system that automatically determines whether an input image contains a **Cat** or **Dog**.

A Convolutional Neural Network is trained using PyTorch and exposed through a REST API. The deployed service accepts an image through the `/predict` endpoint and returns the predicted class and confidence score.

---

## 2. Dataset

The project uses the **Cats vs Dogs image classification dataset**.

The raw dataset is intentionally excluded from the Git repository because of its size.

```text
data/
├── raw/
└── processed/
```

Images are:

- Converted to RGB
- Resized to `128 x 128`
- Converted to PyTorch tensors
- Normalized using ImageNet mean and standard deviation

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

---

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Image Processing | Pillow |
| Experiment Tracking | MLflow |
| API Framework | FastAPI |
| API Server | Uvicorn |
| Testing | Pytest |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Container Registry | GitHub Container Registry |
| Orchestration | Kubernetes |
| Version Control | Git |
| Data Versioning | DVC |
| Health Monitoring | Kubernetes Probes |
| Environment | Conda |

---

## 4. Architecture

```text
                    ┌─────────────────────┐
                    │     Raw Dataset     │
                    │    Cats vs Dogs     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Preprocessing  │
                    │ Resize / Normalize  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Model Training   │
                    │      PyTorch CNN    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Trained Model       │
                    │ best_model.pt       │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
       ┌─────────────────┐          ┌─────────────────┐
       │     MLflow      │          │    FastAPI      │
       │ Experiment      │          │ Model Serving   │
       │ Tracking        │          └────────┬────────┘
       └─────────────────┘                   │
                                             ▼
                                   ┌─────────────────┐
                                   │ Docker Container │
                                   └────────┬────────┘
                                            │
                                            ▼
                                   ┌─────────────────┐
                                   │ GitHub Actions  │
                                   │ CI/CD Pipeline  │
                                   └────────┬────────┘
                                            │
                                            ▼
                                   ┌─────────────────┐
                                   │      GHCR       │
                                   │ Docker Registry │
                                   └────────┬────────┘
                                            │
                                            ▼
                                   ┌─────────────────┐
                                   │    Kubernetes   │
                                   │    Deployment   │
                                   └────────┬────────┘
                                            │
                                            ▼
                                   ┌─────────────────┐
                                   │ Running REST API│
                                   └─────────────────┘
```

---

## 5. Project Structure

```text
cats-dogs-mlops/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── best_model.pt
│
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── log_mlflow.py
│
├── tests/
│   └── test_api.py
│
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── submission/
│   └── ...
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── dvc.yaml
├── .gitignore
└── README.md
```

---

## 6. Data Pipeline

```text
Raw Images
    │
    ▼
Image Loading
    │
    ▼
RGB Conversion
    │
    ▼
Resize to 128 x 128
    │
    ▼
Tensor Conversion
    │
    ▼
Normalization
    │
    ▼
Train / Validation / Test
```

The preprocessing implementation is maintained under:

```text
src/preprocess.py
```

---

## 7. CNN Model

The project uses a custom PyTorch CNN model implemented in:

```text
src/model.py
```

The model performs binary classification:

```text
Class 0 → Cat
Class 1 → Dog
```

The trained model is stored as:

```text
models/best_model.pt
```

The model is included in the repository so the API and Docker deployment can load the trained model without retraining.

---

## 8. Model Training

Model training is implemented in:

```text
src/train.py
```

The training pipeline includes:

- Dataset loading
- Image preprocessing
- CNN initialization
- Training loop
- Validation
- Model evaluation
- Best model selection
- Model checkpoint saving

The best model is saved to:

```text
models/best_model.pt
```

---

## 9. Prediction Pipeline

Prediction functionality is implemented in:

```text
src/predict.py
```

The prediction pipeline:

1. Loads the trained CNN model.
2. Accepts an input image.
3. Converts the image to RGB.
4. Resizes the image to `128 x 128`.
5. Applies the training normalization.
6. Runs inference using the trained CNN.
7. Applies Softmax to obtain class probabilities.
8. Returns the predicted class and confidence.

Example:

```json
{
  "prediction": "Cat",
  "confidence": 99.42
}
```

---

## 10. MLflow Experiment Tracking

MLflow is used for experiment tracking during model development.

The MLflow-related implementation is available under:

```text
src/log_mlflow.py
```

The tracking workflow can record:

- Training parameters
- Model metrics
- Experiment runs
- Model-related artifacts

Local MLflow tracking files are excluded from Git where appropriate.

---

## 11. FastAPI Model Serving

The trained model is exposed using FastAPI.

The API implementation is:

```text
app/app.py
```

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model": "CatsDogsCNN",
  "device": "cpu"
}
```

### Prediction

```http
POST /predict
```

The endpoint accepts an image using multipart form-data.

Example response:

```json
{
  "prediction": "Cat",
  "confidence": 99.42
}
```

### Swagger Documentation

FastAPI provides interactive API documentation at:

```text
http://127.0.0.1:5000/docs
```

---

## 12. Running the API Locally

Activate the project environment:

```bash
conda activate catsdogs-mlops
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI application:

```bash
python app/app.py
```

The API runs on:

```text
http://127.0.0.1:5000
```

---

## 13. API Testing

Automated API tests are implemented using Pytest.

Tests are located at:

```text
tests/test_api.py
```

The tests cover:

- FastAPI application loading
- Health endpoint
- Prediction endpoint
- Image upload handling
- Prediction response validation

Run the tests:

```bash
pytest -q
```

The tests were successfully executed locally:

```text
2 passed
```

The tests are also executed automatically through GitHub Actions.

---

## 14. Docker Containerization

The FastAPI application is containerized using Docker.

The Docker configuration is defined in:

```text
Dockerfile
```

Build the image:

```bash
docker build -t cats-dogs-api:latest .
```

Run the container:

```bash
docker run -p 5000:5000 cats-dogs-api:latest
```

The API can then be accessed at:

```text
http://127.0.0.1:5000
```

---

## 15. Docker Compose

Docker Compose configuration is provided through:

```text
docker-compose.yml
```

Start the application using:

```bash
docker compose up
```

---

## 16. Continuous Integration

GitHub Actions is used to automatically validate the project whenever code is pushed or a pull request is created.

The CI pipeline performs:

```text
Git Push / Pull Request
          │
          ▼
Checkout Repository
          │
          ▼
Set up Python
          │
          ▼
Install Dependencies
          │
          ▼
Run Pytest
          │
          ▼
Build Docker Image
```

The workflow is defined in:

```text
.github/workflows/ci-cd.yml
```

---

## 17. Continuous Delivery

The CI/CD pipeline also publishes the Docker image to GitHub Container Registry.

```text
Code Push
    │
    ▼
Run Tests
    │
    ▼
Build Docker Image
    │
    ▼
Publish Docker Image
    │
    ▼
GitHub Container Registry
```

Published image:

```text
ghcr.io/akshaygund144/cats-dogs-api:latest
```

---

## 18. GitHub Actions Pipeline

The implemented pipeline contains:

```text
┌──────────────┐
│  Run Tests   │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Build Docker     │
│ Image            │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Publish Docker   │
│ Image to GHCR    │
└──────────────────┘
```

The GitHub Actions workflow has been successfully executed with all three stages passing:

```text
✓ Run Tests
✓ Build Docker Image
✓ Publish Docker Image
```

---

## 19. Kubernetes Deployment

The application is deployed to Kubernetes using:

```text
k8s/deployment.yaml
k8s/service.yaml
```

The Kubernetes deployment uses:

```text
ghcr.io/akshaygund144/cats-dogs-api:latest
```

Namespace:

```text
cats-dogs
```

Deployment:

```text
cats-dogs-api
```

---

## 20. Kubernetes Health Checks

The deployment uses readiness and liveness probes.

### Readiness Probe

```text
GET /health
```

### Liveness Probe

```text
GET /health
```

These probes allow Kubernetes to determine whether the application is ready to receive traffic and whether the container remains healthy.

---

## 21. Kubernetes Deployment Verification

The deployment was successfully rolled out using:

```bash
kubectl rollout status deployment/cats-dogs-api -n cats-dogs
```

Successful result:

```text
deployment "cats-dogs-api" successfully rolled out
```

The running pod was verified using:

```bash
kubectl get pods -n cats-dogs
```

Final status:

```text
1/1 Running
```

The deployed container was successfully running the GHCR image:

```text
ghcr.io/akshaygund144/cats-dogs-api:latest
```

---

## 22. Production API Verification

The deployed Kubernetes service was port-forwarded locally and tested.

Health check:

```bash
curl http://127.0.0.1:5011/health
```

Response:

```json
{
  "status": "healthy",
  "model": "CatsDogsCNN",
  "device": "cpu"
}
```

Prediction test:

```bash
curl -X POST -F "image=@C:\Users\Akshay Gund\Projects\cats-dogs-mlops\data\processed\test\Cat\1000.jpg" http://127.0.0.1:5011/predict
```

Example response:

```json
{
  "prediction": "Cat",
  "confidence": 99.42
}
```

This confirms that the trained model can successfully perform image inference from the Kubernetes deployment.

---

## 23. End-to-End MLOps Workflow

```text
                 DATA
                  │
                  ▼
        ┌──────────────────┐
        │ Raw Cats & Dogs  │
        │ Dataset          │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Preprocessing    │
        │ Resize/Normalize │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ CNN Training     │
        │ PyTorch          │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ best_model.pt    │
        └────────┬─────────┘
                 │
        ┌────────┴─────────┐
        │                  │
        ▼                  ▼
   ┌─────────┐       ┌──────────┐
   │ MLflow  │       │ FastAPI  │
   └─────────┘       └────┬─────┘
                          │
                          ▼
                   ┌─────────────┐
                   │   Docker    │
                   └──────┬──────┘
                          │
                          ▼
                   ┌─────────────┐
                   │   Pytest    │
                   └──────┬──────┘
                          │
                          ▼
                   ┌─────────────┐
                   │   GitHub    │
                   │   Actions   │
                   └──────┬──────┘
                          │
                          ▼
                   ┌─────────────┐
                   │    GHCR     │
                   └──────┬──────┘
                          │
                          ▼
                   ┌─────────────┐
                   │ Kubernetes  │
                   └──────┬──────┘
                          │
                          ▼
                   ┌─────────────┐
                   │ Production  │
                   │    API      │
                   └─────────────┘
```

---

## 24. MLOps Lifecycle

```text
1. Data Collection
       ↓
2. Data Preprocessing
       ↓
3. Model Development
       ↓
4. Model Training
       ↓
5. Experiment Tracking
       ↓
6. Model Packaging
       ↓
7. API Development
       ↓
8. Automated Testing
       ↓
9. Containerization
       ↓
10. CI/CD
       ↓
11. Container Registry
       ↓
12. Kubernetes Deployment
       ↓
13. Health Monitoring
       ↓
14. API Validation
```

---

## 25. Version Control

Git is used for source-code versioning.

The repository contains version-controlled:

- Source code
- API implementation
- Automated tests
- Docker configuration
- Kubernetes manifests
- CI/CD workflows
- Trained model artifact
- Project documentation

Large raw datasets and generated files are excluded using `.gitignore`.

---

## 26. Data Versioning

DVC is included in the project to support dataset and pipeline versioning.

The project contains:

```text
dvc.yaml
```

Large raw datasets are intentionally excluded from Git because of repository size considerations.

---

## 27. Model Artifact

The trained model is stored as:

```text
models/best_model.pt
```

The FastAPI application loads the model during startup and places it in evaluation mode before serving predictions.

The model artifact is included in the repository so that the deployed application can load the trained model directly.

---

## 28. Error Handling

The API includes basic error handling for:

- Missing image filename
- Invalid image input
- Image processing errors
- Model inference errors
- Missing model artifact

Errors are returned as API responses instead of causing uncontrolled application termination.

---

## 29. Reproducibility

The project supports reproducible execution through:

- Python dependency management
- Git version control
- DVC configuration
- Consistent image preprocessing
- Version-controlled training and inference code
- MLflow experiment tracking
- Docker containerization
- Automated CI/CD

---

## 30. CI/CD Validation

The GitHub Actions pipeline was successfully validated.

The following stages completed successfully:

```text
✓ Run Tests
✓ Build Docker Image
✓ Publish Docker Image
```

The Docker image was successfully published to GitHub Container Registry.

---

## 31. Deployment Validation

The Kubernetes deployment was successfully validated.

Verified components:

```text
✓ Kubernetes Deployment
✓ Kubernetes Pod
✓ Docker Image Pull
✓ Container Startup
✓ Readiness Probe
✓ Liveness Probe
✓ Health Endpoint
✓ Prediction Endpoint
✓ Model Inference
```

The deployed API successfully returned:

```json
{
  "prediction": "Cat",
  "confidence": 99.42
}
```

---

## 32. Example API Usage

### Health Check

```bash
curl http://127.0.0.1:5011/health
```

Example:

```json
{
  "status": "healthy",
  "model": "CatsDogsCNN",
  "device": "cpu"
}
```

### Image Prediction

Windows:

```bash
curl -X POST -F "image=@C:\path\to\cat.jpg" http://127.0.0.1:5011/predict
```

Linux/macOS:

```bash
curl -X POST \
  -F "image=@/path/to/cat.jpg" \
  http://127.0.0.1:5011/predict
```

Example:

```json
{
  "prediction": "Cat",
  "confidence": 99.42
}
```

---

## 33. Kubernetes Commands

Check the deployment:

```bash
kubectl get deployment cats-dogs-api -n cats-dogs
```

Check pods:

```bash
kubectl get pods -n cats-dogs
```

Check services:

```bash
kubectl get services -n cats-dogs
```

Check rollout:

```bash
kubectl rollout status deployment/cats-dogs-api -n cats-dogs
```

Describe the deployment:

```bash
kubectl describe deployment cats-dogs-api -n cats-dogs
```

Describe a pod:

```bash
kubectl describe pod -n cats-dogs -l app=cats-dogs-api
```

View application logs:

```bash
kubectl logs -n cats-dogs -l app=cats-dogs-api
```

---

## 34. Local Development

Clone the repository:

```bash
git clone https://github.com/akshaygund144/cats-dogs-mlops.git
```

Navigate to the project:

```bash
cd cats-dogs-mlops
```

Activate the environment:

```bash
conda activate catsdogs-mlops
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest -q
```

Run the API:

```bash
python app/app.py
```

---

## 35. Future Improvements

Possible future enhancements include:

- Model performance monitoring
- Automated model retraining
- Model drift detection
- Prometheus metrics
- Grafana dashboards
- Automated Kubernetes deployment from GitHub Actions
- Model registry integration
- Automated DVC pipeline execution
- Horizontal Pod Autoscaling
- HTTPS/TLS configuration
- Authentication and authorization
- Canary or blue-green deployments
- Automated model quality gates

---

## 36. Conclusion

This project demonstrates an end-to-end MLOps implementation for a Cats vs Dogs image classification problem.

The solution integrates:

```text
PyTorch
   +
MLflow
   +
FastAPI
   +
Pytest
   +
Docker
   +
GitHub Actions
   +
GitHub Container Registry
   +
Kubernetes
```

The final system is capable of:

1. Training a CNN image classification model.
2. Saving the trained model as an artifact.
3. Tracking experiments using MLflow.
4. Serving predictions through a FastAPI REST API.
5. Testing the API automatically.
6. Building a Docker image.
7. Running automated CI/CD through GitHub Actions.
8. Publishing the Docker image to GHCR.
9. Deploying the application to Kubernetes.
10. Performing health checks using Kubernetes probes.
11. Performing real image predictions from the deployed API.

---

## Author

**Akshay Gund**

GitHub:

https://github.com/akshaygund144

Project Repository:

https://github.com/akshaygund144/cats-dogs-mlops

---

## Project Status

**End-to-End MLOps Pipeline: Successfully Implemented and Validated**

```text
✓ Data Pipeline
✓ CNN Model
✓ Model Training
✓ Model Artifact
✓ MLflow
✓ FastAPI
✓ Automated Tests
✓ Docker
✓ GitHub Actions CI
✓ GitHub Actions CD
✓ GitHub Container Registry
✓ Kubernetes Deployment
✓ Kubernetes Health Checks
✓ Production API Validation
```
