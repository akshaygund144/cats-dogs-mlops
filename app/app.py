from pathlib import Path
from io import BytesIO

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

from src.model import CatsDogsCNN


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pt"

IMAGE_SIZE = 128

CLASS_NAMES = ["Cat", "Dog"]

DEVICE = torch.device("cpu")


# =========================================================
# FastAPI Application
# =========================================================

app = FastAPI(
    title="Cats vs Dogs CNN API",
    description="CNN-based image classification API for Cats vs Dogs",
    version="1.0.0",
)


# =========================================================
# Image Transformation
# =========================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =========================================================
# Load Model
# =========================================================

def load_model():
    """
    Load the trained Cats vs Dogs CNN model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = CatsDogsCNN(num_classes=2)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        elif "state_dict" in checkpoint:
            model.load_state_dict(
                checkpoint["state_dict"]
            )

        else:
            model.load_state_dict(checkpoint)

    else:
        model = checkpoint

    model.to(DEVICE)
    model.eval()

    return model


# =========================================================
# Model Initialization
# =========================================================

model = None

if MODEL_PATH.exists():
    model = load_model()


# =========================================================
# Health Check Endpoint
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "CatsDogsCNN",
        "device": str(DEVICE),
    }


# =========================================================
# Prediction Endpoint
# =========================================================

@app.post("/predict")
async def predict(
    image: UploadFile = File(...)
):

    # Check whether the model is available
    if model is None:
        return {
            "error": "Model not available."
        }

    # Check filename
    if not image.filename:
        return {
            "error": "Empty filename."
        }

    try:

        # Read uploaded image
        image_data = await image.read()

        # Convert uploaded data to PIL image
        pil_image = Image.open(
            BytesIO(image_data)
        ).convert("RGB")

        # Apply preprocessing
        image_tensor = transform(
            pil_image
        ).unsqueeze(0).to(DEVICE)

        # Model inference
        with torch.no_grad():

            outputs = model(
                image_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted_class = torch.max(
                probabilities,
                dim=1
            )

        # Convert prediction to class label
        predicted_label = CLASS_NAMES[
            predicted_class.item()
        ]

        confidence_value = confidence.item()

        return {
            "prediction": predicted_label,
            "confidence": round(
                confidence_value * 100,
                2
            ),
        }

    except Exception as exc:

        return {
            "error": str(exc)
        }


# =========================================================
# Application Entry Point
# =========================================================

if __name__ == "__main__":

    import uvicorn

    print("=" * 60)
    print("Cats vs Dogs CNN API - FastAPI")
    print("=" * 60)

    print(f"Model: {MODEL_PATH}")
    print(f"Model available: {model is not None}")
    print(f"Device: {DEVICE}")

    print("Endpoints:")
    print("  GET  /health")
    print("  POST /predict")
    print("  GET  /docs")

    print("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
    )