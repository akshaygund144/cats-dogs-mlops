from pathlib import Path

import torch
from flask import Flask, jsonify, request
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
# Flask Application
# =========================================================

app = Flask(__name__)


# =========================================================
# Image Transformation
# =========================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# Load Model
# =========================================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = CatsDogsCNN(
        num_classes=2
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

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


model = load_model()


# =========================================================
# Health Check Endpoint
# =========================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "model": "CatsDogsCNN",
        "device": str(DEVICE)
    })


# =========================================================
# Prediction Endpoint
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return jsonify({
            "error": "No image provided. Use form field 'image'."
        }), 400

    file = request.files["image"]

    if file.filename == "":

        return jsonify({
            "error": "Empty filename."
        }), 400

    try:

        image = Image.open(file).convert("RGB")

        image_tensor = transform(
            image
        ).unsqueeze(0).to(DEVICE)

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

        predicted_label = CLASS_NAMES[
            predicted_class.item()
        ]

        confidence_value = confidence.item()

        return jsonify({
            "prediction": predicted_label,
            "confidence": round(
                confidence_value * 100,
                2
            )
        })

    except Exception as exc:

        return jsonify({
            "error": str(exc)
        }), 500


# =========================================================
# Application Entry Point
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Cats vs Dogs CNN API")
    print("=" * 60)

    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        "Endpoints:"
    )

    print(
        "  GET  /health"
    )

    print(
        "  POST /predict"
    )

    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )