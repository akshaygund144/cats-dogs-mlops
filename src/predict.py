from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.model import CatsDogsCNN


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pt"

IMAGE_SIZE = 128

CLASS_NAMES = [
    "Cat",
    "Dog"
]

DEVICE = torch.device("cpu")


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

    # Handle checkpoint formats used by the project
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

            # The saved file is a raw state_dict
            model.load_state_dict(
                checkpoint
            )

    else:

        # Complete model object
        model = checkpoint

    model.to(DEVICE)

    model.eval()

    return model


# =========================================================
# Prediction Function
# =========================================================

def predict_image(
    model,
    image_path
):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Open image and ensure RGB
    image = Image.open(
        image_path
    ).convert("RGB")

    # Apply preprocessing
    image_tensor = transform(
        image
    )

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE
    )

    # Disable gradient calculation
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

    predicted_index = predicted_class.item()

    predicted_label = CLASS_NAMES[
        predicted_index
    ]

    confidence_percentage = (
        confidence.item() * 100
    )

    return (
        predicted_label,
        confidence_percentage
    )


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 60)
    print("Cats vs Dogs Image Prediction")
    print("=" * 60)

    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        f"Device: {DEVICE}"
    )

    # -----------------------------------------------------
    # Load trained model
    # -----------------------------------------------------

    model = load_model()

    print(
        "Model loaded successfully."
    )

    print()

    # -----------------------------------------------------
    # Ask user for image
    # -----------------------------------------------------

    image_path = input(
        "Enter image path: "
    ).strip()

    if not image_path:
        print(
            "No image path provided."
        )
        return

    # -----------------------------------------------------
    # Make prediction
    # -----------------------------------------------------

    predicted_label, confidence = predict_image(
        model,
        image_path
    )

    # -----------------------------------------------------
    # Display result
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    print(
        f"Image: {image_path}"
    )

    print(
        f"Prediction: {predicted_label}"
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

    print("=" * 60)


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":
    main()