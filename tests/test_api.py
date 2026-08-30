from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from app.app import app


class MockModel:

    def __call__(self, image_tensor):
        import torch

        # Predict Cat with high confidence
        return torch.tensor([[5.0, 1.0]])


def create_test_image():
    image = Image.new(
        "RGB",
        (128, 128),
        color=(120, 120, 120)
    )

    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    return buffer


def test_health():

    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "CatsDogsCNN"


def test_predict():

    with patch("app.app.model", MockModel()):

        client = TestClient(app)

        image = create_test_image()

        response = client.post(
            "/predict",
            files={
                "image": (
                    "test.jpg",
                    image,
                    "image/jpeg"
                )
            }
        )

        assert response.status_code == 200

        data = response.json()

        assert data["prediction"] == "Cat"
        assert "confidence" in data
        assert data["confidence"] > 0