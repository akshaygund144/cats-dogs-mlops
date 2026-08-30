from fastapi.testclient import TestClient
from app.app import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "CatsDogsCNN"
    assert data["device"] == "cpu"


def test_predict():
    image_path = "data/processed/test/Cat/1000.jpg"

    with open(image_path, "rb") as image:
        response = client.post(
            "/predict",
            files={"image": ("1000.jpg", image, "image/jpeg")}
        )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "confidence" in data
    assert data["prediction"] in ["Cat", "Dog"]
    assert 0 <= data["confidence"] <= 100