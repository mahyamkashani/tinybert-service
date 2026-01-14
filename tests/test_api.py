from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_prediction():
    payload = {
        "texts": ["Great product", "Very bad experience"]
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2

    for item in data["predictions"]:
        assert "positive" in item
        assert "negative" in item
        assert 0.0 <= item["positive"] <= 1.0
        assert 0.0 <= item["negative"] <= 1.0
