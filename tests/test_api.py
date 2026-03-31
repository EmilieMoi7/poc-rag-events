from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API RAG opérationnelle"}


def test_ask_success():
    response = client.post(
        "/ask",
        json={"question": "Quels événements musicaux sont prévus à Lille ?"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "question" in data
    assert "response" in data
    assert data["status"] == "success"


def test_ask_empty_question():
    response = client.post(
        "/ask",
        json={"question": "   "}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "La question ne peut pas être vide."