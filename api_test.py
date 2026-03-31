import requests

BASE_URL = "http://127.0.0.1:8000"


def test_root():
    response = requests.get(f"{BASE_URL}/")
    print("GET /")
    print("Status code:", response.status_code)
    print("Response:", response.json())
    print("-" * 50)


def test_ask():
    payload = {
        "question": "Quels événements musicaux sont prévus à Lille ?"
    }
    response = requests.post(f"{BASE_URL}/ask", json=payload)
    print("POST /ask")
    print("Status code:", response.status_code)
    print("Response:", response.json())
    print("-" * 50)


def test_ask_empty():
    payload = {
        "question": "   "
    }
    response = requests.post(f"{BASE_URL}/ask", json=payload)
    print("POST /ask (question vide)")
    print("Status code:", response.status_code)
    print("Response:", response.json())
    print("-" * 50)

def test_rebuild():
    response = requests.post(f"{BASE_URL}/rebuild")
    print("POST /rebuild")
    print("Status code:", response.status_code)
    print("Response:", response.json())
    print("-" * 50)

if __name__ == "__main__":
    test_root()
    test_ask()
    test_ask_empty()
    test_rebuild()