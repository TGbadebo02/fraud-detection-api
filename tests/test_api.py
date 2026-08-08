import os

os.environ["API_KEY"] = "test-key"

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)
AUTH_HEADERS = {"X-API-Key": "test-key"}

#define a sample transaction payload for testing the /predict endpoint
def sample_transaction() -> dict:
    return {
        "transaction_id": "txn_test_001",
        "amount": 950.0,
        "transaction_hour": 2,
        "merchant_category": "electronics",
        "customer_age_days": 10,
        "is_foreign_transaction": True,
        "is_new_merchant": True,
        "previous_chargebacks": 1,
    }

#define a test function to verify that the /health endpoint returns a 200 status code and the expected JSON response
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

#define a test function to verify that the /predict endpoint returns a 200 status code and the expected JSON response structure
def test_predict_returns_fraud_score():
    response = client.post("/predict", json=sample_transaction(), headers=AUTH_HEADERS)

    assert response.status_code == 200
    payload = response.json()
    assert payload["transaction_id"] == "txn_test_001"
    assert 0 <= payload["fraud_score"] <= 1
    assert payload["risk_level"] in {"low", "medium", "high"}
    assert isinstance(payload["reasons"], list)

#define a test function to verify that /predict rejects requests without a valid API key
def test_predict_requires_api_key():
    response = client.post("/predict", json=sample_transaction())
    assert response.status_code == 401

    response = client.post("/predict", json=sample_transaction(), headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401

