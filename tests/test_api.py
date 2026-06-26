from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


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


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_fraud_score():
    response = client.post("/predict", json=sample_transaction())

    assert response.status_code == 200
    payload = response.json()
    assert payload["transaction_id"] == "txn_test_001"
    assert 0 <= payload["fraud_score"] <= 1
    assert payload["risk_level"] in {"low", "medium", "high"}
    assert isinstance(payload["reasons"], list)

