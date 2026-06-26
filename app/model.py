from pathlib import Path

import joblib
import pandas as pd

from app.schemas import Transaction


MODEL_PATH = Path("models/fraud_model.pkl")


class FraudModel:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.pipeline = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Run `python scripts/train_model.py` first."
            )
        self.pipeline = joblib.load(self.model_path)

    def predict(self, transaction: Transaction) -> tuple[float, str, list[str]]:
        if self.pipeline is None:
            self.load()

        row = pd.DataFrame([transaction.model_dump(exclude={"transaction_id"})])
        fraud_score = float(self.pipeline.predict_proba(row)[0][1])
        risk_level = get_risk_level(fraud_score)
        reasons = explain_transaction(transaction, fraud_score)
        return fraud_score, risk_level, reasons


def get_risk_level(fraud_score: float) -> str:
    if fraud_score >= 0.75:
        return "high"
    if fraud_score >= 0.4:
        return "medium"
    return "low"


def explain_transaction(transaction: Transaction, fraud_score: float) -> list[str]:
    reasons: list[str] = []

    if transaction.amount >= 500:
        reasons.append("unusual amount")
    if transaction.transaction_hour <= 5 or transaction.transaction_hour >= 23:
        reasons.append("unusual transaction time")
    if transaction.is_new_merchant:
        reasons.append("new merchant")
    if transaction.is_foreign_transaction:
        reasons.append("foreign location")
    if transaction.previous_chargebacks > 0:
        reasons.append("previous chargebacks")
    if transaction.customer_age_days < 30:
        reasons.append("new customer account")

    if not reasons and fraud_score >= 0.4:
        reasons.append("model detected elevated risk pattern")
    if not reasons:
        reasons.append("normal transaction pattern")

    return reasons


fraud_model = FraudModel()

