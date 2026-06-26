from fastapi import FastAPI

from app.database import init_db, log_prediction
from app.model import fraud_model
from app.schemas import PredictionResponse, Transaction


app = FastAPI(
    title="Fraud Detection API",
    description="Scores payment transactions for fraud risk in real time.",
    version="0.1.0",
)


@app.on_event("startup")
def startup() -> None:
    init_db()
    fraud_model.load()


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction) -> PredictionResponse:
    fraud_score, risk_level, reasons = fraud_model.predict(transaction)
    prediction = PredictionResponse(
        transaction_id=transaction.transaction_id,
        fraud_score=round(fraud_score, 4),
        risk_level=risk_level,
        reasons=reasons,
    )
    log_prediction(transaction, prediction)
    return prediction

