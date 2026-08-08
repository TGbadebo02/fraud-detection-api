from fastapi import Depends, FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.auth import verify_api_key
from app.database import init_db, log_prediction
from app.model import fraud_model
from app.schemas import PredictionResponse, Transaction


app = FastAPI(
    title="Fraud Detection API",
    description="Scores payment transactions for fraud risk in real time.",
    version="0.1.0",
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.on_event("startup")
def startup() -> None:
    init_db()
    fraud_model.load()

#define a health check endpoint to verify that the API is running and responsive
@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

#define a POST endpoint to receive transaction data and return a fraud prediction
@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
def predict(request: Request, transaction: Transaction) -> PredictionResponse:
    fraud_score, risk_level, reasons = fraud_model.predict(transaction)
    prediction = PredictionResponse(
        transaction_id=transaction.transaction_id,
        fraud_score=round(fraud_score, 4),
        risk_level=risk_level,
        reasons=reasons,
    )
    log_prediction(transaction, prediction)
    return prediction

