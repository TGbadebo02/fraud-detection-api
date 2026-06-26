from pydantic import BaseModel, Field


class Transaction(BaseModel):
    transaction_id: str = Field(..., examples=["txn_10001"])
    amount: float = Field(..., gt=0, examples=[249.99])
    transaction_hour: int = Field(..., ge=0, le=23, examples=[23])
    merchant_category: str = Field(..., examples=["electronics"])
    customer_age_days: int = Field(..., ge=0, examples=[180])
    is_foreign_transaction: bool = Field(..., examples=[True])
    is_new_merchant: bool = Field(..., examples=[False])
    previous_chargebacks: int = Field(..., ge=0, examples=[0])


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_score: float
    risk_level: str
    reasons: list[str]

