from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("data/transactions.csv")
RANDOM_SEED = 42
ROW_COUNT = 5000


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-value))


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    merchant_categories = np.array(
        ["grocery", "restaurant", "transport", "pharmacy", "electronics", "travel", "jewelry", "gaming"]
    )
    merchant_probabilities = np.array([0.22, 0.2, 0.15, 0.13, 0.12, 0.08, 0.05, 0.05])

    amount = rng.lognormal(mean=4.2, sigma=1.0, size=ROW_COUNT).round(2)
    amount = np.clip(amount, 3, 5000)
    transaction_hour = rng.integers(0, 24, size=ROW_COUNT)
    merchant_category = rng.choice(merchant_categories, size=ROW_COUNT, p=merchant_probabilities)
    customer_age_days = rng.integers(1, 1800, size=ROW_COUNT)
    is_foreign_transaction = rng.random(ROW_COUNT) < 0.12
    is_new_merchant = rng.random(ROW_COUNT) < 0.18
    previous_chargebacks = rng.poisson(0.08, size=ROW_COUNT)
    previous_chargebacks = np.clip(previous_chargebacks, 0, 4)

    high_risk_category = np.isin(merchant_category, ["electronics", "travel", "jewelry", "gaming"])
    unusual_hour = (transaction_hour <= 5) | (transaction_hour >= 23)
    high_amount = amount > 450
    new_customer = customer_age_days < 45

    risk_score = (
        -4.2
        + 1.25 * high_amount
        + 0.95 * unusual_hour
        + 1.15 * is_foreign_transaction
        + 0.9 * is_new_merchant
        + 1.35 * (previous_chargebacks > 0)
        + 0.8 * new_customer
        + 0.55 * high_risk_category
        + 0.45 * (amount > 1000)
    )
    fraud_probability = sigmoid(risk_score)
    is_fraud = rng.random(ROW_COUNT) < fraud_probability

    data = pd.DataFrame(
        {
            "transaction_id": [f"txn_{index:06d}" for index in range(1, ROW_COUNT + 1)],
            "amount": amount,
            "transaction_hour": transaction_hour,
            "merchant_category": merchant_category,
            "customer_age_days": customer_age_days,
            "is_foreign_transaction": is_foreign_transaction,
            "is_new_merchant": is_new_merchant,
            "previous_chargebacks": previous_chargebacks,
            "is_fraud": is_fraud.astype(int),
        }
    )

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA_PATH, index=False)

    fraud_rate = data["is_fraud"].mean()
    print(f"Wrote {len(data):,} transactions to {DATA_PATH}")
    print(f"Fraud rate: {fraud_rate:.2%}")


if __name__ == "__main__":
    main()

