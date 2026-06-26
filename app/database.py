import sqlite3
from pathlib import Path

from app.schemas import PredictionResponse, Transaction


DATABASE_PATH = Path("predictions.db")


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DATABASE_PATH)


def init_db() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT NOT NULL,
                amount REAL NOT NULL,
                merchant_category TEXT NOT NULL,
                fraud_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                reasons TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def log_prediction(transaction: Transaction, prediction: PredictionResponse) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO predictions (
                transaction_id,
                amount,
                merchant_category,
                fraud_score,
                risk_level,
                reasons
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                transaction.transaction_id,
                transaction.amount,
                transaction.merchant_category,
                prediction.fraud_score,
                prediction.risk_level,
                ", ".join(prediction.reasons),
            ),
        )

