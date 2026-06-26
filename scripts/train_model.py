from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/transactions.csv")
MODEL_PATH = Path("models/fraud_model.pkl")


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    X = data.drop(columns=["transaction_id", "is_fraud"])
    y = data["is_fraud"]

    numeric_features = ["amount", "transaction_hour", "customer_age_days", "previous_chargebacks"]
    categorical_features = ["merchant_category", "is_foreign_transaction", "is_new_merchant"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=150,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    print(classification_report(y_test, predictions))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()

