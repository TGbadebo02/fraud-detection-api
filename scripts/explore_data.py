from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/transactions.csv")
REPORT_PATH = Path("data/fraud_pattern_summary.md")


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    fraud = data[data["is_fraud"] == 1]
    legitimate = data[data["is_fraud"] == 0]

    category_rates = (
        data.groupby("merchant_category")["is_fraud"]
        .agg(["count", "mean"])
        .sort_values("mean", ascending=False)
    )
    category_rates["mean"] = (category_rates["mean"] * 100).round(2)

    summary = [
        "# Fraud Pattern Summary",
        "",
        f"Transactions analyzed: {len(data):,}",
        f"Fraud rate: {data['is_fraud'].mean():.2%}",
        "",
        "## Numeric Patterns",
        "",
        f"Average amount, fraud: ${fraud['amount'].mean():.2f}",
        f"Average amount, legitimate: ${legitimate['amount'].mean():.2f}",
        f"Average customer age in days, fraud: {fraud['customer_age_days'].mean():.1f}",
        f"Average customer age in days, legitimate: {legitimate['customer_age_days'].mean():.1f}",
        "",
        "## Risk Signal Rates",
        "",
        f"Foreign transactions among fraud: {fraud['is_foreign_transaction'].mean():.2%}",
        f"Foreign transactions among legitimate: {legitimate['is_foreign_transaction'].mean():.2%}",
        f"New merchants among fraud: {fraud['is_new_merchant'].mean():.2%}",
        f"New merchants among legitimate: {legitimate['is_new_merchant'].mean():.2%}",
        f"Previous chargebacks among fraud: {(fraud['previous_chargebacks'] > 0).mean():.2%}",
        f"Previous chargebacks among legitimate: {(legitimate['previous_chargebacks'] > 0).mean():.2%}",
        "",
        "## Fraud Rate by Merchant Category",
        "",
        category_rates.to_markdown(),
        "",
    ]

    REPORT_PATH.write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))
    print(f"\nSaved summary to {REPORT_PATH}")


if __name__ == "__main__":
    main()

