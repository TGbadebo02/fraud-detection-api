import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st


DATABASE_PATH = Path("predictions.db")


st.set_page_config(page_title="Fraud Risk Monitor", layout="wide")
st.title("Fraud Risk Monitor")

if not DATABASE_PATH.exists():
    st.info("No predictions logged yet. Start the API and send transactions to /predict.")
    st.stop()

with sqlite3.connect(DATABASE_PATH) as connection:
    data = pd.read_sql_query(
        "SELECT transaction_id, amount, merchant_category, fraud_score, risk_level, reasons, created_at "
        "FROM predictions ORDER BY created_at DESC",
        connection,
    )

if data.empty:
    st.info("No predictions logged yet.")
    st.stop()

total_predictions = len(data)
high_risk = int((data["risk_level"] == "high").sum())
average_score = data["fraud_score"].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Predictions", total_predictions)
col2.metric("High risk", high_risk)
col3.metric("Average fraud score", f"{average_score:.2f}")

st.subheader("Recent predictions")
st.dataframe(data, use_container_width=True)

st.subheader("Risk level breakdown")
st.bar_chart(data["risk_level"].value_counts())

