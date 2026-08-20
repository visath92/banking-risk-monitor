"""Per-transaction feature engineering feeding both the rules engine and the
unsupervised anomaly model."""
from datetime import datetime

import numpy as np
import pandas as pd

NOW = datetime(2026, 8, 20, 9, 0, 0)

# Exported so risk/anomaly_model.py can reference the exact numeric columns
# used for scoring without re-deriving the list.
feature_columns = [
    "amount_zscore",
    "txn_velocity_24h",
    "hour_of_day",
    "is_weekend",
    "days_since_account_opened",
]


def build_transaction_features(transactions: pd.DataFrame, accounts: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    df = transactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    accounts_indexed = accounts.set_index("account_id")
    customers_indexed = customers.set_index("customer_id")

    df["account_status"] = df["account_id"].map(accounts_indexed["status"]).fillna("Unknown")
    df["account_opened_date"] = df["account_id"].map(accounts_indexed["opened_date"])
    df["days_since_account_opened"] = (
        df["timestamp"].dt.normalize() - pd.to_datetime(df["account_opened_date"])
    ).dt.days.fillna(0).clip(lower=0)

    df["customer_kyc_risk"] = df["customer_id"].map(customers_indexed["kyc_risk_flag"]).fillna(False)
    df["customer_risk_category"] = df["customer_id"].map(customers_indexed["risk_category"]).fillna("Unknown")

    per_customer_stats = df.groupby("customer_id")["amount"].agg(["mean", "std"]).rename(
        columns={"mean": "cust_mean_amount", "std": "cust_std_amount"}
    )
    df = df.merge(per_customer_stats, on="customer_id", how="left")
    df["cust_std_amount"] = df["cust_std_amount"].fillna(0).replace(0, 1.0)
    df["amount_zscore"] = (df["amount"] - df["cust_mean_amount"]) / df["cust_std_amount"]

    df = df.sort_values("timestamp")

    def _velocity(group: pd.DataFrame) -> pd.Series:
        counts = group.set_index("timestamp")["transaction_id"].rolling("24h").count()
        counts.index = group.index
        return counts

    velocity = df.groupby("account_id", group_keys=False).apply(_velocity, include_groups=False)
    df["txn_velocity_24h"] = velocity.reindex(df.index)

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["is_weekend"] = df["timestamp"].dt.dayofweek.isin([5, 6]).astype(int)

    return df.sort_index()
