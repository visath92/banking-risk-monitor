"""Declarative business rules for transaction risk.

Each rule is a (name, predicate, points, reason) tuple. predicate takes the
whole features dataframe and returns a boolean mask. apply_rules vectorizes
evaluation and attaches rule_points + rule_reasons to the dataframe.

Point values are split into two tiers: "major" red flags (each strong enough
to cross the High-risk threshold on its own -- closed-account activity,
negative amounts, future-dated transactions, duplicate IDs, invalid
customer/account relationships, and velocity bursts are all unambiguous
problems by themselves) and "contributing" signals (high value, KYC risk,
odd hours) that only push a transaction into High risk when combined with
something else, matching the spec's own example of several contributing
factors compounding into one high-risk case.
"""
from datetime import datetime

import pandas as pd

NOW = datetime(2026, 8, 20, 9, 0, 0)

HIGH_VALUE_ZSCORE_THRESHOLD = 3.0
HIGH_VALUE_ABSOLUTE_THRESHOLD = 500_000
HIGH_VELOCITY_THRESHOLD = 3


def _rule_closed_account(df):
    return df["account_status"] == "Closed", 75, "Closed-account transaction"


def _rule_negative_amount(df):
    return df["amount"] < 0, 75, "Negative transaction amount"


def _rule_future_date(df):
    return pd.to_datetime(df["timestamp"]) > NOW, 72, "Transaction dated in the future"


def _rule_duplicate_id(df):
    return df["transaction_id"].duplicated(keep=False), 75, "Duplicate transaction ID"


def _rule_invalid_relationship(df):
    return df["invalid_customer_relationship"].astype(bool), 75, "Transaction customer does not match account owner"


def _rule_high_velocity(df):
    return df["txn_velocity_24h"] >= HIGH_VELOCITY_THRESHOLD, 72, "Unusual transaction velocity (burst of rapid transactions)"


def _rule_high_value(df):
    mask = (df["amount"] >= HIGH_VALUE_ABSOLUTE_THRESHOLD) | (df["amount_zscore"] >= HIGH_VALUE_ZSCORE_THRESHOLD)
    return mask, 40, "High-value amount relative to customer history"


def _rule_kyc_risk(df):
    return df["customer_kyc_risk"].astype(bool), 32, "KYC-risk customer"


def _rule_odd_hours(df):
    return df["hour_of_day"].isin([0, 1, 2, 3]), 10, "Transaction occurred at an unusual hour"


RULES = [
    ("closed_account", _rule_closed_account),
    ("negative_amount", _rule_negative_amount),
    ("future_date", _rule_future_date),
    ("duplicate_id", _rule_duplicate_id),
    ("invalid_relationship", _rule_invalid_relationship),
    ("high_velocity", _rule_high_velocity),
    ("high_value", _rule_high_value),
    ("kyc_risk", _rule_kyc_risk),
    ("odd_hours", _rule_odd_hours),
]


def apply_rules(features_df: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    df = features_df.copy()

    account_owner = accounts.set_index("account_id")["customer_id"]
    df["invalid_customer_relationship"] = df["account_id"].map(account_owner) != df["customer_id"]

    rule_points = pd.Series(0, index=df.index)
    rule_reasons = pd.Series([[] for _ in range(len(df))], index=df.index)

    for _, rule_fn in RULES:
        mask, points, reason = rule_fn(df)
        rule_points = rule_points + mask.astype(int) * points
        for idx in df.index[mask]:
            rule_reasons.at[idx] = rule_reasons.at[idx] + [reason]

    df["rule_points"] = rule_points
    df["rule_reasons"] = rule_reasons
    return df
