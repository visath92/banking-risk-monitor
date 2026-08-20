"""Combines rule points + normalized anomaly score into a single 0-100 risk
score, bucket, is_flagged boolean, and merged, human-readable reasons list.

is_flagged (== risk_bucket == "High") is the single source of truth used by
both the evaluator and the dashboard -- no other module should re-derive it.
"""
import pandas as pd

from src import config
from src.risk import anomaly_model, rules_engine

MAX_RULE_POINTS = 100  # points saturate the rule component of the score
ANOMALY_WEIGHT = 30  # anomaly score (0-1) contributes up to this many points


def _bucket_for(score_value: float) -> str:
    if score_value <= config.RISK_LOW_MAX:
        return "Low"
    if score_value <= config.RISK_MEDIUM_MAX:
        return "Medium"
    return "High"


def compute_risk_scores(features_df: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    df = rules_engine.apply_rules(features_df, accounts)

    model, scaler = anomaly_model.fit_isolation_forest(df)
    anomaly_scores = anomaly_model.score(model, scaler, df)
    anomaly_reasons = anomaly_model.explain_top_drivers(df)

    rule_component = df["rule_points"].clip(upper=MAX_RULE_POINTS)
    anomaly_component = anomaly_scores * ANOMALY_WEIGHT
    risk_score = (rule_component + anomaly_component).clip(upper=100).round(1)

    df["anomaly_score"] = anomaly_scores.round(3)
    df["risk_score"] = risk_score
    df["risk_bucket"] = risk_score.apply(_bucket_for)
    df["reasons"] = [
        rule_r + anomaly_r for rule_r, anomaly_r in zip(df["rule_reasons"], anomaly_reasons)
    ]
    df["is_flagged"] = df["risk_bucket"] == "High"

    return df
