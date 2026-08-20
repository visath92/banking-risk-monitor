"""Unsupervised anomaly detection (IsolationForest) that catches multivariate
outliers the explicit rules miss. Never trained or evaluated on ground truth.

Explainability without SHAP: for each row IsolationForest flags, report the
1-2 feature columns with the largest |z-score| as "drivers" so every anomaly
still comes with a plain-language reason.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src import config
from src.features.feature_engineering import feature_columns

_DRIVER_LABELS = {
    "amount_zscore": "unusually large amount vs. this customer's history",
    "txn_velocity_24h": "unusually high transaction velocity for this account",
    "hour_of_day": "unusual time-of-day pattern",
    "is_weekend": "unusual weekend activity pattern",
    "days_since_account_opened": "unusual account-age pattern",
}


def fit_isolation_forest(features_df: pd.DataFrame) -> tuple:
    X = features_df[feature_columns].fillna(0).to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=config.IFOREST_CONTAMINATION,
        random_state=config.RANDOM_SEED,
        n_estimators=200,
    )
    model.fit(X_scaled)
    return model, scaler


def score(model: IsolationForest, scaler: StandardScaler, features_df: pd.DataFrame) -> pd.Series:
    X = features_df[feature_columns].fillna(0).to_numpy()
    X_scaled = scaler.transform(X)
    raw_scores = -model.score_samples(X_scaled)  # higher = more anomalous
    normalized = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
    return pd.Series(normalized, index=features_df.index)


def explain_top_drivers(features_df: pd.DataFrame, top_n: int = 2) -> pd.Series:
    zscores = features_df[feature_columns].apply(
        lambda col: (col - col.mean()) / (col.std(ddof=0) or 1.0)
    )
    abs_z = zscores.abs()

    reasons = []
    for idx in features_df.index:
        top_cols = abs_z.loc[idx].sort_values(ascending=False).head(top_n).index
        reasons.append([
            f"Statistical outlier vs. peer group -- {_DRIVER_LABELS.get(col, col)}"
            for col in top_cols if abs_z.loc[idx, col] >= 1.5
        ])
    return pd.Series(reasons, index=features_df.index)
