"""SLA breach detection for production incidents."""
from datetime import datetime

import pandas as pd

from src import config

NOW = datetime(2026, 8, 20, 9, 0, 0)


def detect_sla_breaches(incidents: pd.DataFrame, sla_rules: pd.DataFrame) -> pd.DataFrame:
    df = incidents.copy()
    target = df["severity"].map(config.SLA_TARGET_MINUTES)

    still_open = df["status"] == "Open"
    minutes_open = (NOW - pd.to_datetime(df["opened_at"])).dt.total_seconds() / 60
    breach_while_open = still_open & (minutes_open > target)
    breach_while_resolved = (~still_open) & (df["resolution_minutes"] > target)
    df["is_sla_breach"] = breach_while_open | breach_while_resolved

    df["reasons"] = [
        ["Incident still open past SLA target for its severity"] if o
        else ["Resolution time exceeded SLA target for its severity"] if r
        else []
        for o, r in zip(breach_while_open, breach_while_resolved)
    ]
    return df
