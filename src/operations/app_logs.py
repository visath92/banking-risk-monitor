"""Recurring-failure / technical-hotspot mining over application logs.

This is aggregation-only -- "recurring pattern" is inherently a mining task,
not a per-row classification, so it is not evaluated against per-row ground
truth (unlike transactions, incidents, api_logs, test_cases).
"""
import pandas as pd


def find_recurring_failures(app_logs: pd.DataFrame, error_codes: pd.DataFrame) -> pd.DataFrame:
    df = app_logs[app_logs["level"].isin(["ERROR", "CRITICAL"])].copy()
    df = df.merge(error_codes, on="error_code", how="left")

    hotspots = df.groupby(["service_name", "error_code", "category"]).size().reset_index(name="occurrence_count")
    return hotspots.sort_values("occurrence_count", ascending=False)


def service_error_rates(app_logs: pd.DataFrame) -> pd.DataFrame:
    df = app_logs.copy()
    total = df.groupby("service_name").size().rename("total_logs")
    errors = df[df["level"].isin(["ERROR", "CRITICAL"])].groupby("service_name").size().rename("error_logs")
    summary = pd.concat([total, errors], axis=1).fillna(0)
    summary["error_rate"] = (summary["error_logs"] / summary["total_logs"]).round(3)
    return summary.reset_index().sort_values("error_rate", ascending=False)
