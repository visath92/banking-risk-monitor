"""Slow-API and HTTP 5xx detection over API execution logs."""
import pandas as pd

from src import config


def detect_slow_apis(api_logs: pd.DataFrame) -> pd.DataFrame:
    df = api_logs.copy()
    df["is_slow"] = df["response_time_ms"] > config.API_SLOW_THRESHOLD_MS
    return df


def detect_5xx_failures(api_logs: pd.DataFrame) -> pd.DataFrame:
    df = api_logs.copy()
    df["is_5xx"] = df["http_status"] >= 500
    return df


def endpoint_summary(api_logs: pd.DataFrame) -> pd.DataFrame:
    """Per-endpoint hotspot leaderboard: avg latency, p95 latency, error rate."""
    df = api_logs.copy()
    df["is_5xx"] = df["http_status"] >= 500
    df["is_slow"] = df["response_time_ms"] > config.API_SLOW_THRESHOLD_MS

    summary = df.groupby(["service_name", "endpoint"]).agg(
        request_count=("log_id", "count"),
        avg_response_ms=("response_time_ms", "mean"),
        p95_response_ms=("response_time_ms", lambda s: s.quantile(0.95)),
        slow_count=("is_slow", "sum"),
        error_5xx_count=("is_5xx", "sum"),
    ).reset_index()
    summary["error_rate"] = (summary["error_5xx_count"] / summary["request_count"]).round(3)
    return summary.sort_values("error_5xx_count", ascending=False)
