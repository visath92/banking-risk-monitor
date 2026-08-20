"""Test-failure / quality-issue analysis over software test execution data."""
import pandas as pd


def analyze_test_failures(test_cases: pd.DataFrame) -> pd.DataFrame:
    df = test_cases.copy()
    df["is_failed"] = df["status"] == "Failed"

    summary = df.groupby("module").agg(
        total_tests=("test_id", "count"),
        failed_tests=("is_failed", "sum"),
    ).reset_index()
    summary["failure_rate"] = (summary["failed_tests"] / summary["total_tests"]).round(3)
    return summary.sort_values("failure_rate", ascending=False)
