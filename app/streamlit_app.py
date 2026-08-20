import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.tabs import data_quality, governance, operations, overview, transaction_risk
from src import config
from src.pipeline import run_pipeline

st.set_page_config(page_title="Banking Risk & Incident Monitor", layout="wide")


def _raw_file_mtimes() -> tuple:
    files = [
        os.path.join(config.RAW_DIR, f)
        for f in ["customers.csv", "accounts.csv", "transactions.csv", "incidents.csv",
                  "api_logs.csv", "app_logs.csv", "test_cases.csv"]
    ]
    return tuple(os.path.getmtime(f) if os.path.exists(f) else 0 for f in files)


@st.cache_data(show_spinner="Running pipeline: validation, feature analysis, risk scoring, operations analytics...")
def _cached_pipeline(_mtimes):
    return run_pipeline()


def main():
    st.title("AI-Powered Banking Risk & Production Incident Monitor")
    st.caption(
        "Banking Data -> Data Validation -> Feature Analysis -> Rules + AI Anomaly Detection -> "
        "Risk Score -> Incident Analysis -> AI Insights -> Recommendations -> Human Review"
    )

    result = _cached_pipeline(_raw_file_mtimes())

    tabs = st.tabs(["Overview", "Transaction Risk", "Operations", "Data Quality", "Governance & Evaluation"])
    with tabs[0]:
        overview.render(result)
    with tabs[1]:
        transaction_risk.render(result)
    with tabs[2]:
        operations.render(result)
    with tabs[3]:
        data_quality.render(result)
    with tabs[4]:
        governance.render(result)


if __name__ == "__main__":
    main()
