"""SQLite-backed governance audit trail.

Human review decisions and AI flagging decisions both need to survive app
restarts and be shared across browser sessions, so they live in SQLite rather
than st.session_state (which is transient, per-session UI state only).
"""
import sqlite3
from datetime import datetime

import pandas as pd

from src import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    reviewer_name TEXT,
    note TEXT,
    reviewed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_decision_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_run_id TEXT NOT NULL,
    transaction_id TEXT NOT NULL,
    risk_score REAL NOT NULL,
    risk_bucket TEXT NOT NULL,
    reasons TEXT,
    is_flagged INTEGER NOT NULL,
    logged_at TEXT NOT NULL
);
"""


def init_db(path: str = config.GOVERNANCE_DB_PATH) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def record_review(transaction_id: str, decision: str, reviewer_name: str, note: str,
                   path: str = config.GOVERNANCE_DB_PATH) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "INSERT INTO review_log (transaction_id, decision, reviewer_name, note, reviewed_at) VALUES (?, ?, ?, ?, ?)",
            (transaction_id, decision, reviewer_name, note, datetime.now().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def record_ai_decisions(flagged_df: pd.DataFrame, pipeline_run_id: str,
                         path: str = config.GOVERNANCE_DB_PATH) -> None:
    """Logs only flagged rows, keeping the audit table bounded."""
    if flagged_df.empty:
        return
    conn = sqlite3.connect(path)
    try:
        now = datetime.now().isoformat()
        rows = [
            (pipeline_run_id, row["transaction_id"], float(row["risk_score"]), row["risk_bucket"],
             "; ".join(row["reasons"]), int(row["is_flagged"]), now)
            for _, row in flagged_df.iterrows()
        ]
        conn.executemany(
            "INSERT INTO ai_decision_log (pipeline_run_id, transaction_id, risk_score, risk_bucket, reasons, is_flagged, logged_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def get_review_log(path: str = config.GOVERNANCE_DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(path)
    try:
        return pd.read_sql_query("SELECT * FROM review_log ORDER BY reviewed_at DESC", conn)
    finally:
        conn.close()


def get_ai_decision_log(path: str = config.GOVERNANCE_DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(path)
    try:
        return pd.read_sql_query("SELECT * FROM ai_decision_log ORDER BY logged_at DESC", conn)
    finally:
        conn.close()


def review_coverage_and_override_metrics(path: str = config.GOVERNANCE_DB_PATH) -> dict:
    reviews = get_review_log(path)
    decisions = get_ai_decision_log(path)

    flagged_ids = set(decisions.loc[decisions["is_flagged"] == 1, "transaction_id"])
    reviewed_ids = set(reviews["transaction_id"])
    reviewed_flagged_ids = flagged_ids & reviewed_ids

    coverage = len(reviewed_flagged_ids) / len(flagged_ids) if flagged_ids else 0.0

    overrides = reviews[reviews["transaction_id"].isin(flagged_ids) & (reviews["decision"] == "Dismiss")]
    override_rate = len(overrides) / len(reviewed_flagged_ids) if reviewed_flagged_ids else 0.0

    return {
        "total_flagged": len(flagged_ids),
        "total_reviewed_of_flagged": len(reviewed_flagged_ids),
        "review_coverage": round(coverage, 3),
        "override_rate": round(override_rate, 3),
    }
