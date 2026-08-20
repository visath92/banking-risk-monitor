"""Orchestrates the full flow from the spec:

Banking Data -> Data Validation -> Feature Analysis -> Rules + AI Anomaly
Detection -> Risk Score -> Incident Analysis -> AI Insights ->
Recommendations -> Human Review

run_pipeline() is the single entry point used by both the CLI script and the
Streamlit app (wrapped in st.cache_data there).
"""
import os
import uuid

import pandas as pd

from src import config
from src.data_generation import generate_dataset
from src.evaluation import evaluator
from src.features.feature_engineering import build_transaction_features
from src.governance import audit_store
from src.insights import nlg
from src.operations import run_ops_analytics
from src.risk.risk_scoring import compute_risk_scores
from src.validation.validators import assert_flagged_have_reasons, run_all_validations

logger = config.get_logger(__name__)


class PipelineResult:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _raw_data_exists() -> bool:
    required = ["customers.csv", "accounts.csv", "transactions.csv", "incidents.csv",
                "api_logs.csv", "app_logs.csv", "test_cases.csv"]
    return all(os.path.exists(os.path.join(config.RAW_DIR, f)) for f in required) and os.path.exists(
        config.GROUND_TRUTH_PATH
    )


def _load_raw_data() -> dict:
    if not _raw_data_exists():
        logger.info("Raw data not found -- generating synthetic datasets...")
        generate_dataset.main()

    raw = {
        "customers": pd.read_csv(os.path.join(config.RAW_DIR, "customers.csv")),
        "accounts": pd.read_csv(os.path.join(config.RAW_DIR, "accounts.csv")),
        "transactions": pd.read_csv(os.path.join(config.RAW_DIR, "transactions.csv")),
        "incidents": pd.read_csv(os.path.join(config.RAW_DIR, "incidents.csv")),
        "api_logs": pd.read_csv(os.path.join(config.RAW_DIR, "api_logs.csv")),
        "app_logs": pd.read_csv(os.path.join(config.RAW_DIR, "app_logs.csv")),
        "test_cases": pd.read_csv(os.path.join(config.RAW_DIR, "test_cases.csv")),
        "branches": pd.read_csv(os.path.join(config.REFERENCE_DIR, "branches.csv")),
        "sla_rules": pd.read_csv(os.path.join(config.REFERENCE_DIR, "sla_rules.csv")),
        "error_codes": pd.read_csv(os.path.join(config.REFERENCE_DIR, "error_codes.csv")),
    }
    return raw


def run_pipeline() -> PipelineResult:
    run_id = str(uuid.uuid4())
    raw_data = _load_raw_data()

    logger.info("Running data validation...")
    validation_report = run_all_validations(raw_data)

    logger.info("Building transaction features...")
    features_df = build_transaction_features(raw_data["transactions"], raw_data["accounts"], raw_data["customers"])

    logger.info("Scoring transaction risk (rules + anomaly model)...")
    risk_df = compute_risk_scores(features_df, raw_data["accounts"])

    governance_violations = assert_flagged_have_reasons(risk_df)
    if governance_violations:
        logger.warning("Governance check failed: %d flagged transactions have no reasons", len(governance_violations))

    audit_store.init_db()
    audit_store.record_ai_decisions(risk_df[risk_df["is_flagged"]], run_id)

    risk_df["insight"] = [
        nlg.build_insight(row["transaction_id"], row["risk_score"], row["risk_bucket"], row["reasons"])
        for _, row in risk_df.iterrows()
    ]
    risk_df["recommendation"] = [
        nlg.build_recommendation(row["risk_bucket"], row["reasons"])
        for _, row in risk_df.iterrows()
    ]

    logger.info("Running operations analytics...")
    ops_results = run_ops_analytics(raw_data)
    ops_results["incidents"]["recommendation"] = [
        nlg.build_incident_recommendation(is_breach, severity)
        for is_breach, severity in zip(ops_results["incidents"]["is_sla_breach"], ops_results["incidents"]["severity"])
    ]

    logger.info("Evaluating against ground truth...")
    eval_results = {
        "transactions": evaluator.evaluate("transactions", risk_df, "transaction_id", "is_flagged"),
        "incidents": evaluator.evaluate("incidents", ops_results["incidents"], "incident_id", "is_sla_breach"),
        "api_logs_slow": evaluator.evaluate("api_logs", ops_results["slow_apis"], "log_id", "is_slow"),
        "api_logs_5xx": evaluator.evaluate("api_logs", ops_results["api_5xx"], "log_id", "is_5xx"),
        "test_cases": evaluator.evaluate("test_cases", raw_data["test_cases"].assign(
            is_failed=raw_data["test_cases"]["status"] == "Failed"), "test_id", "is_failed"),
    }

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    risk_df_export = risk_df.copy()
    risk_df_export["reasons"] = risk_df_export["reasons"].apply(lambda r: "; ".join(r))
    risk_df_export.to_csv(os.path.join(config.PROCESSED_DIR, "transaction_risk.csv"), index=False)
    ops_results["incidents"].to_csv(os.path.join(config.PROCESSED_DIR, "incidents.csv"), index=False)
    ops_results["slow_apis"].to_csv(os.path.join(config.PROCESSED_DIR, "api_logs.csv"), index=False)
    ops_results["app_log_hotspots"].to_csv(os.path.join(config.PROCESSED_DIR, "app_log_hotspots.csv"), index=False)
    ops_results["test_failures"].to_csv(os.path.join(config.PROCESSED_DIR, "test_failures.csv"), index=False)

    return PipelineResult(
        run_id=run_id,
        raw_data=raw_data,
        validation_report=validation_report,
        risk_df=risk_df,
        ops_results=ops_results,
        eval_results=eval_results,
        governance_violations=governance_violations,
    )
