"""Data-quality checks over the raw tables.

Each validate_X function returns a list of issue dicts (record_id, issue_type,
severity, description). run_all_validations bundles them into a report the
Data Quality dashboard tab reads directly.
"""
from datetime import datetime

import pandas as pd

NOW = datetime(2026, 8, 20, 9, 0, 0)


def validate_transactions(transactions: pd.DataFrame, accounts: pd.DataFrame) -> list:
    issues = []

    dup_mask = transactions["transaction_id"].duplicated(keep=False)
    for tid in transactions.loc[dup_mask, "transaction_id"].unique():
        issues.append({"record_id": tid, "issue_type": "duplicate_transaction_id", "severity": "High",
                        "description": "transaction_id appears more than once"})

    neg_mask = transactions["amount"] < 0
    for tid in transactions.loc[neg_mask, "transaction_id"]:
        issues.append({"record_id": tid, "issue_type": "negative_amount", "severity": "High",
                        "description": "Transaction amount is negative"})

    ts = pd.to_datetime(transactions["timestamp"])
    future_mask = ts > NOW
    for tid in transactions.loc[future_mask, "transaction_id"]:
        issues.append({"record_id": tid, "issue_type": "future_date", "severity": "Medium",
                        "description": "Transaction timestamp is in the future"})

    account_owner = accounts.set_index("account_id")["customer_id"]
    mismatched = transactions["account_id"].map(account_owner) != transactions["customer_id"]
    for tid in transactions.loc[mismatched, "transaction_id"]:
        issues.append({"record_id": tid, "issue_type": "invalid_customer_relationship", "severity": "High",
                        "description": "customer_id does not match the account's actual owner"})

    account_status = accounts.set_index("account_id")["status"]
    closed_mask = transactions["account_id"].map(account_status) == "Closed"
    for tid in transactions.loc[closed_mask, "transaction_id"]:
        issues.append({"record_id": tid, "issue_type": "closed_account_activity", "severity": "High",
                        "description": "Transaction posted against a closed account"})

    orphan_mask = ~transactions["account_id"].isin(accounts["account_id"])
    for tid in transactions.loc[orphan_mask, "transaction_id"]:
        issues.append({"record_id": tid, "issue_type": "orphaned_account_reference", "severity": "High",
                        "description": "account_id does not exist in accounts table"})

    return issues


def validate_accounts(accounts: pd.DataFrame, customers: pd.DataFrame) -> list:
    issues = []
    orphan_mask = ~accounts["customer_id"].isin(customers["customer_id"])
    for aid in accounts.loc[orphan_mask, "account_id"]:
        issues.append({"record_id": aid, "issue_type": "orphaned_customer_reference", "severity": "High",
                        "description": "customer_id does not exist in customers table"})
    return issues


def validate_incidents(incidents: pd.DataFrame) -> list:
    issues = []
    bad_mask = (incidents["status"] == "Resolved") & incidents["resolved_at"].isna()
    for iid in incidents.loc[bad_mask, "incident_id"]:
        issues.append({"record_id": iid, "issue_type": "missing_resolution_timestamp", "severity": "Medium",
                        "description": "Incident marked Resolved but has no resolved_at timestamp"})
    return issues


class ValidationReport:
    def __init__(self, issues_by_dataset: dict):
        self.issues_by_dataset = issues_by_dataset

    def summary(self) -> pd.DataFrame:
        rows = []
        for dataset, issues in self.issues_by_dataset.items():
            counts = {}
            for issue in issues:
                counts[issue["issue_type"]] = counts.get(issue["issue_type"], 0) + 1
            for issue_type, count in counts.items():
                rows.append({"dataset": dataset, "issue_type": issue_type, "count": count})
        return pd.DataFrame(rows, columns=["dataset", "issue_type", "count"])

    def detail(self, dataset: str) -> pd.DataFrame:
        issues = self.issues_by_dataset.get(dataset, [])
        return pd.DataFrame(issues, columns=["record_id", "issue_type", "severity", "description"])

    def total_issues(self) -> int:
        return sum(len(issues) for issues in self.issues_by_dataset.values())


def run_all_validations(raw_data: dict) -> ValidationReport:
    issues_by_dataset = {
        "transactions": validate_transactions(raw_data["transactions"], raw_data["accounts"]),
        "accounts": validate_accounts(raw_data["accounts"], raw_data["customers"]),
        "incidents": validate_incidents(raw_data["incidents"]),
    }
    return ValidationReport(issues_by_dataset)


def assert_flagged_have_reasons(risk_df: pd.DataFrame) -> list:
    """Governance invariant: every flagged transaction must carry a non-empty
    reasons list. Returns the list of transaction_ids that violate this (should
    always be empty) rather than raising, so the pipeline can surface it as a
    governance metric instead of crashing."""
    flagged = risk_df[risk_df["is_flagged"]]
    violations = flagged[flagged["reasons"].apply(lambda r: not r)]
    return violations["transaction_id"].tolist()
