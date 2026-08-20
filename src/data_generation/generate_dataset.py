"""Synthetic dataset generator for the banking risk / production incident monitor.

Generates all raw tables the spec calls for, deliberately injects the problem
cases the spec names (duplicate transactions, negative amounts, future dates,
invalid customer relationships, closed-account activity, KYC risk, SLA
breaches, slow APIs, HTTP 5xx failures, failed tests), and writes an honest
ground-truth label file recording exactly which records were tampered with and
why. Anomaly injection and label recording always happen together so the
labels can never drift from the data.
"""
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from src import config

logger = config.get_logger(__name__)

NOW = datetime(2026, 8, 20, 9, 0, 0)

BRANCH_REGIONS = ["North", "South", "East", "West", "Central"]
ACCOUNT_TYPES = ["Savings", "Current", "Fixed Deposit", "NRI"]
TXN_TYPES = ["Debit", "Credit", "Transfer", "UPI", "NEFT", "ATM Withdrawal"]
CHANNELS = ["Mobile App", "Internet Banking", "Branch", "ATM", "POS"]
API_ENDPOINTS = [
    "/api/v1/accounts/balance",
    "/api/v1/transactions/create",
    "/api/v1/transactions/list",
    "/api/v1/customers/kyc",
    "/api/v1/auth/login",
    "/api/v1/ledger/post",
    "/api/v1/reports/generate",
    "/api/v1/notifications/send",
]
TEST_MODULES = ["Payments", "Accounts", "Auth", "Ledger", "Notifications", "Reporting", "Fraud", "KYC"]
ERROR_CATEGORIES = ["Timeout", "NullReference", "DBConnection", "ValidationError", "AuthFailure", "OutOfMemory"]


def _seed_everything():
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    Faker.seed(config.RANDOM_SEED)


def _rng():
    return np.random.default_rng(config.RANDOM_SEED)


def generate_reference_data():
    branches = pd.DataFrame(
        {
            "branch_id": [f"BR{idx:03d}" for idx in range(1, 51)],
            "branch_name": [f"{region} Branch {idx}" for idx, region in
                             zip(range(1, 51), (BRANCH_REGIONS * 10)[:50])],
            "region": (BRANCH_REGIONS * 10)[:50],
        }
    )

    sla_rows = [
        {"severity": sev, "target_resolution_minutes": mins, "description": f"{sev} severity incidents"}
        for sev, mins in config.SLA_TARGET_MINUTES.items()
    ]
    sla_rules = pd.DataFrame(sla_rows)

    error_codes = pd.DataFrame(
        {
            "error_code": [f"ERR-{idx:03d}" for idx in range(1, len(ERROR_CATEGORIES) * 3 + 1)],
            "category": (ERROR_CATEGORIES * 3)[: len(ERROR_CATEGORIES) * 3],
            "description": [f"{cat} related failure" for cat in (ERROR_CATEGORIES * 3)[: len(ERROR_CATEGORIES) * 3]],
        }
    )
    return branches, sla_rules, error_codes


def generate_customers(fake: Faker, branches: pd.DataFrame):
    rng = _rng()
    n = config.N_CUSTOMERS
    customer_ids = [f"CUST{idx:06d}" for idx in range(1, n + 1)]
    kyc_risk_flag = rng.random(n) < config.CUSTOMER_KYC_RISK_RATE

    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "name": [fake.name() for _ in range(n)],
            "dob": [fake.date_of_birth(minimum_age=18, maximum_age=85) for _ in range(n)],
            "city": [fake.city() for _ in range(n)],
            "branch_id": rng.choice(branches["branch_id"], size=n),
            "customer_since": [fake.date_between(start_date="-15y", end_date="-1y") for _ in range(n)],
            "kyc_risk_flag": kyc_risk_flag,
            "risk_category": np.where(kyc_risk_flag, "High", rng.choice(["Low", "Medium"], size=n, p=[0.7, 0.3])),
        }
    )

    labels = [
        {"dataset": "customers", "record_id": cid, "is_anomaly": True,
         "anomaly_type": "kyc_risk", "injected_reason": "Customer flagged as KYC risk"}
        for cid in df.loc[kyc_risk_flag, "customer_id"]
    ]
    return df, labels


def generate_accounts(fake: Faker, customers: pd.DataFrame, branches: pd.DataFrame):
    rng = _rng()
    n = config.N_ACCOUNTS
    account_ids = [f"ACC{idx:06d}" for idx in range(1, n + 1)]
    owner_customer_ids = rng.choice(customers["customer_id"], size=n)
    status = rng.choice(
        ["Active", "Closed", "Dormant"], size=n,
        p=[1 - config.ACCOUNT_CLOSED_RATE - 0.03, config.ACCOUNT_CLOSED_RATE, 0.03],
    )
    opened_dates = [fake.date_between(start_date="-15y", end_date="-30d") for _ in range(n)]

    df = pd.DataFrame(
        {
            "account_id": account_ids,
            "customer_id": owner_customer_ids,
            "branch_id": rng.choice(branches["branch_id"], size=n),
            "account_type": rng.choice(ACCOUNT_TYPES, size=n),
            "status": status,
            "opened_date": opened_dates,
            "balance": np.round(rng.exponential(scale=75_000, size=n), 2),
        }
    )
    return df, []


def _inject_transaction_anomalies(df: pd.DataFrame, accounts: pd.DataFrame, customers: pd.DataFrame, rng):
    labels = []
    n = len(df)
    account_owner = accounts.set_index("account_id")["customer_id"]
    closed_accounts = accounts.loc[accounts["status"] == "Closed", "account_id"].to_numpy()
    kyc_risk_customers = customers.loc[customers["kyc_risk_flag"], "customer_id"].to_numpy()
    active_accounts_by_customer = (
        accounts[accounts["status"] == "Active"]
        .drop_duplicates("customer_id")
        .set_index("customer_id")["account_id"]
    )

    n_anomalous = int(n * config.TXN_ANOMALY_RATE)
    anomaly_types = [
        "duplicate_transaction_id", "negative_amount", "future_date",
        "invalid_customer_relationship", "closed_account_activity",
        "kyc_high_value", "unusual_velocity",
    ]
    per_type = max(1, n_anomalous // len(anomaly_types))

    all_idx = rng.choice(n, size=min(n, per_type * len(anomaly_types)), replace=False)
    slices = np.array_split(all_idx, len(anomaly_types))
    extra_rows = []

    for anomaly_type, idx in zip(anomaly_types, slices):
        idx = idx.tolist()
        if not idx:
            continue

        if anomaly_type == "duplicate_transaction_id":
            for i in idx:
                dup = df.iloc[i].copy()
                extra_rows.append(dup)
                labels.append({"dataset": "transactions", "record_id": dup["transaction_id"], "is_anomaly": True,
                                "anomaly_type": "duplicate_transaction_id",
                                "injected_reason": "Duplicate transaction_id inserted"})

        elif anomaly_type == "negative_amount":
            df.loc[df.index[idx], "amount"] = -np.abs(df.loc[df.index[idx], "amount"])
            for i in idx:
                labels.append({"dataset": "transactions", "record_id": df.iloc[i]["transaction_id"],
                                "is_anomaly": True, "anomaly_type": "negative_amount",
                                "injected_reason": "Amount set negative"})

        elif anomaly_type == "future_date":
            future_offsets = rng.integers(1, 90, size=len(idx))
            new_dates = [NOW + timedelta(days=int(off)) for off in future_offsets]
            df.loc[df.index[idx], "timestamp"] = new_dates
            for i in idx:
                labels.append({"dataset": "transactions", "record_id": df.iloc[i]["transaction_id"],
                                "is_anomaly": True, "anomaly_type": "future_date",
                                "injected_reason": "Transaction timestamp set in the future"})

        elif anomaly_type == "invalid_customer_relationship":
            for i in idx:
                acc_id = df.iloc[i]["account_id"]
                true_owner = account_owner.get(acc_id)
                wrong_customer = rng.choice(customers["customer_id"])
                tries = 0
                while wrong_customer == true_owner and tries < 5:
                    wrong_customer = rng.choice(customers["customer_id"])
                    tries += 1
                df.iloc[i, df.columns.get_loc("customer_id")] = wrong_customer
                labels.append({"dataset": "transactions", "record_id": df.iloc[i]["transaction_id"],
                                "is_anomaly": True, "anomaly_type": "invalid_customer_relationship",
                                "injected_reason": "customer_id does not match account owner"})

        elif anomaly_type == "closed_account_activity" and len(closed_accounts) > 0:
            chosen_accounts = rng.choice(closed_accounts, size=len(idx))
            for pos, i in enumerate(idx):
                acc_id = chosen_accounts[pos]
                df.iloc[i, df.columns.get_loc("account_id")] = acc_id
                df.iloc[i, df.columns.get_loc("customer_id")] = account_owner.get(acc_id)
                labels.append({"dataset": "transactions", "record_id": df.iloc[i]["transaction_id"],
                                "is_anomaly": True, "anomaly_type": "closed_account_activity",
                                "injected_reason": "Transaction posted on a closed account"})

        elif anomaly_type == "kyc_high_value" and len(kyc_risk_customers) > 0:
            chosen_customers = rng.choice(kyc_risk_customers, size=len(idx))
            for pos, i in enumerate(idx):
                cust_id = chosen_customers[pos]
                acc_id = active_accounts_by_customer.get(cust_id)
                if acc_id is None:
                    continue
                df.iloc[i, df.columns.get_loc("customer_id")] = cust_id
                df.iloc[i, df.columns.get_loc("account_id")] = acc_id
                df.iloc[i, df.columns.get_loc("amount")] = rng.uniform(500_000, 2_000_000)
                labels.append({"dataset": "transactions", "record_id": df.iloc[i]["transaction_id"],
                                "is_anomaly": True, "anomaly_type": "kyc_high_value",
                                "injected_reason": "High-value transaction by a KYC-risk customer"})

        elif anomaly_type == "unusual_velocity":
            groups = np.array_split(idx, max(1, len(idx) // 4))
            for group in groups:
                if len(group) == 0:
                    continue
                acc_id = df.iloc[group[0]]["account_id"]
                base_time = df.iloc[group[0]]["timestamp"]
                for pos, i in enumerate(group):
                    df.iloc[i, df.columns.get_loc("account_id")] = acc_id
                    df.iloc[i, df.columns.get_loc("customer_id")] = account_owner.get(acc_id)
                    df.iloc[i, df.columns.get_loc("timestamp")] = base_time + timedelta(minutes=pos * 2)
                    df.iloc[i, df.columns.get_loc("amount")] = rng.uniform(20_000, 150_000)
                    labels.append({"dataset": "transactions", "record_id": df.iloc[i]["transaction_id"],
                                    "is_anomaly": True, "anomaly_type": "unusual_velocity",
                                    "injected_reason": "Burst of rapid transactions on the same account"})

    if extra_rows:
        df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)
    return df, labels


def generate_transactions(fake: Faker, accounts: pd.DataFrame, customers: pd.DataFrame):
    rng = _rng()
    n = config.N_TRANSACTIONS
    transaction_ids = [f"TX{idx:07d}" for idx in range(1, n + 1)]
    # Normal transactions only ever occur on non-closed accounts -- closed-account
    # activity is reserved entirely for the deliberate injection step below, so it
    # stays a genuine anomaly signal instead of an ~8% baseline false-positive rate.
    eligible_accounts = accounts.loc[accounts["status"] != "Closed", "account_id"]
    chosen_accounts = rng.choice(eligible_accounts, size=n)
    account_owner = accounts.set_index("account_id")["customer_id"]
    timestamps = [NOW - timedelta(minutes=int(m)) for m in rng.integers(0, 60 * 24 * 365, size=n)]

    df = pd.DataFrame(
        {
            "transaction_id": transaction_ids,
            "account_id": chosen_accounts,
            "customer_id": [account_owner.get(a) for a in chosen_accounts],
            "amount": np.round(rng.exponential(scale=8_000, size=n) + 100, 2),
            "txn_type": rng.choice(TXN_TYPES, size=n),
            "channel": rng.choice(CHANNELS, size=n),
            "timestamp": timestamps,
        }
    )

    df, labels = _inject_transaction_anomalies(df, accounts, customers, rng)
    return df, labels


def generate_incidents(sla_rules: pd.DataFrame):
    rng = _rng()
    n = config.N_INCIDENTS
    severities = rng.choice(list(config.SLA_TARGET_MINUTES.keys()), size=n, p=[0.1, 0.2, 0.35, 0.35])
    opened_at = [NOW - timedelta(minutes=int(m)) for m in rng.integers(0, 60 * 24 * 180, size=n)]

    breach_mask = rng.random(n) < config.INCIDENT_SLA_BREACH_RATE
    resolution_minutes = np.empty(n)
    for i in range(n):
        target = config.SLA_TARGET_MINUTES[severities[i]]
        if breach_mask[i]:
            resolution_minutes[i] = rng.uniform(target * 1.2, target * 4)
        else:
            resolution_minutes[i] = rng.uniform(target * 0.1, target * 0.95)

    still_open = rng.random(n) < 0.05
    resolved_at = [
        None if still_open[i] else opened_at[i] + timedelta(minutes=float(resolution_minutes[i]))
        for i in range(n)
    ]
    breach_mask = breach_mask | still_open

    df = pd.DataFrame(
        {
            "incident_id": [f"INC{idx:06d}" for idx in range(1, n + 1)],
            "service_name": rng.choice(config.SERVICE_NAMES, size=n),
            "severity": severities,
            "opened_at": opened_at,
            "resolved_at": resolved_at,
            "resolution_minutes": np.where(still_open, np.nan, resolution_minutes),
            "status": np.where(still_open, "Open", "Resolved"),
            "description": rng.choice(
                ["Service outage", "Latency spike", "Data sync failure", "Failed deployment",
                 "Database connection errors", "Memory leak"], size=n,
            ),
        }
    )

    labels = [
        {"dataset": "incidents", "record_id": inc_id, "is_anomaly": True, "anomaly_type": "sla_breach",
         "injected_reason": "Resolution time exceeded SLA target for severity"}
        for inc_id in df.loc[breach_mask, "incident_id"]
    ]
    return df, labels


def generate_api_logs():
    rng = _rng()
    n = config.N_API_LOGS
    timestamps = [NOW - timedelta(minutes=int(m)) for m in rng.integers(0, 60 * 24 * 60, size=n)]

    response_time_ms = rng.gamma(shape=2.0, scale=150, size=n) + 50
    http_status = rng.choice([200, 200, 200, 201, 400, 404], size=n)

    slow_mask = rng.random(n) < config.API_SLOW_RATE
    response_time_ms[slow_mask] = rng.uniform(config.API_SLOW_THRESHOLD_MS * 1.1, config.API_SLOW_THRESHOLD_MS * 4,
                                               size=slow_mask.sum())

    fivexx_mask = rng.random(n) < config.API_5XX_RATE
    http_status = http_status.astype(object)
    http_status[fivexx_mask] = rng.choice([500, 502, 503], size=fivexx_mask.sum())

    df = pd.DataFrame(
        {
            "log_id": [f"API{idx:07d}" for idx in range(1, n + 1)],
            "service_name": rng.choice(config.SERVICE_NAMES, size=n),
            "endpoint": rng.choice(API_ENDPOINTS, size=n),
            "timestamp": timestamps,
            "response_time_ms": np.round(response_time_ms, 1),
            "http_status": http_status.astype(int),
        }
    )

    labels = []
    for log_id in df.loc[slow_mask, "log_id"]:
        labels.append({"dataset": "api_logs", "record_id": log_id, "is_anomaly": True, "anomaly_type": "slow_api",
                        "injected_reason": "Response time exceeded slow-API threshold"})
    for log_id in df.loc[fivexx_mask, "log_id"]:
        labels.append({"dataset": "api_logs", "record_id": log_id, "is_anomaly": True, "anomaly_type": "http_5xx",
                        "injected_reason": "HTTP 5xx server error response"})
    return df, labels


def generate_app_logs(fake: Faker, error_codes: pd.DataFrame):
    rng = _rng()
    n = config.N_APP_LOGS
    timestamps = [NOW - timedelta(minutes=int(m)) for m in rng.integers(0, 60 * 24 * 60, size=n)]
    levels = rng.choice(["INFO", "WARNING", "ERROR", "CRITICAL"], size=n, p=[0.7, 0.15, 0.12, 0.03])
    error_code = np.where(
        np.isin(levels, ["ERROR", "CRITICAL"]),
        rng.choice(error_codes["error_code"], size=n),
        None,
    )

    df = pd.DataFrame(
        {
            "log_id": [f"APP{idx:07d}" for idx in range(1, n + 1)],
            "service_name": rng.choice(config.SERVICE_NAMES, size=n),
            "timestamp": timestamps,
            "level": levels,
            "error_code": error_code,
            "message": [
                f"{lvl} in service" if ec is None else f"{lvl}: {ec} occurred during request processing"
                for lvl, ec in zip(levels, error_code)
            ],
        }
    )
    # Application logs are used for aggregate hotspot mining, not per-row
    # ground-truth evaluation -- "recurring pattern" isn't a per-row label.
    return df, []


def generate_test_cases():
    rng = _rng()
    n = config.N_TEST_CASES
    executed_at = [NOW - timedelta(minutes=int(m)) for m in rng.integers(0, 60 * 24 * 30, size=n)]
    failed_mask = rng.random(n) < config.TEST_FAIL_RATE
    status = np.where(failed_mask, "Failed", "Passed")

    df = pd.DataFrame(
        {
            "test_id": [f"TC{idx:06d}" for idx in range(1, n + 1)],
            "module": rng.choice(TEST_MODULES, size=n),
            "test_name": [f"test_{i}" for i in range(1, n + 1)],
            "executed_at": executed_at,
            "status": status,
            "duration_ms": np.round(rng.gamma(shape=2.0, scale=400, size=n), 1),
            "failure_reason": np.where(
                failed_mask,
                rng.choice(["AssertionError", "Timeout", "Environment error", "Data mismatch"], size=n),
                None,
            ),
        }
    )

    labels = [
        {"dataset": "test_cases", "record_id": tid, "is_anomaly": True, "anomaly_type": "test_failure",
         "injected_reason": "Test case failed"}
        for tid in df.loc[failed_mask, "test_id"]
    ]
    return df, labels


def main():
    _seed_everything()
    fake = Faker()
    Faker.seed(config.RANDOM_SEED)

    os.makedirs(config.RAW_DIR, exist_ok=True)
    os.makedirs(config.REFERENCE_DIR, exist_ok=True)
    os.makedirs(config.GROUND_TRUTH_DIR, exist_ok=True)

    logger.info("Generating reference data...")
    branches, sla_rules, error_codes = generate_reference_data()
    branches.to_csv(os.path.join(config.REFERENCE_DIR, "branches.csv"), index=False)
    sla_rules.to_csv(os.path.join(config.REFERENCE_DIR, "sla_rules.csv"), index=False)
    error_codes.to_csv(os.path.join(config.REFERENCE_DIR, "error_codes.csv"), index=False)

    all_labels = []

    logger.info("Generating customers...")
    customers, labels = generate_customers(fake, branches)
    all_labels += labels
    customers.to_csv(os.path.join(config.RAW_DIR, "customers.csv"), index=False)

    logger.info("Generating accounts...")
    accounts, labels = generate_accounts(fake, customers, branches)
    all_labels += labels
    accounts.to_csv(os.path.join(config.RAW_DIR, "accounts.csv"), index=False)

    logger.info("Generating transactions...")
    transactions, labels = generate_transactions(fake, accounts, customers)
    all_labels += labels
    transactions.to_csv(os.path.join(config.RAW_DIR, "transactions.csv"), index=False)

    logger.info("Generating incidents...")
    incidents, labels = generate_incidents(sla_rules)
    all_labels += labels
    incidents.to_csv(os.path.join(config.RAW_DIR, "incidents.csv"), index=False)

    logger.info("Generating API logs...")
    api_logs, labels = generate_api_logs()
    all_labels += labels
    api_logs.to_csv(os.path.join(config.RAW_DIR, "api_logs.csv"), index=False)

    logger.info("Generating application logs...")
    app_logs, labels = generate_app_logs(fake, error_codes)
    all_labels += labels
    app_logs.to_csv(os.path.join(config.RAW_DIR, "app_logs.csv"), index=False)

    logger.info("Generating test cases...")
    test_cases, labels = generate_test_cases()
    all_labels += labels
    test_cases.to_csv(os.path.join(config.RAW_DIR, "test_cases.csv"), index=False)

    labels_df = pd.DataFrame(all_labels, columns=["dataset", "record_id", "is_anomaly", "anomaly_type", "injected_reason"])
    labels_df.to_csv(config.GROUND_TRUTH_PATH, index=False)

    logger.info("Done. Row counts: customers=%d accounts=%d transactions=%d incidents=%d api_logs=%d app_logs=%d test_cases=%d ground_truth_labels=%d",
                len(customers), len(accounts), len(transactions), len(incidents), len(api_logs), len(app_logs),
                len(test_cases), len(labels_df))


if __name__ == "__main__":
    main()
