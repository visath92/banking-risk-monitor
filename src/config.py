"""Single source of truth for paths, sizes, thresholds, and the random seed."""
import logging
import os

RANDOM_SEED = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
REFERENCE_DIR = os.path.join(RAW_DIR, "reference")
GROUND_TRUTH_DIR = os.path.join(DATA_DIR, "ground_truth")
GROUND_TRUTH_PATH = os.path.join(GROUND_TRUTH_DIR, "labels.csv")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
GOVERNANCE_DB_PATH = os.path.join(DATA_DIR, "governance.db")

# Dataset row counts, per the spec's dataset summary table.
N_CUSTOMERS = 10_000
N_ACCOUNTS = 15_000
N_TRANSACTIONS = 25_000
N_INCIDENTS = 10_000
N_API_LOGS = 15_000
N_APP_LOGS = 20_000
N_TEST_CASES = 5_000

# Per-dataset deliberate-anomaly injection rates. Kept generous enough that the
# operational dashboards (SLA breaches, slow APIs, failed tests) have enough
# positive cases to be visually and statistically meaningful.
TXN_ANOMALY_RATE = 0.03
INCIDENT_SLA_BREACH_RATE = 0.12
API_SLOW_RATE = 0.07
API_5XX_RATE = 0.07
TEST_FAIL_RATE = 0.10
CUSTOMER_KYC_RISK_RATE = 0.06
ACCOUNT_CLOSED_RATE = 0.08

# Risk score buckets (0-100 scale).
RISK_LOW_MAX = 39
RISK_MEDIUM_MAX = 69  # >= 70 is High

# IsolationForest contamination. Deliberately a separate, hand-picked modeling
# constant -- NOT derived from TXN_ANOMALY_RATE above, so the model can never
# be accused of "peeking" at the data-generation parameters.
IFOREST_CONTAMINATION = 0.05

# Operational thresholds used by both data generation (to inject realistic
# breaches) and the operations analytics modules (to detect them).
API_SLOW_THRESHOLD_MS = 2000
SLA_TARGET_MINUTES = {"Critical": 60, "High": 240, "Medium": 1440, "Low": 4320}

SERVICE_NAMES = [
    "payments-api",
    "accounts-service",
    "auth-service",
    "ledger-engine",
    "notification-service",
    "fraud-detection-service",
    "reporting-service",
    "mobile-gateway",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
