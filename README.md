# AI-Powered Banking Risk & Production Incident Monitor

An explainable Streamlit dashboard that ingests synthetic banking/production
data and flags risky transactions, SLA breaches, slow APIs, HTTP failures,
recurring application errors, and failed tests -- with a stated reason and
recommendation for every flag, and a human-in-the-loop review workflow. The
AI detects and recommends; a human makes the final decision.

## Pipeline

```
Banking Data -> Data Validation -> Feature Analysis -> Rules + AI Anomaly
Detection -> Risk Score -> Incident Analysis -> AI Insights ->
Recommendations -> Human Review
```

- **Rules engine** (`src/risk/rules_engine.py`): fully explainable business
  rules (closed-account activity, negative amounts, future dates, duplicate
  IDs, invalid customer/account relationships, high value, KYC risk,
  unusual velocity, odd hours).
- **Anomaly model** (`src/risk/anomaly_model.py`): unsupervised
  IsolationForest catches multivariate outliers the rules miss, explained via
  the largest per-feature z-score drivers -- no external LLM calls anywhere,
  the app is fully offline and reproducible.
- **Operations analytics** (`src/operations/`): SLA breach detection, slow
  API / HTTP 5xx detection, recurring application-error hotspot mining, and
  test-failure analysis.
- **Evaluation** (`src/evaluation/evaluator.py`): every detector is checked
  against a ground-truth label file written once at data-generation time and
  never read by the detection code, so precision/recall/F1 reflect genuine
  detection quality.
- **Governance** (`src/governance/audit_store.py`): a SQLite audit trail of
  every AI flagging decision and every human review decision, plus an
  automated "every flag has a reason" explainability check and human review
  coverage/override metrics.

## Setup

```bash
pip install -r requirements.txt
```

## Generate the synthetic datasets

No real data is required -- everything is generated locally:

```bash
python scripts/generate_data.py
```

This writes `data/raw/*.csv`, `data/raw/reference/*.csv`, and
`data/ground_truth/labels.csv` (10k customers, 15k accounts, 25k
transactions, 10k incidents, 15k API logs, 20k app logs, 5k test cases, plus
reference data), deliberately injecting duplicate transactions, negative
amounts, future dates, invalid customer relationships, closed-account
activity, KYC risk, SLA breaches, slow APIs, HTTP 5xx failures, and failed
tests.

## Run the pipeline standalone (sanity check)

```bash
python scripts/run_pipeline.py
```

Prints row counts, data-quality issues, risk bucket counts, operations
summaries, and precision/recall/F1 against ground truth.

## Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

Datasets are generated automatically on first run if `data/raw` is empty.
