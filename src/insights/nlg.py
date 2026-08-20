"""Deterministic, template-based natural-language generation for explainable
insights and recommendations. No external LLM call -- fully offline and
reproducible, per the "AI detects and recommends, human decides" principle."""

_RECOMMENDATION_RULES = [
    ("Closed-account transaction", "Freeze further activity on this account and verify why it accepted a transaction while closed."),
    ("KYC-risk customer", "Verify the customer's KYC status before allowing this transaction to proceed."),
    ("Transaction customer does not match account owner", "Investigate the account/customer relationship for possible identity or data error."),
    ("High-value amount relative to customer history", "Prioritize for manual investigation given the unusually large amount."),
    ("Unusual transaction velocity (multiple rapid transactions)", "Review recent account activity for signs of automated or fraudulent bursts."),
]


def build_insight(transaction_id: str, risk_score: float, risk_bucket: str, reasons: list) -> str:
    reasons_text = "; ".join(reasons) if reasons else "No specific reasons triggered"
    return f"Transaction {transaction_id} -- {risk_bucket} Risk: {risk_score:.0f}/100. Reasons: {reasons_text}."


def build_recommendation(risk_bucket: str, reasons: list) -> str:
    if risk_bucket != "High":
        return "No action required -- transaction falls within normal risk parameters."

    actions = [action for trigger, action in _RECOMMENDATION_RULES if any(trigger in r for r in reasons)]
    if not actions:
        actions = ["Prioritize this transaction for manual investigation before further processing."]

    return "Recommendation: " + " Also, ".join(actions)


def build_incident_recommendation(is_sla_breach: bool, severity: str) -> str:
    if not is_sla_breach:
        return "No action required -- incident is within its SLA target."
    return f"Recommendation: Escalate this {severity}-severity incident immediately -- it has breached its SLA resolution target."


def build_ops_recommendation(kind: str) -> str:
    templates = {
        "slow_api": "Recommendation: Investigate this endpoint for performance regressions (e.g. DB query plans, downstream dependency latency).",
        "http_5xx": "Recommendation: Investigate recent deployments or dependency failures causing server errors on this endpoint.",
        "app_log_hotspot": "Recommendation: Prioritize root-cause analysis for this recurring error -- it is a technical hotspot.",
        "test_failure_module": "Recommendation: Review this module's test suite and recent changes -- failure rate exceeds acceptable quality threshold.",
    }
    return templates.get(kind, "Recommendation: Review this finding for further investigation.")
