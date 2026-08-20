from src.operations.incidents import detect_sla_breaches
from src.operations.api_logs import detect_slow_apis, detect_5xx_failures
from src.operations.app_logs import find_recurring_failures
from src.operations.test_cases import analyze_test_failures


def run_ops_analytics(raw_data: dict) -> dict:
    incidents_result = detect_sla_breaches(raw_data["incidents"], raw_data["sla_rules"])
    slow_apis = detect_slow_apis(raw_data["api_logs"])
    api_5xx = detect_5xx_failures(raw_data["api_logs"])
    app_log_hotspots = find_recurring_failures(raw_data["app_logs"], raw_data["error_codes"])
    test_failures = analyze_test_failures(raw_data["test_cases"])

    return {
        "incidents": incidents_result,
        "slow_apis": slow_apis,
        "api_5xx": api_5xx,
        "app_log_hotspots": app_log_hotspots,
        "test_failures": test_failures,
    }
