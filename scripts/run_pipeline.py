#!/usr/bin/env python3
"""CLI: run the full pipeline standalone and print a summary. Useful as a fast
sanity check before ever touching Streamlit."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_pipeline


def main():
    result = run_pipeline()

    print("\n=== Row counts ===")
    for name, df in result.raw_data.items():
        print(f"  {name}: {len(df)}")

    print("\n=== Data quality issues ===")
    print(result.validation_report.summary().to_string(index=False))
    print(f"  total issues: {result.validation_report.total_issues()}")

    print("\n=== Transaction risk ===")
    bucket_counts = result.risk_df["risk_bucket"].value_counts()
    print(bucket_counts.to_string())
    print(f"  flagged (High): {int(result.risk_df['is_flagged'].sum())}")
    print(f"  governance violations (flagged w/ no reasons): {len(result.governance_violations)}")

    print("\n=== Operations ===")
    print(f"  SLA breaches: {int(result.ops_results['incidents']['is_sla_breach'].sum())} / {len(result.ops_results['incidents'])}")
    print(f"  Slow API calls: {int(result.ops_results['slow_apis']['is_slow'].sum())} / {len(result.ops_results['slow_apis'])}")
    print(f"  5xx API calls: {int(result.ops_results['api_5xx']['is_5xx'].sum())} / {len(result.ops_results['api_5xx'])}")
    print(f"  App log hotspot rows: {len(result.ops_results['app_log_hotspots'])}")
    print(f"  Test modules analyzed: {len(result.ops_results['test_failures'])}")

    print("\n=== Evaluation vs. ground truth ===")
    for name, eval_result in result.eval_results.items():
        d = eval_result.as_dict()
        print(f"  {name}: precision={d['precision']} recall={d['recall']} f1={d['f1']} "
              f"(tp={d['true_positives']} fp={d['false_positives']} fn={d['false_negatives']})")
        if not eval_result.by_type.empty:
            print(eval_result.by_type.to_string(index=False))

    print("\nSample high-risk transaction insight:")
    high_risk = result.risk_df[result.risk_df["is_flagged"]]
    if len(high_risk):
        row = high_risk.iloc[0]
        print(f"  {row['insight']}")
        print(f"  {row['recommendation']}")


if __name__ == "__main__":
    main()
