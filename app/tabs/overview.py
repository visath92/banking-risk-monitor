import streamlit as st

from app import components, palette


def render(result):
    components.governance_banner()

    risk_df = result.risk_df
    ops = result.ops_results

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        components.kpi_tile("Transactions analyzed", f"{len(risk_df):,}")
    with col2:
        pct = 100 * risk_df["is_flagged"].mean()
        components.kpi_tile("High-risk transactions", f"{int(risk_df['is_flagged'].sum()):,}", f"{pct:.1f}% of all transactions")
    with col3:
        sla_rate = 100 * ops["incidents"]["is_sla_breach"].mean()
        components.kpi_tile("SLA breach rate", f"{sla_rate:.1f}%", f"{int(ops['incidents']['is_sla_breach'].sum()):,} of {len(ops['incidents']):,} incidents")
    with col4:
        test_summary = ops["test_failures"]
        overall_fail_rate = 100 * test_summary["failed_tests"].sum() / test_summary["total_tests"].sum()
        components.kpi_tile("Test failure rate", f"{overall_fail_rate:.1f}%")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        avg_latency = ops["slow_apis"]["response_time_ms"].mean()
        components.kpi_tile("Avg API latency", f"{avg_latency:,.0f} ms")
    with col6:
        slow_rate = 100 * ops["slow_apis"]["is_slow"].mean()
        components.kpi_tile("Slow API rate", f"{slow_rate:.1f}%")
    with col7:
        err_rate = 100 * ops["api_5xx"]["is_5xx"].mean()
        components.kpi_tile("API 5xx rate", f"{err_rate:.1f}%")
    with col8:
        open_incidents = int((ops["incidents"]["status"] == "Open").sum())
        components.kpi_tile("Open incidents", f"{open_incidents:,}")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        bucket_counts = risk_df["risk_bucket"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
        fig = components.status_bar_chart(
            bucket_counts.index.tolist(), bucket_counts.values.astype(int).tolist(),
            palette.RISK_BUCKET_COLOR, "Transactions by risk bucket",
        )
        components.render_chart(fig)

    with c2:
        sev_counts = ops["incidents"]["severity"].value_counts().reindex(["Low", "Medium", "High", "Critical"]).fillna(0)
        fig = components.status_bar_chart(
            sev_counts.index.tolist(), sev_counts.values.astype(int).tolist(),
            palette.SEVERITY_COLOR, "Incidents by severity",
        )
        components.render_chart(fig)
