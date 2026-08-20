import streamlit as st

from app import components
from src.operations.api_logs import endpoint_summary
from src.operations.app_logs import service_error_rates


def render(result):
    ops = result.ops_results
    raw = result.raw_data

    st.subheader("Incidents & SLA")
    breaches = ops["incidents"][ops["incidents"]["is_sla_breach"]].sort_values("opened_at", ascending=False)
    st.write(f"**{len(breaches):,}** of {len(ops['incidents']):,} incidents have breached (or are on track to breach) their SLA.")
    by_service = breaches.groupby("service_name").size().sort_values(ascending=False)
    if len(by_service):
        fig = components.ranked_bar_chart(by_service.index.tolist(), by_service.values.tolist(),
                                           "SLA breaches by service")
        components.render_chart(fig)
    st.dataframe(
        breaches[["incident_id", "service_name", "severity", "status", "resolution_minutes", "recommendation"]].head(200),
        use_container_width=True, height=260,
    )

    st.divider()
    st.subheader("API performance")
    endpoint_stats = endpoint_summary(raw["api_logs"])
    top_slow = endpoint_stats.sort_values("avg_response_ms", ascending=False).head(10)
    fig = components.ranked_bar_chart(
        [f"{s} {e}" for s, e in zip(top_slow["service_name"], top_slow["endpoint"])],
        top_slow["avg_response_ms"].round(0).tolist(),
        "Slowest endpoints (avg response ms)",
    )
    components.render_chart(fig)
    st.dataframe(endpoint_stats, use_container_width=True, height=260)

    st.divider()
    st.subheader("Application log hotspots")
    hotspots = ops["app_log_hotspots"].head(15)
    if len(hotspots):
        fig = components.ranked_bar_chart(
            [f"{s} / {c}" for s, c in zip(hotspots["service_name"], hotspots["category"])],
            hotspots["occurrence_count"].tolist(),
            "Top recurring error hotspots",
        )
        components.render_chart(fig)
    st.dataframe(hotspots, use_container_width=True, height=220)

    err_rates = service_error_rates(raw["app_logs"])
    st.caption("Error-log rate by service")
    st.dataframe(err_rates, use_container_width=True, height=220)

    st.divider()
    st.subheader("Test quality")
    test_failures = ops["test_failures"]
    fig = components.ranked_bar_chart(
        test_failures["module"].tolist(), (test_failures["failure_rate"] * 100).round(1).tolist(),
        "Test failure rate by module (%)",
    )
    components.render_chart(fig)
    st.dataframe(test_failures, use_container_width=True, height=220)
