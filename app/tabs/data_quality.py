import streamlit as st

from app import components


def render(result):
    report = result.validation_report

    st.write(f"**{report.total_issues():,}** data-quality issues detected across the raw datasets.")

    summary = report.summary()
    if not summary.empty:
        by_type = summary.groupby("issue_type")["count"].sum().sort_values(ascending=False)
        fig = components.ranked_bar_chart(by_type.index.tolist(), by_type.values.tolist(), "Issues by type")
        components.render_chart(fig)

    for dataset in report.issues_by_dataset:
        detail = report.detail(dataset)
        if detail.empty:
            continue
        with st.expander(f"{dataset} -- {len(detail)} issue(s)"):
            st.dataframe(detail, use_container_width=True, height=260)
