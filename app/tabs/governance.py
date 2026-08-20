import streamlit as st

from app import components
from src.governance import audit_store


def render(result):
    components.governance_banner()

    st.subheader("Human review coverage")
    metrics = audit_store.review_coverage_and_override_metrics()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        components.kpi_tile("Flagged transactions (all runs logged)", f"{metrics['total_flagged']:,}")
    with col2:
        components.kpi_tile("Reviewed so far", f"{metrics['total_reviewed_of_flagged']:,}")
    with col3:
        components.kpi_tile("Review coverage", f"{100 * metrics['review_coverage']:.1f}%")
    with col4:
        components.kpi_tile("Override rate (Dismissed)", f"{100 * metrics['override_rate']:.1f}%")

    st.divider()
    st.subheader("Explainability completeness check")
    violations = result.governance_violations
    if violations:
        st.error(f"{len(violations)} flagged transaction(s) have no recorded reason -- explainability invariant violated.")
        st.write(violations[:20])
    else:
        st.success("Every flagged transaction has at least one recorded reason. Explainability invariant holds.")

    st.divider()
    st.subheader("Evaluation vs. ground truth")
    st.caption(
        "Detections are compared against a ground-truth label file written at data-generation "
        "time and never read by the detection logic itself, so these numbers reflect actual "
        "detection quality rather than a self-consistent score."
    )
    for name, eval_result in result.eval_results.items():
        d = eval_result.as_dict()
        st.markdown(f"**{name}** -- precision `{d['precision']}` / recall `{d['recall']}` / F1 `{d['f1']}` "
                    f"(TP={d['true_positives']}, FP={d['false_positives']}, FN={d['false_negatives']})")
        if not eval_result.by_type.empty:
            st.dataframe(eval_result.by_type, use_container_width=True, height=150)

    st.divider()
    st.subheader("Human review log")
    reviews = audit_store.get_review_log()
    st.dataframe(reviews, use_container_width=True, height=220)

    st.subheader("AI decision log (flagged transactions only)")
    decisions = audit_store.get_ai_decision_log()
    st.dataframe(decisions, use_container_width=True, height=220)
