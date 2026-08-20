import streamlit as st

from app import components
from src.governance import audit_store


def render(result):
    risk_df = result.risk_df

    st.caption("Filter, drill into a transaction, and record a human review decision. "
               "The AI only recommends -- your decision is what gets logged as final.")

    col1, col2, col3 = st.columns(3)
    with col1:
        bucket_filter = st.multiselect("Risk bucket", ["Low", "Medium", "High"], default=["High"])
    with col2:
        status_filter = st.multiselect("Account status", sorted(risk_df["account_status"].unique().tolist()))
    with col3:
        min_score = st.slider("Minimum risk score", 0, 100, 0)

    filtered = risk_df.copy()
    if bucket_filter:
        filtered = filtered[filtered["risk_bucket"].isin(bucket_filter)]
    if status_filter:
        filtered = filtered[filtered["account_status"].isin(status_filter)]
    filtered = filtered[filtered["risk_score"] >= min_score]
    filtered = filtered.sort_values("risk_score", ascending=False)

    st.write(f"**{len(filtered):,}** transactions match the current filters.")

    display_cols = ["transaction_id", "account_id", "customer_id", "amount", "risk_score", "risk_bucket", "account_status"]
    st.dataframe(filtered[display_cols].head(500), use_container_width=True, height=320)

    st.divider()
    st.subheader("Review a transaction")

    options = filtered["transaction_id"].head(500).tolist()
    if not options:
        st.warning("No transactions match the current filters.")
        return

    selected_id = st.selectbox("Transaction ID", options)
    row = filtered[filtered["transaction_id"] == selected_id].iloc[0]

    left, right = st.columns([2, 1])
    with left:
        st.markdown(f"**Risk:** {components.risk_badge(row['risk_bucket'])} -- {row['risk_score']:.0f}/100")
        st.markdown(f"**Insight:** {row['insight']}")
        st.markdown(f"**{row['recommendation']}**")
        with st.expander("Transaction details"):
            st.json({
                "transaction_id": row["transaction_id"],
                "account_id": row["account_id"],
                "customer_id": row["customer_id"],
                "amount": float(row["amount"]),
                "txn_type": row["txn_type"],
                "channel": row["channel"],
                "timestamp": str(row["timestamp"]),
                "account_status": row["account_status"],
                "customer_kyc_risk": bool(row["customer_kyc_risk"]),
            })

    with right:
        with st.form(key=f"review_form_{selected_id}"):
            decision = st.radio("Decision", ["Escalate", "Approve", "Dismiss"], horizontal=False)
            reviewer_name = st.text_input("Reviewer name", value="")
            note = st.text_area("Note", value="", height=80)
            submitted = st.form_submit_button("Submit review")
            if submitted:
                audit_store.record_review(selected_id, decision, reviewer_name or "Unspecified", note)
                st.success(f"Recorded '{decision}' for {selected_id}.")
                st.rerun()

        past_reviews = audit_store.get_review_log()
        past_reviews = past_reviews[past_reviews["transaction_id"] == selected_id]
        if not past_reviews.empty:
            st.caption("Review history for this transaction:")
            st.dataframe(past_reviews[["decision", "reviewer_name", "note", "reviewed_at"]], use_container_width=True)
