"""Small reusable dashboard widgets shared across tabs."""
import plotly.graph_objects as go
import streamlit as st

from app import palette

_STATUS_ICON = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}


def kpi_tile(label: str, value: str, help_text: str = None):
    st.metric(label, value, help=help_text)


def risk_badge(bucket: str) -> str:
    icon = _STATUS_ICON.get(bucket, "")
    return f"{icon} {bucket}"


def _base_layout(fig: go.Figure, title: str, height: int = 320):
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=palette.GRIDLINE, gridwidth=1)
    return fig


def status_bar_chart(labels: list, values: list, color_map: dict, title: str):
    """Bar chart where each bar's identity IS a status -- color + visible label
    together, never color alone (icon/label baked into the categorical axis)."""
    colors = [color_map.get(lbl, palette.MUTED_INK) for lbl in labels]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, text=values, textposition="outside"))
    return _base_layout(fig, title)


def ranked_bar_chart(labels: list, values: list, title: str, horizontal: bool = True):
    """Single-hue bar chart for plain magnitude comparison across categories
    (a ranking, not an identity comparison) -- one sequential hue, per the
    dataviz color-formula rule."""
    if horizontal:
        fig = go.Figure(go.Bar(y=labels, x=values, orientation="h", marker_color=palette.SEQUENTIAL_BLUE,
                                text=values, textposition="outside"))
        fig.update_yaxes(autorange="reversed")
    else:
        fig = go.Figure(go.Bar(x=labels, y=values, marker_color=palette.SEQUENTIAL_BLUE,
                                text=values, textposition="outside"))
    return _base_layout(fig, title, height=max(320, 32 * len(labels)))


def render_chart(fig: go.Figure):
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def governance_banner():
    st.info(
        "**AI recommends -- a human decides.** Every risk score and recommendation on this "
        "dashboard is a suggestion for review, not an automated action. All data shown is "
        "synthetically generated for this demo; no real customer data is used.",
        icon="ℹ️",
    )
