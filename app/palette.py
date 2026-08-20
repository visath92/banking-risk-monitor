"""Validated color palette (see the dataviz skill's references/palette.md).
Status colors are reserved for risk/severity/health states; the sequential
blue ramp is for plain magnitude bars; the categorical order is fixed and
only used when chart identity (not magnitude) is the point.
"""

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

CATEGORICAL = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]

SEQUENTIAL_BLUE = "#2a78d6"

RISK_BUCKET_COLOR = {"Low": STATUS["good"], "Medium": STATUS["warning"], "High": STATUS["critical"]}
SEVERITY_COLOR = {"Low": STATUS["good"], "Medium": STATUS["warning"], "High": STATUS["serious"], "Critical": STATUS["critical"]}

MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
